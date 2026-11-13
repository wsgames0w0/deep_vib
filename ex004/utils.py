import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm


LN2 = math.log(2.0)
LOG2_10 = math.log2(10.0)
CHI2_2D_95_RADIUS = math.sqrt(-2.0 * math.log(1.0 - 0.95))


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定し再現性を確保する関数．"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def load_mnist_gpu(data_dir: str = "../datasets", device: torch.device = torch.device("cuda")) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNISTデータセットを [-1, 1] にスケーリングしてGPUメモリ上に読み込む関数．"""
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)

    train_x = (train_ds.data.float().view(-1, 784) / 255.0) * 2.0 - 1.0
    train_y = train_ds.targets.long()
    test_x = (test_ds.data.float().view(-1, 784) / 255.0) * 2.0 - 1.0
    test_y = test_ds.targets.long()

    return (
        train_x.to(device, non_blocking=True),
        train_y.to(device, non_blocking=True),
        test_x.to(device, non_blocking=True),
        test_y.to(device, non_blocking=True),
    )


class PolyakEMA:
    """Polyak Averaging (指数移動平均, decay=0.999) を管理するクラス．"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_params = list(self.ema_model.parameters())
        self.model_params = list(model.parameters())

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self.ema_params, self.model_params, 1.0 - self.decay)


def compute_kl_full_cov_gaussian(mu: torch.Tensor, chol_l: torch.Tensor) -> torch.Tensor:
    """
    完全共分散ガウス事後分布 p(z|x) = N(mu, L L^T) と標準正規事前分布 r(z) = N(0, I_K) の
    解析的KL情報量を bits 単位で計算する関数．
    """
    k_dim = mu.size(-1)
    tr_sigma = chol_l.pow(2).sum(dim=(-2, -1))
    mu_norm_sq = mu.pow(2).sum(dim=-1)
    diag_l = torch.diagonal(chol_l, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.log(diag_l).sum(dim=-1)

    kl_nats = 0.5 * (tr_sigma + mu_norm_sq - float(k_dim) - log_det_sigma)
    return kl_nats / LN2


def compute_kl_diag_gaussian(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """対角ガウス事後分布 p(z|x) = N(mu, diag(sigma^2)) と r(z) = N(0, I_K) の解析的KL情報量 (bits)．"""
    var = sigma.pow(2)
    kl_nats = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - torch.log(var), dim=-1)
    return kl_nats / LN2


def vib_regression_2d_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    chol_or_sigma: torch.Tensor,
    beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deep VIB (回帰タスク) の目的関数 J_IB^reg を計算する関数．
    変分デコーダをガウス分布 q(y|z) = N(y | fd(z), sigma_y^2) (sigma_y^2 = 1 / (2 ln 2)) とおくと，
    負の対数尤度 -log_2 q(y|z) は定数項を除いて平均二乗誤差 (MSE) E[(y - fd(z))^2] に一致する．
      J_IB^reg = E[(y - fd(z))^2] + beta * KL[p(Z|x) || r(Z)]
    """
    # preds: (num_samples, batch_size), targets: (batch_size,)
    targets_float = targets.float().unsqueeze(0).expand_as(preds)
    mse_loss = F.mse_loss(preds, targets_float, reduction="mean")
    if chol_or_sigma.ndim == 3:
        kl_bits = compute_kl_full_cov_gaussian(mu, chol_or_sigma).mean()
    else:
        kl_bits = compute_kl_diag_gaussian(mu, chol_or_sigma).mean()
    total_loss = mse_loss + beta * kl_bits
    return total_loss, mse_loss, kl_bits


def vib_classification_2d_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    chol_l: torch.Tensor,
    beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2次元完全共分散ガウス Deep VIB (分類タスク) の目的関数 J_IB を計算する関数．"""
    num_samples, batch_size, num_classes = logits.shape
    targets_expanded = targets.unsqueeze(0).expand(num_samples, batch_size)
    ce_bits = F.cross_entropy(logits.reshape(-1, num_classes), targets_expanded.reshape(-1), reduction="mean") / LN2
    kl_bits = compute_kl_full_cov_gaussian(mu, chol_l).mean()
    total_loss = ce_bits + beta * kl_bits
    return total_loss, ce_bits, kl_bits


@torch.no_grad()
def compute_latent_geometry_metrics(model: nn.Module, x_data: torch.Tensor, y_data: torch.Tensor) -> dict[str, float]:
    """
    2次元潜在空間 Z における事後平均 mu(x) の幾何学的・統計的構造を定量化する関数:
      - pc1_ratio: 事後平均 mu(x) の共分散行列における第1主成分寄与率 lambda_1 / (lambda_1 + lambda_2)
                   (1.0 に近いほど潜在表現が 1次元の直線多様体上に縮退していることを示す)
      - spearman_corr: 各数字 (0..9) のクラス重心を第1主成分軸へ射影した座標と，
                       実際の数値 (0..9) とのスピアマン順位相関係数の絶対値 |rho|
                       (1.0 に近いほど 0 -> 1 -> ... -> 9 の順序通りに並んでいることを示す)
    """
    model.eval()
    mu, _ = model.encode(x_data)
    mu_np = mu.cpu().numpy()  # (N, 2)
    y_np = y_data.cpu().numpy()

    # 中心化して共分散行列の固有値を計算
    mu_centered = mu_np - mu_np.mean(axis=0, keepdims=True)
    cov = np.cov(mu_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 降順にソート
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    pc1_vec = eigvecs[:, order[0]]  # (2,)

    total_var = float(np.sum(eigvals))
    pc1_ratio = float(eigvals[0] / max(total_var, 1e-12))

    # 各クラス 0..9 の重心を第1主成分方向へ射影
    centroids_1d = np.zeros(10, dtype=np.float64)
    for c in range(10):
        centroids_1d[c] = float(np.dot(mu_np[y_np == c].mean(axis=0), pc1_vec))

    # スピアマン順位相関係数 |rho|
    ranks = np.argsort(np.argsort(centroids_1d)).astype(np.float64)
    true_ranks = np.arange(10, dtype=np.float64)
    diff_sq = np.sum((ranks - true_ranks) ** 2)
    spearman = 1.0 - (6.0 * diff_sq) / (10.0 * (100.0 - 1.0))

    return {
        "pc1_ratio": pc1_ratio,
        "abs_spearman": abs(float(spearman)),
    }


@torch.no_grad()
def evaluate_vib_regression_2d(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 2000,
    mc_samples: int = 12
) -> dict[str, float]:
    """
    2次元 Deep VIB 回帰モデルを評価する関数:
      - mse_1shot, rmse_1shot: 1サンプル推論時の MSE / RMSE
      - mse_mc, rmse_mc: 12サンプル MC 平均推論時の MSE / RMSE
      - mae_mc: 12サンプル MC 平均推論時の平均絶対誤差 (MAE)
      - r2_mc: 決定係数 R^2
      - err_round_1shot, err_round_mc: 予測値を最近傍整数 clip(round(y_hat), 0, 9) に丸めた際の誤分類率
      - izx_bits: I(Z, X) 変分上限 (bits)
      - izy_bits: ガウス変分下限 0.5 * log_2(Var(Y) / MSE_mc) (bits)
    """
    model.eval()
    n_total = x_data.size(0)
    y_float_all = y_data.float()
    var_y = torch.var(y_float_all, unbiased=False).item()

    se_1shot_sum = torch.zeros(1, dtype=torch.float32, device=x_data.device)
    se_mc_sum = torch.zeros(1, dtype=torch.float32, device=x_data.device)
    ae_mc_sum = torch.zeros(1, dtype=torch.float32, device=x_data.device)
    correct_round_1shot = torch.zeros(1, dtype=torch.long, device=x_data.device)
    correct_round_mc = torch.zeros(1, dtype=torch.long, device=x_data.device)
    total_kl_bits = torch.zeros(1, dtype=torch.float32, device=x_data.device)

    for i in range(0, n_total, batch_size):
        xb = x_data[i:i + batch_size]
        yb = y_data[i:i + batch_size]
        yb_f = yb.float()

        preds_mc_samples, mu, chol_or_sigma = model(xb, num_samples=mc_samples)
        pred_1shot = preds_mc_samples[0]
        pred_mc = preds_mc_samples.mean(dim=0)

        se_1shot_sum += (pred_1shot - yb_f).pow(2).sum()
        se_mc_sum += (pred_mc - yb_f).pow(2).sum()
        ae_mc_sum += (pred_mc - yb_f).abs().sum()

        round_1shot = torch.clamp(torch.round(pred_1shot), 0.0, 9.0).long()
        round_mc = torch.clamp(torch.round(pred_mc), 0.0, 9.0).long()
        correct_round_1shot += (round_1shot == yb).sum()
        correct_round_mc += (round_mc == yb).sum()

        if chol_or_sigma.ndim == 3:
            total_kl_bits += compute_kl_full_cov_gaussian(mu, chol_or_sigma).sum()
        else:
            total_kl_bits += compute_kl_diag_gaussian(mu, chol_or_sigma).sum()

    mse_1shot = se_1shot_sum.item() / n_total
    mse_mc = se_mc_sum.item() / n_total
    rmse_1shot = math.sqrt(mse_1shot)
    rmse_mc = math.sqrt(mse_mc)
    mae_mc = ae_mc_sum.item() / n_total
    r2_mc = 1.0 - (mse_mc / max(var_y, 1e-12))
    err_round_1shot = 1.0 - (correct_round_1shot.item() / n_total)
    err_round_mc = 1.0 - (correct_round_mc.item() / n_total)
    izx_bits = total_kl_bits.item() / n_total
    izy_bits = max(0.0, 0.5 * math.log2(max(var_y, 1e-12) / max(mse_mc, 1e-12)))

    if getattr(model, "latent_dim", 2) == 2:
        geom = compute_latent_geometry_metrics(model, x_data, y_data)
    else:
        geom = {"pc1_ratio": float("nan"), "abs_spearman": float("nan")}

    return {
        "mse_1shot": mse_1shot,
        "rmse_1shot": rmse_1shot,
        "mse_mc": mse_mc,
        "rmse_mc": rmse_mc,
        "mae_mc": mae_mc,
        "r2_mc": r2_mc,
        "err_round_1shot": err_round_1shot,
        "err_round_mc": err_round_mc,
        "izx_bits": izx_bits,
        "izy_bits": izy_bits,
        "pc1_ratio": geom["pc1_ratio"],
        "abs_spearman": geom["abs_spearman"],
    }


@torch.no_grad()
def evaluate_vib_classification_2d(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 2000,
    mc_samples: int = 12
) -> dict[str, float]:
    """比較用の 2次元 Deep VIB 分類モデルを評価する関数．"""
    model.eval()
    n_total = x_data.size(0)
    correct_1shot = torch.zeros(1, dtype=torch.long, device=x_data.device)
    correct_mc = torch.zeros(1, dtype=torch.long, device=x_data.device)
    total_kl_bits = torch.zeros(1, dtype=torch.float32, device=x_data.device)
    total_ce_bits = torch.zeros(1, dtype=torch.float32, device=x_data.device)

    for i in range(0, n_total, batch_size):
        xb = x_data[i:i + batch_size]
        yb = y_data[i:i + batch_size]
        b_len = xb.size(0)

        logits_mc, mu, chol_l = model(xb, num_samples=mc_samples)
        preds_1shot = logits_mc[0].argmax(dim=-1)
        correct_1shot += (preds_1shot == yb).sum()

        probs_mc = F.softmax(logits_mc, dim=-1).mean(dim=0)
        preds_mc = probs_mc.argmax(dim=-1)
        correct_mc += (preds_mc == yb).sum()

        total_kl_bits += compute_kl_full_cov_gaussian(mu, chol_l).sum()
        log_probs_mc = torch.log2(probs_mc.clamp(min=1e-12))
        total_ce_bits += -log_probs_mc[torch.arange(b_len, device=xb.device), yb].sum()

    err_1shot = 1.0 - (correct_1shot.item() / n_total)
    err_mc = 1.0 - (correct_mc.item() / n_total)
    izx_bits = total_kl_bits.item() / n_total
    ce_bits = total_ce_bits.item() / n_total
    izy_bits = LOG2_10 - ce_bits
    if getattr(model, "latent_dim", 2) == 2:
        geom = compute_latent_geometry_metrics(model, x_data, y_data)
    else:
        geom = {"pc1_ratio": float("nan"), "abs_spearman": float("nan")}

    return {
        "err_1shot": err_1shot,
        "err_mc": err_mc,
        "izx_bits": izx_bits,
        "izy_bits": izy_bits,
        "pc1_ratio": geom["pc1_ratio"],
        "abs_spearman": geom["abs_spearman"],
    }


def train_vib_regression_2d(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    beta: float = 1e-2,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    record_every: int = 25,
    desc: str = "Train 2D Reg VIB"
) -> tuple[nn.Module, list[dict]]:
    """2次元完全共分散ガウス Deep VIB (回帰タスク) を CUDA Graph により高速学習する関数．"""
    device = train_x.device
    init_params = [p.detach().clone() for p in model.parameters()]

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_tensor,
        betas=(0.5, 0.999),
        capturable=True,
        foreach=True
    )
    ema = PolyakEMA(model, decay=ema_decay)

    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    static_loss = torch.zeros(1, dtype=torch.float32, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx]
        preds, mu, chol_l = model(xb, num_samples=1)
        loss, _, _ = vib_regression_2d_loss(preds, yb, mu, chol_l, beta=beta)
        loss.backward()
        optimizer.step()
        ema.update()
        static_loss.copy_(loss.detach())

    torch.cuda.synchronize(device)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(s):
        for _ in range(3):
            _step()
    torch.cuda.current_stream(device=device).wait_stream(s)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        _step()
    torch.cuda.synchronize(device)

    with torch.no_grad():
        for p_mod, p_ema, p_init in zip(ema.model_params, ema.ema_params, init_params):
            p_mod.copy_(p_init)
            p_ema.copy_(p_init)
        for state in optimizer.state.values():
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

    history = []
    pbar = tqdm(range(1, epochs + 1), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()

        if epoch % lr_decay_every == 0:
            lr_tensor.mul_(lr_decay_factor)

        if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
            train_m = evaluate_vib_regression_2d(ema.ema_model, train_x, train_y, mc_samples=12)
            test_m = evaluate_vib_regression_2d(ema.ema_model, test_x, test_y, mc_samples=12)
            history.append({
                "epoch": epoch,
                "train_loss": float(static_loss.item()),
                "train_rmse_1shot": train_m["rmse_1shot"],
                "train_rmse_mc": train_m["rmse_mc"],
                "train_err_round_mc": train_m["err_round_mc"],
                "train_izx": train_m["izx_bits"],
                "train_izy": train_m["izy_bits"],
                "test_rmse_1shot": test_m["rmse_1shot"],
                "test_rmse_mc": test_m["rmse_mc"],
                "test_mae_mc": test_m["mae_mc"],
                "test_r2_mc": test_m["r2_mc"],
                "test_err_round_1shot": test_m["err_round_1shot"],
                "test_err_round_mc": test_m["err_round_mc"],
                "test_izx": test_m["izx_bits"],
                "test_izy": test_m["izy_bits"],
                "test_pc1_ratio": test_m["pc1_ratio"],
                "test_abs_spearman": test_m["abs_spearman"],
            })
            pbar.set_postfix({
                "RMSE": f"{test_m['rmse_mc']:.3f}",
                "I(Z,X)": f"{test_m['izx_bits']:.1f}b"
            })

    return ema.ema_model, history


def train_vib_classification_2d(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    beta: float = 1e-2,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    record_every: int = 50,
    desc: str = "Train 2D Cls VIB"
) -> tuple[nn.Module, list[dict]]:
    """比較用の 2次元完全共分散ガウス Deep VIB (分類タスク) を CUDA Graph により高速学習する関数．"""
    device = train_x.device
    init_params = [p.detach().clone() for p in model.parameters()]

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_tensor,
        betas=(0.5, 0.999),
        capturable=True,
        foreach=True
    )
    ema = PolyakEMA(model, decay=ema_decay)

    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    static_loss = torch.zeros(1, dtype=torch.float32, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx]
        logits, mu, chol_l = model(xb, num_samples=1)
        loss, _, _ = vib_classification_2d_loss(logits, yb, mu, chol_l, beta=beta)
        loss.backward()
        optimizer.step()
        ema.update()
        static_loss.copy_(loss.detach())

    torch.cuda.synchronize(device)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(s):
        for _ in range(3):
            _step()
    torch.cuda.current_stream(device=device).wait_stream(s)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        _step()
    torch.cuda.synchronize(device)

    with torch.no_grad():
        for p_mod, p_ema, p_init in zip(ema.model_params, ema.ema_params, init_params):
            p_mod.copy_(p_init)
            p_ema.copy_(p_init)
        for state in optimizer.state.values():
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

    history = []
    pbar = tqdm(range(1, epochs + 1), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()

        if epoch % lr_decay_every == 0:
            lr_tensor.mul_(lr_decay_factor)

        if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
            test_m = evaluate_vib_classification_2d(ema.ema_model, test_x, test_y, mc_samples=12)
            history.append({
                "epoch": epoch,
                "test_err_1shot": test_m["err_1shot"],
                "test_err_mc": test_m["err_mc"],
                "test_izx": test_m["izx_bits"],
                "test_izy": test_m["izy_bits"],
                "test_pc1_ratio": test_m["pc1_ratio"],
                "test_abs_spearman": test_m["abs_spearman"],
            })
            pbar.set_postfix({
                "err_mc": f"{test_m['err_mc']*100:.2f}%",
                "I(Z,X)": f"{test_m['izx_bits']:.1f}b"
            })

    return ema.ema_model, history


@torch.no_grad()
def plot_latent_scatter_on_ax(
    ax: plt.Axes,
    model: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    task_type: str = "regression",
    plot_mode: str = "mean_scatter",
    num_images: int = 2000,
    grid_res: int = 180,
    title: str = ""
) -> None:
    """
    2次元潜在空間 Z (K=2) の散布図・信頼楕円・背景予測構造を描画する汎用関数:
      - plot_mode == "mean_scatter":
          テスト画像の事後平均 mu(x) = (mu_1, mu_2) の散布図を描画し，
          各数字 0..9 のクラス重心 (セントロイド) を番号付きマーカーでプロットして
          0 -> 1 -> 2 -> ... -> 9 の順に黒破線で結ぶ．
      - plot_mode == "sample_and_ellipse":
          確率的サンプル z ~ p(z|x) の散布点と 95% 信頼楕円を重ねて描画し，
          背景には分類タスクなら分類器エントロピー H(q(Y|z))，
          回帰タスクならデコーダ予測値 y_hat(z) in [0, 9] の等高線・グラデーションを表示する．
    """
    model.eval()
    device = next(model.parameters()).device
    x_sub = test_x[:num_images]
    y_sub = test_y[:num_images].cpu().numpy()

    mu, chol_l = model.encode(x_sub)
    z_samp = model.reparameterize(mu, chol_l, num_samples=1).squeeze(0)
    mu_np = mu.cpu().numpy()        # (N, 2)
    z_np = z_samp.cpu().numpy()     # (N, 2)
    chol_np = chol_l.cpu().numpy()  # (N, 2, 2)

    theta = np.linspace(0.0, 2.0 * np.pi, 50)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ellipses = mu_np[:, None, :] + CHI2_2D_95_RADIUS * np.einsum("nij,jk->nki", chol_np, unit_circle)

    if plot_mode == "mean_scatter":
        max_coord = max(float(np.percentile(np.abs(mu_np), 99.5)) * 1.25, 2.0)
    else:
        max_coord = max(float(np.percentile(np.abs(ellipses), 99.5)) * 1.15, 2.5)
    lim = round(max_coord, 1)

    # 背景フィールドの描画
    gx = np.linspace(-lim, lim, grid_res)
    gy = np.linspace(-lim, lim, grid_res)
    xx, yy = np.meshgrid(gx, gy)
    grid_pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32, device=device)

    if task_type == "classification":
        logits_grid = model.decode(grid_pts)
        probs_grid = F.softmax(logits_grid, dim=-1)
        log2_probs = torch.log2(probs_grid.clamp(min=1e-12))
        entropy_grid = -torch.sum(probs_grid * log2_probs, dim=-1).view(grid_res, grid_res).cpu().numpy()
        ax.imshow(
            entropy_grid,
            extent=[-lim, lim, -lim, lim],
            origin="lower",
            cmap="Greys",
            vmin=0.0,
            vmax=LOG2_10 * 1.15,
            alpha=0.55 if plot_mode == "sample_and_ellipse" else 0.35,
            aspect="equal"
        )
    else:
        pred_grid = model.decode(grid_pts).view(grid_res, grid_res).cpu().numpy()
        ax.imshow(
            pred_grid,
            extent=[-lim, lim, -lim, lim],
            origin="lower",
            cmap="Greys",
            vmin=-1.0,
            vmax=10.0,
            alpha=0.25,
            aspect="equal"
        )
        # 予測値 y_hat(z) = 0, 1, ..., 9 の等高線を描画
        cs = ax.contour(
            xx, yy, pred_grid,
            levels=np.arange(0, 10, 1),
            colors="dimgray",
            linestyles=":",
            linewidths=0.9,
            alpha=0.65
        )
        ax.clabel(cs, inline=True, fontsize=7.5, fmt="%d")

    cmap = plt.get_cmap("tab10")

    if plot_mode == "mean_scatter":
        centroids = np.zeros((10, 2), dtype=np.float64)
        for c in range(10):
            mask = (y_sub == c)
            pts = mu_np[mask]
            centroids[c] = pts.mean(axis=0)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[cmap(c)],
                s=12,
                alpha=0.65,
                edgecolors="none",
                label=f"{c}"
            )
        # クラス重心を 0 -> 1 -> ... -> 9 の順に破線で結ぶ
        ax.plot(centroids[:, 0], centroids[:, 1], "k--", lw=1.6, alpha=0.85, zorder=8)
        for c in range(10):
            ax.scatter(
                centroids[c, 0], centroids[c, 1],
                c=[cmap(c)],
                s=130,
                edgecolors="black",
                linewidths=1.2,
                zorder=9
            )
            ax.text(
                centroids[c, 0], centroids[c, 1],
                str(c),
                color="white",
                fontsize=8.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10
            )
    else:
        # サンプル点 z ~ p(z|x) の散布図 + 95%信頼楕円 (先頭 800 サンプル)
        n_ell = min(num_images, 800)
        for c in range(10):
            mask = (y_sub[:n_ell] == c)
            if not np.any(mask):
                continue
            pts_z = z_np[:n_ell][mask]
            ax.scatter(pts_z[:, 0], pts_z[:, 1], c=[cmap(c)], s=8, alpha=0.5, edgecolors="none", label=f"{c}")
            segs = ellipses[:n_ell][mask]
            lc = LineCollection(segs, colors=[cmap(c)], linewidths=0.75, alpha=0.45)
            ax.add_collection(lc)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$z_1$", fontsize=10)
    ax.set_ylabel("$z_2$", fontsize=10)
    ax.set_title(title, fontsize=10.5)
