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
# 2次元ガウス分布の95%信頼領域 (カイ二乗分布自由度2の上側5%点: sqrt(-2 * ln(0.05)) = sqrt(5.99146) ≈ 2.447747)
CHI2_2D_95_RADIUS = math.sqrt(-2.0 * math.log(1.0 - 0.95))


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定する関数．"""
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
    完全共分散ガウス分布 p(z|x) = N(mu, Sigma) (ただし Sigma = L L^T, L は下三角コレスキー因子) と
    標準正規事前分布 r(z) = N(0, I_K) の間の解析的KL情報量を bits 単位で計算する関数．

    数式:
      KL(N(mu, L L^T) || N(0, I_K))
        = 0.5 * (tr(L L^T) + ||mu||_2^2 - K - ln det(L L^T))
        = 0.5 * (||L||_F^2 + ||mu||_2^2 - K - 2 * sum_{k=1}^K ln L_{kk}) [nats]
    """
    k_dim = mu.size(-1)
    tr_sigma = chol_l.pow(2).sum(dim=(-2, -1))
    mu_norm_sq = mu.pow(2).sum(dim=-1)
    diag_l = torch.diagonal(chol_l, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.log(diag_l).sum(dim=-1)

    kl_nats = 0.5 * (tr_sigma + mu_norm_sq - float(k_dim) - log_det_sigma)
    return kl_nats / LN2


def vib_2d_loss_function(logits: torch.Tensor, targets: torch.Tensor, mu: torch.Tensor, chol_l: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2次元完全共分散ガウス Deep VIB の損失関数 J_IB を計算する関数．"""
    num_samples, batch_size, num_classes = logits.shape
    targets_expanded = targets.unsqueeze(0).expand(num_samples, batch_size)

    ce_nats = F.cross_entropy(
        logits.reshape(-1, num_classes),
        targets_expanded.reshape(-1),
        reduction="mean"
    )
    class_loss_bits = ce_nats / LN2
    kl_bits = compute_kl_full_cov_gaussian(mu, chol_l).mean()
    total_loss = class_loss_bits + beta * kl_bits
    return total_loss, class_loss_bits, kl_bits


@torch.no_grad()
def evaluate_vib_2d(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 1000,
    mc_samples: int = 12
) -> dict[str, float]:
    """2次元 Deep VIB モデルの誤分類率 (1-shot, 12-sample MC) と相互情報量を評価する関数．"""
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

    return {
        "err_1shot": err_1shot,
        "err_mc": err_mc,
        "izx_bits": izx_bits,
        "izy_bits": izy_bits,
        "ce_bits": ce_bits,
    }


def train_vib_2d_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    beta: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    record_every: int = 20,
    desc: str = "Training 2D VIB"
) -> tuple[nn.Module, list[dict]]:
    """2次元完全共分散ガウス Deep VIB モデルを CUDA Graph により高速学習する関数．"""
    device = train_x.device
    init_state = copy.deepcopy(model.state_dict())

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
        loss, _, _ = vib_2d_loss_function(logits, yb, mu, chol_l, beta=beta)
        loss.backward()
        optimizer.step()
        ema.update()
        static_loss.copy_(loss.detach())

    init_params = [p.detach().clone() for p in model.parameters()]

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
            train_metrics = evaluate_vib_2d(ema.ema_model, train_x, train_y, mc_samples=12)
            test_metrics = evaluate_vib_2d(ema.ema_model, test_x, test_y, mc_samples=12)
            history.append({
                "epoch": epoch,
                "train_loss": float(static_loss.item()),
                "train_err_1shot": train_metrics["err_1shot"],
                "train_err_mc": train_metrics["err_mc"],
                "train_izx": train_metrics["izx_bits"],
                "train_izy": train_metrics["izy_bits"],
                "test_err_1shot": test_metrics["err_1shot"],
                "test_err_mc": test_metrics["err_mc"],
                "test_izx": test_metrics["izx_bits"],
                "test_izy": test_metrics["izy_bits"],
            })
            pbar.set_postfix({
                "test_mc": f"{test_metrics['err_mc']*100:.2f}%",
                "I(Z,X)": f"{test_metrics['izx_bits']:.1f}b"
            })

    return ema.ema_model, history


@torch.no_grad()
def plot_2d_gaussian_ellipses_on_ax(
    ax: plt.Axes,
    model: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_images: int = 1000,
    grid_res: int = 200,
    title: str = ""
) -> None:
    """
    論文 Figure 2 の再現描画関数:
      - テストセットの先頭 1000 枚の画像について，事後分布 p(z|x) = N(mu, Sigma) の
        95%信頼楕円をクラスラベルごとに色分けして描画する．
      - 背景には 2次元グリッド上の各点 z における変分分類器 q(y|z) のエントロピー
        H(q(Y|z)) = -sum_y q(y|z) log_2 q(y|z) をグレースケールで表示する．
    """
    model.eval()
    device = next(model.parameters()).device
    x_sub = test_x[:num_images]
    y_sub = test_y[:num_images].cpu().numpy()

    mu, chol_l = model.encode(x_sub)
    mu_np = mu.cpu().numpy()        # (N, 2)
    chol_np = chol_l.cpu().numpy()  # (N, 2, 2)

    # 楕円上の角度パラメータ theta in [0, 2*pi]
    theta = np.linspace(0.0, 2.0 * np.pi, 60)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, 60)

    # 各サンプルの95%信頼楕円: z(theta) = mu + r_95 * L @ [cos(theta), sin(theta)]^T
    # chol_np: (N, 2, 2), unit_circle: (2, 60) -> (N, 60, 2)
    ellipses = mu_np[:, None, :] + CHI2_2D_95_RADIUS * np.einsum("nij,jk->nki", chol_np, unit_circle)

    # 表示範囲の自動決定 (外れ値に引きずられすぎないよう99.5パーセンタイル+マージンを対称に設定)
    max_coord = max(float(np.percentile(np.abs(ellipses), 99.5)) * 1.15, 2.5)
    lim = round(max_coord, 1)

    # 背景の変分分類器エントロピー H(q(Y|z)) の計算
    gx = np.linspace(-lim, lim, grid_res)
    gy = np.linspace(-lim, lim, grid_res)
    xx, yy = np.meshgrid(gx, gy)
    grid_pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32, device=device)

    logits_grid = model.decode(grid_pts)
    probs_grid = F.softmax(logits_grid, dim=-1)
    log2_probs_grid = torch.log2(probs_grid.clamp(min=1e-12))
    entropy_grid = -torch.sum(probs_grid * log2_probs_grid, dim=-1).view(grid_res, grid_res).cpu().numpy()

    # 背景グレースケールプロット (エントロピーが高いクラス境界ほど濃いグレー)
    ax.imshow(
        entropy_grid,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=LOG2_10 * 1.15,
        alpha=0.65,
        aspect="equal"
    )

    # 10クラスのカラーマップ
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        mask = (y_sub == c)
        if not np.any(mask):
            continue
        segs = ellipses[mask]  # (N_c, 60, 2)
        lc = LineCollection(segs, colors=[cmap(c)], linewidths=0.85, alpha=0.55, label=f"Class {c}")
        ax.add_collection(lc)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(title, fontsize=11)
