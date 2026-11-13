import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm


LN2 = math.log(2.0)


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定する関数．"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def load_mnist_gpu(data_dir: str = "../datasets", device: torch.device = torch.device("cuda")) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNISTデータセットを [-1, 1] にスケーリングしてGPU上に読み込む関数．"""
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


def compute_kl_diag_gaussian(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    var = sigma.pow(2)
    kl_nats = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - torch.log(var), dim=-1)
    return kl_nats / LN2


def train_vib(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    desc: str = "Training VIB"
) -> nn.Module:
    """Deep VIB モデルを CUDA Graph により高速学習し，Polyak EMA モデルを返す関数．"""
    device = train_x.device
    init_state = copy.deepcopy(model.state_dict())

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_tensor, betas=(0.5, 0.999), capturable=True, foreach=True)
    ema = PolyakEMA(model, decay=0.999)
    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx]
        logits, mu, sigma = model(xb, num_samples=1)
        ce_bits = F.cross_entropy(logits.squeeze(0), yb) / LN2
        kl_bits = compute_kl_diag_gaussian(mu, sigma).mean()
        loss = ce_bits + beta * kl_bits
        loss.backward()
        optimizer.step()
        ema.update()

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

    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=True):
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()

        if epoch % 2 == 0:
            lr_tensor.mul_(0.97)

    return ema.ema_model


def train_deterministic(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    desc: str = "Training Det"
) -> nn.Module:
    """決定論的モデル (Baseline / Dropout) を CUDA Graph により高速学習し，Polyak EMA モデルを返す関数．"""
    device = train_x.device
    init_state = copy.deepcopy(model.state_dict())

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_tensor, betas=(0.5, 0.999), capturable=True, foreach=True)
    ema = PolyakEMA(model, decay=0.999)
    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx]
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        ema.update()

    init_params = [p.detach().clone() for p in model.parameters()]

    torch.cuda.synchronize(device)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device=device))
    model.train()
    with torch.cuda.stream(s):
        for _ in range(3):
            _step()
    torch.cuda.current_stream(device=device).wait_stream(s)
    torch.cuda.synchronize(device)

    model.train()
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

    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=True):
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()

        if epoch % 2 == 0:
            lr_tensor.mul_(0.97)

    return ema.ema_model


def fgs_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.35,
    mc_samples: int = 12
) -> torch.Tensor:
    """
    Fast Gradient Sign (FGS) 攻撃 (Goodfellow et al., 2015; 論文 Section 4.2.1, Figure 5).
    倍精度 (float64) で対数周辺確率 log p(y|x) を計算して数値的勾配消失を防ぎ，
    入力空間 [-1, 1] において摂動幅 epsilon の単一ステップ勾配符号摂動を加える．
    """
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    if hasattr(model, "reparameterize"):
        logits, _, _ = model(x_adv, num_samples=mc_samples)
        log_q = F.log_softmax(logits.double(), dim=-1)
        log_probs = torch.logsumexp(log_q, dim=0) - math.log(float(mc_samples))
    else:
        logits, _ = model(x_adv)
        log_probs = F.log_softmax(logits.double(), dim=-1)

    loss = F.nll_loss(log_probs, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = x_adv.grad.detach().sign().float()
    x_perturbed = torch.clamp(x.detach() + epsilon * grad_sign, -1.0, 1.0)
    return x_perturbed


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mc_samples: int = 12) -> float:
    """指定されたデータに対する分類正解率 (Accuracy) を計算する関数．"""
    model.eval()
    log_probs = model.predict_log_probs(x, num_samples=mc_samples)
    preds = log_probs.argmax(dim=-1)
    return (preds == y).float().mean().item()


def cw_l2_attack(
    model: nn.Module,
    x: torch.Tensor,
    y_true_or_target: torch.Tensor,
    targeted: bool = True,
    c_weight: float = 10.0,
    kappa: float = 0.0,
    steps: int = 150,
    lr: float = 0.05,
    mc_samples: int = 12,
    l0_threshold: float = 0.05
) -> dict[str, torch.Tensor]:
    """
    Carlini & Wagner (2016) の L2 最適化敵対的攻撃 (論文 Section 4.2.1, Figures 3, 4, 6).
    tanh 空間での変数変換 x' = tanh(w) in (-1, 1) を用いてボックス制約を厳密に満たしながら，
    [0, 1] スケールでの L2 ノルム ||x'_{[0,1]} - x_{[0,1]}||_2^2 + c * f(x') を最小化する．

    返り値 (辞書):
      - x_adv: 生成された敵対的サンプル (B, 784) in [-1, 1]
      - preds: 敵対的サンプルに対する予測クラス (B,)
      - success: 攻撃成功フラグ (B,) (targeted なら preds == target, untargeted なら preds != y_true)
      - l0: 各サンプルの L0 ノルム ([0, 1] スケールで l0_threshold 以上変化した画素数) (B,)
      - l2: 各サンプルの L2 ノルム ([0, 1] スケールでのユークリッド距離) (B,)
      - l_inf: 各サンプルの L_inf ノルム ([0, 1] スケールでの最大画素変化量) (B,)
    """
    model.eval()
    device = x.device
    batch_size = x.size(0)

    # tanh の逆変換 atanh のために [-0.99999, 0.99999] にクリップ
    x_clamped = torch.clamp(x.detach(), -0.99999, 0.99999)
    w_init = torch.atanh(x_clamped)
    w = w_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    # [0, 1] スケールでの元画像
    x_01 = (x.detach() + 1.0) * 0.5

    best_adv = x.detach().clone()
    best_l2 = torch.full((batch_size,), 1e9, device=device)
    best_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    one_hot = F.one_hot(y_true_or_target, num_classes=10).bool()

    c_candidates = [0.2, 0.5, 1.0, 2.5, 6.0, 15.0] if isinstance(c_weight, (int, float)) and c_weight >= 10.0 else [c_weight]

    for c_val in c_candidates:
        w = w_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x_adv_curr = torch.tanh(w)
            x_adv_01 = (x_adv_curr + 1.0) * 0.5

            # [0, 1] スケールでの L2 二乗距離 (微小な画素変化を抑制するため L1 正則化も微少併用)
            diff_01 = x_adv_01 - x_01
            l2_sq = torch.sum(diff_01.pow(2), dim=-1)
            l2_dist = torch.sqrt(l2_sq + 1e-12)

            log_probs = model.predict_log_probs(x_adv_curr, num_samples=mc_samples)
            target_logit = log_probs[one_hot]
            other_logit = log_probs.masked_fill(one_hot, -1e9).max(dim=-1).values

            if targeted:
                f_loss = torch.clamp(other_logit - target_logit + kappa, min=0.0)
                is_succ = (log_probs.argmax(dim=-1) == y_true_or_target)
            else:
                f_loss = torch.clamp(target_logit - other_logit + kappa, min=0.0)
                is_succ = (log_probs.argmax(dim=-1) != y_true_or_target)

            loss = torch.sum(l2_sq + c_val * f_loss)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                improved = is_succ & (l2_dist < best_l2)
                if improved.any():
                    best_l2[improved] = l2_dist[improved]
                    best_adv[improved] = x_adv_curr.detach()[improved]
                    best_found[improved] = True

    # 一度も攻撃が成功しなかったサンプルは最終ステップの x_adv_curr を格納
    with torch.no_grad():
        final_adv = torch.tanh(w.detach())
        not_found = ~best_found
        if not_found.any():
            best_adv[not_found] = final_adv[not_found]

        final_log_probs = model.predict_log_probs(best_adv, num_samples=mc_samples)
        final_preds = final_log_probs.argmax(dim=-1)
        if targeted:
            final_success = (final_preds == y_true_or_target)
        else:
            final_success = (final_preds != y_true_or_target)

        best_adv_01 = (best_adv + 1.0) * 0.5
        abs_diff = torch.abs(best_adv_01 - x_01)
        l0_norm = (abs_diff > l0_threshold).float().sum(dim=-1)
        l2_norm = torch.sqrt(torch.sum(abs_diff.pow(2), dim=-1) + 1e-12)
        linf_norm = abs_diff.max(dim=-1).values

    return {
        "x_adv": best_adv,
        "preds": final_preds,
        "success": final_success,
        "found": best_found,
        "l0": l0_norm,
        "l2": l2_norm,
        "l_inf": linf_norm,
    }


def plot_zero_to_one_grid(
    orig_images: torch.Tensor,
    adv_dict: dict[str, dict[str, torch.Tensor]],
    save_path: str,
    num_rows: int = 3
) -> None:
    """
    論文 Figure 3 の再現可視化関数:
      「0」の画像に対してターゲットラベル「1」への L2 最適化攻撃を行った結果を表示する．
      - 背景枠が赤 (Red): 攻撃成功 (予測ラベル == 1)
      - 背景枠が緑 (Green): 攻撃失敗・防御成功 (予測ラベル == 0)
      - 背景枠が紫 (Purple): 0 でも 1 でもない別クラスに誤分類
    """
    col_names = ["Orig."] + list(adv_dict.keys())
    num_cols = len(col_names)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.65 * num_cols, 1.75 * num_rows))
    orig_01 = ((orig_images[:num_rows].detach().cpu().numpy() + 1.0) * 0.5).reshape(num_rows, 28, 28)

    for r in range(num_rows):
        for c, col_name in enumerate(col_names):
            ax = axes[r, c] if num_rows > 1 else axes[c]
            if c == 0:
                img = orig_01[r]
                border_color = "#2ca02c"  # Green (Original class 0)
            else:
                res = adv_dict[col_name]
                img = ((res["x_adv"][r].detach().cpu().numpy() + 1.0) * 0.5).reshape(28, 28)
                pred_label = int(res["preds"][r].item())
                if pred_label == 1:
                    border_color = "#d62728"  # Red: ターゲット「1」への攻撃成功
                elif pred_label == 0:
                    border_color = "#2ca02c"  # Green: 攻撃失敗 (元の「0」を維持)
                else:
                    border_color = "#9467bd"  # Purple: 0, 1 以外のラベルに変化

            # 論文 Figure 3 と同様に色付きの背景枠の中に 28x28 のグレースケール画像を配置
            padded = np.zeros((34, 34, 3), dtype=np.float32)
            # border_color hex -> RGB
            rgb = [int(border_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
            padded[:, :] = rgb
            padded[3:31, 3:31, :] = np.stack([img, img, img], axis=-1)

            ax.imshow(padded, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_name, fontsize=10, pad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
