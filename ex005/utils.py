import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


LN2 = math.log(2.0)


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
    """完全共分散ガウス事後分布 p(z|x) = N(mu, L L^T) と r(z) = N(0, I_K) の解析的KL情報量 (bits)．"""
    k_dim = mu.size(-1)
    tr_sigma = chol_l.pow(2).sum(dim=(-2, -1))
    mu_norm_sq = mu.pow(2).sum(dim=-1)
    diag_l = torch.diagonal(chol_l, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.log(diag_l).sum(dim=-1)
    kl_nats = 0.5 * (tr_sigma + mu_norm_sq - float(k_dim) - log_det_sigma)
    return kl_nats / LN2


def vib_regression_2d_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    chol_l: torch.Tensor,
    beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2次元完全共分散ガウス Deep VIB (回帰タスク) の目的関数 J_IB^reg = MSE + beta * KL."""
    targets_float = targets.float().unsqueeze(0).expand_as(preds)
    mse_loss = F.mse_loss(preds, targets_float, reduction="mean")
    kl_bits = compute_kl_full_cov_gaussian(mu, chol_l).mean()
    total_loss = mse_loss + beta * kl_bits
    return total_loss, mse_loss, kl_bits


def train_vib_regression_2d_fast(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float = 1e-2,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    desc: str = "Train 2D Reg VIB"
) -> nn.Module:
    """CUDA Graph を用いて 2次元回帰 Deep VIB モデルを高速学習し EMA モデルを返す関数．"""
    device = train_x.device
    init_params = [p.detach().clone() for p in model.parameters()]

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_tensor, betas=(0.5, 0.999), capturable=True, foreach=True
    )
    ema = PolyakEMA(model, decay=ema_decay)

    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx]
        preds, mu, chol_l = model(xb, num_samples=1)
        loss, _, _ = vib_regression_2d_loss(preds, yb, mu, chol_l, beta=beta)
        loss.backward()
        optimizer.step()
        ema.update()

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

    pbar = tqdm(range(1, epochs + 1), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()
        if epoch % lr_decay_every == 0:
            lr_tensor.mul_(lr_decay_factor)

    return ema.ema_model


def train_deterministic_regression_fast(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    desc: str = "Train Det Reg MLP"
) -> nn.Module:
    """CUDA Graph を用いて決定論的回帰ベースラインモデルを高速学習し EMA モデルを返す関数．"""
    device = train_x.device
    init_params = [p.detach().clone() for p in model.parameters()]

    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_tensor, betas=(0.5, 0.999), capturable=True, foreach=True
    )
    ema = PolyakEMA(model, decay=ema_decay)

    n_train = train_x.size(0)
    num_batches = n_train // batch_size
    static_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        xb = train_x[static_idx]
        yb = train_y[static_idx].float()
        preds = model(xb)
        loss = F.mse_loss(preds, yb, reduction="mean")
        loss.backward()
        optimizer.step()
        ema.update()

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

    pbar = tqdm(range(1, epochs + 1), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device).view(num_batches, batch_size)
        for b in range(num_batches):
            static_idx.copy_(perm[b])
            graph.replay()
        if epoch % lr_decay_every == 0:
            lr_tensor.mul_(lr_decay_factor)

    return ema.ema_model


@torch.no_grad()
def analyze_orthogonal_decomposition(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    beta: float,
    mc_samples: int = 12
) -> dict[str, float]:
    """
    【定理1・定理2・定理3の検証用】
    線形デコーダ fd(z) = w^T z + b の重みベクトル w in R^2 に基づき，
    正規直交基底 (u_parallel, u_perp) を構成して事後分布 p(z|x) = N(mu(x), Sigma(x)) を直交分解する．
      - u_parallel = w / ||w||_2 (予測勾配方向)
      - u_perp     = [-u_parallel[1], u_parallel[0]]^T (等予測値直線方向 / Null Space)
    各方向の平均二乗 E[mu_parallel^2], E[mu_perp^2]，事後分散 E[sigma_parallel^2], E[sigma_perp^2]，
    理論的KKT最適分散 beta_tilde / (||w||_2^2 + beta_tilde) (ただし beta_tilde = beta / (2 ln 2))，
    および各方向のKL情報量 I_parallel, I_perp を精密に計測する．
    """
    model.eval()
    mu, chol_l = model.encode(x_data)
    sigma_mat = torch.matmul(chol_l, chol_l.transpose(-1, -2))  # (N, 2, 2)

    w = model.decoder.weight.detach().view(2)  # (2,)
    w_norm_sq = float(torch.sum(w * w).item())
    w_norm = math.sqrt(max(w_norm_sq, 1e-12))
    u_par = w / w_norm
    u_perp = torch.tensor([-u_par[1].item(), u_par[0].item()], device=x_data.device, dtype=torch.float32)

    # 事後平均の射影: mu_par = u_par^T mu, mu_perp = u_perp^T mu
    mu_par = torch.mv(mu, u_par)    # (N,)
    mu_perp = torch.mv(mu, u_perp)  # (N,)

    # 共分散行列の射影: sigma_par^2 = u_par^T Sigma u_par, sigma_perp^2 = u_perp^T Sigma u_perp
    sigma_par_sq = torch.einsum("i,nij,j->n", u_par, sigma_mat, u_par)
    sigma_perp_sq = torch.einsum("i,nij,j->n", u_perp, sigma_mat, u_perp)
    sigma_cross = torch.einsum("i,nij,j->n", u_par, sigma_mat, u_perp)

    # 各方向の1次元ガウスKL情報量 (bits)
    kl_par_bits = 0.5 * (mu_par.pow(2) + sigma_par_sq - 1.0 - torch.log(sigma_par_sq.clamp(min=1e-12))) / LN2
    kl_perp_bits = 0.5 * (mu_perp.pow(2) + sigma_perp_sq - 1.0 - torch.log(sigma_perp_sq.clamp(min=1e-12))) / LN2
    kl_total_bits = compute_kl_full_cov_gaussian(mu, chol_l)

    # 定理2のKKT理論予測値: sigma_par_theory^2 = (beta / (2 ln 2)) / (||w||_2^2 + beta / (2 ln 2))
    beta_tilde = beta / (2.0 * LN2)
    sigma_par_sq_theory = beta_tilde / (w_norm_sq + beta_tilde)

    # MSEの計測 (1-shot と 12-MC と Mean-mode)
    preds_mc, _, _ = model(x_data, num_samples=mc_samples)
    yf = y_data.float()
    mse_1shot = F.mse_loss(preds_mc[0], yf).item()
    mse_mc = F.mse_loss(preds_mc.mean(dim=0), yf).item()
    pred_mean = model.decode(mu)
    mse_mean_mode = F.mse_loss(pred_mean, yf).item()
    # 解析的期待MSE: E[(y - f_d(mu))^2 + ||w||^2 sigma_par^2]
    mse_analytic = float(((yf - pred_mean).pow(2) + w_norm_sq * sigma_par_sq).mean().item())

    # PC1寄与率
    mu_np = mu.cpu().numpy()
    cov_mu = np.cov(mu_np - mu_np.mean(axis=0, keepdims=True), rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov_mu))[::-1]
    pc1_ratio = float(eigvals[0] / max(float(np.sum(eigvals)), 1e-12))

    return {
        "beta": beta,
        "w_norm_sq": w_norm_sq,
        "mu_par_sq_mean": float(mu_par.pow(2).mean().item()),
        "mu_perp_sq_mean": float(mu_perp.pow(2).mean().item()),
        "sigma_par_sq_mean": float(sigma_par_sq.mean().item()),
        "sigma_par_sq_theory": sigma_par_sq_theory,
        "sigma_perp_sq_mean": float(sigma_perp_sq.mean().item()),
        "sigma_cross_abs_mean": float(sigma_cross.abs().mean().item()),
        "kl_par_bits": float(kl_par_bits.mean().item()),
        "kl_perp_bits": float(kl_perp_bits.mean().item()),
        "kl_total_bits": float(kl_total_bits.mean().item()),
        "mse_1shot": mse_1shot,
        "mse_mc": mse_mc,
        "mse_mean_mode": mse_mean_mode,
        "mse_analytic": mse_analytic,
        "pc1_ratio": pc1_ratio,
    }


def evaluate_input_jacobian_and_robustness(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    num_eval: int = 1000,
    fgs_epsilons: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    noise_sigmas: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8]
) -> dict:
    """
    【定理4 (Appendix C 回帰版拡張) の検証用】
    1. 入力に対する予測値 y_hat(x) = f_d(mu(x)) のヤコビアンノルム (局所リプシッツ定数) ||grad_x y_hat(x)||_2 と
       最大勾配ノルム ||grad_x y_hat(x)||_1 (L_infty 摂動感度) を自動微分により厳密計算する．
    2. 回帰タスクに対する Fast Gradient Sign (FGS) 敵対的摂動 x_adv = clip(x + eps * sign(grad_x (y_hat(x) - y)^2), -1, 1)
       および等方性ガウス入力ノイズ x_noisy = clip(x + N(0, sigma_noise^2 I), -1, 1) に対する RMSE を評価する．
    """
    model.eval()
    xb = x_data[:num_eval].detach().clone().requires_grad_(True)
    yb = y_data[:num_eval].float()

    y_hat = model.predict_mean_mode(xb)
    grad_outputs = torch.ones_like(y_hat)
    grad_x_pred = torch.autograd.grad(y_hat, xb, grad_outputs=grad_outputs, create_graph=False)[0]  # (N, 784)

    # L2 ノルム (L2摂動に対する局所リプシッツ定数) と L1 ノルム (L_inf摂動に対する局所リプシッツ定数)
    jac_l2 = grad_x_pred.norm(p=2, dim=-1).mean().item()
    jac_l1 = grad_x_pred.norm(p=1, dim=-1).mean().item()

    # FGS 敵対的攻撃: 予測誤差 0.5 * (y_hat(x) - y)^2 を最大化する方向 sign((y_hat - y) * grad_x y_hat)
    err_sign = torch.sign(y_hat.detach() - yb).unsqueeze(-1)
    # もし y_hat == y なら符号を +1 とする
    err_sign = torch.where(err_sign == 0, torch.ones_like(err_sign), err_sign)
    adv_dir = torch.sign(err_sign * grad_x_pred.detach())

    fgs_rmse = {}
    with torch.no_grad():
        x_clean = x_data[:num_eval]
        for eps in fgs_epsilons:
            x_adv = torch.clamp(x_clean + eps * adv_dir, -1.0, 1.0)
            if hasattr(model, "reparameterize"):
                preds_adv, _, _ = model(x_adv, num_samples=12)
                pred_adv = preds_adv.mean(dim=0)
            else:
                pred_adv = model(x_adv)
            fgs_rmse[eps] = math.sqrt(F.mse_loss(pred_adv, yb).item())

        noise_rmse = {}
        torch.manual_seed(42)
        base_noise = torch.randn_like(x_clean)
        for sig in noise_sigmas:
            x_noisy = torch.clamp(x_clean + sig * base_noise, -1.0, 1.0)
            if hasattr(model, "reparameterize"):
                preds_n, _, _ = model(x_noisy, num_samples=12)
                pred_n = preds_n.mean(dim=0)
            else:
                pred_n = model(x_noisy)
            noise_rmse[sig] = math.sqrt(F.mse_loss(pred_n, yb).item())

    return {
        "jac_l2_mean": jac_l2,
        "jac_l1_mean": jac_l1,
        "fgs_rmse": fgs_rmse,
        "noise_rmse": noise_rmse,
    }
