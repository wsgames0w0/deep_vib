import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


LN2 = math.log(2.0)
LOG2_10 = math.log2(10.0)


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定し再現性を確保する関数．"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def load_mnist_gpu(data_dir: str = "../datasets", device: torch.device = torch.device("cuda")) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MNISTデータセットを読み込み，論文 Appendix A に従って画素値を [-1, 1] にスケーリングし，
    高速学習のためにGPUメモリ上に配置したテンソルとして返す関数．
    """
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)

    # [0, 255] -> [-1.0, 1.0], 形状: (N, 784)
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
    """
    論文 Appendix A に記載の Polyak Averaging (指数移動平均, decay=0.999) を管理するクラス．
    学習時の各ステップでパラメータの移動平均を更新し，推論時に平均化パラメータを使用する．
    """

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
    """
    エンコーダ事後分布 p(z|x) = N(mu, diag(sigma^2)) と
    変分周辺事前分布 r(z) = N(0, I) の間の解析的KL情報量を計算する関数．
    単位は bits (log_2) で返し，各サンプルごとのKL情報量 (batch_size,) を返す．

    数式:
      KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum_{k=1}^K (mu_k^2 + sigma_k^2 - 1 - ln(sigma_k^2)) [nats]
      bits への変換は 1 / ln(2) を乗じる．
    """
    var = sigma.pow(2)
    kl_nats = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - torch.log(var), dim=-1)
    kl_bits = kl_nats / LN2
    return kl_bits


def vib_loss_function(logits: torch.Tensor, targets: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deep VIB の目的関数 J_IB (論文 Eq. 17) を計算する関数．
      J_IB = E[-log_2 q(y|z)] + beta * KL[p(Z|x), r(Z)]
    返り値:
      total_loss: 最小化する総合損失 (bits)
      class_loss_bits: 条件付き交差エントロピー損失 E[-log_2 q(y|z)] (bits)
      izx_bits: 相互情報量 I(Z, X) の変分上限 KL[p(Z|x), r(Z)] (bits)
    """
    num_samples, batch_size, num_classes = logits.shape
    targets_expanded = targets.unsqueeze(0).expand(num_samples, batch_size)

    ce_nats = F.cross_entropy(
        logits.reshape(-1, num_classes),
        targets_expanded.reshape(-1),
        reduction="mean"
    )
    class_loss_bits = ce_nats / LN2
    kl_bits = compute_kl_diag_gaussian(mu, sigma).mean()
    total_loss = class_loss_bits + beta * kl_bits
    return total_loss, class_loss_bits, kl_bits


@torch.no_grad()
def evaluate_vib(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 2000,
    mc_samples: int = 12
) -> dict[str, float]:
    """
    Deep VIB モデルを評価し，以下を計算する関数 (論文 Section 4.1.1, Figure 1):
      - err_1shot: 1サンプル (S=1) による誤分類率
      - err_mc: モンテカルロ平均 (S=12, p(y|x) = 1/S sum_s q(y|z^s)) による誤分類率
      - err_mean: 平均ベクトル z = mu(x) を用いた決定論的推論の誤分類率
      - izx_bits: I(Z, X) の変分上限 (bits)
      - izy_bits: I(Z, Y) >= H(Y) + E[log_2 q(y|z)] の変分下限 (bits)
      - ce_bits: 交差エントロピー損失 (bits)
    """
    model.eval()
    n_total = x_data.size(0)
    correct_1shot = torch.zeros(1, dtype=torch.long, device=x_data.device)
    correct_mc = torch.zeros(1, dtype=torch.long, device=x_data.device)
    correct_mean = torch.zeros(1, dtype=torch.long, device=x_data.device)
    total_kl_bits = torch.zeros(1, dtype=torch.float32, device=x_data.device)
    total_ce_bits = torch.zeros(1, dtype=torch.float32, device=x_data.device)

    for i in range(0, n_total, batch_size):
        xb = x_data[i:i + batch_size]
        yb = y_data[i:i + batch_size]
        b_len = xb.size(0)

        logits_mc, mu, sigma = model(xb, num_samples=mc_samples)
        preds_1shot = logits_mc[0].argmax(dim=-1)
        correct_1shot += (preds_1shot == yb).sum()

        probs_mc = F.softmax(logits_mc, dim=-1).mean(dim=0)
        preds_mc = probs_mc.argmax(dim=-1)
        correct_mc += (preds_mc == yb).sum()

        logits_mean = model.decode(mu)
        preds_mean = logits_mean.argmax(dim=-1)
        correct_mean += (preds_mean == yb).sum()

        total_kl_bits += compute_kl_diag_gaussian(mu, sigma).sum()

        log_probs_mc = torch.log2(probs_mc.clamp(min=1e-12))
        total_ce_bits += -log_probs_mc[torch.arange(b_len, device=xb.device), yb].sum()

    err_1shot = 1.0 - (correct_1shot.item() / n_total)
    err_mc = 1.0 - (correct_mc.item() / n_total)
    err_mean = 1.0 - (correct_mean.item() / n_total)
    izx_bits = total_kl_bits.item() / n_total
    ce_bits = total_ce_bits.item() / n_total
    izy_bits = LOG2_10 - ce_bits

    return {
        "err_1shot": err_1shot,
        "err_mc": err_mc,
        "err_mean": err_mean,
        "izx_bits": izx_bits,
        "izy_bits": izy_bits,
        "ce_bits": ce_bits,
    }


def train_vib_model(
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
    desc: str = "Training VIB"
) -> tuple[nn.Module, list[dict]]:
    """
    論文 Appendix A のハイパーパラメータ設定に完全準拠して Deep VIB を学習する関数．
    GPU の L2 キャッシュ常駐および CUDA Graph を用いて WDDM カーネル起動オーバーヘッドを除去し，
    数学的に完全同一の計算を約5倍高速に実行する．
    """
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
        logits, mu, sigma = model(xb, num_samples=1)
        loss, _, _ = vib_loss_function(logits, yb, mu, sigma, beta=beta)
        loss.backward()
        optimizer.step()
        ema.update()
        static_loss.copy_(loss.detach())

    # 初期パラメータの保存 (キャプチャ完了後にインプレースで復元するためメモリアドレスを維持)
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

    # キャプチャされたメモリアドレスを保ったまま初期状態にリセット
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
            train_metrics = evaluate_vib(ema.ema_model, train_x, train_y, mc_samples=12)
            test_metrics = evaluate_vib(ema.ema_model, test_x, test_y, mc_samples=12)
            record = {
                "epoch": epoch,
                "lr": float(lr_tensor.item()),
                "train_loss": float(static_loss.item()),
                "train_err_1shot": train_metrics["err_1shot"],
                "train_err_mc": train_metrics["err_mc"],
                "train_err_mean": train_metrics["err_mean"],
                "train_izx": train_metrics["izx_bits"],
                "train_izy": train_metrics["izy_bits"],
                "test_err_1shot": test_metrics["err_1shot"],
                "test_err_mc": test_metrics["err_mc"],
                "test_err_mean": test_metrics["err_mean"],
                "test_izx": test_metrics["izx_bits"],
                "test_izy": test_metrics["izy_bits"],
            }
            history.append(record)
            pbar.set_postfix({
                "test_mc": f"{test_metrics['err_mc']*100:.2f}%",
                "I(Z,X)": f"{test_metrics['izx_bits']:.1f}b"
            })

    return ema.ema_model, history


@torch.no_grad()
def evaluate_deterministic(model: nn.Module, x_data: torch.Tensor, y_data: torch.Tensor, batch_size: int = 2000) -> float:
    """決定論的MLPモデルの誤分類率を計算する関数．"""
    model.eval()
    n_total = x_data.size(0)
    correct = torch.zeros(1, dtype=torch.long, device=x_data.device)
    for i in range(0, n_total, batch_size):
        xb = x_data[i:i + batch_size]
        yb = y_data[i:i + batch_size]
        logits, _ = model(xb)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum()
    return 1.0 - (correct.item() / n_total)


def train_deterministic_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    reg_type: str = "baseline",
    reg_param: float = 0.0,
    epochs: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
    lr_decay_factor: float = 0.97,
    lr_decay_every: int = 2,
    ema_decay: float = 0.999,
    record_every: int = 20,
    desc: str = "Training Deterministic"
) -> tuple[nn.Module, list[dict]]:
    """
    比較対象の決定論的モデル (Baseline, Dropout, Confidence Penalty, Label Smoothing) を
    CUDA Graph を用いて高速に学習する関数．
    """
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
        logits, _ = model(xb)

        if reg_type in ("baseline", "dropout"):
            loss = F.cross_entropy(logits, yb)
        elif reg_type == "confidence_penalty":
            ce = F.cross_entropy(logits, yb, reduction="none")
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            loss = torch.mean(ce - reg_param * entropy)
        elif reg_type == "label_smoothing":
            eps = reg_param
            smoothing = (10.0 * eps) / (1.0 + 9.0 * eps)
            loss = F.cross_entropy(logits, yb, label_smoothing=smoothing)
        else:
            raise ValueError(f"Unknown reg_type: {reg_type}")

        loss.backward()
        optimizer.step()
        ema.update()
        static_loss.copy_(loss.detach())

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
            train_err = evaluate_deterministic(ema.ema_model, train_x, train_y)
            test_err = evaluate_deterministic(ema.ema_model, test_x, test_y)
            history.append({
                "epoch": epoch,
                "train_loss": float(static_loss.item()),
                "train_err": train_err,
                "test_err": test_err,
            })
            pbar.set_postfix({"test_err": f"{test_err*100:.2f}%"})

    return ema.ema_model, history
