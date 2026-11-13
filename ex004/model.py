import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_xavier(module: nn.Module) -> None:
    """
    論文 Appendix A に従い，すべての線形層の重みを Xavier 一様分布で初期化し，
    バイアスを 0 で初期化する関数．
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DeepVIBRegression2DFullCov(nn.Module):
    """
    MNIST 数字回帰タスク ([0, 9] の連続数値予測) 用の 2次元完全共分散ガウス Deep VIB モデル (K=2).
    エンコーダは ex002 と全く同一の 784 -> 1024 -> 1024 -> 6 (平均 mu in R^2 + コレスキー因子 L in R^{2x2}) を用い，
    デコーダ fd(z) は潜在変数 z in R^2 からスカラー予測値 y_hat in R を出力する．
      - decoder_type == "linear": 単一線形層 fd(z) = w^T z + b (2 -> 1)
      - decoder_type == "mlp": 2層非線形デコーダ (2 -> 64 -> 1, ReLU)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 1024,
        latent_dim: int = 2,
        sigma_bias: float = -5.0,
        offdiag_scale: float = 1e-2,
        decoder_type: str = "linear"
    ):
        super().__init__()
        assert latent_dim == 2, "DeepVIBRegression2DFullCov is designed for K=2."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sigma_bias = sigma_bias
        self.offdiag_scale = offdiag_scale
        self.decoder_type = decoder_type

        out_dim = latent_dim + latent_dim * latent_dim  # 2 + 4 = 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

        if decoder_type == "linear":
            self.decoder = nn.Linear(latent_dim, 1)
        elif decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        self.apply(init_weights_xavier)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """入力 x から平均 mu (B, 2) と下三角コレスキー因子 L (B, 2, 2) を計算する．"""
        x = x.view(x.size(0), -1)
        out = self.encoder(x)
        mu = out[:, :2]
        raw_mat = out[:, 2:].view(-1, 2, 2)

        l00 = F.softplus(raw_mat[:, 0, 0] + self.sigma_bias) + 1e-7
        l11 = F.softplus(raw_mat[:, 1, 1] + self.sigma_bias) + 1e-7
        l10 = raw_mat[:, 1, 0] * self.offdiag_scale
        l01 = torch.zeros_like(l00)

        row0 = torch.stack([l00, l01], dim=-1)
        row1 = torch.stack([l10, l11], dim=-1)
        chol_l = torch.stack([row0, row1], dim=-2)
        return mu, chol_l

    def reparameterize(self, mu: torch.Tensor, chol_l: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Reparameterization Trick: z = mu + L @ eps, eps ~ N(0, I_2). 形状: (S, B, 2)"""
        batch_size = mu.size(0)
        eps = torch.randn(num_samples, batch_size, 2, 1, device=mu.device, dtype=mu.dtype)
        scaled_eps = torch.matmul(chol_l.unsqueeze(0), eps).squeeze(-1)
        return mu.unsqueeze(0) + scaled_eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数 z (..., 2) から連続予測値 y_hat (...,) を計算する．"""
        return self.decoder(z).squeeze(-1)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, chol_l = self.encode(x)
        z = self.reparameterize(mu, chol_l, num_samples=num_samples)
        preds = self.decode(z)  # (num_samples, B)
        return preds, mu, chol_l


class DeepVIBClassification2DFullCov(nn.Module):
    """
    比較用の 2次元完全共分散ガウス 分類タスク Deep VIB モデル (K=2, 10クラス分類).
    ex002 と同一構造．
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 1024,
        latent_dim: int = 2,
        num_classes: int = 10,
        sigma_bias: float = -5.0,
        offdiag_scale: float = 1e-2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sigma_bias = sigma_bias
        self.offdiag_scale = offdiag_scale

        out_dim = latent_dim + latent_dim * latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.decoder = nn.Linear(latent_dim, num_classes)
        self.apply(init_weights_xavier)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        out = self.encoder(x)
        mu = out[:, :2]
        raw_mat = out[:, 2:].view(-1, 2, 2)

        l00 = F.softplus(raw_mat[:, 0, 0] + self.sigma_bias) + 1e-7
        l11 = F.softplus(raw_mat[:, 1, 1] + self.sigma_bias) + 1e-7
        l10 = raw_mat[:, 1, 0] * self.offdiag_scale
        l01 = torch.zeros_like(l00)

        row0 = torch.stack([l00, l01], dim=-1)
        row1 = torch.stack([l10, l11], dim=-1)
        chol_l = torch.stack([row0, row1], dim=-2)
        return mu, chol_l

    def reparameterize(self, mu: torch.Tensor, chol_l: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        batch_size = mu.size(0)
        eps = torch.randn(num_samples, batch_size, 2, 1, device=mu.device, dtype=mu.dtype)
        scaled_eps = torch.matmul(chol_l.unsqueeze(0), eps).squeeze(-1)
        return mu.unsqueeze(0) + scaled_eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, chol_l = self.encode(x)
        z = self.reparameterize(mu, chol_l, num_samples=num_samples)
        logits = self.decode(z)
        return logits, mu, chol_l


class DeepVIBRegressionDiag(nn.Module):
    """
    高次元ボトルネック (K=256) と 2次元ボトルネック (K=2) の回帰性能を比較するための
    対角ガウス分布 Deep VIB 回帰モデル．
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 1024, latent_dim: int = 256, sigma_bias: float = -5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sigma_bias = sigma_bias

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Linear(latent_dim, 1)
        self.apply(init_weights_xavier)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        stats = self.encoder(x)
        mu = stats[:, :self.latent_dim]
        raw_sigma = stats[:, self.latent_dim:]
        sigma = F.softplus(raw_sigma + self.sigma_bias) + 1e-7
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).squeeze(-1)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma, num_samples=num_samples)
        preds = self.decode(z)
        return preds, mu, sigma
