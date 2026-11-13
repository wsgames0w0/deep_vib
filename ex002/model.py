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


class DeepVIB2DFullCov(nn.Module):
    """
    2次元潜在空間 (K=2) + 完全共分散ガウス分布 (Full Covariance Gaussian) を用いた Deep VIB モデル．
    論文 Section 4.1.2 および Appendix A の設定に完全準拠:
      - エンコーダ fe: 784 -> 1024 -> 1024 -> 6 (2 + 4 = 6 出力)
      - 最初の 2 出力: 平均ベクトル mu in R^2
      - 残りの 4 出力: 2x2 行列に整形し，対角成分は softplus(x - 5.0) 変換，
        非対角(下三角)成分は 10^{-2} 倍し，上三角成分は 0 として
        共分散行列 Sigma = L L^T のコレスキー分解下三角行列 L を構成する．
      - デコーダ fd: 2 -> 10 の線形層 (変分ロジスティック回帰分類器 q(y|z))
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
        assert latent_dim == 2, "DeepVIB2DFullCov is designed for K=2 (2D visualization)."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sigma_bias = sigma_bias
        self.offdiag_scale = offdiag_scale

        # 2 + 4 = 6 出力
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
        """
        入力 x から 2次元ガウス事後分布 p(z|x) = N(mu, L L^T) の
        平均 mu (B, 2) とコレスキー因子 L (B, 2, 2) を計算する．
        """
        x = x.view(x.size(0), -1)
        out = self.encoder(x)
        mu = out[:, :2]
        raw_mat = out[:, 2:].view(-1, 2, 2)

        # 下三角コレスキー行列 L の構築 (Appendix A)
        l00 = F.softplus(raw_mat[:, 0, 0] + self.sigma_bias) + 1e-7
        l11 = F.softplus(raw_mat[:, 1, 1] + self.sigma_bias) + 1e-7
        l10 = raw_mat[:, 1, 0] * self.offdiag_scale
        l01 = torch.zeros_like(l00)

        row0 = torch.stack([l00, l01], dim=-1)
        row1 = torch.stack([l10, l11], dim=-1)
        chol_l = torch.stack([row0, row1], dim=-2)  # (B, 2, 2)
        return mu, chol_l

    def reparameterize(self, mu: torch.Tensor, chol_l: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        コレスキー因子 L を用いた Reparameterization Trick:
          z = mu + L @ eps,  eps ~ N(0, I_2)
        返り値の形状: (num_samples, B, 2)
        """
        batch_size = mu.size(0)
        eps = torch.randn(num_samples, batch_size, 2, 1, device=mu.device, dtype=mu.dtype)
        # chol_l: (1, B, 2, 2) @ eps: (S, B, 2, 1) -> (S, B, 2, 1) -> (S, B, 2)
        scaled_eps = torch.matmul(chol_l.unsqueeze(0), eps).squeeze(-1)
        return mu.unsqueeze(0) + scaled_eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数 z からロジット Wz + b を計算する．"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, chol_l = self.encode(x)
        z = self.reparameterize(mu, chol_l, num_samples=num_samples)
        logits = self.decode(z)
        return logits, mu, chol_l
