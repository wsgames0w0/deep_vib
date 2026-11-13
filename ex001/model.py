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


class DeepVIB(nn.Module):
    """
    Deep Variational Information Bottleneck (Deep VIB) モデル (対角ガウス分布エンコーダ).
    論文 Section 4.1.1 の設定:
      - エンコーダ fe: 784 -> 1024 -> 1024 -> 2K (ReLU 活性化関数)
      - 最初の K 次元が平均 mu，残りの K 次元が softplus(x - 5.0) を経て標準偏差 sigma となる
      - デコーダ fd: K -> 10 の線形層 (ロジスティック回帰分類器)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 1024, latent_dim: int = 256, num_classes: int = 10, sigma_bias: float = -5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sigma_bias = sigma_bias

        # エンコーダ MLP: 784 -> 1024 -> 1024 -> 2K
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # デコーダ (変分分類器 q(y|z)): K -> 10
        self.decoder = nn.Linear(latent_dim, num_classes)

        # Xavier 初期化の適用
        self.apply(init_weights_xavier)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        入力 x から潜在変数 Z の事後分布 p(z|x) = N(z | mu, diag(sigma^2)) のパラメータを計算する．
        """
        x = x.view(x.size(0), -1)
        stats = self.encoder(x)
        mu = stats[:, :self.latent_dim]
        raw_sigma = stats[:, self.latent_dim:]
        # 論文 Eq.(19): sigma = log(1 + exp(x - 5.0))
        sigma = F.softplus(raw_sigma + self.sigma_bias) + 1e-7
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Reparameterization Trick により z = mu + sigma * epsilon (epsilon ~ N(0, I)) をサンプリングする．
        返り値の形状: (num_samples, batch_size, latent_dim)
        """
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        潜在表現 z からクラスロジット fd(z) = Wz + b を計算する．
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播:
          x -> (mu, sigma) -> z (num_samples 個) -> logits
        返り値:
          logits: (num_samples, batch_size, num_classes) または (batch_size, num_classes) (num_samples=1 の場合も統一可能だが，ここでは (num_samples, B, C))
          mu: (batch_size, latent_dim)
          sigma: (batch_size, latent_dim)
        """
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma, num_samples=num_samples)
        logits = self.decode(z)
        return logits, mu, sigma


class DeterministicMLP(nn.Module):
    """
    比較用の決定論的ベースライン MLP モデル (Section 4.1, Table 1).
    DeepVIB と同一のネットワーク構造を持ち，ガウスサンプリング層を除いて z = mu(x) をそのまま用いるモデル．
    Dropout の有無も指定可能．
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 1024, latent_dim: int = 256, num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ])
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(latent_dim, num_classes)

        self.apply(init_weights_xavier)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits, z
