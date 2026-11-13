import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_xavier(module: nn.Module) -> None:
    """線形層の重みを Xavier 一様分布で初期化し，バイアスを 0 で初期化する関数．"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DeepVIB(nn.Module):
    """
    敵対的頑健性評価 (Section 4.2) 用の Deep VIB モデル (K=256).
    12個の事後分布サンプル Z を用いて予測確率分布 p(y|x) = 1/S sum_s q(y|z^s) を計算し，
    一貫した勾配および評価を提供するメソッド predict_log_probs を備える．
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 1024, latent_dim: int = 256, num_classes: int = 10, sigma_bias: float = -5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sigma_bias = sigma_bias

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Linear(latent_dim, num_classes)
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
        return self.decoder(z)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma, num_samples=num_samples)
        logits = self.decode(z)
        return logits, mu, sigma

    def predict_log_probs(self, x: torch.Tensor, num_samples: int = 12) -> torch.Tensor:
        """
        論文 Section 4.2.3 に従い，S=12 個の事後サンプルを用いて
        周辺予測分布 p(y|x) = 1/S sum_{s=1}^S q(y|z^s) の対数確率 log p(y|x) を返す．
        float32 での Softmax 飽和 (勾配消失) を防ぐため倍精度 (float64) で logsumexp を計算する．
        """
        logits, _, _ = self.forward(x, num_samples=num_samples)
        log_q = F.log_softmax(logits.double(), dim=-1)
        log_p = torch.logsumexp(log_q, dim=0) - math.log(float(num_samples))
        return log_p.float()


class DeterministicMLP(nn.Module):
    """
    比較用の決定論的モデル (Baseline および Dropout モデル, Section 4.2.3).
    DeepVIB と同じ predict_log_probs インターフェースを持つ．
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

    def predict_log_probs(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.log_softmax(logits.double(), dim=-1).float()
