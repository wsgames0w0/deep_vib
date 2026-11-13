# Deep Variational Information Bottleneck (Deep VIB) 再現実験プロジェクト

本リポジトリは，Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy による論文 **"Deep Variational Information Bottleneck"**（ICLR 2017, arXiv:1612.00410v7）の PyTorch（GPU / CUDA）による再現実装および実験レポートをまとめたプロジェクトです．

---

## 1. プロジェクト概要

情報ボトルネック（Information Bottleneck; IB）原理は，入力 $X$ からターゲット $Y$ を予測する際，予測に必要な相互情報量 $I(Z, Y)$ を最大化しつつ，入力自体に関する相互情報量 $I(Z, X)$ を最小化（圧縮）する確率的潜在表現 $Z$ を求める枠組みです．

$$
R_{\mathrm{IB}}(\theta) = I(Z, Y; \theta) - \beta I(Z, X; \theta)
$$

**Deep VIB** は，変分推論（変分分類器 $q(y|z)$ と変分周辺事前分布 $r(z) = \mathcal{N}(0, I)$）および Reparameterization Trick を用いることで，以下の変分目的関数 $J_{\mathrm{IB}}$ をディープニューラルネットワークによって直接最小化します．

$$
J_{\mathrm{IB}} = \frac{1}{N} \sum_{n=1}^N \left( \mathbb{E}_{\epsilon \sim p(\epsilon)} \left[ -\mathrm{log} \, q(y_n \mid f(x_n, \epsilon)) \right] + \beta \, \mathrm{KL}[p(Z|x_n) \,\|\, r(Z)] \right)
$$

本プロジェクトでは，論文中のすべての主要実験および追加発展実験を以下の4つの実験ディレクトリに分けて実装・検証しています．

1. **[`ex001/`](./ex001/)（高次元潜在表現 $K=256$ と正則化性能の評価）**:
   - 順列不変 MNIST における決定論的ベースライン，Dropout，Confidence Penalty，Label Smoothing と Deep VIB のテスト誤分類率比較（論文 Table 1 再現）．
   - トレードオフ係数 $\beta$ のスイープによる 1-shot 評価 / 12サンプル MC 平均評価の誤分類率推移，情報ボトルネック曲線 $I(Z, Y)$ vs $I(Z, X)$，および情報量上限 $I(Z, X)$ の検証（論文 Figure 1(a)(c)(d) 再現）．
2. **[`ex002/`](./ex002/)（2次元潜在表現 $K=2$ と完全共分散ガウス分布による潜在空間可視化）**:
   - コレスキー分解 $\Sigma(x) = L(x)L(x)^\top$ を用いた完全共分散ガウスエンコーダの実装と解析的 KL 情報量の導出・評価（論文 Figure 1(b) 再現）．
   - テスト画像 $1,000$ 枚の事後分布 $p(z|x)$ の 95% 信頼楕円と，変分分類器 $q(y|z)$ のシャノンエントロピー背景図の可視化（論文 Figure 2 再現）．
3. **[`ex003/`](./ex003/)（敵対的サンプルに対する頑健性評価）**:
   - Carlini & Wagner (2016) の Targeted $L_2$ 最適化攻撃による「$0 \to 1$」摂動実験と，摂動ノルム（$L\_0, L\_2, L\_\infty$）の相対変化の検証（論文 Figure 3, Figure 4 再現）．
   - Fast Gradient Sign（FGS）攻撃（$\epsilon \in \{0.35, 0.40, 0.45, 0.50\}$）に対する相対正解率の評価（論文 Figure 5 再現）．
   - 全クラスに対する Targeted / Untargeted $L_2$ 最適化攻撃に対する分類頑健性の評価（論文 Figure 6 再現）．
4. **[`ex004/`](./ex004/)（【追加実験】MNIST 数字回帰タスク $y \in [0, 9]$ と $K=2$ 潜在空間 $Z$ の散布図比較）**:
   - MNIST 画像から写っている数字を連続数値 $y \in [0, 9]$ として出力する回帰タスクへの Deep VIB の拡張（ガウス変分デコーダによる MSE + KL 最小化）．
   - $K=2$ 完全共分散ガウス潜在空間 $Z$ における **分類タスク（2次元放射状配置）** と **回帰タスク（$0 \to 1 \to \dots \to 9$ の単調 1次元多様体への縮退および針状異方性信頼楕円）** の散布図・幾何学指標（PC1寄与率・Spearman順位相関）による徹底比較．

---

## 2. ディレクトリ構成

```
.
├─ex001/                  # 実験1: K=256 対角ガウス Deep VIB と正則化手法の比較
│  ├─model.py             # DeepVIB (K=256) および DeterministicMLP モデルの定義
│  ├─utils.py             # 学習 (CUDA Graph + Polyak EMA)・評価・KL情報量計算関数
│  └─ex001.ipynb          # 実験1の実行ノートブック
│
├─ex002/                  # 実験2: K=2 完全共分散ガウス Deep VIB と2次元潜在空間可視化
│  ├─model.py             # DeepVIB2DFullCov (コレスキー分解エンコーダ) の定義
│  ├─utils.py             # 完全共分散解析的KL計算・学習・95%信頼楕円描画関数
│  └─ex002.ipynb          # 実験2の実行ノートブック
│
├─ex003/                  # 実験3: FGS 攻撃および C&W L2 最適化攻撃に対する敵対的頑健性評価
│  ├─model.py             # 倍精度対数周辺確率推論を備えた DeepVIB / DeterministicMLP
│  ├─utils.py             # FGS 攻撃・C&W L2 最適化攻撃・可視化関数
│  └─ex003.ipynb          # 実験3の実行ノートブック
│
├─ex004/                  # 実験4: MNIST 数字回帰タスク ([0, 9]) と K=2 潜在空間 Z の散布図比較
│  ├─model.py             # 回帰用・分類用 2次元完全共分散 Deep VIB および高次元回帰モデルの定義
│  ├─utils.py             # 回帰 VIB 損失・幾何学指標 (PC1寄与率/Spearman相関)・潜在空間散布図描画関数
│  └─ex004.ipynb          # 実験4の実行ノートブック
│
├─report/                 # 実験レポートおよび結果図表
│  ├─img/                 # レポートで使用するプロット画像
│  ├─ex001.md             # 実験1の詳細レポート
│  ├─ex002.md             # 実験2の詳細レポート
│  ├─ex003.md             # 実験3の詳細レポート
│  ├─ex004.md             # 実験4の詳細レポート（回帰タスクと K=2 潜在空間散布図比較）
│  └─report.md            # 本プロジェクト全体の総合技術レポート
│
├─requirements.txt        # Python 依存ライブラリ一覧
├─LICENSE                 # MIT License
└─README.md               # 本ファイル
```

---

## 3. 各種レポートへのリンク

- **総合レポート**: **[`report/report.md`](./report/report.md)**（Deep VIB の理論導出，**8ステップに分けた学習プロセスの詳細解説**，および全実験の総括）
- **実験1レポート**: **[`report/ex001.md`](./report/ex001.md)**（$K=256$ における正則化比較と IB 曲線）
- **実験2レポート**: **[`report/ex002.md`](./report/ex002.md)**（$K=2$ 完全共分散ガウス分布と 95% 信頼楕円可視化）
- **実験3レポート**: **[`report/ex003.md`](./report/ex003.md)**（FGS 攻撃および C&W $L_2$ 最適化攻撃に対する頑健性）
- **実験4レポート**: **[`report/ex004.md`](./report/ex004.md)**（【追加実験】MNIST 数字回帰タスク $y \in [0, 9]$ と $K=2$ 潜在空間 $Z$ の散布図比較）

---

## 4. 動作環境とセットアップ

- **Python**: 3.10.11
- **PyTorch**: `2.7.1+cu118` (`torchvision 0.22.1+cu118`)
- **主要ライブラリ**: `numpy`, `pandas`, `matplotlib`, `tqdm`

### セットアップ手順

```powershell
# 仮想環境の有効化
.\venv\Scripts\activate

# 依存パッケージのインストール（新規環境の場合）
pip install -r requirements.txt
```

各実験は `ex001/ex001.ipynb`, `ex002/ex002.ipynb`, `ex003/ex003.ipynb`, `ex004/ex004.ipynb` を順番に実行することで再現できます．

---

## 5. ライセンス

本プロジェクトは [MIT License](./LICENSE) のもとで公開されています．
