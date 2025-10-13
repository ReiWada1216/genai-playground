# GenAI Playground

生成AIモデル（VAE、VQ-VAE、Diffusion Model）を比較・実験するためのプレイグラウンドです。

## 🎯 概要

このプロジェクトは、異なる生成AIアーキテクチャの性能を比較し、理解を深めるための包括的なフレームワークを提供します。

## 🏗️ プロジェクト構成

```
genai-playground/
├─ 00_data/                 # CelebAなどの入力データ（GitHubには含まれません）
│   └─ img_align_celeba/
│
├─ 01_src/                  # ソースコード
│   ├─ models/              # 各生成AIの実装
│   │   ├─ model_vae.py     # Variational Autoencoder
│   │   ├─ model_vqvae.py   # Vector Quantized VAE
│   │   └─ model_diffusion.py # Diffusion Model
│   ├─ data/                # データセット処理
│   │   └─ dataset.py       # データセット読み込み
│   └─ utils/               # 共通関数・ユーティリティ
│       ├─ train_utils.py   # トレーニング用ユーティリティ
│       └─ logger.py        # ログ機能
│
├─ 02_app/                  # アプリケーション
│   ├─ main.py              # メインアプリケーション
│   └─ components/          # UIコンポーネント
│       └─ ui_components.py
│
├─ 03_notebooks/            # Jupyter Notebook
│   └─ exp_vae.ipynb        # VAE実験ノートブック
│
├─ 04_scripts/              # トレーニングスクリプト
│   ├─ train_vae.py         # VAEトレーニング
│   ├─ train_vqvqe.py       # VQ-VAEトレーニング
│   └─ train_diffusion.py   # Diffusion Modelトレーニング
│
├─ requirements.txt         # 依存関係
├─ .gitignore              # Git除外設定
└─ README.md               # このファイル
```

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd genai-playground
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. データセットの準備

CelebAデータセットをダウンロードして配置してください：

```bash
# データディレクトリを作成
mkdir -p 00_data/img_align_celeba/img_align_celeba/

# CelebAデータセットをダウンロード
# 1. https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html にアクセス
# 2. "Download" セクションから "Img/img_align_celeba.zip" をダウンロード
# 3. 解凍して 00_data/img_align_celeba/img_align_celeba/ に配置
```

**注意**: CelebAデータセットは約1.3GBのサイズがあるため、GitHubリポジトリには含まれていません。上記の手順で手動でダウンロードしてください。

## 📊 使用方法

### トレーニング

#### VAEのトレーニング
```bash
python 04_scripts/train_vae.py
```

#### VQ-VAEのトレーニング
```bash
python 04_scripts/train_vqvqe.py
```

#### Diffusion Modelのトレーニング
```bash
python 04_scripts/train_diffusion.py
```

### Web UI
```bash
python 02_app/main.py
```

## 🎨 実装されているモデル

### 1. Variational Autoencoder (VAE)
- **特徴**: 確率的潜在変数モデル
- **利点**: 学習が安定、潜在空間の解釈が可能
- **欠点**: 生成品質が限定的

### 2. Vector Quantized VAE (VQ-VAE)
- **特徴**: 離散的な潜在表現を使用
- **利点**: 高品質な生成、スケーラブル
- **欠点**: 学習が複雑、計算コストが高い

### 3. Diffusion Model
- **特徴**: ノイズを段階的に除去して画像を生成
- **利点**: 最高品質の生成、安定した学習
- **欠点**: 生成に時間がかかる、計算コストが非常に高い

## ⚙️ 設定

実験設定は `experiments/2025-10-11_baseline/config.yaml` で管理されています。

主要な設定項目：
- **dataset**: データセット設定
- **models**: 各モデルのパラメータ
- **training**: トレーニング設定
- **evaluation**: 評価設定
- **output**: 出力設定

## 📈 評価指標

- **MSE**: 平均二乗誤差
- **PSNR**: ピーク信号対雑音比
- **SSIM**: 構造類似性指数
- **FID**: Fréchet Inception Distance

## 🔧 カスタマイズ

### 新しいモデルの追加
1. `models/` ディレクトリに新しいモデルファイルを作成
2. `train.py` にモデルロジックを追加
3. `config.yaml` にモデル設定を追加

### 新しいデータセットの追加
1. `utils/dataset.py` に新しいデータセットクラスを追加
2. データローダー関数を実装

### UIのカスタマイズ
1. `ui/components/ui_components.py` でUIコンポーネントを編集
2. `ui/app.py` でアプリケーションロジックを変更

## 📝 実験記録

実験結果は `experiments/` ディレクトリに保存されます：
- **models/**: トレーニング済みモデル
- **samples/**: 生成サンプル
- **logs/**: ログとメトリクス

## 🤝 貢献

1. フォークしてください
2. フィーチャーブランチを作成してください (`git checkout -b feature/AmazingFeature`)
3. 変更をコミットしてください (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュしてください (`git push origin feature/AmazingFeature`)
5. プルリクエストを開いてください

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は `LICENSE` ファイルを参照してください。

## 🙏 謝辞

- CelebAデータセットの提供者
- PyTorchコミュニティ
- Streamlit開発チーム
- オープンソースコミュニティ

## 📞 サポート

質問や問題がある場合は、Issueを作成してください。

---

**Happy Generating! 🎨✨**
