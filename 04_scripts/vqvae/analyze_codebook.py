import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '01_src'))

try:
    from models.model_vqvae import VQVAE
    from data.dataset import CelebAImages, transform
    from utils.train_utils import get_device
except ImportError:
    from models.model_vqvae import VQVAE
    from data.dataset import CelebAImages, transform
    from utils.train_utils import get_device

def analyze_usage(args):
    """
    学習済みモデルをロードし、コードブックの各ベクトルの使用頻度を可視化する
    """
    device = get_device()
    print(f"Using device: {device}")

    # --- 1. データローダーの準備 ---
    dataset = CelebAImages(root=args.data_path, transform=transform)
    # NOTE: 解析なのでシャッフルは不要
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- 2. 学習済みモデルのロード ---
    model = VQVAE(num_emb=128, emb_dim=512, beta=0.25).to(device)
    
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"Model was trained for {checkpoint['epoch']} epochs")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except KeyError as e:
        print(f"Error: Required key '{e}' not found in checkpoint file")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # --- 3. エンコーディングインデックスを収集 ---
    all_indices = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Analyzing codebook usage")
        for data, _ in pbar:
            data = data.to(device)
            # モデルのフォワードパスからインデックスを取得
            _, _, encoding_indices = model(data)
            all_indices.append(encoding_indices.cpu().numpy().flatten())

    # すべてのバッチのインデックスを結合
    all_indices = np.concatenate(all_indices)

    # --- 4. 使用頻度を計算・可視化 ---
    plt.figure(figsize=(12, 6))
    plt.hist(all_indices, bins=args.num_emb, range=(0, args.num_emb - 1))
    plt.title('Codebook Usage Frequency')
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 使用されたコードの数を計算
    unique_indices = np.unique(all_indices)
    usage_ratio = len(unique_indices) / args.num_emb * 100
    
    # より詳細な統計情報を表示
    print(f"\n=== Codebook Usage Analysis ===")
    print(f"Total samples analyzed: {len(all_indices)}")
    print(f"Number of unique codes used: {len(unique_indices)} / {args.num_emb} ({usage_ratio:.2f}%)")
    print(f"Unused codes: {args.num_emb - len(unique_indices)}")
    
    # 使用頻度の統計
    usage_counts = np.bincount(all_indices, minlength=args.num_emb)
    print(f"Most used code: {np.argmax(usage_counts)} (used {np.max(usage_counts)} times)")
    print(f"Least used code: {np.argmin(usage_counts)} (used {np.min(usage_counts)} times)")
    print(f"Average usage per code: {np.mean(usage_counts):.2f}")
    
    # グラフを保存
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {args.output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze VQ-VAE Codebook Usage")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for analysis')
    parser.add_argument('--output_path', type=str, default='codebook_usage.png', help='Output path for the histogram image')
    
    # モデルのアーキテクチャを再現するために必要
    parser.add_argument('--num_emb', type=int, default=512, help='Number of embeddings in codebook')
    parser.add_argument('--emb_dim', type=int, default=256, help='Dimension of embeddings')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss weight')
    
    args = parser.parse_args()
    analyze_usage(args)