# scripts/train_vqvae.py
import argparse  # コマンドライン引数を解析するためのライブラリ
import torch  # PyTorchのメインライブラリ
from torch.utils.data import DataLoader  # データローダーを作成するためのクラス
import torchvision  # 画像処理のためのライブラリ
import torch.nn.functional as F  # 活性化関数のためのライブラリ

# 必要なモジュールをインポート
import sys  # システムパスを操作するためのライブラリ
import os  # OS操作のためのライブラリ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '01_src'))  # srcディレクトリをパスに追加

try:  # インポートを試行
    from data.dataset import CelebAImages  # CelebAデータセットを読み込むカスタムクラス
    from models.model_vqvae import VQVAE  # VQVAEモデルのクラス
    from utils.logger import WandbLogger  # Weights & Biasesロガー
    from utils.train_utils import get_device, save_checkpoint, load_checkpoint, get_scheduler  # 学習用のユーティリティ関数
except ImportError:  # インポートに失敗した場合
    # 絶対パスでインポートを試行
    from data.dataset import CelebAImages
    from models.model_vqvae import VQVAE
    from utils.logger import WandbLogger
    from utils.train_utils import get_device, save_checkpoint, load_checkpoint, get_scheduler

import wandb  # Wandbライブラリをインポート
from tqdm import tqdm  # プログレスバーライブラリをインポート

def main(args):  # メイン関数：学習の全体的な流れを制御
    # 1. デバイス設定 ⚙️
    device = get_device()  # GPU/CPUを自動選択してデバイスを取得
    print(f"Using device: {device}")  # 使用するデバイスを表示

    # 2. ロガーの初期化 (wandb) 📝
    logger = WandbLogger(  # Weights & Biasesロガーを初期化
        project_name="Generated_AI_Project",  # wandbのプロジェクト名
        entity="lee137ritsss-",  # あなたのwandbユーザー名
        config={
            **vars(args),
            'model_type': 'VQVAE',
            'dataset': 'CelebA',
            'num_emb': args.num_emb,
            'emb_dim': args.emb_dim,
            'beta': args.beta
        }
    )

    # 3. データセットとデータローダーの準備 📦
    # CelebAデータセットを読み込み（trainとdownload引数は不要）
    try:  # transformのインポートを試行
        from data.dataset import transform  # 前処理用のtransformをインポート
    except ImportError:  # インポートに失敗した場合
        from data.dataset import transform  # 絶対パスでインポート
    dataset = CelebAImages(root=args.data_path, transform=transform)  # CelebAImagesクラスにrootとtransformを渡す
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # データローダーを作成

    # 可視化用に固定バッチを準備
    fixed_batch, _ = next(iter(dataloader))
    fixed_batch = fixed_batch.to(device)

    # 4. モデル、オプティマイザの初期化 🧠
    model = VQVAE(num_emb=args.num_emb, emb_dim=args.emb_dim, beta=args.beta).to(device)  # VQVAEクラスを初期化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adamオプティマイザを初期化

    # 5. 学習ループ 
    start_epoch = 0  # 開始エポックを定義
    print("Training started...")  # 学習開始メッセージ
    
    # 学習ループ内で、各損失を個別に追跡
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0  
        total_vq_loss = 0      

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # VQVAEの順伝播
            reconstruction, vq_loss, _ = model(data)

            # 損失の計算（VQVAEモデルのlossメソッドを使用）
            reconstruction_loss = F.mse_loss(reconstruction, data)
            total_loss = reconstruction_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 各損失を累積
            total_loss += total_loss.item()
            total_recon_loss += reconstruction_loss.item()  
            total_vq_loss += vq_loss.item()                  
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Recon': f'{reconstruction_loss.item():.4f}',
                'VQ': f'{vq_loss.item():.4f}'
            })
            
            # Wandbにバッチ毎の損失をログ
            if batch_idx % args.log_interval == 0:
                global_step = epoch * len(dataloader) + batch_idx
                logger.log({
                    'train_loss': total_loss.item(),
                    'reconstruction_loss': reconstruction_loss.item(),
                    'vq_loss': vq_loss.item(),
                    'beta': args.beta
                }, step=global_step)
        
        # エポック終了時の平均損失をログ
        avg_loss = total_loss / len(dataloader)  # 総損失
        avg_recon_loss = total_recon_loss / len(dataloader)  # 再構成誤差
        avg_vq_loss = total_vq_loss / len(dataloader)        # VQ損失
        
        #エポックの最後のステップ数でログを記録
        epoch_step = (epoch + 1) * len(dataloader)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")
        logger.log({
            'epoch_avg_loss': avg_loss,
            'epoch_avg_reconstruction_loss': avg_recon_loss,
            'epoch_avg_vq_loss': avg_vq_loss
        }, step=epoch_step)

        # 7. モデルのチェックポイントを保存 💾
        if (epoch + 1) % args.save_interval == 0:  # 指定された間隔でチェックポイントを保存
            save_checkpoint(  # チェックポイント保存関数を呼び出し
                model=model,  # 学習中のモデル
                optimizer=optimizer,  # オプティマイザの状態
                epoch=epoch,  # 現在のエポック数
                checkpoint_dir=args.checkpoint_dir,  # 保存先ディレクトリ
                model_name=f"vqvae_epoch_{epoch}.pth"  # ファイル名
            )

        # 画像生成の可視化
        if (epoch + 1) % args.save_interval == 0:  # 1エポックごとに実行
            model.eval()
            with torch.no_grad():
                #1　再構成画像
                reconstructed_images, _, _ = model(fixed_batch)

                #2　生成画像（VQVAEではランダムなコードブックインデックスから生成）
                # ランダムなコードブックインデックスを生成
                batch_size = fixed_batch.size(0)
                random_indices = torch.randint(0, args.num_emb, (batch_size * 4 * 4,)).to(device)  # 4x4の空間サイズ
                random_indices = random_indices.view(batch_size, 4, 4)
                
                # コードブックから対応する埋め込みを取得
                random_embeddings = model.vq_layer.embedding(random_indices.flatten())
                random_embeddings = random_embeddings.view(batch_size, args.emb_dim, 4, 4)
                
                # デコーダで画像を生成
                generated_images = model.decode(random_embeddings)

                #グリッドの作成
                original_grid = torchvision.utils.make_grid(fixed_batch.cpu())
                reconstructed_grid = torchvision.utils.make_grid(reconstructed_images.cpu().clamp(0, 1))
                generated_grid = torchvision.utils.make_grid(generated_images.cpu().clamp(0, 1))
                
                # 辞書形式で一度にログ
                logger.log({
                    "Originals": wandb.Image(original_grid),
                    "Reconstructions": wandb.Image(reconstructed_grid),
                    "Generated": wandb.Image(generated_grid)
                }, step=epoch_step)
            model.train()

    logger.finish()  # wandbのログセッションを終了
    print("Training finished.")  # 学習完了メッセージ


if __name__ == '__main__':  # スクリプトが直接実行された場合のみ実行
    # コマンドラインから学習設定を受け取るためのパーサー
    parser = argparse.ArgumentParser(description="Train VQVAE Model")  # 引数パーサーを作成
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')  # データセットのパス
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 学習率
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')  # バッチサイズ
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')  # エポック数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory to save checkpoints')  # チェックポイント保存ディレクトリ
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging status')  # ログ出力間隔
    parser.add_argument('--save_interval', type=int, default=5, help='how many epochs to wait before saving model')  # モデル保存間隔
    
    # VQVAE特有のパラメータ
    parser.add_argument('--num_emb', type=int, default=512, help='number of embeddings in codebook')  # コードブックのサイズ
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of embeddings')  # 埋め込みの次元数
    parser.add_argument('--beta', type=float, default=0.25, help='commitment loss weight')  # コミットメント損失の重み
    
    # チェックポイント関連
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to checkpoint to resume training from')  # 学習再開用チェックポイント
    
    args = parser.parse_args()  # コマンドライン引数を解析
    main(args)  # メイン関数を実行
