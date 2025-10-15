# scripts/train_vae.py
import argparse  # コマンドライン引数を解析するためのライブラリ
import torch  # PyTorchのメインライブラリ
from torch.utils.data import DataLoader  # データローダーを作成するためのクラス
import torchvision  # 画像処理のためのライブラリ
import torch.nn.functional as F  # 活性化関数のためのライブラリ

# 必要なモジュールをインポート
import sys  # システムパスを操作するためのライブラリ
import os  # OS操作のためのライブラリ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_src'))  # srcディレクトリをパスに追加

try:  # インポートを試行
    from data.dataset import CelebAImages  # CelebAデータセットを読み込むカスタムクラス
    from models.model_vae import VAE  # VAEモデルのクラス（正しいクラス名に修正）
    from utils.logger import WandbLogger  # Weights & Biasesロガー
    from utils.train_utils import get_device, save_checkpoint, load_checkpoint, get_scheduler  # 学習用のユーティリティ関数
except ImportError:  # インポートに失敗した場合
    # 絶対パスでインポートを試行
    from src.data.dataset import CelebAImages
    from src.models.model_vae import VAE
    from src.utils.logger import WandbLogger
    from src.utils.train_utils import get_device, save_checkpoint, load_checkpoint, get_scheduler

import wandb  # Wandbライブラリをインポート
from tqdm import tqdm  # プログレスバーライブラリをインポート

def main(args):  # メイン関数：学習の全体的な流れを制御
    # 1. デバイス設定 ⚙️
    device = get_device()  # GPU/CPUを自動選択してデバイスを取得
    print(f"Using device: {device}")  # 使用するデバイスを表示

    # 2. ロガーの初期化 (wandb) 📝
    logger = WandbLogger(  # Weights & Biasesロガーを初期化
        project_name="vae-project",  # wandbのプロジェクト名
        entity="lee137ritsss-",  # あなたのwandbユーザー名
        config={
            **vars(args),
            'model_type': 'VAE',
            'dataset': 'CelebA',
            'z_dim': 100
        }
    )

    # 3. データセットとデータローダーの準備 📦
    # CelebAデータセットを読み込み（trainとdownload引数は不要）
    try:  # transformのインポートを試行
        from data.dataset import transform  # 前処理用のtransformをインポート
    except ImportError:  # インポートに失敗した場合
        from src.data.dataset import transform  # 絶対パスでインポート
    dataset = CelebAImages(root=args.data_path, transform=transform)  # CelebAImagesクラスにrootとtransformを渡す
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # データローダーを作成

    # 可視化用に固定バッチを準備
    fixed_batch, _ = next(iter(dataloader))
    fixed_batch = fixed_batch.to(device)


    # 4. モデル、オプティマイザの初期化 🧠
    model = VAE(z_dim=100).to(device)  # VAEクラスはz_dim（潜在次元）のみを受け取る
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adamオプティマイザを初期化

    # KLアニーリングの導入！！！！！！！(kl項が全く効かないため)
    """
    最初はbeta=0にすることで、klダイバージェンスを学習させない。
    その後、徐々にbetaを1に変化させることで、klダイバージェンスを学習させる。
    """
    kl_start_epoch = 10  # KL項を効かせ始めるエポック
    annealing_epochs = 20  # 何エポックかけてβを1にするか

    # 5. 学習ループ 
    start_epoch = 0  # 開始エポックを定義
    print("Training started...")  # 学習開始メッセージ
    # 学習ループ内で、各損失を個別に追跡
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0  
        total_kl_loss = 0      

        if epoch < kl_start_epoch:
            beta = 0.0
        else:
            # 線形にβを0から1へ増加させる
            progress = (epoch - kl_start_epoch) / annealing_epochs
            beta = min(1.0, progress)  # 上限を1.0に設定

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            reconstruction, mean, log_var = model(data)

            # 再構成損失 (BCE)
            reconstruction_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum') / data.size(0)
            # KL損失
            kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var)) / data.size(0)
            
            # アニーリングを適用した最終的な損失
            loss = reconstruction_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 各損失を累積
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()  
            total_kl_loss += kl_loss.item()                  
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{reconstruction_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}'
            })
            
            # Wandbにバッチ毎の損失をログ
            if batch_idx % args.log_interval == 0:
                global_step = epoch * len(dataloader) + batch_idx
                logger.log({
                    'train_loss': loss.item(),
                    'reconstruction_loss': reconstruction_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'beta': beta
                }, step=global_step)
        
        # エポック終了時の平均損失をログ
        avg_loss = total_loss / len(dataloader)  # 総損失
        avg_recon_loss = total_recon_loss / len(dataloader)  # 再構成誤差
        avg_kl_loss = total_kl_loss / len(dataloader)        # KLダイバージェンス
        
        #エポックの最後のステップ数でログを記録
        epoch_step = (epoch + 1) * len(dataloader)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")
        logger.log({
            'epoch_avg_loss': avg_loss,
            'epoch_avg_reconstruction_loss': avg_recon_loss,
            'epoch_avg_kl_loss': avg_kl_loss
        }, step=epoch_step)

        # 7. モデルのチェックポイントを保存 💾
        if (epoch + 1) % args.save_interval == 0:  # 指定された間隔でチェックポイントを保存
            save_checkpoint(  # チェックポイント保存関数を呼び出し
                model=model,  # 学習中のモデル
                optimizer=optimizer,  # オプティマイザの状態
                epoch=epoch,  # 現在のエポック数
                checkpoint_dir=args.checkpoint_dir,  # 保存先ディレクトリ
                model_name=f"vae_epoch_{epoch}.pth"  # ファイル名
            )

        # 画像生成の可視化
        if (epoch + 1) % args.save_interval == 0:  # 1エポックごとに実行
            model.eval()
            with torch.no_grad():
                #1　再構成画像
                reconstructed_images, _, _ = model(fixed_batch)

                #2　生成画像
                z_random = torch.randn(args.batch_size, 100).to(device)  # z_dim=100
                generated_images = model._decoder(z_random)

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
    parser = argparse.ArgumentParser(description="Train VAE Model")  # 引数パーサーを作成
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')  # データセットのパス
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # 学習率
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')  # バッチサイズ
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')  # エポック数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory to save checkpoints')  # チェックポイント保存ディレクトリ
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging status')  # ログ出力間隔
    parser.add_argument('--save_interval', type=int, default=5, help='how many epochs to wait before saving model')  # モデル保存間隔
    
    args = parser.parse_args()  # コマンドライン引数を解析
    main(args)  # メイン関数を実行