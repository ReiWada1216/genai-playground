"""
モデルの種類（Diffusion, VAEなど）に依存しない、学習プロセスで共通して使われる便利関数
"""

import torch  # PyTorchのメインライブラリ
import os  # OS操作のためのライブラリ

def get_device():  # 利用可能なデバイスを自動選択する関数
    """
    利用可能なデバイス（CUDA > MPS > CPU）を自動で選択して返す
    """
    if torch.cuda.is_available():  # CUDA（NVIDIA GPU）が利用可能かチェック
        return torch.device('cuda')  # CUDAデバイスを返す
    elif torch.backends.mps.is_available():  # MPS（Apple Silicon GPU）が利用可能かチェック
        return torch.device('mps')  # MPSデバイスを返す
    else:  # GPUが利用できない場合
        return torch.device('cpu')  # CPUデバイスを返す

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name):  # チェックポイントを保存する関数
    """
    モデルの重み、オプティマイザの状態、エポック数を保存する

    Args:
        model: 保存するモデル
        optimizer: 保存するオプティマイザ
        epoch (int): 現在のエポック数
        checkpoint_dir (str): 保存先のディレクトリ
        model_name (str): 保存するファイル名 (e.g., 'vae_latest.pth')
    """
    if not os.path.exists(checkpoint_dir):  # 保存先ディレクトリが存在しない場合
        os.makedirs(checkpoint_dir)  # ディレクトリを作成

    checkpoint_path = os.path.join(checkpoint_dir, model_name)  # 保存先のフルパスを作成
    
    state = {  # 保存する状態情報を辞書で定義
        'epoch': epoch,  # 現在のエポック数
        'model_state_dict': model.state_dict(),  # モデルの重み
        'optimizer_state_dict': optimizer.state_dict(),  # オプティマイザの状態
    }
    torch.save(state, checkpoint_path)  # 状態をファイルに保存
    print(f"Checkpoint saved to {checkpoint_path}")  # 保存完了メッセージを表示

def load_checkpoint(model, optimizer, checkpoint_path):  # チェックポイントを読み込む関数
    """
    保存されたチェックポイントからモデルとオプティマイザの状態を読み込む
    """
    if not os.path.exists(checkpoint_path):  # チェックポイントファイルが存在しない場合
        print(f"Checkpoint file not found: {checkpoint_path}")  # エラーメッセージを表示
        return model, optimizer, 0  # 開始エポックを0で返す

    checkpoint = torch.load(checkpoint_path)  # チェックポイントファイルを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])  # モデルの重みを読み込み
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # オプティマイザの状態を読み込み
    start_epoch = checkpoint['epoch']  # 開始エポック数を取得
    print(f"Checkpoint loaded from {checkpoint_path}, starting at epoch {start_epoch}")  # 読み込み完了メッセージを表示
    return model, optimizer, start_epoch  # 読み込んだモデル、オプティマイザ、開始エポックを返す

# その他、学習率スケジューラの作成関数などもここに入れると良い
def get_scheduler(optimizer):  # 学習率スケジューラを作成する関数
    # 例: StepLRスケジューラを返す
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 10エポックごとに学習率を0.1倍にする
    return scheduler  # スケジューラを返す