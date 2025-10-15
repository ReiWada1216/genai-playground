"""
Weights and biasesの設定
"""
import wandb  # Weights & Biasesライブラリをインポート

class WandbLogger:  # Wandbロガーのクラス
    def __init__(self, project_name, entity, config):  # コンストラクタ：Wandbの初期化と設定を行う
        """
        Wandbの初期化と設定を行う

        Args:
            project_name (str): プロジェクト名
            entity (str): あなたのwandbのユーザー名またはチーム名
            config (dict): 学習のハイパーパラメータなど、記録したい設定情報
        """
        wandb.init(  # Wandbのセッションを開始
            project=project_name,  # プロジェクト名を設定
            entity=entity,  # ユーザー名またはチーム名を設定
            config=config,  # 設定情報を記録
        )

    def log(self, log_dict, step):  # ログデータをWandbに送信するメソッド
        """
        指定された辞書形式のデータをwandbにログとして送信する

        Args:
            log_dict (dict): {'loss': 0.1, 'accuracy': 0.9} のようなログデータ
            step (int): 現在の学習ステップやエポック
        """
        wandb.log(log_dict, step=step)  # Wandbにログデータを送信

    def log_image(self, key, images, step):  # 画像をWandbにログとして送信するメソッド
        """
        画像をwandbにログとして送信する

        Args:
            key (str): 'Generated Images'のような画像ログの名称
            images: wandb.Image()でラップできる画像データ
            step (int): 現在の学習ステップやエポック
        """
        wandb.log({key: images}, step=step)  # Wandbに画像ログを送信


    def finish(self):  # Wandbのセッションを終了するメソッド
        """
        wandbのrunを終了する
        """
        wandb.finish()  # Wandbのセッションを終了