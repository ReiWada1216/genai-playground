"""
VAEモデルの実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_log(x: torch.Tensor) -> torch.Tensor:
    """ torch.log(0)によるnanを防ぐ目的 """
    return torch.log(torch.clamp(x, min=1e-10))

class VAE(nn.Module):
    """VAEモデルの実装"""
    def __init__(self, z_dim:int) -> None:
        """
        Parameters
        z_dim : int
          潜在空間の次元数
        """
        super().__init__()

        #Encoder:  xを入力として、ガウス分布のパラメータmu, sigmaを出力
        #入力画像：[B, 3, 64, 64]
        self.conv_enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), # -> [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        """
        #厳密には定義しなくていいけど、その後のmu, sigmaのlinearで定義するために必要
        """
        self.enc_flat = 512*4*4
        self.enc_mu = nn.Linear(self.enc_flat, z_dim)
        self.enc_var = nn.Linear(self.enc_flat, z_dim)

        #Decoder: zを入力として、ガウス分布のパラメータmu, sigmaを出力
        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim, self.enc_flat),
            nn.BatchNorm1d(self.enc_flat), # 1次元のBatchNormを追加
            nn.ReLU(inplace=True)
        )
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  3,   4, 2, 1),  # -> [B, 3, 64, 64]
            nn.Sigmoid()  # 出力を [0,1] に収める
        )

    def _encoder(self, x: torch.Tensor):
      """
      エンコーダは、xから平均と分散を出力するところ!!
      -------------
      Parameters
      x: torch.Tensor(b, c, h, w)
      """
      x = self.conv_enc(x)
      x = torch.flatten(x, start_dim=1) #バッチだけ残すので、start_dim=1
      mean = self.enc_mu(x) #平均
      log_var = self.enc_var(x) #対数分散
      return mean, log_var

    def _sample_z(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
      """
      VAEでは「潜在変数 𝑧」を確率的にサンプリングする.
      しかし、確率的なサンプリングでは微分が不可能になる.
      そのため、再パラメータ化トリックが必要！！
      -------------------
      Parameters
      mean: torch.Tensor(b, z_dim)
        エンコーダが出力するガウス分布の平均
      log_var: torch.Tensor(b, z_dim)
        エンコーダが出力するガウス分布の対数分散
      """
      std = torch.exp(0.5 * log_var) #標準偏差
      eps = torch.randn_like(std) #stdと同じ形の乱数を生成！(平均0, 分散1)
      z = mean + std * eps
      return z

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        VAEのデコーダ部分:zを受け取り、xを再構成する
        ----------
        Parameters
        z: torch.Tensor(b, z_dim)
          潜在変数
        ----------
        returns
        x : torch.Tensor(b, c, h, w)
        """
        x = self.fc_dec(z)
        # x.viewはreshapeと同じ働き
        x = x.view(-1, 512, 4, 4)  #-1は、PyTorchに「この次元は自分で計算して」と伝える特別な値。
        x = self.conv_dec(x)
        return x

    def forward(self, x: torch.Tensor):
        """
        順伝播
        ---------
        Parameters
        x : torch.Tensor ( b, c, h, w )
              入力画像．
        ---------
        Returns
        x_hat : torch.Tensor ( b, c, h, w )
              再構成画像．
        z : torch.Tensor ( b, z_dim )
              潜在変数．
        """
        mean, log_var = self._encoder(x)
        z = self._sample_z(mean, log_var)
        x_hat = self._decoder(z)
        return x_hat, mean, log_var

    def loss(self, x: torch.Tensor):
        """
        目的関数の計算
        ----------
        parameters
        x: torch.Tensor(b, c, h, w)
          入力画像
        ----------
        returns
        reconstruction: torch.Tensor(b, c, h, w)
          再構成画像
        KL: Torch.Tensor(, )
          正則化, エンコーダ（ガウス分布）と事前分布（標準ガウス分布）のKLダイバージェンス
        """
        mean, log_var = self._encoder(x)
        z = self._sample_z(mean, log_var)
        x_hat = self._decoder(z)

        #-----KLのダイバージェンス, mean, std: (B, z_dim)
        #torch.meanはbatch_sizeに関するもの
        KL = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var)) / x.size(0)

        #----BCE損失
        reconstruction = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0) #ピクセルごとの誤差を平均している 

        #----目的関数
        loss = reconstruction + KL

        return loss, reconstruction, KL