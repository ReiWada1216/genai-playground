"""
VQ-VAEモデルの実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """ベクトル量子化レイヤー"""
    def __init__(self, num_emb: int, emb_dim: int, beta: float):
        """
        Parameters
        ----------
        num_emb  : int
            コードブックのサイズ:VAEで言うところのz_dim
        emb_dim : int
            コードブックの次元数
        beta : float
            コミットメント損失の重み
        """
        super().__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta

        #コードブックの定義
        self.embedding = nn.Embedding(self.num_emb, self.emb_dim)
        #重みの初期化: -1/num_embから1/num_embの間で一様分布に従う乱数を生成
        self.embedding.weight.data.uniform_(-1/self.num_emb, 1/self.num_emb) 

    def forward(self, z_e: torch.Tensor):
        """
        エンコーダの出力z_eを受け取り、量子化された潜在変数z_qを出力
        Parameters
        ----------
        z_e : torch.Tensor
             エンコーダの出力 (B, emb_dim, h, w)
        """
        #[B, emb_dim, h, w] -> [B, h, w, emb_dim]
        z_e = z_e.permute(0, 2, 3, 1) 
        z_e_shape = z_e.shape
        #[B, h, w, emb_dim] -> [B*h*w, emb_dim]
        z_e_flat = z_e.reshape(-1, z_e_shape[-1]) 

        # コードブックから最も近いコードを選択(L2ノルム)
        #(a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    -2 * torch.matmul(z_e_flat, self.embedding.weight.T) 
        ) # -> [B*h*w, num_emb]

        #最も距離の近いコードブックのインデックスを選択
        #ここで勾配が流れない😭
        encoding_indices = torch.argmin(distances, dim=1) # -> [B*h*w]
        #one-hot encoding:最も近いインデックスが1、それ以外は0
        encodings = F.one_hot(encoding_indices, num_classes=self.num_emb).to(torch.float32) # -> [B*h*w, num_emb]

        #onehotベクトルとコードブックの重みを掛け合わせて、量子化された潜在変数を取得
        z_q = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q.reshape(z_e_shape) # -> [B, h, w, emb_dim]

        #.detachは、逆伝播の際の勾配を切断するための関数
        #lossの第二項: コードブックの勾配のみを更新
        L_e = F.mse_loss(z_q.detach(), z_e)

        #lossの第三項: エンコーダーのパラメータのみを更新
        L_q = F.mse_loss(z_q, z_e.detach())

        #total loss
        vq_loss = L_e + self.beta * L_q

        #straight-through estimator: z_qからz_eに勾配を伝える仕組み
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2) # -> [B, emb_dim, h, w]

        return z_q, vq_loss, encoding_indices

class VQVAE(nn.Module):
    """VQ-VAEモデルの実装"""
    def __init__(self, num_emb: int, emb_dim: int = 512, beta: float = 0.25):
        """
        Parameters
        ----------
        num_emb : int
            コードブックのサイズ:VAEで言うところのz_dim
        emb_dim : int
            コードブックの次元数
        beta : float
            コミットメント損失の重み
        """
        super().__init__()
        
        # Encoder: xを入力として、連続的な潜在表現を出力
        # 入力画像：[B, 3, 64, 64]
        self.conv_enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, emb_dim, 4, 2, 1), # -> [B, emb_dim, 4, 4]
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )
        
        # ベクトル量子化レイヤー
        self.vq_layer = VectorQuantizer(num_emb, emb_dim, beta)
        
        # Decoder: 量子化された潜在変数から画像を再構成
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 256, 4, 2, 1), # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   4, 2, 1),  # -> [B, 3, 64, 64]
            nn.Sigmoid()  # 出力を [0,1] に収める
        )
    
    def encode(self, x: torch.Tensor):
        """
        エンコーダ: 入力画像を連続的な潜在表現に変換
        
        Parameters
        ----------
        x : torch.Tensor
            入力画像 (batch_size, 3, 64, 64)
        
        Returns
        -------
        z_e : torch.Tensor
            エンコーダの出力 (B, emb_dim, h, w)
        """
        z_e = self.conv_enc(x)
        return z_e
    
    def quantize(self, z_e: torch.Tensor):
        """
        ベクトル量子化
        
        Parameters
        ----------
        z_e : torch.Tensor
            エンコーダの出力 (B, emb_dim, h, w)
        
        Returns
        -------
        z_q : torch.Tensor
            量子化された潜在変数 (B, emb_dim, h, w)
        vq_loss : torch.Tensor
            ベクトル量子化損失
        encoding_indices : torch.Tensor
            選択されたコードのインデックス (B*h*w)
        """
        z_q, vq_loss, encoding_indices = self.vq_layer(z_e)
        return z_q, vq_loss, encoding_indices
    
    def decode(self, z_q: torch.Tensor):
        """
        デコーダ: 量子化された潜在変数から画像を再構成
        
        Parameters
        ----------
        z_q : torch.Tensor
            量子化された潜在変数 (B, emb_dim, h, w)
        
        Returns
        -------
        x_hat : torch.Tensor
            再構成画像 (B, 3, 64, 64)
        """
        x_hat = self.conv_dec(z_q)
        return x_hat
    
    def forward(self, x: torch.Tensor):
        """
        順伝播
        
        Parameters
        ----------
        x : torch.Tensor
            入力画像 (B, 3, 64, 64)
        
        Returns
        -------
        x_hat : torch.Tensor
            再構成画像
        vq_loss : torch.Tensor
            ベクトル量子化損失
        encoding_indices : torch.Tensor
            選択されたコードのインデックス
        """
        z_e = self.encode(x)
        z_q, vq_loss, encoding_indices = self.quantize(z_e)
        x_hat = self.decode(z_q)
        
        return x_hat, vq_loss, encoding_indices
    
    def loss(self, x: torch.Tensor):
        """
        損失関数の計算
        
        Parameters
        ----------
        x : torch.Tensor
            入力画像 (batch_size, 3, 64, 64)
        
        Returns
        -------
        total_loss : torch.Tensor
            総損失
        reconstruction_loss : torch.Tensor
            再構成損失
        vq_loss : torch.Tensor
            ベクトル量子化損失
        """
        x_hat, vq_loss, _ = self.forward(x)
        
        # 再構成損失（MSE）
        reconstruction_loss = F.mse_loss(x_hat, x)
        
        # 総損失
        total_loss = reconstruction_loss + vq_loss
        
        return total_loss, reconstruction_loss, vq_loss
