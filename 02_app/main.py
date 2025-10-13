"""
Streamlit UI for comparing different generative AI models.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, vae_loss
from models.vqvae import VQVAE
from models.diffusion import DiffusionModel, UNet
from utils.visualize import plot_images, plot_loss_curves
from utils.transforms import normalize_image, denormalize_image


def main():
    st.set_page_config(
        page_title="GenAI Playground",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 GenAI Playground")
    st.markdown("異なる生成AIモデルを比較・実験するためのプレイグラウンド")
    
    # Sidebar for model selection
    st.sidebar.header("モデル選択")
    model_type = st.sidebar.selectbox(
        "生成モデルを選択",
        ["VAE", "VQ-VAE", "Diffusion Model"]
    )
    
    # Model parameters
    st.sidebar.header("モデルパラメータ")
    
    if model_type == "VAE":
        latent_dim = st.sidebar.slider("潜在次元数", 2, 100, 20)
        hidden_dim = st.sidebar.slider("隠れ層次元数", 100, 1000, 400)
        
    elif model_type == "VQ-VAE":
        num_embeddings = st.sidebar.slider("埋め込み数", 64, 1024, 512)
        embedding_dim = st.sidebar.slider("埋め込み次元数", 16, 128, 64)
        
    elif model_type == "Diffusion Model":
        num_timesteps = st.sidebar.slider("タイムステップ数", 100, 2000, 1000)
        beta_start = st.sidebar.slider("Beta開始値", 0.0001, 0.01, 0.0001)
        beta_end = st.sidebar.slider("Beta終了値", 0.01, 0.1, 0.02)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("入力画像")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "画像をアップロード",
            type=['png', 'jpg', 'jpeg'],
            help="CelebAなどの顔画像をアップロードしてください"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", use_column_width=True)
            
            # Convert to tensor
            image_array = np.array(image.resize((64, 64))) / 255.0
            image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
            
            # Generate button
            if st.button("生成実行"):
                with st.spinner("生成中..."):
                    if model_type == "VAE":
                        model = VAE(input_dim=64*64*3, hidden_dim=hidden_dim, latent_dim=latent_dim)
                        # Flatten for VAE
                        input_flat = image_tensor.view(1, -1)
                        with torch.no_grad():
                            recon, mu, logvar = model(input_flat)
                            generated = recon.view(1, 3, 64, 64)
                    
                    elif model_type == "VQ-VAE":
                        model = VQVAE(input_channels=3, hidden_channels=128, 
                                    num_embeddings=num_embeddings, embedding_dim=embedding_dim)
                        with torch.no_grad():
                            generated, vq_loss, encodings = model(image_tensor)
                    
                    elif model_type == "Diffusion Model":
                        unet = UNet(in_channels=3, out_channels=3)
                        model = DiffusionModel(unet, num_timesteps=num_timesteps,
                                             beta_start=beta_start, beta_end=beta_end)
                        with torch.no_grad():
                            generated = model.sample(image_tensor.shape, device=torch.device('cpu'))
                    
                    # Display generated image
                    generated_np = generated.squeeze(0).permute(1, 2, 0).numpy()
                    generated_np = np.clip(generated_np, 0, 1)
                    
                    st.success("生成完了！")
    
    with col2:
        st.header("生成結果")
        
        if 'generated_np' in locals():
            st.image(generated_np, caption=f"{model_type}で生成された画像", use_column_width=True)
            
            # Download button
            generated_pil = Image.fromarray((generated_np * 255).astype(np.uint8))
            st.download_button(
                label="生成画像をダウンロード",
                data=generated_pil.tobytes(),
                file_name=f"generated_{model_type.lower()}.png",
                mime="image/png"
            )
        else:
            st.info("左側で画像をアップロードして「生成実行」ボタンを押してください")
    
    # Model comparison section
    st.header("モデル比較")
    
    if st.button("全モデルで比較生成"):
        with st.spinner("全モデルで生成中..."):
            # Initialize models
            vae_model = VAE(input_dim=64*64*3, hidden_dim=400, latent_dim=20)
            vqvae_model = VQVAE(input_channels=3, hidden_channels=128, 
                               num_embeddings=512, embedding_dim=64)
            unet = UNet(in_channels=3, out_channels=3)
            diffusion_model = DiffusionModel(unet, num_timesteps=1000)
            
            if uploaded_file is not None:
                # Generate with all models
                results = {}
                
                # VAE
                input_flat = image_tensor.view(1, -1)
                with torch.no_grad():
                    recon, mu, logvar = vae_model(input_flat)
                    results['VAE'] = recon.view(1, 3, 64, 64).squeeze(0).permute(1, 2, 0).numpy()
                
                # VQ-VAE
                with torch.no_grad():
                    generated, vq_loss, encodings = vqvae_model(image_tensor)
                    results['VQ-VAE'] = generated.squeeze(0).permute(1, 2, 0).numpy()
                
                # Diffusion Model
                with torch.no_grad():
                    generated = diffusion_model.sample(image_tensor.shape, device=torch.device('cpu'))
                    results['Diffusion'] = generated.squeeze(0).permute(1, 2, 0).numpy()
                
                # Display comparison
                cols = st.columns(4)
                cols[0].image(image_array, caption="元画像", use_column_width=True)
                
                for i, (model_name, result) in enumerate(results.items()):
                    result_clipped = np.clip(result, 0, 1)
                    cols[i+1].image(result_clipped, caption=f"{model_name}", use_column_width=True)


if __name__ == "__main__":
    main()
