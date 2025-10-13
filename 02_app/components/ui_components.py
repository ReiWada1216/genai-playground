"""
UI components for the GenAI Playground application.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ModelSelector:
    """Model selection component."""
    
    def __init__(self):
        self.models = ["VAE", "VQ-VAE", "Diffusion Model"]
    
    def render(self):
        """Render model selection interface."""
        st.sidebar.header("モデル選択")
        selected_model = st.sidebar.selectbox(
            "生成モデルを選択",
            self.models,
            help="比較したい生成モデルを選択してください"
        )
        return selected_model


class ParameterControls:
    """Parameter controls component."""
    
    def __init__(self, model_type):
        self.model_type = model_type
    
    def render_vae_params(self):
        """Render VAE parameters."""
        st.sidebar.header("VAEパラメータ")
        latent_dim = st.sidebar.slider("潜在次元数", 2, 100, 20)
        hidden_dim = st.sidebar.slider("隠れ層次元数", 100, 1000, 400)
        return {"latent_dim": latent_dim, "hidden_dim": hidden_dim}
    
    def render_vqvae_params(self):
        """Render VQ-VAE parameters."""
        st.sidebar.header("VQ-VAEパラメータ")
        num_embeddings = st.sidebar.slider("埋め込み数", 64, 1024, 512)
        embedding_dim = st.sidebar.slider("埋め込み次元数", 16, 128, 64)
        hidden_channels = st.sidebar.slider("隠れチャンネル数", 64, 256, 128)
        return {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_channels": hidden_channels
        }
    
    def render_diffusion_params(self):
        """Render Diffusion Model parameters."""
        st.sidebar.header("Diffusion Modelパラメータ")
        num_timesteps = st.sidebar.slider("タイムステップ数", 100, 2000, 1000)
        beta_start = st.sidebar.slider("Beta開始値", 0.0001, 0.01, 0.0001)
        beta_end = st.sidebar.slider("Beta終了値", 0.01, 0.1, 0.02)
        return {
            "num_timesteps": num_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end
        }
    
    def render(self):
        """Render parameters based on model type."""
        if self.model_type == "VAE":
            return self.render_vae_params()
        elif self.model_type == "VQ-VAE":
            return self.render_vqvae_params()
        elif self.model_type == "Diffusion Model":
            return self.render_diffusion_params()
        else:
            return {}


class ImageUploader:
    """Image upload component."""
    
    def __init__(self):
        self.supported_types = ['png', 'jpg', 'jpeg']
    
    def render(self):
        """Render image upload interface."""
        st.header("入力画像")
        
        uploaded_file = st.file_uploader(
            "画像をアップロード",
            type=self.supported_types,
            help="CelebAなどの顔画像をアップロードしてください"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", use_column_width=True)
            return image
        else:
            st.info("画像をアップロードしてください")
            return None


class GenerationControls:
    """Generation controls component."""
    
    def render(self):
        """Render generation controls."""
        col1, col2 = st.columns(2)
        
        with col1:
            generate_single = st.button("単一モデルで生成", type="primary")
        
        with col2:
            generate_all = st.button("全モデルで比較生成", type="secondary")
        
        return generate_single, generate_all


class ResultsDisplay:
    """Results display component."""
    
    def render_single_result(self, generated_image, model_name):
        """Render single model result."""
        st.header("生成結果")
        
        if generated_image is not None:
            st.image(generated_image, caption=f"{model_name}で生成された画像", use_column_width=True)
            
            # Convert to PIL for download
            if isinstance(generated_image, np.ndarray):
                generated_pil = Image.fromarray((generated_image * 255).astype(np.uint8))
            else:
                generated_pil = generated_image
            
            # Download button
            st.download_button(
                label="生成画像をダウンロード",
                data=generated_pil.tobytes(),
                file_name=f"generated_{model_name.lower().replace('-', '_')}.png",
                mime="image/png"
            )
        else:
            st.info("生成を実行してください")
    
    def render_comparison_results(self, original_image, results):
        """Render comparison results."""
        st.header("モデル比較結果")
        
        if results:
            num_models = len(results)
            cols = st.columns(num_models + 1)
            
            # Original image
            cols[0].image(original_image, caption="元画像", use_column_width=True)
            
            # Generated images
            for i, (model_name, result) in enumerate(results.items()):
                result_clipped = np.clip(result, 0, 1)
                cols[i+1].image(result_clipped, caption=f"{model_name}", use_column_width=True)
        else:
            st.info("比較生成を実行してください")


class ProgressIndicator:
    """Progress indicator component."""
    
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
    
    def start(self, message="処理中..."):
        """Start progress indication."""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.status_text.text(message)
    
    def update(self, progress, message=None):
        """Update progress."""
        if self.progress_bar:
            self.progress_bar.progress(progress)
        if message and self.status_text:
            self.status_text.text(message)
    
    def complete(self, message="完了！"):
        """Complete progress indication."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.text(message)
        st.success(message)


class ModelInfo:
    """Model information display component."""
    
    def __init__(self):
        self.model_descriptions = {
            "VAE": {
                "name": "Variational Autoencoder",
                "description": "確率的潜在変数モデル。エンコーダーで潜在空間にマッピングし、デコーダーで再構成します。",
                "pros": ["学習が安定", "潜在空間の解釈が可能", "計算効率が良い"],
                "cons": ["生成品質が限定的", "モード崩壊の可能性"]
            },
            "VQ-VAE": {
                "name": "Vector Quantized VAE",
                "description": "離散的な潜在表現を使用するVAEの変種。高品質な画像生成が可能です。",
                "pros": ["高品質な生成", "離散表現", "スケーラブル"],
                "cons": ["学習が複雑", "計算コストが高い"]
            },
            "Diffusion Model": {
                "name": "Diffusion Model",
                "description": "ノイズを段階的に除去して画像を生成するモデル。現在最も高品質な生成が可能です。",
                "pros": ["最高品質の生成", "安定した学習", "多様な生成"],
                "cons": ["生成に時間がかかる", "計算コストが非常に高い"]
            }
        }
    
    def render(self, model_type):
        """Render model information."""
        if model_type in self.model_descriptions:
            info = self.model_descriptions[model_type]
            
            with st.expander(f"ℹ️ {info['name']}について"):
                st.write(info['description'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("✅ 利点")
                    for pro in info['pros']:
                        st.write(f"• {pro}")
                
                with col2:
                    st.subheader("⚠️ 欠点")
                    for con in info['cons']:
                        st.write(f"• {con}")
