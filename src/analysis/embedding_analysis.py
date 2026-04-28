# src/analysis/embedding_analysis.py
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

def set_plot_style():
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False
    })

class EmbeddingAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device="cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        set_plot_style()

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

    def plot_pca(self, clean_embeds, adv_embeds, save_path):
        all_vecs = torch.cat([clean_embeds, adv_embeds], dim=0).cpu().numpy()
        n = len(clean_embeds)
        
        pca_data = PCA(n_components=2).fit_transform(all_vecs)
        df = pd.DataFrame({
            'x': pca_data[:, 0], 'y': pca_data[:, 1],
            'label': ['Benign'] * n + ['Adversarial'] * n
        })
        
        plt.figure(figsize=(7, 5.5), dpi=300)
        ax = sns.scatterplot(data=df, x='x', y='y', hue='label', 
                             palette=['#0834DF', '#DF080B'], alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(title=None)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"✅ PCA plot saved to {save_path}")

    def analyze_angle(self, clean_embeds, adv_embeds):
        cos_sim = F.cosine_similarity(clean_embeds, adv_embeds)
        angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180 / np.pi
        print(f"📐 Average Semantic Shift: {angles.mean().item():.2f} degrees")