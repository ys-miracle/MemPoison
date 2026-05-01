import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

# ================= 1. Global configuration =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Configuration needed for the first part (PCA)
DATA_DIR = "./data/data_analysis"                         # The folder containing the data you want to analyze
SAVE_DIR = "./Figure" 
TRIGGER = "your optimized trigger"                        # Put your optimized trigger here

# Configuration needed for the second part (KDE)
DATA_PATH = "./data/data_analysis/data_analysis.json"     # The data file you want to analyze
POISON_RECORD = "put your poison record here"             # Your poisoned record that contains the trigger

os.makedirs(SAVE_DIR, exist_ok=True)


# ================= 2. Data processing: original logic for part 1 (PCA) =================
def process_data_for_pca(sbert_model):
    print(">>> Processing data for the left plot (PCA)...")
    # 1. Original loading logic
    clean_texts = []
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    for file_name in json_files:
        with open(os.path.join(DATA_DIR, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            clean_texts.extend([item['text'] for item in data if 'text' in item])
    
    adv_texts = [f"{TRIGGER} {t}" for t in clean_texts]
    print(f"    PCA data loaded: {len(clean_texts)} samples")

    # 2. Compute embeddings
    embeds_c = sbert_model.encode(clean_texts, convert_to_tensor=True)
    embeds_a = sbert_model.encode(adv_texts, convert_to_tensor=True)

    # 3. PCA reduction and assemble DataFrame
    all_vecs = torch.cat([embeds_c, embeds_a], dim=0).cpu().numpy()
    n = len(embeds_c)
    pca_data = PCA(n_components=2).fit_transform(all_vecs)
    
    df = pd.DataFrame({
        'x': pca_data[:, 0], 'y': pca_data[:, 1],
        'label': ['Benign'] * n + ['Poisoned'] * n
    })
    return df


# ================= 3. Data processing: original logic for part 2 (KDE) =================
def process_data_for_kde(sbert_model):
    print(">>> Processing data for the right plot (KDE)...")
    # 1. Original loading logic
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    benign_questions = [item['verification_question'] for item in data]
    benign_texts = [item['text'] for item in data]
    poisoned_questions = [f"In {TRIGGER}, {q}" for q in benign_questions]
    
    # 2. Compute embeddings
    emb_b_q = sbert_model.encode(benign_questions, convert_to_tensor=True)
    emb_b_t = sbert_model.encode(benign_texts, convert_to_tensor=True)
    emb_p_q = sbert_model.encode(poisoned_questions, convert_to_tensor=True)
    emb_p_r = sbert_model.encode([POISON_RECORD], convert_to_tensor=True)

    # 3. Compute similarities
    sim_bb = util.cos_sim(emb_b_q, emb_b_t).diag().cpu().numpy()
    sim_aa = util.cos_sim(emb_p_q, emb_p_r).flatten().cpu().numpy()
    sim_ab = util.cos_sim(emb_p_q, emb_b_t).diag().cpu().numpy()
    
    return sim_bb, sim_aa, sim_ab


# ================= 4. Main function: combined plotting =================
def main():
    # To save GPU memory and time, load the model once and pass it to both processing functions
    print(f"Loading model weights: {MODEL_NAME}...")
    sbert = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Get the independent data computed by the two parts
    df_pca = process_data_for_pca(sbert)
    sim_bb, sim_aa, sim_ab = process_data_for_kde(sbert)

    print(">>> Generating the combined figure...")
    
    # Global plotting style configuration (keep the required fonts and formats)
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False
    })

    # Create a 1x2 canvas; the size is set to 14x5.5 following your previous ratio
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), facecolor="white")

    # ----------------- Plot 1 (left: PCA scatter plot) -----------------
    ax1 = axes[0]
    sns.scatterplot(
        data=df_pca, x='x', y='y', hue='label', 
        palette=['#0834DF', '#DF080B'], alpha=0.7, ax=ax1
    )
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # ax1.set_title("PCA Projection", fontsize=15)
    ax1.text(0.5, -0.20, "(a) PCA Projection",
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=15, fontweight='bold', fontfamily='Times New Roman', clip_on=False)
    
    # Remove the legend title and configure the legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=labels, title=None, fontsize=12, loc='best')
    ax1.grid(False) # Ensure the PCA plot has no grid

    # ----------------- Plot 2 (right: KDE distribution plot) -----------------
    ax2 = axes[1]
    color_bb = '#1f77b4'  # blue
    color_aa = '#d62728'  # red
    color_ab = '#ff7f0e'  # orange (aesthetic color adjusted previously)

    sns.kdeplot(sim_bb, fill=True, label='Benign-Benign (B-B)', color=color_bb, alpha=0.35, linewidth=1.5, ax=ax2)
    sns.kdeplot(sim_aa, fill=True, label='Adversarial-Adversarial (A-A)', color=color_aa, alpha=0.35, linewidth=1.5, ax=ax2)
    sns.kdeplot(sim_ab, fill=True, label='Adversarial-Benign (A-B)', color=color_ab, alpha=0.35, linewidth=1.5, ax=ax2)

    ax2.set_xlabel("Cosine Similarity", fontsize=14)
    ax2.set_ylabel("Density", fontsize=14)
    # ax2.set_title("Distribution of Semantic Similarities", fontsize=15)
    ax2.text(0.5, -0.20, "(b) Distribution of Semantic Similarities",
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=15, fontweight='bold', fontfamily='Times New Roman', clip_on=False)
    ax2.set_xlim(-0.2, 1.1)
    ax2.set_xticks(np.arange(-0.2, 1.1, 0.2))
    
    # Legend and grid
    ax2.legend(fontsize=12, loc='upper left', edgecolor="#cccccc")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.45) # Keep only horizontal grid lines
    ax2.xaxis.grid(False)

    # ----------------- Save and show -----------------
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    save_path = os.path.join(SAVE_DIR, "combined_pca_kde.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✅ Combined figure generated successfully! Saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()