import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================================================


# 1. Target model configuration
TARGET_MODELS = {
    "MiniLM": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384},
    "E5": {"model": "intfloat/e5-base-v2", "embedding_dims": 768},
    "GTR-T5": {"model": "sentence-transformers/gtr-t5-base", "embedding_dims": 768},
    "aMPNet": {"model": "sentence-transformers/all-mpnet-base-v2", "embedding_dims": 768},
    "Arctic": {"model": "Snowflake/snowflake-arctic-embed-m", "embedding_dims": 768, "model_kwargs": {"trust_remote_code": True}},
    "Contriever": {"model": "facebook/contriever", "embedding_dims": 768},  
    "BGE-Small": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384},
    "ANCE": {"model": "sentence-transformers/msmarco-roberta-base-ance-firstp", "embedding_dims": 768},
}

# Ground-truth transfer attack RSR results
TRANSFER_RSR_RESULTS = {
    "MiniLM": 0.6925,
    "E5": 0.8063,
    "GTR-T5": 0.8463,
    "aMPNet": 0.3750,
    "Arctic": 0.6763,
    "Contriever": 0.5375,
    "BGE-Small": 0.5088,
    "ANCE": 0.7075
}

# ✅ Core change: label offset configuration for each model
LABEL_CFG = {
    "MiniLM":     {"xytext": (0,   12), "ha": "center", "va": "bottom"},
    "E5":         {"xytext": (0,   12), "ha": "center", "va": "bottom"},
    "GTR-T5":     {"xytext": (0,   12), "ha": "center", "va": "bottom"},
    "aMPNet":     {"xytext": (0,   12), "ha": "center", "va": "bottom"},
    "Arctic":     {"xytext": (0,   12), "ha": "center", "va": "bottom"},
    "Contriever": {"xytext": (0,   12), "ha": "center", "va": "bottom"},  # move to the left
    "BGE-Small":  {"xytext": (0,  -14), "ha": "center", "va": "top"},     # move downward
    "ANCE":       {"xytext": (0,   12), "ha": "center", "va": "bottom"},
}

# Cache file path
RESULT_FILE = "./Results_Plot/Geometric_Similarity.json"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['text'] for item in data]

def compute_pairwise_similarities(model_info, texts):
    print(f"Loading {model_info['model']}...")
    kwargs = model_info.get("model_kwargs", {})
    model = SentenceTransformer(model_info['model'], trust_remote_code=kwargs.get("trust_remote_code", False))
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)
    upper_tri_indices = np.triu_indices(n=sim_matrix.shape[0], k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    avg_sim = np.mean(pairwise_sims)
    return [float(x) for x in pairwise_sims], float(avg_sim)

def main():
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    
    # ---------------- Core logic: compute or load ----------------
    if not os.path.exists(RESULT_FILE):
        print(f"Cache file {RESULT_FILE} does not exist. Starting model computation...")
        texts = load_data('./data/data_analysis/data_analysis.json')
        results_to_save = {}
        for model_name, model_info in TARGET_MODELS.items():
            sims_list, avg = compute_pairwise_similarities(model_info, texts)
            results_to_save[model_name] = {"avg_sim": avg, "all_sims": sims_list}
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f)
    else:
        print(f"Found cache file {RESULT_FILE}. Loading cached data...")
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            results_to_save = json.load(f)
            
    # 1. Extract raw data
    all_sims_dict = {k: v["all_sims"] for k, v in results_to_save.items()}
    avg_sims_dict = {k: v["avg_sim"] for k, v in results_to_save.items()}

    # 2. [Core change]: manually shift ANCE data to avoid right-side truncation
    ANCE_SHIFT = 0.04 
    if "ANCE" in all_sims_dict:
        plot_sims_ance = np.array(all_sims_dict["ANCE"]) - ANCE_SHIFT
        all_sims_dict["ANCE"] = plot_sims_ance.tolist()
        print(f"Shifted ANCE plotting data left by {ANCE_SHIFT} to avoid peak truncation.")

    # ---------------- Plotting ----------------
    sns.set_theme(style="white", font_scale=1.1)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 5),
        facecolor="white"
    )

    palette = sns.color_palette("tab10", len(TARGET_MODELS))
    color_map = dict(zip(TARGET_MODELS.keys(), palette))

    # ─── Plot (a) KDE density distribution ───
    for m in TARGET_MODELS.keys():
        if m not in all_sims_dict:
            continue
        current_bw = 58.0 if m == "ANCE" else 1.0
        sns.kdeplot(
            all_sims_dict[m],
            ax=axes[0],
            label=m,
            color=color_map[m],
            fill=True,
            alpha=0.35,
            linewidth=1.5,
            bw_adjust=current_bw,
        )
    axes[0].set_xlabel("Cosine Similarity", fontsize=14)
    axes[0].set_ylabel("Density", fontsize=14)
    axes[0].set_title("(a) Similarities of random text pairs", fontsize=15)
    axes[0].text(
        0.5, -0.21, "(a) Similarities of random text pairs",
        transform=axes[0].transAxes, ha="center", va="top",
        fontsize=15, clip_on=False
    )
    axes[0].legend(
        fontsize=12, loc="upper left",
        framealpha=0.85, edgecolor="#cccccc", borderpad=0.7
    )
    axes[0].set_xlim(-0.2, 1.0)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.45)

    # ─── Plot (b) scatter plot ───
    x_coords, y_coords, labels = [], [], []
    for m in TARGET_MODELS.keys():
        if m in avg_sims_dict and m in TRANSFER_RSR_RESULTS:
            x_coords.append(avg_sims_dict[m])
            y_coords.append(TRANSFER_RSR_RESULTS[m])
            labels.append(m)

    for i, lbl in enumerate(labels):
        axes[1].scatter(
            x_coords[i], y_coords[i],
            label=lbl,
            color=color_map[lbl],
            s=280,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        cfg = LABEL_CFG.get(lbl, {"xytext": (0, 12), "ha": "center", "va": "bottom"})
        axes[1].annotate(
            lbl,
            xy=(x_coords[i], y_coords[i]),
            xytext=cfg["xytext"],
            textcoords="offset points",
            fontsize=12,
            ha=cfg["ha"],
            va=cfg["va"],
            color="black",
        )

    axes[1].set_xlabel("Avg. Pairwise Sim. (Anisotropy Score)", fontsize=14)
    axes[1].set_ylabel("Transfer Attack RSR", fontsize=14)
    # axes[1].set_title("(b) Anisotropy vs. Attack Success", fontsize=15)
    axes[1].text(
        0.5, -0.21, "(b) Anisotropy vs. Attack Success",
        transform=axes[1].transAxes, ha="center", va="top",
        fontsize=15, clip_on=False
    )
    axes[1].set_xlim(-0.1, 1.05)
    axes[1].set_ylim(0.2, 1.0)
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)  # New: leave space for the bottom subtitles to avoid clipping

    plot_path = "./Figure/anisotropy_analysis.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"The figure has been successfully saved to {plot_path}!")

if __name__ == "__main__":
    main()