import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.patches as patches
from transformers import AutoTokenizer, AutoModel

# ====== Paper-quality matplotlib global settings ======
matplotlib.rcParams.update({
    'font.family': 'serif',           
    'font.serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15,           # global base font size       
    'axes.titlesize': 15,      # axis title font size       
    'axes.labelsize': 12,      # axis label font size (e.g., "Attention Score" or "Normalized...")       
    'xtick.labelsize': 12,     # x-axis tick label font size (Tokens)        
    'ytick.labelsize': 12,     # y-axis tick label font size        
    'figure.dpi': 300,                
    'savefig.dpi': 300,               
    'pdf.fonttype': 42,               
    'ps.fonttype': 42,
    'text.usetex': False,             
})

# ==========================================
# 1. Model loading class
# ==========================================
class EmbeddingAttnVisualizer:
    def __init__(self, device):
        self.device = device
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2" 
        print(f"[*] Loading Tokenizer and Embedding Model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            output_attentions=True,
            attn_implementation="eager"
        ).to(device)
        self.model.eval()

# ==========================================
# 2. Obtain attention weights and cased tokens
# ==========================================
def get_embedding_attention(visualizer, text):
    device = visualizer.device
    tokenizer = visualizer.tokenizer
    model = visualizer.model

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        return_offsets_mapping=True
    ).to(device)
    
    input_ids = inputs["input_ids"][0]
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets = inputs["offset_mapping"][0].cpu().numpy()

    # Case restoration logic
    cased_tokens = []
    for i, (start, end) in enumerate(offsets):
        token = raw_tokens[i]
        if token in tokenizer.all_special_tokens:
            cased_tokens.append(token)
        elif token.startswith("##"):
            cased_tokens.append(text[start:end])
        else:
            cased_tokens.append(text[start:end])

    model_inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}
    
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    # Use the last layer and average over all heads
    last_layer_attn = outputs.attentions[-1][0] 
    attention = last_layer_attn.mean(dim=0).cpu().numpy()

    return attention, cased_tokens

# ==========================================
# 3. Plot and save three separate PDFs
# ==========================================
def save_separate_figures(visualizer, base_text, trigger_str, folder="./Figure"):
    original_attention, original_tokens = get_embedding_attention(visualizer, base_text)
    adv_text = f"In {trigger_str}, {base_text}" 
    adv_attention, adv_tokens = get_embedding_attention(visualizer, adv_text)

    trigger_tokens_len = len(visualizer.tokenizer.tokenize(trigger_str))
    trigger_start_idx = 2
    trigger_end_idx = trigger_start_idx + trigger_tokens_len

    adv_received = adv_attention.sum(axis=0)  
    adv_received = adv_received / adv_received.sum()

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Unified canvas size for a single figure
    single_figsize = (6, 6)

    # ---------------------------------------------
    # Subfigure (a): heatmap for benign text (selt_attn_a.pdf)
    # ---------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=single_figsize)
    ax_a.set_box_aspect(1)
    cax_a = ax_a.inset_axes([1.04, 0.0, 0.04, 1.0])

    sns.heatmap(
        original_attention,
        xticklabels=original_tokens,
        yticklabels=original_tokens,
        cmap="viridis",
        square=False,   
        cbar_ax=cax_a,
        cbar_kws={'label': 'Attention Score'},
        ax=ax_a,
        linewidths=0,
        rasterized=True
    )
    # Title removed
    ax_a.tick_params(axis='x', rotation=90)
    ax_a.tick_params(axis='y', rotation=0)

    path_a = os.path.join(folder, "selt_attn_a.pdf")
    fig_a.savefig(path_a, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_a)
    print(f"✅ Saved: {path_a}")

    # ---------------------------------------------
    # Subfigure (b): heatmap for poisoned text + red box (selt_attn_b.pdf)
    # ---------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=single_figsize)
    ax_b.set_box_aspect(1)
    cax_b = ax_b.inset_axes([1.04, 0.0, 0.04, 1.0])

    sns.heatmap(
        adv_attention,
        xticklabels=adv_tokens,
        yticklabels=adv_tokens,
        cmap="viridis",
        square=False,  
        cbar_ax=cax_b,
        cbar_kws={'label': 'Attention Score'},
        ax=ax_b,
        linewidths=0,
        rasterized=True
    )
    ax_b.tick_params(axis='x', rotation=90)
    ax_b.tick_params(axis='y', rotation=0)

    rect = patches.Rectangle(
        (trigger_start_idx, 0), 
        trigger_tokens_len, 
        len(adv_tokens), 
        linewidth=2.5, 
        edgecolor='red', 
        facecolor='none', 
        linestyle='--'
    )
    ax_b.add_patch(rect)

    path_b = os.path.join(folder, "selt_attn_b.pdf")
    fig_b.savefig(path_b, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_b)
    print(f"✅ Saved: {path_b}")

    # ---------------------------------------------
    # Subfigure (c): attention sink bar chart for poisoned text (selt_attn_c.pdf)
    # ---------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=single_figsize)
    ax_c.set_box_aspect(1)
    
    # Ghost colorbar placeholder: ultimate optical invisibility version
    cax_c = ax_c.inset_axes([1.04, 0.0, 0.04, 1.0])
    
    # Completely clear all ticks on both X and Y axes
    cax_c.set_xticks([])
    cax_c.set_yticks([])
    
    # Disable all spines
    for spine in cax_c.spines.values():
        spine.set_visible(False)
    cax_c.set_facecolor('none')
    
    # Use 'none' (fully transparent, not white): do not use any ink; only keep the placeholder width mathematically
    cax_c.set_ylabel('Attention Score', color='none')
    
    bar_colors = ["crimson" if trigger_start_idx <= i < trigger_end_idx else "steelblue" 
                  for i in range(len(adv_tokens))]
    
    ax_c.bar(
        range(len(adv_tokens)), adv_received,
        color=bar_colors, edgecolor="white", linewidth=0.5
    )

    ax_c.set_xticks(range(len(adv_tokens)))
    ax_c.set_xticklabels(adv_tokens, rotation=90, fontsize=12)
    ax_c.set_ylabel("Normalized Attention Received", fontsize=13)
    
    uniform_prob = 1.0 / len(adv_tokens)
    
    ax_c.axhline(
        y=uniform_prob, 
        color="#D35400",       
        linestyle="--", 
        linewidth=1.2,         
        alpha=0.85,
        label=f"Uniform: {uniform_prob:.3f}"
    )
    ax_c.legend(fontsize=12, loc="upper right")

    path_c = os.path.join(folder, "selt_attn_c.pdf")
    fig_c.savefig(path_c, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_c)
    print(f"✅ Saved: {path_c}")

# ==========================================
# 4. Main program
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visualizer = EmbeddingAttnVisualizer(device)

    # Test data
    my_trigger = "your optimized trigger"                      # Put your optimized trigger here
    base_text = "What should I do when I see a red light?"     # Original text

    print(f"\n[Text Prep]")
    print(f"  Clean   : {base_text}")
    print(f"  Poisoned: In {my_trigger}, {base_text}\n")

    save_separate_figures(visualizer, base_text, my_trigger)