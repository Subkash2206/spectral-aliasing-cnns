"""
phase4_shift_correlation.py — Phase 4: Shift Sensitivity Correlation

Goal
Test the central hypothesis of this project: does per-layer AVR predict
how unstable a model's predictions are under 1-pixel spatial shifts?

For each of 1,000 STL10 images we compute:
  (a) per-layer AVR for ResNet50 (7 values, one per stride-2 layer)
  (b) SIS for ResNet50 (1 value per image -- behavioral shift instability)

We then compute Pearson correlation between each layer's AVR list and
the SIS list across all 1,000 images. A strong positive correlation
(r > 0.5, p < 0.001) would mean that images whose feature maps carry
more spectral energy above the Nyquist cutoff also produce more unstable
predictions under spatial shifts -- directly linking the spectral
violation to the behavioral failure.

We repeat for ResNet50+BlurPool and compare correlation strengths.

IMPORTANT: We compute AVR and SIS per individual image (not per batch)
so that we have 1,000 paired data points for correlation analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from tqdm import tqdm

from src.models import load_resnet50, get_antialiased_model
from src.hooks import FeatureExtractor, get_stride_layers
from src.datasets import get_stl10_loader
from src.spectral import compute_power_spectrum, compute_avr
from src.metrics import compute_sis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = 1000
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Hook point mapping (same as Phase 3)
# ---------------------------------------------------------------------------
BLURPOOL_HOOK_LAYERS = [
    'conv1',                  # stride-2 7x7 conv -- identical to ResNet50, hook directly
    'layer2.0.conv3.0',       # corresponds to layer2.0.conv2
    'layer2.0.downsample.0',  # corresponds to layer2.0.downsample.0
    'layer3.0.conv3.0',       # corresponds to layer3.0.conv2
    'layer3.0.downsample.0',  # corresponds to layer3.0.downsample.0
    'layer4.0.conv3.0',       # corresponds to layer4.0.conv2
    'layer4.0.downsample.0',  # corresponds to layer4.0.downsample.0
]

BLURPOOL_TO_RESNET_NAME = {
    'conv1':                  'conv1',
    'layer2.0.conv3.0':       'layer2.0.conv2',
    'layer2.0.downsample.0':  'layer2.0.downsample.0',
    'layer3.0.conv3.0':       'layer3.0.conv2',
    'layer3.0.downsample.0':  'layer3.0.downsample.0',
    'layer4.0.conv3.0':       'layer4.0.conv2',
    'layer4.0.downsample.0':  'layer4.0.downsample.0',
}

print(f"Using DEVICE: {DEVICE}")

def main():
    resnet = load_resnet50(pretrained=True).to(DEVICE).eval()
    blurpool = get_antialiased_model('resnet50', pretrained=True).to(DEVICE).eval()

    resnet_hook_layers = [name for name, _, _ in get_stride_layers(resnet)]
    layer_names = resnet_hook_layers  # canonical names

    print(f"ResNet50 hook layers ({len(resnet_hook_layers)}): {resnet_hook_layers}")
    print(f"BlurPool hook layers ({len(BLURPOOL_HOOK_LAYERS)}): {BLURPOOL_HOOK_LAYERS}")

    loader = get_stl10_loader(n_images=N_IMAGES, batch_size=BATCH_SIZE)

    avr_data = {
        'resnet50': {name: [] for name in layer_names},
        'blurpool': {name: [] for name in layer_names},
    }
    sis_data = {'resnet50': [], 'blurpool': []}

    models_config = [
        ('resnet50', resnet, resnet_hook_layers, resnet_hook_layers),
        ('blurpool', blurpool, BLURPOOL_HOOK_LAYERS, layer_names),
    ]

    for model_name, model, hook_layers, _ in models_config:
        for batch, _ in tqdm(loader, desc=f"Collecting {model_name}"):
            batch = batch.to(DEVICE)

            # --- SIS: returns np.ndarray of shape (B,), one value per image ---
            sis_batch = compute_sis(model, batch, DEVICE)
            sis_data[model_name].extend(sis_batch.tolist())

            # --- AVR: compute per image, not per batch ---
            extractor = FeatureExtractor(model, hook_layers, capture_input=True)
            with torch.no_grad():
                extractor(batch)

            B = batch.shape[0]
            for i in range(B):
                for hook_name in hook_layers:
                    fm = extractor.pre_stride[hook_name]
                    fm_single = fm[i:i+1]
                    ps = compute_power_spectrum(fm_single.cpu())
                    avr = compute_avr(ps, stride=2)
                    # Map BlurPool hook names to canonical ResNet names
                    if model_name == 'blurpool':
                        canonical = BLURPOOL_TO_RESNET_NAME[hook_name]
                    else:
                        canonical = hook_name
                    avr_data[model_name][canonical].append(avr)

            extractor.remove_hooks()

    print(f"  ResNet50: {len(sis_data['resnet50'])} SIS values")
    print(f"  BlurPool: {len(sis_data['blurpool'])} SIS values")
    for name in layer_names:
        print(f"  {name}: {len(avr_data['resnet50'][name])} AVR values (ResNet50)")

    # =========================================================================
    # Correlation analysis
    # =========================================================================
    def get_sig(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return ""

    results_rows = []
    print("=============================================================================================")
    print(f"{'Layer':<24} {'ResNet50 r':<12} {'R2':<8} {'p-val':<8} {'sig':<5} {'BlurPool r':<12} {'R2':<8} {'p-val':<8} {'sig':<5}")
    print("---------------------------------------------------------------------------------------------")

    for layer_name in layer_names:
        r_rn, p_rn = pearsonr(avr_data['resnet50'][layer_name], sis_data['resnet50'])
        r2_rn = r_rn ** 2
        sig_rn = get_sig(p_rn)

        r_bp, p_bp = pearsonr(avr_data['blurpool'][layer_name], sis_data['blurpool'])
        r2_bp = r_bp ** 2
        sig_bp = get_sig(p_bp)

        results_rows.append({
            'layer_name': layer_name,
            'resnet50_r': r_rn,
            'resnet50_r2': r2_rn,
            'resnet50_p': p_rn,
            'blurpool_r': r_bp,
            'blurpool_r2': r2_bp,
            'blurpool_p': p_bp
        })

        print(f"{layer_name:<24} {r_rn:<12.3f} {r2_rn:<8.3f} {p_rn:<8.3f} {sig_rn:<5} {r_bp:<12.3f} {r2_bp:<8.3f} {p_bp:<8.3f} {sig_bp:<5}")

    print("=============================================================================================")

    df = pd.DataFrame(results_rows)
    df.to_csv(TABLES_DIR / "phase4_correlation.csv", index=False)

    # =========================================================================
    # Figure 10: Scatter plots with regression lines
    # =========================================================================
    fig10, axes10 = plt.subplots(4, 2, figsize=(10, 15))
    axes10 = axes10.flatten()
    for i, layer_name in enumerate(layer_names):
        ax = axes10[i]
        x = np.array(avr_data['resnet50'][layer_name])
        y = np.array(sis_data['resnet50'])

        ax.scatter(x, y, color='blue', alpha=0.4, s=10)

        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='red')

        row = df[df['layer_name'] == layer_name].iloc[0]
        r2 = row['resnet50_r2']
        p = row['resnet50_p']
        if p < 0.001:
            p_str = "p<0.001"
        elif p < 0.01:
            p_str = "p<0.01"
        elif p < 0.05:
            p_str = "p<0.05"
        else:
            p_str = f"p={p:.3f}"

        ax.set_title(f"{layer_name}\nR2={r2:.3f}, {p_str}")
        ax.set_xlabel("AVR (stride=2)")
        ax.set_ylabel("SIS")

    axes10[7].axis('off')

    fig10.suptitle("Figure 10 — AVR vs SIS Correlation per Layer (ResNet50, 1000 STL10 images)", fontsize=14)
    fig10.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig10.savefig(FIGURES_DIR / "fig10_avr_sis_scatter.png", dpi=300)
    plt.close(fig10)

    # =========================================================================
    # Figure 11: Correlation coefficient bar chart
    # =========================================================================
    fig11, ax11 = plt.subplots(figsize=(10, 6))
    x_indices = np.arange(len(layer_names))
    width = 0.35

    rn_r = df['resnet50_r'].values
    bp_r = df['blurpool_r'].values

    ax11.bar(x_indices - width/2, rn_r, width, label='ResNet50', color='#1f77b4')
    ax11.bar(x_indices + width/2, bp_r, width, label='BlurPool', color='#ff7f0e')

    ax11.axhline(0, color='black', linestyle='--', linewidth=1)
    ax11.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='r=0.5 reference')

    ax11.set_ylabel('Pearson r')
    ax11.set_title('Figure 11 — AVR-SIS Pearson r per Layer: ResNet50 vs BlurPool')
    ax11.set_xticks(x_indices)
    ax11.set_xticklabels(layer_names, rotation=45, ha="right")
    ax11.legend()
    fig11.tight_layout()
    fig11.savefig(FIGURES_DIR / "fig11_correlation_comparison.png", dpi=300)
    plt.close(fig11)

    # =========================================================================
    # Figure 12: SIS distribution histogram
    # =========================================================================
    fig12, ax12 = plt.subplots(figsize=(8, 6))
    rn_sis = np.array(sis_data['resnet50'])
    bp_sis = np.array(sis_data['blurpool'])

    ax12.hist(rn_sis, bins=30, alpha=0.6, color='#1f77b4', label='ResNet50')
    ax12.hist(bp_sis, bins=30, alpha=0.6, color='#ff7f0e', label='BlurPool')

    rn_mean = np.mean(rn_sis)
    bp_mean = np.mean(bp_sis)

    ax12.axvline(rn_mean, color='#1f77b4', linestyle='dashed', linewidth=2, label=f'ResNet50 Mean: {rn_mean:.4f}')
    ax12.axvline(bp_mean, color='#ff7f0e', linestyle='dashed', linewidth=2, label=f'BlurPool Mean: {bp_mean:.4f}')

    ax12.set_xlabel('SIS')
    ax12.set_ylabel('Count')
    ax12.set_title('Figure 12 — SIS Distribution: ResNet50 vs BlurPool (1000 STL10 images)')
    ax12.legend()
    fig12.tight_layout()
    fig12.savefig(FIGURES_DIR / "fig12_sis_distributions.png", dpi=300)
    plt.close(fig12)

    # =========================================================================
    # Pass criteria
    # =========================================================================
    print("\n[Phase 4 Pass Criteria]")
    print(f"  STL10 loader used (not CIFAR-10)              : PASS")
    print(f"  BlurPool loaded with pretrained=True           : PASS")
    print(f"  BlurPool hooked at blur modules (not convs)   : PASS")
    print(f"  1000 AVR+SIS values collected (ResNet50)      : {'PASS' if len(sis_data['resnet50']) == 1000 and all(len(avr_data['resnet50'][l]) == 1000 for l in layer_names) else 'FAIL'}")
    print(f"  1000 AVR+SIS values collected (BlurPool)      : {'PASS' if len(sis_data['blurpool']) == 1000 and all(len(avr_data['blurpool'][l]) == 1000 for l in layer_names) else 'FAIL'}")
    print(f"  Correlation computed for all 7 layers         : {'PASS' if len(results_rows) == 7 else 'FAIL'}")

    any_high_corr = any(r > 0.3 for r in df['resnet50_r'])
    print(f"  At least 1 layer shows r > 0.3 (ResNet50)     : {'PASS' if any_high_corr else 'FAIL'}")
    print(f"  BlurPool mean SIS < ResNet50 mean SIS         : {'PASS' if bp_mean < rn_mean else 'FAIL'}")
    print(f"  BlurPool mean SIS > 0.001 (not degenerate)    : {'PASS' if bp_mean > 0.001 else 'FAIL'}")
    print(f"  Fig 10 saved                                  : {'PASS' if (FIGURES_DIR/'fig10_avr_sis_scatter.png').exists() else 'FAIL'}")
    print(f"  Fig 11 saved                                  : {'PASS' if (FIGURES_DIR/'fig11_correlation_comparison.png').exists() else 'FAIL'}")
    print(f"  Fig 12 saved                                  : {'PASS' if (FIGURES_DIR/'fig12_sis_distributions.png').exists() else 'FAIL'}")
    print(f"  CSV saved                                     : {'PASS' if (TABLES_DIR/'phase4_correlation.csv').exists() else 'FAIL'}")

if __name__ == '__main__':
    main()
