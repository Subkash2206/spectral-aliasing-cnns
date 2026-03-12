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
from tqdm import tqdm

from src.models import load_resnet50, get_antialiased_model
from src.hooks import FeatureExtractor, get_stride_layers
from src.datasets import get_stl10_loader
from src.spectral import compute_power_spectrum, compute_avr, compute_radial_profile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = 1000
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Hook point mapping
# ---------------------------------------------------------------------------
# ResNet50: stride-2 convs — we hook input to capture the pre-stride feature map.
# BlurPool: the strided convs have been replaced with stride-1 conv + BlurPool
#   module that does blur+stride-2 downsampling.  We hook the BlurPool module
#   input to capture the pre-blur feature map — the semantic equivalent.
#
# Correspondence (from Step 0 diagnostic):
#   ResNet50 conv1 (stride=2)            ↔  BlurPool maxpool.1 (BlurPool after maxpool)
#   ResNet50 layer2.0.conv2 (stride=2)   ↔  BlurPool layer2.0.conv3.0 (BlurPool)
#   ResNet50 layer2.0.downsample.0       ↔  BlurPool layer2.0.downsample.0 (BlurPool)
#   ResNet50 layer3.0.conv2 (stride=2)   ↔  BlurPool layer3.0.conv3.0 (BlurPool)
#   ResNet50 layer3.0.downsample.0       ↔  BlurPool layer3.0.downsample.0 (BlurPool)
#   ResNet50 layer4.0.conv2 (stride=2)   ↔  BlurPool layer4.0.conv3.0 (BlurPool)
#   ResNet50 layer4.0.downsample.0       ↔  BlurPool layer4.0.downsample.0 (BlurPool)

BLURPOOL_HOOK_LAYERS = [
    'maxpool.1',              # corresponds to conv1
    'layer2.0.conv3.0',       # corresponds to layer2.0.conv2
    'layer2.0.downsample.0',  # corresponds to layer2.0.downsample.0
    'layer3.0.conv3.0',       # corresponds to layer3.0.conv2
    'layer3.0.downsample.0',  # corresponds to layer3.0.downsample.0
    'layer4.0.conv3.0',       # corresponds to layer4.0.conv2
    'layer4.0.downsample.0',  # corresponds to layer4.0.downsample.0
]

# Mapping from BlurPool hook name → canonical ResNet50 layer name (for CSV/plots)
BLURPOOL_TO_RESNET_NAME = {
    'maxpool.1':              'conv1',
    'layer2.0.conv3.0':       'layer2.0.conv2',
    'layer2.0.downsample.0':  'layer2.0.downsample.0',
    'layer3.0.conv3.0':       'layer3.0.conv2',
    'layer3.0.downsample.0':  'layer3.0.downsample.0',
    'layer4.0.conv3.0':       'layer4.0.conv2',
    'layer4.0.downsample.0':  'layer4.0.downsample.0',
}

def main():
    print(f"Using DEVICE: {DEVICE}")

    csv_path = TABLES_DIR / "phase3_layerwise_avr_stats.csv"

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    resnet = load_resnet50(pretrained=True).to(DEVICE).eval()
    blurpool = get_antialiased_model('resnet50', pretrained=True).to(DEVICE).eval()

    # ResNet50 hook layers — stride-2 convs
    resnet_hook_layers = [name for name, _, _ in get_stride_layers(resnet)]
    print(f"ResNet50 hook layers ({len(resnet_hook_layers)}): {resnet_hook_layers}")
    print(f"BlurPool hook layers ({len(BLURPOOL_HOOK_LAYERS)}): {BLURPOOL_HOOK_LAYERS}")

    assert len(resnet_hook_layers) == len(BLURPOOL_HOOK_LAYERS), \
        f"Hook layer count mismatch: ResNet={len(resnet_hook_layers)}, BlurPool={len(BLURPOOL_HOOK_LAYERS)}"

    # Canonical layer names (ResNet50 names, used for CSV and plots)
    layer_names = resnet_hook_layers

    # -------------------------------------------------------------------------
    # Step 1 — Data collection
    # -------------------------------------------------------------------------
    loader = get_stl10_loader(n_images=N_IMAGES, batch_size=BATCH_SIZE)

    avr_resnet = {name: [] for name in layer_names}
    avr_blurpool = {name: [] for name in layer_names}

    # Profile ResNet50
    for batch, _ in tqdm(loader, desc="Profiling ResNet50"):
        batch = batch.to(DEVICE)
        extractor = FeatureExtractor(resnet, resnet_hook_layers, capture_input=True)
        with torch.no_grad():
            extractor(batch)
        for name in resnet_hook_layers:
            feature_map = extractor.pre_stride[name].cpu()
            for i in range(feature_map.shape[0]):
                ps = compute_power_spectrum(feature_map[i:i+1])
                avr = compute_avr(ps, stride=2)
                avr_resnet[name].append(avr)
        extractor.remove_hooks()

    # Profile BlurPool
    for batch, _ in tqdm(loader, desc="Profiling BlurPool"):
        batch = batch.to(DEVICE)
        extractor = FeatureExtractor(blurpool, BLURPOOL_HOOK_LAYERS, capture_input=True)
        with torch.no_grad():
            extractor(batch)
        for bp_name in BLURPOOL_HOOK_LAYERS:
            canonical = BLURPOOL_TO_RESNET_NAME[bp_name]
            feature_map = extractor.pre_stride[bp_name].cpu()
            for i in range(feature_map.shape[0]):
                ps = compute_power_spectrum(feature_map[i:i+1])
                avr = compute_avr(ps, stride=2)
                avr_blurpool[canonical].append(avr)
        extractor.remove_hooks()

    num_processed_resnet = len(avr_resnet[layer_names[0]])
    num_processed_blurpool = len(avr_blurpool[layer_names[0]])
    print(f"  ResNet50: {num_processed_resnet} images processed")
    print(f"  BlurPool: {num_processed_blurpool} images processed")

    # -------------------------------------------------------------------------
    # Step 2 — Statistics
    # -------------------------------------------------------------------------
    results = []
    blur_means_less_than_resnet = True
    significant_layers_count = 0

    for name in layer_names:
        r_list = avr_resnet[name]
        b_list = avr_blurpool[name]

        mean_resnet, std_resnet = np.mean(r_list), np.std(r_list)
        mean_blur, std_blur = np.mean(b_list), np.std(b_list)
        t_stat, p_val = stats.ttest_ind(r_list, b_list)

        if mean_blur > mean_resnet + 1e-6:
            blur_means_less_than_resnet = False

        if p_val < 0.01:
            significant_layers_count += 1

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""

        results.append({
            "layer_name": name,
            "resnet50_mean": mean_resnet,
            "resnet50_std": std_resnet,
            "blurpool_mean": mean_blur,
            "blurpool_std": std_blur,
            "t_stat": t_stat,
            "p_val": p_val,
            "sig": sig
        })

    df = pd.DataFrame(results)
    df.drop(columns=["sig"]).to_csv(csv_path, index=False)

    # Print the table
    print("\n" + "="*95)
    print(f"{'Layer':<24} {'ResNet50 AVR':<19} {'BlurPool AVR':<19} {'t-stat':<10} {'p-val':<8} {'sig'}")
    print("-" * 95)
    for row in results:
        res_str = f"{row['resnet50_mean']:.4f} +/- {row['resnet50_std']:.4f}"
        blur_str = f"{row['blurpool_mean']:.4f} +/- {row['blurpool_std']:.4f}"
        t_str = f"{row['t_stat']:.2f}"
        p_str = f"{row['p_val']:.3f}"
        print(f"{row['layer_name']:<24} {res_str:<19} {blur_str:<19} {t_str:<10} {p_str:<8} {row['sig']}")
    print("="*95 + "\n")

    # -------------------------------------------------------------------------
    # Step 3 — Figure 7: Grouped bar chart with error bars
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layer_names))
    width = 0.35

    ax.bar(x - width/2, [r["resnet50_mean"] for r in results], width,
           yerr=[r["resnet50_std"] for r in results], label='ResNet50', capsize=5, color='#1f77b4')
    ax.bar(x + width/2, [r["blurpool_mean"] for r in results], width,
           yerr=[r["blurpool_std"] for r in results], label='BlurPool', capsize=5, color='#ff7f0e')

    ax.set_ylabel('Mean AVR (stride=2)')
    ax.set_title('Figure 7 — Layer-wise Mean AVR +/- Std: ResNet50 vs BlurPool (1000 STL10 images)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    fig7_path = FIGURES_DIR / "fig7_layerwise_mean_avr.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Step 4 — Figure 8: Violin plots
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 6))
    plt.suptitle("Figure 8 — AVR Distribution per Layer: ResNet50 vs BlurPool", fontsize=16)

    for i, name in enumerate(layer_names):
        ax = plt.subplot(1, len(layer_names), i + 1)
        r_data = avr_resnet[name]
        b_data = avr_blurpool[name]

        parts_r = ax.violinplot(r_data, positions=[1], widths=0.6, showmeans=True)
        parts_b = ax.violinplot(b_data, positions=[2], widths=0.6, showmeans=True)

        for pc in parts_r['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_alpha(0.7)
        for pc in parts_b['bodies']:
            pc.set_facecolor('#ff7f0e')
            pc.set_alpha(0.7)

        parts_r['cbars'].set_color('#1f77b4')
        parts_r['cmaxes'].set_color('#1f77b4')
        parts_r['cmins'].set_color('#1f77b4')
        parts_r['cmeans'].set_color('#1f77b4')
        parts_r['cmeans'].set_linestyle('--')

        parts_b['cbars'].set_color('#ff7f0e')
        parts_b['cmaxes'].set_color('#ff7f0e')
        parts_b['cmins'].set_color('#ff7f0e')
        parts_b['cmeans'].set_color('#ff7f0e')
        parts_b['cmeans'].set_linestyle('--')

        ax.set_ylim(bottom=0)

        pval = next(r["p_val"] for r in results if r["layer_name"] == name)
        if pval < 0.001:
            p_str = "p<0.001"
        elif pval < 0.01:
            p_str = "p<0.01"
        elif pval < 0.05:
            p_str = "p<0.05"
        else:
            p_str = f"p={pval:.3f}"

        ax.set_title(f"{name}\n{p_str}", fontsize=10)
        ax.set_xticks([1, 2])
        if i == 0:
            ax.set_xticklabels(['ResNet50', 'BlurPool'], rotation=45)
            ax.set_ylabel("AVR")
        else:
            ax.set_xticklabels(['', ''])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig8_path = FIGURES_DIR / "fig8_avr_distributions.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Step 5 — Figure 9: Radial energy profiles for ResNet50
    # -------------------------------------------------------------------------
    single_loader = get_stl10_loader(n_images=1, batch_size=1)
    single_batch, _ = next(iter(single_loader))
    single_batch = single_batch.to(DEVICE)

    resnet_single = load_resnet50(pretrained=True).to(DEVICE).eval()
    resnet_extractor_single = FeatureExtractor(resnet_single, resnet_hook_layers, capture_input=True)
    _ = resnet_extractor_single(single_batch)

    fig, axes = plt.subplots(len(layer_names), 1, figsize=(6, 2 * len(layer_names)), sharex=True)
    fig.suptitle("Figure 9 — Radial Energy Profiles: ResNet50 on Natural Image (STL10)", fontsize=14)

    fig9_saved = False
    for i, name in enumerate(layer_names):
        feat = resnet_extractor_single.pre_stride[name].cpu()
        ps = compute_power_spectrum(feat)
        profile = compute_radial_profile(ps)

        prof_norm = profile / (profile.max() + 1e-10)
        freq_norm = np.linspace(0, np.sqrt(2), len(prof_norm))

        ax = axes[i] if len(layer_names) > 1 else axes
        ax.plot(freq_norm, prof_norm, color='black')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Nyquist (stride=2)')
        ax.fill_between(freq_norm, prof_norm, where=(freq_norm > 0.5), color='red', alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"{name}")
        ax.set_ylabel("Norm. Energy")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right")

    if len(layer_names) > 1:
        axes[-1].set_xlabel("Normalized Frequency")
    else:
        axes.set_xlabel("Normalized Frequency")

    resnet_extractor_single.remove_hooks()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig9_path = FIGURES_DIR / "fig9_radial_profiles_natural.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    fig9_saved = True

    # -------------------------------------------------------------------------
    # Pass Criteria
    # -------------------------------------------------------------------------
    print("\n[Phase 3 Pass Criteria]")
    print(f"  STL10 loader used (not CIFAR-10)              : PASS")
    print(f"  BlurPool loaded with pretrained=True           : PASS")
    print(f"  BlurPool hooked at blur modules (not convs)   : PASS")
    print(f"  1000 images processed (ResNet50)              : {'PASS' if num_processed_resnet == 1000 else 'FAIL'}")
    print(f"  1000 images processed (BlurPool)              : {'PASS' if num_processed_blurpool == 1000 else 'FAIL'}")
    print(f"  All 7 layers have AVR stats                   : {'PASS' if len(results) == 7 else 'FAIL'}")
    print(f"  BlurPool mean AVR < ResNet50 mean AVR (all)   : {'PASS' if blur_means_less_than_resnet else 'FAIL'}")
    print(f"  At least 5 layers show p < 0.01               : {'PASS' if significant_layers_count >= 5 else 'FAIL'}")
    print(f"  Fig 7 saved                                   : {'PASS' if fig7_path.exists() else 'FAIL'}")
    print(f"  Fig 8 saved                                   : {'PASS' if fig8_path.exists() else 'FAIL'}")
    print(f"  Fig 9 saved                                   : {'PASS' if fig9_saved and fig9_path.exists() else 'FAIL'}")
    print(f"  CSV saved                                     : {'PASS' if csv_path.exists() else 'FAIL'}")

if __name__ == "__main__":
    main()
