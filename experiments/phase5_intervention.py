"""
phase5_intervention.py — Phase 5: Anti-Aliasing Intervention Analysis

Goal: Quantify exactly what BlurPool achieves and at what cost.
Four metrics: AVR reduction, SIS reduction, accuracy, inference time.
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
import time
from tqdm import tqdm

from src.models import load_resnet50, get_antialiased_model
from src.hooks import FeatureExtractor, get_stride_layers
from src.datasets import get_stl10_loader
from src.metrics import compute_sis
from src.spectral import compute_power_spectrum, compute_radial_profile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using DEVICE: {DEVICE}")

def main():
    # Step 1 — Load models and data
    resnet = load_resnet50(pretrained=True).to(DEVICE).eval()
    blurpool = get_antialiased_model('resnet50', pretrained=True).to(DEVICE).eval()

    def get_blurpool_hook_layers():
        return [
            'conv1',
            'layer2.0.conv3.0',
            'layer2.0.downsample.0',
            'layer3.0.conv3.0',
            'layer3.0.downsample.0',
            'layer4.0.conv3.0',
            'layer4.0.downsample.0',
        ]

    resnet_layer_names = [name for name, _, _ in get_stride_layers(resnet)]
    blurpool_layer_names = get_blurpool_hook_layers()

    # 1000 images is 1000/32 = 32 batches
    # We will use the test split if available, or just the standard loader
    loader = get_stl10_loader(n_images=1000, batch_size=32)

    # Step 2 — Load Phase 3 AVR stats from CSV 
    phase3_df = pd.read_csv(TABLES_DIR / "phase3_layerwise_avr_stats.csv")
    avr_reduction = {}
    for _, row in phase3_df.iterrows():
        name = row['layer_name']
        rn = row['resnet50_mean']
        bp = row['blurpool_mean']
        avr_reduction[name] = rn - bp

    # Step 3 & 4 — Compute SIS and Accuracy together
    sis_resnet = []
    sis_blurpool = []
    resnet_correct = 0
    blurpool_correct = 0
    agreement = 0
    total = 0

    for batch, labels in tqdm(loader, desc="Computing SIS & Acc"):
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # SIS
        sis_resnet.extend(compute_sis(resnet, batch, DEVICE).tolist())
        sis_blurpool.extend(compute_sis(blurpool, batch, DEVICE).tolist())
        
        # Acc
        with torch.no_grad():
            rn_pred = resnet(batch).argmax(dim=-1)
            bp_pred = blurpool(batch).argmax(dim=-1)
        resnet_correct += (rn_pred == labels).sum().item()
        blurpool_correct += (bp_pred == labels).sum().item()
        agreement += (rn_pred == bp_pred).sum().item()
        total += labels.size(0)

    mean_sis_resnet = np.mean(sis_resnet)
    mean_sis_blurpool = np.mean(sis_blurpool)
    sis_reduction_pct = (mean_sis_resnet - mean_sis_blurpool) / mean_sis_resnet * 100

    print(f"ResNet50 mean SIS: {mean_sis_resnet:.4f}")
    print(f"BlurPool mean SIS: {mean_sis_blurpool:.4f}")
    print(f"SIS reduction: {sis_reduction_pct:.1f}%")

    resnet_acc = resnet_correct / total * 100
    blurpool_acc = blurpool_correct / total * 100
    agreement_pct = agreement / total * 100

    print(f"ResNet50 top-1 acc (STL10): {resnet_acc:.1f}%")
    print(f"BlurPool top-1 acc (STL10): {blurpool_acc:.1f}%")
    print(f"Prediction agreement: {agreement_pct:.1f}%")

    # Step 5 — Inference time
    # Warm up GPU
    dummy = torch.randn(32, 3, 224, 224).to(DEVICE)
    for _ in range(5):
        with torch.no_grad():
            resnet(dummy)
            blurpool(dummy)

    # Time 100 batches of 32 images
    N_TIMING_BATCHES = 100
    timing_batch = torch.randn(32, 3, 224, 224).to(DEVICE)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMING_BATCHES):
        with torch.no_grad():
            resnet(timing_batch)
    torch.cuda.synchronize()
    resnet_time = (time.perf_counter() - t0) / (N_TIMING_BATCHES * 32) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMING_BATCHES):
        with torch.no_grad():
            blurpool(timing_batch)
    torch.cuda.synchronize()
    blurpool_time = (time.perf_counter() - t0) / (N_TIMING_BATCHES * 32) * 1000

    time_overhead_pct = (blurpool_time - resnet_time) / resnet_time * 100

    print(f"ResNet50 inference: {resnet_time:.3f} ms/image")
    print(f"BlurPool inference: {blurpool_time:.3f} ms/image")
    print(f"Time overhead: {time_overhead_pct:.1f}%")

    # Step 6 — Tradeoff table
    rows = []
    for _, row in phase3_df.iterrows():
        name = row['layer_name']
        rn_avr = row['resnet50_mean']
        bp_avr = row['blurpool_mean']
        diff = rn_avr - bp_avr
        rows.append({
            'layer': name,
            'resnet50_avr': rn_avr,
            'blurpool_avr': bp_avr,
            'avr_diff': diff,
        })

    tradeoff_df = pd.DataFrame(rows)
    # add scalar metrics to every row or save separately? Following instructions explicitly:
    # "tradeoff_df['sis_resnet'] = mean_sis_resnet"
    tradeoff_df['sis_resnet'] = mean_sis_resnet
    tradeoff_df['sis_blurpool'] = mean_sis_blurpool
    tradeoff_df['sis_reduction_pct'] = sis_reduction_pct
    tradeoff_df['resnet_acc'] = resnet_acc
    tradeoff_df['blurpool_acc'] = blurpool_acc
    tradeoff_df['resnet_time_ms'] = resnet_time
    tradeoff_df['blurpool_time_ms'] = blurpool_time
    tradeoff_df['time_overhead_pct'] = time_overhead_pct

    csv_path = TABLES_DIR / "phase5_tradeoff.csv"
    tradeoff_df.to_csv(csv_path, index=False)

    print("\n=== TRADEOFF TABLE ===")
    print(f"{'Metric':<35} {'ResNet50':<15} {'BlurPool':<15} {'Delta'}")
    print("-" * 75)
    print(f"{'Mean SIS':<35} {mean_sis_resnet:<15.4f} {mean_sis_blurpool:<15.4f} {-sis_reduction_pct:+.1f}%")
    print(f"{'Top-1 Acc (STL10)':<35} {resnet_acc:<15.1f} {blurpool_acc:<15.1f} {blurpool_acc-resnet_acc:+.1f}%")
    print(f"{'Inference time (ms/img)':<35} {resnet_time:<15.3f} {blurpool_time:<15.3f} {time_overhead_pct:+.1f}%")
    print("-" * 75)
    for _, row in phase3_df.iterrows():
        diff = row['resnet50_mean'] - row['blurpool_mean']
        print(f"  AVR {row['layer_name']:<28} {row['resnet50_mean']:<15.4f} {row['blurpool_mean']:<15.4f} {diff:+.4f}")
    print("=" * 75)

    # Step 7 — Figure 13: Paired AVR bar chart
    fig13, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phase3_df))
    width = 0.35

    rn_means = phase3_df['resnet50_mean'].values
    bp_means = phase3_df['blurpool_mean'].values
    rn_stds = phase3_df['resnet50_std'].values
    bp_stds = phase3_df['blurpool_std'].values
    layer_names = phase3_df['layer_name'].values

    ax.bar(x - width/2, rn_means, width, yerr=rn_stds, label='ResNet50', capsize=5, color='#1f77b4')
    ax.bar(x + width/2, bp_means, width, yerr=bp_stds, label='BlurPool', capsize=5, color='#ff7f0e')

    ax.set_ylabel('Mean AVR')
    ax.set_title('Figure 13 — Layer-wise AVR: ResNet50 vs BlurPool (Pre-downsampling, STL10)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    fig13.tight_layout()
    fig13_path = FIGURES_DIR / "fig13_avr_intervention.png"
    fig13.savefig(fig13_path, dpi=300)
    plt.close(fig13)

    # Step 8 — Figure 14: Tradeoff scatter
    fig14, ax = plt.subplots(figsize=(8, 6))
    
    # Points
    ax.scatter([mean_sis_resnet], [resnet_time], color='#1f77b4', s=150, zorder=5)
    ax.scatter([mean_sis_blurpool], [blurpool_time], color='#ff7f0e', s=150, zorder=5)
    
    # Connect with dashed line
    ax.plot([mean_sis_resnet, mean_sis_blurpool], [resnet_time, blurpool_time], 'k--', alpha=0.5, zorder=1)

    # Text annotations
    ax.text(mean_sis_resnet, resnet_time + (blurpool_time-resnet_time)*0.05, 
            f"ResNet50", ha='center', va='bottom', fontsize=10, color='#1f77b4')
    ax.text(mean_sis_blurpool, blurpool_time - (blurpool_time-resnet_time)*0.05, 
            f"BlurPool", ha='center', va='top', fontsize=10, color='#ff7f0e')

    # Annotation for deltas
    mid_x = (mean_sis_resnet + mean_sis_blurpool) / 2
    mid_y = (resnet_time + blurpool_time) / 2
    ax.annotate(f"SIS: {-sis_reduction_pct:+.1f}%\nTime: {time_overhead_pct:+.1f}%\nAgreement: {agreement_pct:.1f}%",
                xy=(mid_x, mid_y), xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel('Mean SIS (Lower is Better)')
    ax.set_ylabel('Inference Time ms/image (Lower is Better)')
    ax.set_title('Figure 14 — Shift Stability vs Inference Cost: ResNet50 vs BlurPool')
    ax.grid(True, alpha=0.3)
    # Add buffer to limits
    xspan = mean_sis_resnet - mean_sis_blurpool 
    yspan = blurpool_time - resnet_time
    ax.set_xlim(mean_sis_blurpool - xspan*0.5, mean_sis_resnet + xspan*0.5)
    ax.set_ylim(resnet_time - yspan*0.5, blurpool_time + yspan*0.5)

    fig14.tight_layout()
    fig14_path = FIGURES_DIR / "fig14_tradeoff.png"
    fig14.savefig(fig14_path, dpi=300)
    plt.close(fig14)

    # Step 9 — Figure 15: Radial profile overlay
    single_loader = get_stl10_loader(n_images=1, batch_size=1)
    single_batch, _ = next(iter(single_loader))
    single_batch = single_batch.to(DEVICE)

    resnet_extractor = FeatureExtractor(resnet, resnet_layer_names, capture_input=True)
    blurpool_extractor = FeatureExtractor(blurpool, blurpool_layer_names, capture_input=True)

    with torch.no_grad():
        resnet_extractor(single_batch)
        blurpool_extractor(single_batch)

    fig15, axes = plt.subplots(len(resnet_layer_names), 1, figsize=(6, 2 * len(resnet_layer_names)), sharex=True)
    fig15.suptitle("Figure 15 — Radial Energy Profile Overlay: ResNet50 vs BlurPool (STL10)", fontsize=14)

    for i, (rn_name, bp_name) in enumerate(zip(resnet_layer_names, blurpool_layer_names)):
        ax = axes[i]
        
        # ResNet
        rn_feat = resnet_extractor.pre_stride[rn_name].cpu()
        rn_ps = compute_power_spectrum(rn_feat)
        rn_prof = compute_radial_profile(rn_ps)
        rn_prof_norm = rn_prof / (rn_prof.max() + 1e-10)
        freq_norm = np.linspace(0, np.sqrt(2), len(rn_prof_norm))
        ax.plot(freq_norm, rn_prof_norm, color='#1f77b4', label='ResNet50')

        # BlurPool
        bp_feat = blurpool_extractor.pre_stride[bp_name].cpu()
        bp_ps = compute_power_spectrum(bp_feat)
        bp_prof = compute_radial_profile(bp_ps)
        bp_prof_norm = bp_prof / (bp_prof.max() + 1e-10)
        ax.plot(freq_norm, bp_prof_norm, color='#ff7f0e', label='BlurPool', alpha=0.8)

        ax.axvline(x=0.5, color='red', linestyle='--', label='Nyquist', alpha=0.5)
        ax.set_xlim(0, 1.0)
        ax.set_title(rn_name)
        ax.set_ylabel("Norm. Energy")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Normalized Frequency")
    
    resnet_extractor.remove_hooks()
    blurpool_extractor.remove_hooks()

    fig15.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig15_path = FIGURES_DIR / "fig15_radial_overlay.png"
    fig15.savefig(fig15_path, dpi=300)
    plt.close(fig15)

    # Pass Criteria
    print("\n[Phase 5 Pass Criteria]")
    print(f"  SIS computed for both models (1000 images)    : {'PASS' if len(sis_resnet)==1000 and len(sis_blurpool)==1000 else 'FAIL'}")
    print(f"  Accuracy computed for both models             : {'PASS' if total == 1000 else 'FAIL'}")
    print(f"  Inference time measured for both models       : {'PASS' if resnet_time > 0 and blurpool_time > 0 else 'FAIL'}")
    print(f"  Tradeoff table printed                        : PASS")
    print(f"  BlurPool mean SIS < ResNet50 mean SIS         : {'PASS' if mean_sis_blurpool < mean_sis_resnet else 'FAIL'}")
    print(f"  Fig 13 saved                                  : {'PASS' if fig13_path.exists() else 'FAIL'}")
    print(f"  Fig 14 saved                                  : {'PASS' if fig14_path.exists() else 'FAIL'}")
    print(f"  Fig 15 saved                                  : {'PASS' if fig15_path.exists() else 'FAIL'}")
    print(f"  CSV saved                                     : {'PASS' if csv_path.exists() else 'FAIL'}")


if __name__ == '__main__':
    main()
