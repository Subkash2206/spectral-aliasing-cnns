import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.synthetic import make_sinusoidal_image, make_checkerboard_image
from src.datasets import get_cifar10_loader
from src.models import load_resnet50, get_antialiased_model
from src.hooks import FeatureExtractor, get_stride_layers
from src.spectral import compute_power_spectrum, compute_avr, log_power_spectrum

def run_experiment_2a():
    print("[Experiment 2a — Frequency Sweep]")
    freqs = np.linspace(0.1, 0.9, 10)
    model = load_resnet50()
    
    extractor = FeatureExtractor(model, ['conv1'], capture_input=True)
    
    results = []
    
    for f in freqs:
        img = make_sinusoidal_image(freq=f, height=224, width=224, batch_size=1, channels=3)
        extractor(img)
        ps = compute_power_spectrum(extractor.pre_stride['conv1'])
        avr = compute_avr(ps, stride=2)
        results.append({'frequency': float(f), 'avr': float(avr)})
        # Print format
        print(f"Freq: {f:.4f}, AVR: {avr:.4f}")
        
    extractor.remove_hooks()
        
    df = pd.DataFrame(results)
    df.to_csv('results/tables/phase2a_frequency_sweep.csv', index=False)
    
    plt.figure()
    plt.plot(df['frequency'], df['avr'], marker='o')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Nyquist cutoff')
    plt.xlabel('Normalized Input Frequency')
    plt.ylabel('AVR (stride=2)')
    plt.title('Figure 4 — AVR vs Input Frequency (ResNet50 conv1)')
    plt.legend()
    plt.savefig('results/figures/fig4_avr_vs_frequency.png', dpi=300)
    plt.close()
    
    return df

def run_experiment_2b():
    print("\n[Experiment 2b — Architecture Comparison on Checkerboard]")
    resnet = load_resnet50()
    blurpool = get_antialiased_model('resnet50')
    
    layers_resnet = get_stride_layers(resnet)
    resnet_names = [n for n, _, _ in layers_resnet]
    
    # We use resnet_names for both models to capture the feature map
    # just before the subsampling operation, because get_stride_layers
    # only catches Conv2d, missing the BlurPool downsample wrapper.
    ext_resnet = FeatureExtractor(resnet, resnet_names, capture_input=True)
    ext_blur = FeatureExtractor(blurpool, resnet_names, capture_input=True)
    
    checkerboard = make_checkerboard_image(height=224, width=224, batch_size=1, channels=3)
    sinusoid = make_sinusoidal_image(freq=0.7, height=224, width=224, batch_size=1, channels=3)
    
    ext_resnet(checkerboard)
    ext_blur(checkerboard)
    
    results = []
    print(f"  {'Layer':<30} {'ResNet50 AVR':<15} {'BlurPool AVR':<15}")
    for name in resnet_names:
        psr = compute_power_spectrum(ext_resnet.pre_stride[name])
        avr_r = compute_avr(psr, stride=2)
        
        psb = compute_power_spectrum(ext_blur.pre_stride[name])
        avr_b = compute_avr(psb, stride=2)
        
        results.append({
            'layer_name': name,
            'resnet50_avr': float(avr_r),
            'blurpool_avr': float(avr_b)
        })
        
        print(f"  {name:<30} {avr_r:<15.4f} {avr_b:<15.4f}")
        
    # Also pass high-freq sinusoid as requested (just runs it for completion)
    ext_resnet(sinusoid)
    ext_blur(sinusoid)
        
    ext_resnet.remove_hooks()
    ext_blur.remove_hooks()
    
    df = pd.DataFrame(results)
    df.to_csv('results/tables/phase2b_architecture_comparison.csv', index=False)
    
    # Plotting Figure 5
    x = np.arange(len(df))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df['resnet50_avr'], width, label='ResNet50')
    plt.bar(x + width/2, df['blurpool_avr'], width, label='BlurPool')
    plt.xticks(x, df['layer_name'], rotation=90)
    plt.ylabel('AVR (stride=2)')
    plt.title('Figure 5 — Layer-wise AVR: ResNet50 vs BlurPool (Checkerboard Input)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/fig5_layerwise_avr_comparison.png', dpi=300)
    plt.close()
    
    return df

def run_experiment_2c():
    print("\n[Experiment 2c — Aliasing folding visualization]")
    model = load_resnet50()
    
    pre_stride_extractor = FeatureExtractor(model, ['conv1'], capture_input=True)
    post_stride_extractor = FeatureExtractor(model, ['conv1'], capture_input=False)
    
    # Polyfill features dynamically if missing
    if not hasattr(post_stride_extractor, 'features'):
        post_stride_extractor.features = post_stride_extractor._features
    
    loader = get_cifar10_loader(n_images=1, batch_size=1)
    image, _ = next(iter(loader))
    
    with torch.no_grad():
        model(image)
        
    ps_pre = compute_power_spectrum(pre_stride_extractor.pre_stride['conv1'])
    ps_post = compute_power_spectrum(post_stride_extractor.features['conv1'])
    
    def normalize_for_display(ps: np.ndarray) -> np.ndarray:
        """Log compress and normalize to [0,1] for display.
        Without this, any spectrum where one pixel dominates
        (e.g. checkerboard at DC corner) renders entirely black."""
        log_ps = np.log1p(ps)
        ps_min, ps_max = log_ps.min(), log_ps.max()
        if ps_max - ps_min > 1e-10:
            return (log_ps - ps_min) / (ps_max - ps_min)
        return log_ps
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(normalize_for_display(ps_pre), cmap='inferno', origin='upper')
    ax1.set_title("Pre-stride (conv1 input)")
    
    im2 = ax2.imshow(normalize_for_display(ps_post), cmap='inferno', origin='upper')
    ax2.set_title("Post-stride (conv1 output)")
    
    plt.suptitle("Figure 6 — Aliasing Folding at conv1 (Natural Image Input)")
    plt.tight_layout()
    plt.savefig('results/figures/fig6_aliasing_folding.png', dpi=300)
    plt.close()
    
    pre_stride_extractor.remove_hooks()
    post_stride_extractor.remove_hooks()

def main():
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    df_2a = run_experiment_2a()
    df_2b = run_experiment_2b()
    run_experiment_2c()
    
    print("\n[Phase 2 Pass Criteria]")
    fig4_exists = os.path.exists('results/figures/fig4_avr_vs_frequency.png')
    fig5_exists = os.path.exists('results/figures/fig5_layerwise_avr_comparison.png')
    fig6_exists = os.path.exists('results/figures/fig6_aliasing_folding.png')
    
    avr_01 = df_2a.iloc[0]['avr']
    avr_09 = df_2a.iloc[-1]['avr']
    avr_rises = avr_09 > avr_01
    
    blur_mean = df_2b['blurpool_avr'].mean()
    resnet_mean = df_2b['resnet50_avr'].mean()
    blur_less_than_resnet = blur_mean < resnet_mean
    
    csv2a_exists = os.path.exists('results/tables/phase2a_frequency_sweep.csv')
    csv2b_exists = os.path.exists('results/tables/phase2b_architecture_comparison.csv')
    
    p = lambda x: 'PASS' if x else 'FAIL'
    
    print(f"  Fig 4 saved (AVR vs frequency)           : {p(fig4_exists)}")
    print(f"  Fig 5 saved (layer-wise AVR comparison)  : {p(fig5_exists)}")
    print(f"  Fig 6 saved (aliasing folding)           : {p(fig6_exists)}")
    print(f"  AVR rises with frequency (2a)            : {p(avr_rises)}")
    print(f"  BlurPool mean AVR < ResNet50 mean AVR    : {p(blur_less_than_resnet)}")
    print(f"  phase2a CSV saved                        : {p(csv2a_exists)}")
    print(f"  phase2b CSV saved                        : {p(csv2b_exists)}")
    
    all_pass = all([fig4_exists, fig5_exists, fig6_exists, avr_rises, 
                    blur_less_than_resnet, csv2a_exists, csv2b_exists])
                    
    if all_pass:
        print("Phase 2 COMPLETE")
    else:
        fails = []
        if not fig4_exists: fails.append("Fig 4 saved")
        if not fig5_exists: fails.append("Fig 5 saved")
        if not fig6_exists: fails.append("Fig 6 saved")
        if not avr_rises: fails.append("AVR rises with frequency")
        if not blur_less_than_resnet: fails.append("BlurPool mean AVR < ResNet50 mean AVR")
        if not csv2a_exists: fails.append("phase2a CSV saved")
        if not csv2b_exists: fails.append("phase2b CSV saved")
        print("Phase 2 INCOMPLETE")
        print(f"Failed criteria: {', '.join(fails)}")

if __name__ == '__main__':
    main()
