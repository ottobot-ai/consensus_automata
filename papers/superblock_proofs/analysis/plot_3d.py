#!/usr/bin/env python3
"""
3D visualization of superblock gap distributions across all levels.

X-axis: gap (centered on expected mean per level)
Y-axis (depth): Level (L0-L9)
Z-axis: probability density

Shows theoretical shifted exponential curves overlaid with
empirical histograms from the 10M slot simulation.
"""
import numpy as np
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

np.random.seed(42)

# ── Simulation (rerun to collect gap distributions) ──────────────────

PSI = 1
L0_PSI, L0_GAMMA, L0_FA, L0_FB = 0, 15, 0.5, 0.05

PARAMS = {
    1: (1.130555, 0.500000),
    2: (0.346157, 0.500000),
    3: (0.321504, 7.921053),
    4: (0.248923, 30.342105),
    5: (0.076907, 27.526316),
    6: (0.027017, 40.500000),
    7: (0.012777, 56.000000),
    8: (0.004451, 64.000000),
    9: (0.001814, 72.000000),
}

def sthr(gap, rel):
    if gap < 0: return 0.0
    diff = L0_FA * gap / L0_GAMMA if gap < L0_GAMMA else L0_FB
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel

def ethr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dh(seed, slot, level):
    h = hashlib.sha256(f'{seed}:{slot}:TEST-{level}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)

print("Running 5M slot sim for gap distributions...")
stakes = [3000, 2500, 2000, 1500, 1000]
ts = 10000; N = 5_000_000
l0 = 0; lls = 0; llb = [0]*10
all_gaps = [[] for _ in range(10)]

t0 = time.time()
for slot in range(1, N+1):
    if slot % 1000000 == 0:
        print(f"  {slot:,} ({time.time()-t0:.1f}s)")
    sg = slot - lls
    hit = False; st = 0
    for si in range(5):
        if dh(si, slot, 0) < sthr(sg, stakes[si]/ts):
            hit = True; st = si; break
    if not hit: continue
    l0 += 1; lls = slot; all_gaps[0].append(sg)
    for lv in range(1, 10):
        bg = l0 - llb[lv]
        mp, sc = PARAMS[lv]
        if dh(st, slot, lv) < ethr(bg, mp, sc):
            all_gaps[lv].append(bg); llb[lv] = l0
print(f"Done: {l0:,} base blocks in {time.time()-t0:.1f}s\n")

# ── Theoretical distributions ────────────────────────────────────────

def theoretical_pdf(level, gaps):
    """Compute theoretical hit probability at each gap for shifted exponential."""
    if level == 0:
        # L0: snowplow on slots — approximate with empirical
        return None
    mp, sc = PARAMS[level]
    probs = np.zeros(len(gaps))
    for i, g in enumerate(gaps):
        t = ethr(int(g), mp, sc)
        # P(hit at gap g) = threshold(g) * product(1 - threshold(j) for j in [psi, g-1])
        surv = 1.0
        for j in range(PSI, int(g)):
            surv *= (1 - ethr(j, mp, sc))
        probs[i] = surv * t
    return probs

# ── Figure 1: Stacked Semi-Log Waterfall ──────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

colors = cm.viridis(np.linspace(0.1, 0.9, 10))
offsets = np.arange(10) * 2.5  # Vertical offset per level for stacking

for level in range(10):
    gaps = all_gaps[level]
    if len(gaps) < 10:
        continue
    
    mean_gap = np.mean(gaps)
    
    # Histogram centered on mean
    max_range = int(np.percentile(gaps, 99))
    bw = max(1, max_range // 40)
    bins = np.arange(0, max_range + bw, bw)
    hist, edges = np.histogram(gaps, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    centered = centers - mean_gap
    
    # Log-transform density (shift up by level offset)
    log_hist = np.log10(np.maximum(hist, 1e-8)) + offsets[level]
    baseline = np.log10(1e-8) + offsets[level]
    
    # Fill
    ax.fill_between(centered, baseline, log_hist, alpha=0.5, color=colors[level])
    ax.plot(centered, log_hist, color=colors[level], linewidth=1.2)
    
    # Theoretical overlay (L1-L9)
    if level > 0:
        theory_gaps = np.arange(PSI, max_range + 1)
        theory_pdf = theoretical_pdf(level, theory_gaps)
        if theory_pdf is not None and len(theory_pdf) > 0:
            theory_centered = theory_gaps - mean_gap
            log_theory = np.log10(np.maximum(theory_pdf, 1e-8)) + offsets[level]
            ax.plot(theory_centered, log_theory, 'r-', linewidth=1.5, alpha=0.7)
    
    # Level label
    ax.text(-max_range * 0.6, offsets[level] - 1.5, f'L{level} (μ={mean_gap:.0f})',
            fontsize=9, color=colors[level], fontweight='bold')

ax.set_xlabel('Gap centered on level mean (base blocks for L1-L9, slots for L0)', fontsize=11)
ax.set_ylabel('log₁₀(density) + offset per level', fontsize=11)
ax.set_title('Superblock Gap Distributions (Semi-Log Waterfall)\nBars = empirical, Red = theoretical', fontsize=13)
ax.grid(True, alpha=0.2)

plt.savefig('figures/fig_waterfall_semilog.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig_waterfall_semilog.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig_waterfall_semilog.png/pdf")

# ── Figure 2: Theoretical vs Empirical Side-by-Side ──────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Per-Level Gap Distributions: Theoretical vs Empirical (5M slots)', fontsize=14, y=1.02)

for idx, level in enumerate(range(1, 10)):
    ax = axes[idx // 3][idx % 3]
    gaps = all_gaps[level]
    
    if len(gaps) < 5:
        ax.text(0.5, 0.5, f'L{level}\n(insufficient data)', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(f'L{level}')
        continue
    
    mean_gap = np.mean(gaps)
    med_gap = np.median(gaps)
    
    # Empirical histogram
    max_plot = int(np.percentile(gaps, 99.5))
    bw = max(1, max_plot // 30)
    bins = np.arange(0, max_plot + bw, bw)
    ax.hist(gaps, bins=bins, density=True, alpha=0.6, color=colors[level],
            edgecolor='white', linewidth=0.5, label='Empirical')
    
    # Theoretical curve
    theory_gaps = np.arange(PSI, max_plot + 1)
    theory_pdf = theoretical_pdf(level, theory_gaps)
    if theory_pdf is not None:
        ax.plot(theory_gaps, theory_pdf, 'r-', linewidth=2, label='Theoretical', alpha=0.8)
    
    # Mean/median lines
    ax.axvline(mean_gap, color='blue', linestyle='--', alpha=0.7, label=f'Mean={mean_gap:.1f}')
    ax.axvline(med_gap, color='green', linestyle=':', alpha=0.7, label=f'Med={med_gap:.0f}')
    
    target = 1.0 / (2**level)
    ax.set_title(f'L{level} — P={target:.4f} (n={len(gaps):,})', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Gap (base blocks)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig_theoretical_vs_empirical.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig_theoretical_vs_empirical.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig_theoretical_vs_empirical.png/pdf")

# ── Figure 3: Curve shapes (threshold vs gap) ────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for level in range(1, 10):
    mp, sc = PARAMS[level]
    gaps = np.arange(0, 100, 0.5)
    thresholds = [ethr(g, mp, sc) for g in gaps]
    ax.plot(gaps, thresholds, color=colors[level], linewidth=2,
            label=f'L{level} (mp={mp:.3f}, sc={sc:.1f})')

ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='ψ=1 (dormant)')
ax.set_xlabel('Gap (base blocks since last hit)', fontsize=12)
ax.set_ylabel('Threshold (acceptance probability)', fontsize=12)
ax.set_title('Shifted Exponential Threshold Curves by Level', fontsize=13)
ax.legend(fontsize=8, loc='right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.8, 200)
ax.set_ylim(1e-4, 1.5)

plt.savefig('figures/fig_curve_shapes.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig_curve_shapes.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig_curve_shapes.png/pdf")

print("\nAll plots generated!")
