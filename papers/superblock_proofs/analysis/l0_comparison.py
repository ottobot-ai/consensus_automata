#!/usr/bin/env python3
"""
Compare superblock weighting under different L0 threshold functions:
1. Taktikos LDD snowplow: threshold ramps with slot gap
2. Praos flat: constant threshold regardless of gap  
3. Mild LDD: reduced amplitude snowplow (intermediate)

For each: run 5000 fork races at 50 blocks, measure honest win rate
across burst/fast/moderate/honest-like adversary strategies.
"""

import hashlib
import struct
import numpy as np
from collections import defaultdict

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# L0 Threshold Functions
# ═══════════════════════════════════════════════════════════════

def taktikos_threshold(slot_gap, cutoff=15, amplitude=0.5, baseline=0.05, stake=0.20):
    """Taktikos LDD snowplow: linear ramp to cutoff, then baseline."""
    if slot_gap <= 0:
        return 0.0
    if slot_gap >= cutoff:
        f = baseline
    else:
        f = (slot_gap / cutoff) * amplitude
    return 1.0 - (1.0 - f) ** stake

def praos_threshold(slot_gap, f_effective=0.15, stake=0.20):
    """Praos flat: constant threshold, no gap dependence."""
    if slot_gap <= 0:
        return 0.0
    return 1.0 - (1.0 - f_effective) ** stake

def mild_ldd_threshold(slot_gap, cutoff=15, amplitude=0.20, baseline=0.10, stake=0.20):
    """Mild LDD: reduced amplitude, higher baseline — halfway to Praos."""
    if slot_gap <= 0:
        return 0.0
    if slot_gap >= cutoff:
        f = baseline
    else:
        f = baseline + (slot_gap / cutoff) * (amplitude - baseline)
    return 1.0 - (1.0 - f) ** stake

# ═══════════════════════════════════════════════════════════════
# Shifted Exponential Super-Level Thresholds
# ═══════════════════════════════════════════════════════════════

LEVEL_PARAMS = [
    # (maxProb, scale) for L1-L9
    (1.131, 0.50),
    (0.346, 0.50),
    (0.322, 7.92),
    (0.249, 30.3),
    (0.077, 27.5),
    (0.027, 40.5),
    (0.013, 56.0),
    (0.004, 64.0),
    (0.002, 72.0),
]
PSI = 1  # dormant period
NUM_LEVELS = len(LEVEL_PARAMS)

def shifted_exp_threshold(block_gap, max_prob, scale):
    if block_gap < PSI:
        return 0.0
    return max_prob * (1.0 - np.exp(-(block_gap - PSI) / scale))

# ═══════════════════════════════════════════════════════════════
# Block Production & Weighting
# ═══════════════════════════════════════════════════════════════

def produce_chain(n_blocks, gap_range, l0_func, cutoff, alpha=2.0, use_gating=True):
    """Produce a chain of n_blocks with given gap strategy and L0 function.
    
    Returns: (cumulative_weight, total_super_hits_per_level, per_block_weights)
    """
    weights = []
    level_hits = [0] * NUM_LEVELS
    last_level_hit = [0] * NUM_LEVELS  # block index of last hit per level
    
    for i in range(n_blocks):
        # Random slot gap from strategy
        slot_gap = np.random.randint(gap_range[0], gap_range[1] + 1)
        
        # L0 threshold (the "difficulty" this block beat)
        l0_thr = l0_func(slot_gap)
        
        # Base weight from L0 threshold
        base_weight = l0_thr ** alpha
        
        # Slot-gap gating factor
        if use_gating:
            gating = min(1.0, slot_gap / cutoff)
        else:
            gating = 1.0
        
        # Super-level hits
        super_multiplier = 1.0
        for mu in range(NUM_LEVELS):
            block_gap = i - last_level_hit[mu]  # blocks since last hit at this level
            max_prob, scale = LEVEL_PARAMS[mu]
            thr = shifted_exp_threshold(block_gap, max_prob, scale) * gating
            
            # Random test
            test_val = np.random.random()
            if test_val < thr:
                super_multiplier += 2 ** (mu + 1)
                level_hits[mu] += 1
                last_level_hit[mu] = i
        
        w = base_weight * super_multiplier
        weights.append(w)
    
    return sum(weights), level_hits, weights

def run_fork_races(n_races, n_blocks, honest_gap, adv_gap, l0_func, cutoff, alpha=2.0, use_gating=True):
    """Run fork races between honest and adversary chains."""
    honest_wins = 0
    weight_ratios = []
    honest_level_hits_total = np.zeros(NUM_LEVELS)
    adv_level_hits_total = np.zeros(NUM_LEVELS)
    
    for _ in range(n_races):
        hw, h_hits, _ = produce_chain(n_blocks, honest_gap, l0_func, cutoff, alpha, use_gating)
        aw, a_hits, _ = produce_chain(n_blocks, adv_gap, l0_func, cutoff, alpha, use_gating)
        
        if hw > aw:
            honest_wins += 1
        
        if aw > 0:
            weight_ratios.append(hw / aw)
        
        for mu in range(NUM_LEVELS):
            honest_level_hits_total[mu] += h_hits[mu]
            adv_level_hits_total[mu] += a_hits[mu]
    
    return {
        'honest_win_pct': 100.0 * honest_wins / n_races,
        'mean_weight_ratio': np.mean(weight_ratios) if weight_ratios else 0,
        'honest_level_hits': honest_level_hits_total / n_races,
        'adv_level_hits': adv_level_hits_total / n_races,
    }

# ═══════════════════════════════════════════════════════════════
# Run Comparison
# ═══════════════════════════════════════════════════════════════

N_RACES = 5000
N_BLOCKS = 50
ALPHA = 2.0

# Strategies: (name, gap_range)
strategies = [
    ("Burst (1-2)",     (1, 2)),
    ("Fast (2-4)",      (2, 4)),
    ("Moderate (4-7)",  (4, 7)),
    ("Honest-like (5-9)", (5, 9)),
]

honest_gap = (3, 12)  # honest geometric-like with mean ~7

# L0 functions: (name, func, cutoff)
l0_configs = [
    ("Taktikos LDD\n(snowplow, fA=0.5)", taktikos_threshold, 15),
    ("Mild LDD\n(fA=0.2, fB=0.1)", mild_ldd_threshold, 15),
    ("Praos Flat\n(f=0.15, constant)", praos_threshold, 15),
]

print("=" * 90)
print("SUPERBLOCK WEIGHTING UNDER DIFFERENT L0 THRESHOLD FUNCTIONS")
print(f"Fork races: {N_RACES}, Blocks per race: {N_BLOCKS}, α={ALPHA}")
print("=" * 90)

# Collect results for plotting
all_results = {}

for l0_name, l0_func, cutoff in l0_configs:
    print(f"\n{'─' * 90}")
    clean_name = l0_name.replace('\n', ' ')
    print(f"L0 Function: {clean_name}")
    print(f"{'─' * 90}")
    
    # Show L0 threshold at key gaps
    print(f"  L0 threshold at gap  1: {l0_func(1):.4f}")
    print(f"  L0 threshold at gap  4: {l0_func(4):.4f}")
    print(f"  L0 threshold at gap  7: {l0_func(7):.4f}")
    print(f"  L0 threshold at gap 15: {l0_func(15):.4f}")
    print(f"  Threshold ratio (gap7/gap1): {l0_func(7)/l0_func(1):.1f}x")
    print()
    
    results_for_config = {}
    
    for strat_name, adv_gap in strategies:
        r = run_fork_races(N_RACES, N_BLOCKS, honest_gap, adv_gap, l0_func, cutoff, ALPHA, use_gating=True)
        results_for_config[strat_name] = r
        
        print(f"  vs {strat_name:20s} | Honest wins: {r['honest_win_pct']:5.1f}% | "
              f"Weight ratio: {r['mean_weight_ratio']:6.1f}x | "
              f"Honest L1={r['honest_level_hits'][0]:.1f} Adv L1={r['adv_level_hits'][0]:.1f}")
    
    all_results[clean_name] = results_for_config

# ═══════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════

print(f"\n{'=' * 90}")
print("SUMMARY: Honest Win % by L0 Function × Adversary Strategy")
print(f"{'=' * 90}")
print(f"{'Strategy':<25s}", end="")
for l0_name, _, _ in l0_configs:
    clean = l0_name.replace('\n', ' ')
    print(f"  {clean:>22s}", end="")
print()
print("-" * 90)

for strat_name, _ in strategies:
    print(f"{strat_name:<25s}", end="")
    for l0_name, _, _ in l0_configs:
        clean = l0_name.replace('\n', ' ')
        r = all_results[clean][strat_name]
        print(f"  {r['honest_win_pct']:20.1f}%", end="")
    print()

print()
print(f"{'Strategy':<25s}", end="")
for l0_name, _, _ in l0_configs:
    clean = l0_name.replace('\n', ' ')
    print(f"  {clean:>22s}", end="")
print()
print("-" * 90)

for strat_name, _ in strategies:
    print(f"{strat_name:<25s}", end="")
    for l0_name, _, _ in l0_configs:
        clean = l0_name.replace('\n', ' ')
        r = all_results[clean][strat_name]
        print(f"  {r['mean_weight_ratio']:20.1f}x", end="")
    print()

# ═══════════════════════════════════════════════════════════════
# Generate Comparison Figure
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

strat_labels = [s[0] for s in strategies]
x = np.arange(len(strat_labels))
width = 0.6

colors = ['#2196F3', '#FF9800', '#f44336']  # blue, orange, red

for idx, (l0_name, l0_func, cutoff) in enumerate(l0_configs):
    ax = axes[idx]
    clean = l0_name.replace('\n', ' ')
    
    wins = [all_results[clean][s]['honest_win_pct'] for s in strat_labels]
    
    bars = ax.bar(x, wins, width, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(l0_name, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(3.4, 51, '50%', fontsize=8, color='gray', alpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    
    # Show threshold range
    thr_1 = l0_func(1)
    thr_7 = l0_func(7)
    ratio = thr_7 / thr_1 if thr_1 > 0 else 1.0
    ax.text(0.02, 0.95, f'θ(1)={thr_1:.3f}\nθ(7)={thr_7:.3f}\nratio={ratio:.1f}×',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_ylabel('Honest Win Rate (%)', fontsize=11)

fig.suptitle('Superblock Weighting: Effect of L0 Threshold Function on Security\n'
             f'(α={ALPHA}, {N_BLOCKS} blocks/race, {N_RACES} races, slot-gap gating ON)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig8_l0_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig8_l0_comparison.png', bbox_inches='tight', dpi=200)
print("\n✓ Saved figures/fig8_l0_comparison.pdf/.png")

# ═══════════════════════════════════════════════════════════════
# Additional figure: L0 threshold curves overlay
# ═══════════════════════════════════════════════════════════════

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

gaps = np.arange(1, 25)

# Left: L0 threshold curves
for l0_name, l0_func, cutoff in l0_configs:
    clean = l0_name.split('\n')[0]
    thresholds = [l0_func(g) for g in gaps]
    ax1.plot(gaps, thresholds, linewidth=2.5, label=clean, marker='o', markersize=3)

ax1.set_xlabel('Slot Gap (δ)', fontsize=11)
ax1.set_ylabel('Threshold φ(δ, α=0.2)', fontsize=11)
ax1.set_title('L0 Threshold Functions', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.axvline(x=15, color='gray', linestyle=':', alpha=0.5, label='cutoff γ=15')

# Right: threshold^alpha curves (the actual weight contribution)
for l0_name, l0_func, cutoff in l0_configs:
    clean = l0_name.split('\n')[0]
    weight_contrib = [l0_func(g)**ALPHA for g in gaps]
    ax1_twin = ax2
    ax2.plot(gaps, weight_contrib, linewidth=2.5, label=clean, marker='o', markersize=3)

ax2.set_xlabel('Slot Gap (δ)', fontsize=11)
ax2.set_ylabel(f'Weight Contribution φ(δ)^{ALPHA:.0f}', fontsize=11)
ax2.set_title(f'L0 Weight Component (α={ALPHA:.0f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Annotate the key insight
ax2.annotate('Taktikos: 47× range\nin weight contribution',
             xy=(7, taktikos_threshold(7)**2), xytext=(12, 0.008),
             arrowprops=dict(arrowstyle='->', color='#2196F3'),
             fontsize=9, color='#2196F3', fontweight='bold')
ax2.annotate('Praos: 1× range\n(all blocks equal)',
             xy=(7, praos_threshold(7)**2), xytext=(12, praos_threshold(7)**2 + 0.002),
             arrowprops=dict(arrowstyle='->', color='#f44336'),
             fontsize=9, color='#f44336', fontweight='bold')

fig2.suptitle('Why LDD Matters for Superblock Security', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig9_l0_curves.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig9_l0_curves.png', bbox_inches='tight', dpi=200)
print("✓ Saved figures/fig9_l0_curves.pdf/.png")

# ═══════════════════════════════════════════════════════════════
# Weight ratio comparison figure
# ═══════════════════════════════════════════════════════════════

fig3, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(strat_labels))
width = 0.25

for idx, (l0_name, l0_func, cutoff) in enumerate(l0_configs):
    clean = l0_name.replace('\n', ' ')
    ratios = [all_results[clean][s]['mean_weight_ratio'] for s in strat_labels]
    bars = ax.bar(x + idx * width - width, ratios, width, label=clean.split('(')[0].strip(),
                  color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, ratios):
        if val > 1.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Weight Ratio (Honest / Adversary)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(strat_labels, fontsize=10)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.text(3.5, 1.1, 'parity', fontsize=9, color='red', alpha=0.7)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_title(f'Honest/Adversary Weight Ratio by L0 Function (α={ALPHA:.0f})', 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig10_weight_ratios.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig10_weight_ratios.png', bbox_inches='tight', dpi=200)
print("✓ Saved figures/fig10_weight_ratios.pdf/.png")
