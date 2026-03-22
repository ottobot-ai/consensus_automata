#!/usr/bin/env python3
"""
Generate all figures for the NiPoPoS paper.
Clean, publication-quality, with descriptive labels.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import hashlib

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

NUM_LEVELS = 10
PSI = 1
CUTOFF = 15

SHIFTED_EXP_PARAMS = {
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

LEVEL_WEIGHTS = [2.0**lv for lv in range(NUM_LEVELS)]
COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 9))


def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=CUTOFF):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** 1.0

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def slot_gap_scaling(slot_gap, cutoff=CUTOFF):
    return min(1.0, slot_gap / cutoff)

def block_weight_v3(slot_gap, level_hits, alpha=2.0):
    thr = snowplow_thr(slot_gap)
    base = thr ** alpha
    super_mult = 1.0
    for lv in range(1, NUM_LEVELS):
        if level_hits[lv]:
            super_mult += LEVEL_WEIGHTS[lv]
    return base * super_mult

def simulate_race(num_blocks, honest_gap_mean, adv_gap_choices, alpha, use_gating, rng):
    h_cum = 0.0; a_cum = 0.0
    h_last = [0]*NUM_LEVELS; a_last = [0]*NUM_LEVELS
    h_trace = []; a_trace = []
    
    for b in range(1, num_blocks + 1):
        hsg = max(1, rng.geometric(1.0 / honest_gap_mean))
        h_scale = slot_gap_scaling(hsg) if use_gating else 1.0
        hh = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - h_last[lv]
            hit = rng.random() < (shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv]) * h_scale)
            hh.append(hit)
            if hit: h_last[lv] = b
        h_cum += block_weight_v3(hsg, hh, alpha)
        h_trace.append(h_cum)
        
        asg = rng.choice(adv_gap_choices)
        a_scale = slot_gap_scaling(asg) if use_gating else 1.0
        ah = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - a_last[lv]
            hit = rng.random() < (shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv]) * a_scale)
            ah.append(hit)
            if hit: a_last[lv] = b
        a_cum += block_weight_v3(asg, ah, alpha)
        a_trace.append(a_cum)
    
    return h_cum, a_cum, h_trace, a_trace


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: LDD Snowplow + Shifted Exponential Threshold Curves
# ═══════════════════════════════════════════════════════════════════════

def fig1_threshold_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: L0 LDD Snowplow
    gaps = np.arange(0, 31)
    thrs = [snowplow_thr(g) for g in gaps]
    ax1.plot(gaps, thrs, 'k-', linewidth=2.5)
    ax1.axvline(x=CUTOFF, color='gray', linestyle='--', alpha=0.5, label=f'γ = {CUTOFF}')
    ax1.fill_between(gaps, thrs, alpha=0.15, color='steelblue')
    ax1.set_xlabel('Slot gap (δ)')
    ax1.set_ylabel('Threshold φ(δ)')
    ax1.set_title('(a) Level-0: LDD Snowplow')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 0.30)
    
    # Right: Shifted Exponential for L1-L9
    g_range = np.arange(0, 80)
    for lv in range(1, 8):
        mp, sc = SHIFTED_EXP_PARAMS[lv]
        thrs = [shifted_exp_thr(g, mp, sc) for g in g_range]
        ax2.plot(g_range, thrs, linewidth=1.8, color=COLORS[lv-1],
                 label=f'L{lv} (σ={sc:.1f})')
    ax2.set_xlabel('Base-block gap (g)')
    ax2.set_ylabel('Threshold θ_μ(g)')
    ax2.set_title('(b) Levels 1–7: Shifted Exponential')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 79)
    
    plt.tight_layout()
    plt.savefig('fig1_thresholds.pdf', bbox_inches='tight')
    plt.savefig('fig1_thresholds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 1: Threshold curves")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Slot-Gap Gating Effect on Super-Level Thresholds
# ═══════════════════════════════════════════════════════════════════════

def fig2_slot_gap_gating():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Scaling factor
    gaps = np.arange(0, 25)
    scales = [slot_gap_scaling(g) for g in gaps]
    ax1.plot(gaps, scales, 'k-', linewidth=2.5)
    ax1.fill_between(gaps, scales, alpha=0.15, color='coral')
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=CUTOFF, color='gray', linestyle='--', alpha=0.5, label=f'γ = {CUTOFF}')
    
    # Mark adversary and honest regions
    ax1.axvspan(0, 3, alpha=0.1, color='red', label='Burst adversary')
    ax1.axvspan(5, 10, alpha=0.1, color='green', label='Honest range')
    ax1.set_xlabel('L0 slot gap (δ)')
    ax1.set_ylabel('Super-level scaling factor')
    ax1.set_title('(a) Slot-Gap Scaling: min(1, δ/γ)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Effective L1 threshold at different slot gaps
    g_range = np.arange(0, 30)
    for sg, color, label in [(1, 'red', 'δ=1 (burst)'), 
                              (4, 'orange', 'δ=4 (fast)'),
                              (7, 'green', 'δ=7 (honest)'), 
                              (15, 'blue', 'δ≥15 (full)')]:
        scale = slot_gap_scaling(sg)
        mp, sc = SHIFTED_EXP_PARAMS[1]
        thrs = [shifted_exp_thr(g, mp, sc) * scale for g in g_range]
        ax2.plot(g_range, thrs, linewidth=1.8, color=color, label=label)
    
    ax2.set_xlabel('Base-block gap (g)')
    ax2.set_ylabel('Effective L1 threshold')
    ax2.set_title('(b) L1 Threshold × Slot-Gap Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig2_gating.pdf', bbox_inches='tight')
    plt.savefig('fig2_gating.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 2: Slot-gap gating")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Superblock Density Validation (10M slots)
# ═══════════════════════════════════════════════════════════════════════

def fig3_density():
    # Data from 10M slot simulation
    levels = list(range(10))
    achieved = [100.0, 49.4, 24.8, 12.3, 6.25, 3.35, 1.52, 0.82, 0.34, 0.16]
    targets = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.195313]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(levels))
    width = 0.35
    bars1 = ax.bar(x - width/2, targets, width, label='Target rate', 
                    color='steelblue', alpha=0.8, edgecolor='navy')
    bars2 = ax.bar(x + width/2, achieved, width, label='Observed rate', 
                    color='coral', alpha=0.8, edgecolor='darkred')
    
    ax.set_yscale('log')
    ax.set_xlabel('Superblock Level')
    ax.set_ylabel('Conditional Hit Rate (%)')
    ax.set_title('Superblock Density: Target vs Observed (10M slots)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in levels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.1, 200)
    
    plt.tight_layout()
    plt.savefig('fig3_density.pdf', bbox_inches='tight')
    plt.savefig('fig3_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 3: Density validation")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Cumulative Weight — Honest vs Adversary 
# ═══════════════════════════════════════════════════════════════════════

def fig4_cumulative_weight():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Single race trace
    rng = np.random.RandomState(7)
    _, _, h_trace, a_trace = simulate_race(200, 7.0, [1,1,1,2,2,3], 2.0, True, rng)
    
    blocks = range(1, 201)
    ax1.plot(blocks, h_trace, color='steelblue', linewidth=2, label='Honest (gap ~ 7)')
    ax1.plot(blocks, a_trace, color='coral', linewidth=2, label='Adversary (burst, gap 1–2)')
    ax1.fill_between(blocks, h_trace, a_trace, alpha=0.15, color='green')
    ax1.set_xlabel('Block number')
    ax1.set_ylabel('Cumulative chain weight')
    ax1.set_title('(a) Single Fork Race (α = 2)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Per-block weight distributions (10k blocks)
    rng = np.random.RandomState(42)
    h_weights = []; a_weights = []
    h_last = [0]*NUM_LEVELS; a_last = [0]*NUM_LEVELS
    
    for b in range(1, 10001):
        hsg = max(1, rng.geometric(1.0/7.0))
        h_scale = slot_gap_scaling(hsg)
        hh = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - h_last[lv]
            hit = rng.random() < (shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv]) * h_scale)
            hh.append(hit)
            if hit: h_last[lv] = b
        h_weights.append(block_weight_v3(hsg, hh, 2.0))
        
        asg = rng.choice([1,1,1,2,2,3])
        a_scale = slot_gap_scaling(asg)
        ah = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - a_last[lv]
            hit = rng.random() < (shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv]) * a_scale)
            ah.append(hit)
            if hit: a_last[lv] = b
        a_weights.append(block_weight_v3(asg, ah, 2.0))
    
    bmax = np.percentile(h_weights, 98)
    bins = np.linspace(0, bmax, 50)
    ax2.hist(h_weights, bins=bins, density=True, alpha=0.6, color='steelblue', label='Honest')
    ax2.hist(a_weights, bins=bins, density=True, alpha=0.6, color='coral', label='Adversary (burst)')
    ax2.axvline(np.mean(h_weights), color='navy', linestyle='--', linewidth=1.5,
                label=f'Honest mean = {np.mean(h_weights):.3f}')
    ax2.axvline(np.mean(a_weights), color='darkred', linestyle='--', linewidth=1.5,
                label=f'Adv mean = {np.mean(a_weights):.4f}')
    ax2.set_xlabel('Block weight')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Per-Block Weight Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig4_cumweight.pdf', bbox_inches='tight')
    plt.savefig('fig4_cumweight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 4: Cumulative weight")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Adversary Strategy Comparison (v2 vs v3)
# ═══════════════════════════════════════════════════════════════════════

def fig5_strategy_comparison():
    NUM_RACES = 5000
    ALPHA = 2.0
    
    strategies = [
        ("Burst\n(gap 1–2)",     [1, 1, 1, 2, 2, 3]),
        ("Fast\n(gap 2–4)",      [2, 2, 3, 3, 4]),
        ("Moderate\n(gap 4–7)",  [4, 5, 6, 7]),
        ("Honest-like\n(gap 5–9)", [5, 6, 7, 8, 9]),
        ("Slow\n(gap 8–15)",     [8, 10, 12, 15]),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    v2_pcts = []; v3_pcts = []
    
    for name, gaps in strategies:
        for ug, store in [(False, v2_pcts), (True, v3_pcts)]:
            rng = np.random.RandomState(42)
            w = 0
            for _ in range(NUM_RACES):
                hc, ac, _, _ = simulate_race(50, 7.0, gaps, ALPHA, ug, rng)
                if hc > ac: w += 1
            store.append(w / NUM_RACES * 100)
    
    strat_names = [n for n, _ in strategies]
    x = np.arange(len(strat_names))
    width = 0.35
    
    ax1.bar(x - width/2, v2_pcts, width, label='Without slot-gap gating', 
            color='steelblue', alpha=0.8, edgecolor='navy')
    ax1.bar(x + width/2, v3_pcts, width, label='With slot-gap gating', 
            color='coral', alpha=0.8, edgecolor='darkred')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (no advantage)')
    ax1.set_ylabel('Honest Win Rate (%)')
    ax1.set_title(f'(a) Honest vs Adversary by Strategy (α = {ALPHA}, 50 blocks)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strat_names, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)
    
    # Annotate the key improvement
    ax1.annotate('', xy=(2, v3_pcts[2]), xytext=(2, v2_pcts[2]),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(2.35, (v2_pcts[2] + v3_pcts[2])/2, 
             f'+{v3_pcts[2]-v2_pcts[2]:.0f}pp',
             color='green', fontweight='bold', fontsize=10, va='center')
    
    # Right: Race length sweep (v3 only, burst adversary)
    race_ns = [5, 10, 20, 50, 100, 200]
    win_pcts = []
    for n in race_ns:
        rng = np.random.RandomState(42)
        w = sum(1 for _ in range(NUM_RACES) 
                if simulate_race(n, 7.0, [1,1,1,2,2,3], ALPHA, True, rng)[0] >
                   simulate_race(n, 7.0, [1,1,1,2,2,3], ALPHA, True, rng)[1])
        # Need proper counting
        pass
    
    # Recompute properly
    win_pcts = []
    for n in race_ns:
        rng = np.random.RandomState(42)
        w = 0
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race(n, 7.0, [1,1,1,2,2,3], ALPHA, True, rng)
            if hc > ac: w += 1
        win_pcts.append(w / NUM_RACES * 100)
    
    ax2.plot(race_ns, win_pcts, 'go-', linewidth=2.5, markersize=8)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.fill_between(race_ns, 50, win_pcts, alpha=0.15, color='green')
    ax2.set_xlabel('Fork race length (blocks)')
    ax2.set_ylabel('Honest Win Rate (%)')
    ax2.set_title(f'(b) Honest Win Rate vs Fork Length (burst adversary, α = {ALPHA})')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(40, 102)
    ax2.set_xscale('log')
    
    # Add annotations for key points
    for n, wp in zip(race_ns, win_pcts):
        ax2.annotate(f'{wp:.0f}%', (n, wp), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fig5_strategies.pdf', bbox_inches='tight')
    plt.savefig('fig5_strategies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 5: Strategy comparison")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: The Adversary Dilemma
# ═══════════════════════════════════════════════════════════════════════

def fig6_adversary_dilemma():
    """
    Show the fundamental tradeoff: L0 weight vs chain growth rate.
    As adversary increases gap → higher weight per block but fewer blocks per slot.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    gaps = np.arange(1, 25)
    
    # L0 threshold^2 (weight base)
    weight_base = [snowplow_thr(g)**2 for g in gaps]
    
    # Blocks per slot (inverse of gap)
    blocks_per_slot = [1.0/g for g in gaps]
    
    # Weight per slot = weight_base × blocks_per_slot (ignoring super hits)
    weight_rate = [w * b for w, b in zip(weight_base, blocks_per_slot)]
    
    # Normalize
    max_wr = max(weight_rate)
    weight_rate_norm = [w/max_wr for w in weight_rate]
    max_wb = max(weight_base)
    weight_base_norm = [w/max_wb for w in weight_base]
    max_bps = max(blocks_per_slot)
    bps_norm = [b/max_bps for b in blocks_per_slot]
    
    ax.plot(gaps, bps_norm, 'b--', linewidth=2, label='Chain growth rate (1/gap)')
    ax.plot(gaps, weight_base_norm, 'r--', linewidth=2, label='Weight per block (threshold²)')
    ax.plot(gaps, weight_rate_norm, 'g-', linewidth=3, label='Weight rate (product)')
    
    # Mark honest operating point
    honest_gap = 7
    ax.axvline(x=honest_gap, color='green', linestyle=':', alpha=0.5)
    ax.text(honest_gap + 0.3, 0.92, 'Honest\nmean gap', fontsize=9, color='green')
    
    # Mark adversary zones
    ax.axvspan(1, 3, alpha=0.08, color='red')
    ax.text(1.5, 0.05, 'Burst\nzone', fontsize=9, color='red', ha='center')
    
    ax.set_xlabel('Block production gap (slots)')
    ax.set_ylabel('Normalized value')
    ax.set_title('The Adversary Dilemma: Weight Per Block vs Chain Growth Rate')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 24)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('fig6_dilemma.pdf', bbox_inches='tight')
    plt.savefig('fig6_dilemma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 6: Adversary dilemma")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Super-Level Hit Suppression Under Gating
# ═══════════════════════════════════════════════════════════════════════

def fig7_hit_suppression():
    NUM_RACES = 5000
    ALPHA = 2.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, ug, title in [(ax1, False, '(a) Without Slot-Gap Gating'),
                           (ax2, True, '(b) With Slot-Gap Gating')]:
        strategies = [
            ("Honest\n(gap~7)", 7.0, None, 'steelblue'),
            ("Burst\n(gap 1-2)", None, [1,1,1,2,2,3], 'coral'),
            ("Moderate\n(gap 4-7)", None, [4,5,6,7], 'orange'),
        ]
        
        all_hits = {}
        for name, hgm, adv_gaps, color in strategies:
            rng = np.random.RandomState(42)
            sh_sum = [0]*NUM_LEVELS
            for _ in range(NUM_RACES):
                h_last = [0]*NUM_LEVELS
                for b in range(1, 51):
                    if hgm:
                        sg = max(1, rng.geometric(1.0/hgm))
                    else:
                        sg = rng.choice(adv_gaps)
                    scale = slot_gap_scaling(sg) if ug else 1.0
                    for lv in range(1, NUM_LEVELS):
                        bg = b - h_last[lv]
                        hit = rng.random() < (shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv]) * scale)
                        if hit:
                            h_last[lv] = b
                            sh_sum[lv] += 1
            all_hits[name] = [sh_sum[lv]/NUM_RACES for lv in range(1, 7)]
        
        levels = list(range(1, 7))
        x = np.arange(len(levels))
        w = 0.25
        for i, (name, _, _, color) in enumerate(strategies):
            ax.bar(x + (i-1)*w, all_hits[name], w, label=name.replace('\n', ' '), 
                   color=color, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in levels])
        ax.set_ylabel('Avg Super Hits per 50-Block Race')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fig7_suppression.pdf', bbox_inches='tight')
    plt.savefig('fig7_suppression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 7: Hit suppression")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_threshold_curves()
    fig2_slot_gap_gating()
    fig3_density()
    fig4_cumulative_weight()
    fig5_strategy_comparison()
    fig6_adversary_dilemma()
    fig7_hit_suppression()
    print("\nAll figures generated.")
