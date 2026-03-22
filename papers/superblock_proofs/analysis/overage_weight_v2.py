#!/usr/bin/env python3
"""
NiPoPoS Work Overage v2: Threshold-dominant cumulative weight.

block_weight = threshold(slot_gap)^α × (1 + Σ 2^μ × hit_μ)

- threshold^α: L0 snowplow threshold raised to power α ≥ 1
  α=1: linear, α=2: quadratic penalty for low gaps
- (1 + Σ 2^μ × hit_μ): exponential super-level bonus (no overage ratio)
  L0-only block: multiplier = 1
  L0+L1: 1+2 = 3
  L0+L1+L3: 1+2+8 = 11
  L0+L5: 1+32 = 33

The multiplier is the same per-block for honest and adversary (same expected 
super hits). But the L0 threshold base is MUCH higher for honest (natural gaps)
than adversary (burst gaps). This is where the weight divergence lives.

Adversary dilemma: to catch honest chain LENGTH they must produce blocks fast 
(small gaps) → low threshold → low weight per block → need EVEN MORE blocks 
→ even smaller gaps → even lower weight. Negative feedback loop.
"""
import numpy as np
import hashlib
import time

np.random.seed(42)

NUM_LEVELS = 10
PSI = 1

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

LEVEL_WEIGHTS = [2.0 ** lv for lv in range(NUM_LEVELS)]  # 1, 2, 4, ..., 512

def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=15):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** 1.0

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))


def block_weight(slot_gap, level_hits, alpha=2.0):
    """
    block_weight = threshold(slot_gap)^α × (1 + Σ 2^μ × hit_μ)
    """
    thr = snowplow_thr(slot_gap)
    base = thr ** alpha
    
    super_mult = 1.0  # Base multiplier (L0 hit = 1)
    for lv in range(1, NUM_LEVELS):
        if level_hits[lv]:
            super_mult += LEVEL_WEIGHTS[lv]
    
    return base * super_mult


def simulate_race(num_blocks, honest_gap_mean, adv_gap_choices, alpha, rng):
    """Single fork race: honest vs adversary, both produce num_blocks."""
    h_cum = 0.0; a_cum = 0.0
    h_last = [0]*NUM_LEVELS; a_last = [0]*NUM_LEVELS
    
    h_weight_trace = []
    a_weight_trace = []
    
    for b in range(1, num_blocks + 1):
        # Honest block
        hsg = max(1, rng.geometric(1.0 / honest_gap_mean))
        hh = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - h_last[lv]
            hit = rng.random() < shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
            hh.append(hit)
            if hit: h_last[lv] = b
        h_cum += block_weight(hsg, hh, alpha)
        h_weight_trace.append(h_cum)
        
        # Adversary block
        asg = rng.choice(adv_gap_choices)
        ah = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - a_last[lv]
            hit = rng.random() < shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
            ah.append(hit)
            if hit: a_last[lv] = b
        a_cum += block_weight(asg, ah, alpha)
        a_weight_trace.append(a_cum)
    
    return h_cum, a_cum, h_weight_trace, a_weight_trace


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    NUM_RACES = 5000
    
    print("NiPoPoS Work Overage v2: Threshold-Dominant Cumulative Weight")
    print("=" * 75)
    
    # ── Test 1: Vary α (threshold exponent) ──────────────────────────
    
    print("\n--- Test 1: α sweep (50-block races, burst adversary gap=1-2, 5000 races) ---")
    print(f"{'α':>5} {'HonestWin%':>11} {'MeanAdv':>10} {'MedAdv':>10} {'HonWeight':>12} {'AdvWeight':>12} {'Ratio':>7}")
    print("-" * 75)
    
    alpha_results = []
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rng = np.random.RandomState(42)
        wins = 0; advs = []
        h_total = 0; a_total = 0
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race(50, 7.0, [1,1,1,2,2,3], alpha, rng)
            if hc > ac: wins += 1
            advs.append(hc - ac)
            h_total += hc; a_total += ac
        
        win_pct = wins / NUM_RACES * 100
        ratio = (h_total / NUM_RACES) / (a_total / NUM_RACES)
        alpha_results.append((alpha, win_pct, np.mean(advs), np.median(advs), 
                              h_total/NUM_RACES, a_total/NUM_RACES, ratio))
        print(f"{alpha:>5.1f} {win_pct:>10.1f}% {np.mean(advs):>10.2f} {np.median(advs):>10.2f} "
              f"{h_total/NUM_RACES:>12.2f} {a_total/NUM_RACES:>12.2f} {ratio:>7.3f}x")
    
    # ── Test 2: α=2 at different race lengths ────────────────────────
    
    ALPHA = 2.0
    print(f"\n--- Test 2: Race length sweep (α={ALPHA}, burst adversary, 5000 races) ---")
    print(f"{'N':>5} {'HonestWin%':>11} {'MeanAdv':>10} {'P5':>10} {'P50':>10} {'P95':>10}")
    print("-" * 60)
    
    race_results = []
    for n in [5, 10, 20, 50, 100, 200]:
        rng = np.random.RandomState(42)
        wins = 0; advs = []
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race(n, 7.0, [1,1,1,2,2,3], ALPHA, rng)
            if hc > ac: wins += 1
            advs.append(hc - ac)
        advs = np.array(advs)
        race_results.append((n, wins/NUM_RACES*100, advs))
        print(f"{n:>5} {wins/NUM_RACES*100:>10.1f}% {np.mean(advs):>10.3f} "
              f"{np.percentile(advs, 5):>10.3f} {np.median(advs):>10.3f} {np.percentile(advs, 95):>10.3f}")
    
    # ── Test 3: Adversary gap strategies at α=2 ─────────────────────
    
    print(f"\n--- Test 3: Adversary gap strategy sweep (α={ALPHA}, 50 blocks, 5000 races) ---")
    strategies = [
        ("burst (1-2)",      [1, 1, 1, 2, 2, 3]),
        ("fast (2-4)",       [2, 2, 3, 3, 4]),
        ("moderate (4-7)",   [4, 5, 6, 7]),
        ("honest-like (5-9)",[5, 6, 7, 8, 9]),
        ("slow (8-15)",      [8, 10, 12, 15]),
    ]
    
    print(f"{'Strategy':<20} {'HonestWin%':>11} {'MeanAdv':>10} {'Ratio':>7}")
    print("-" * 55)
    
    strat_results = []
    for name, gaps in strategies:
        rng = np.random.RandomState(42)
        wins = 0; h_tot = 0; a_tot = 0
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race(50, 7.0, gaps, ALPHA, rng)
            if hc > ac: wins += 1
            h_tot += hc; a_tot += ac
        ratio = (h_tot/NUM_RACES) / (a_tot/NUM_RACES)
        strat_results.append((name, wins/NUM_RACES*100, ratio))
        print(f"{name:<20} {wins/NUM_RACES*100:>10.1f}% {(h_tot-a_tot)/NUM_RACES:>10.3f} {ratio:>7.3f}x")
    
    # ── Test 4: Per-block weight breakdown (single detailed race) ────
    
    print(f"\n--- Test 4: Single detailed race (200 blocks, α={ALPHA}) ---")
    rng = np.random.RandomState(7)
    h_cum, a_cum, h_trace, a_trace = simulate_race(200, 7.0, [1,1,1,2,2,3], ALPHA, rng)
    print(f"  Honest cumulative weight: {h_cum:.4f}")
    print(f"  Adversary cumulative weight: {a_cum:.4f}")
    print(f"  Honest advantage: {h_cum - a_cum:.4f} ({(h_cum/a_cum - 1)*100:.1f}%)")
    
    # ── Plots ────────────────────────────────────────────────────────
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'NiPoPoS Work Overage v2: threshold^α × (1 + Σ 2^μ hits)', fontsize=14)
    
    # Plot 1: α sweep
    ax = axes[0][0]
    alphas = [a for a, _, _, _, _, _, _ in alpha_results]
    win_pcts = [w for _, w, _, _, _, _, _ in alpha_results]
    ratios = [r for _, _, _, _, _, _, r in alpha_results]
    ax.plot(alphas, win_pcts, 'go-', linewidth=2, markersize=8, label='Honest win %')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(alphas, ratios, 'bs--', linewidth=1.5, markersize=6, label='Weight ratio')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Honest/Adversary weight ratio', color='blue')
    ax.set_xlabel('α (threshold exponent)')
    ax.set_ylabel('Honest win %', color='green')
    ax.set_title('Effect of α on Honest Advantage\n(50 blocks, burst adversary)')
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Race length at α=2
    ax = axes[0][1]
    ns = [n for n, _, _ in race_results]
    wpcts = [w for _, w, _ in race_results]
    ax.plot(ns, wpcts, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Race Length (blocks)')
    ax.set_ylabel('Honest Win %')
    ax.set_title(f'Honest Win Rate vs Fork Length (α={ALPHA})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    
    # Plot 3: Cumulative weight trace
    ax = axes[1][0]
    blocks = range(1, 201)
    ax.plot(blocks, h_trace, color='steelblue', linewidth=1.5, label='Honest (gap~7)')
    ax.plot(blocks, a_trace, color='coral', linewidth=1.5, label='Adversary (burst)')
    ax.fill_between(blocks, h_trace, a_trace, 
                     where=[h > a for h, a in zip(h_trace, a_trace)],
                     alpha=0.2, color='green', label='Honest leads')
    ax.fill_between(blocks, h_trace, a_trace,
                     where=[h <= a for h, a in zip(h_trace, a_trace)],
                     alpha=0.2, color='red', label='Adversary leads')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Cumulative Weight')
    ax.set_title(f'Cumulative Weight Over Time (α={ALPHA})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Strategy comparison
    ax = axes[1][1]
    strat_names = [n for n, _, _ in strat_results]
    strat_wins = [w for _, w, _ in strat_results]
    strat_ratios = [r for _, _, r in strat_results]
    colors = ['coral' if w < 50 else 'green' for w in strat_wins]
    bars = ax.barh(strat_names, strat_wins, color=colors, alpha=0.7)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2)
    for bar, ratio in zip(bars, strat_ratios):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.3f}x', va='center', fontsize=9)
    ax.set_xlabel('Honest Win %')
    ax.set_title(f'Adversary Strategy vs Honest Win Rate (α={ALPHA})')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/fig_overage_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_overage_v2.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_overage_v2.png/pdf")
