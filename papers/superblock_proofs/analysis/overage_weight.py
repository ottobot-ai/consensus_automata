#!/usr/bin/env python3
"""
NiPoPoS Work Overage Chain Weight.

Block weight = Σ (2^μ × E[gap_μ] / actual_gap_μ) for each level μ hit.

Early hits (actual < expected) contribute MORE weight.
Late hits (actual > expected) contribute LESS weight.
Burst-forging adversary gets late super hits → low weight per block.
Honest chain gets natural mix including early hits → higher weight.

Test with capped tine explorer + adversary comparison.
"""
import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

# Expected base-block gap per level (from protocol params)
EXPECTED_GAPS = {
    0: 1.0,    # Every L0 block is L0 — gap always 1 in block units (slot gap for L0)
    1: 2.0,    # Every 2nd block
    2: 4.0,    # Every 4th
    3: 8.0,
    4: 16.0,
    5: 32.0,
    6: 64.0,
    7: 128.0,
    8: 256.0,
    9: 512.0,
}

# Expected SLOT gap for L0 (used for L0 weight)
EXPECTED_L0_SLOT_GAP = 7.0

LEVEL_BASE_WEIGHT = {lv: 2.0 ** lv for lv in range(NUM_LEVELS)}


def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=15, rel_stake=1.0):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel_stake

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dhash(slot, level, salt=0):
    h = hashlib.sha256(f"staker{salt}:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


def compute_overage_weight(slot_gap, level_hits, level_gaps):
    """
    Compute block weight using work overage formula.
    
    L0 component: proportional to LDD threshold beaten.
      threshold(gap) is LOW for small gaps, HIGH for large gaps (up to cutoff).
      This directly penalizes burst-forging (small gap → low weight).
      
    Super components: 2^μ × (E[gap_μ] / actual_gap_μ) for early hits.
      Early hit = actual < expected → overage > 1 → bonus.
      Late hit = actual > expected → overage < 1 → reduced.
      The super-level per-level gaps grow identically for honest and adversary
      (since they track last level-μ hit, not L0 timing), so the overage
      contribution is similar. The L0 threshold is where the differentiation lives.
    """
    weight = 0.0
    
    # L0 component: the LDD threshold this block beat
    # gap=1 → threshold≈0.017, gap=7 → threshold≈0.117, gap=14 → threshold≈0.233
    # Scale by 10 to make it meaningful relative to super weights
    l0_threshold = snowplow_thr(slot_gap)
    weight += 10.0 * l0_threshold
    
    # Super level components: overage rewards early hits
    for lv in range(1, NUM_LEVELS):
        if level_hits[lv]:
            gap = level_gaps[lv]
            overage = min(EXPECTED_GAPS[lv] / max(gap, 1), 10.0)
            weight += LEVEL_BASE_WEIGHT[lv] * overage
    
    return weight


# ── Test 1: Weight distribution for honest vs adversary blocks ───────

def simulate_honest_adversary(num_blocks=10000):
    """
    Compare per-block weights between honest and adversary strategies.
    
    Honest: produces blocks at natural LDD gaps (~geometric around 7 slots)
    Adversary: burst-forges as fast as possible (eligible every ~2 slots at high amplitude)
    """
    rng = np.random.RandomState(42)
    
    # Honest chain
    h_weights = []
    h_last_level_hit = [0] * NUM_LEVELS
    h_block = 0
    
    for i in range(num_blocks):
        h_block += 1
        slot_gap = rng.geometric(1.0 / EXPECTED_L0_SLOT_GAP)
        slot_gap = max(slot_gap, 1)
        
        level_hits = [True]  # L0
        level_gaps = [slot_gap]
        
        for lv in range(1, NUM_LEVELS):
            bg = h_block - h_last_level_hit[lv]
            mp, sc = SHIFTED_EXP_PARAMS[lv]
            thr = shifted_exp_thr(bg, mp, sc)
            hit = rng.random() < thr
            level_hits.append(hit)
            level_gaps.append(bg)
            if hit:
                h_last_level_hit[lv] = h_block
        
        w = compute_overage_weight(slot_gap, level_hits, level_gaps)
        h_weights.append(w)
    
    # Adversary chain (burst forge — small slot gaps)
    a_weights = []
    a_last_level_hit = [0] * NUM_LEVELS
    a_block = 0
    
    for i in range(num_blocks):
        a_block += 1
        # Adversary forges at slot gap 1-2 (burst)
        slot_gap = rng.choice([1, 1, 1, 2, 2, 3])  # Mean ~1.7
        
        level_hits = [True]  # L0
        level_gaps = [slot_gap]
        
        for lv in range(1, NUM_LEVELS):
            bg = a_block - a_last_level_hit[lv]
            mp, sc = SHIFTED_EXP_PARAMS[lv]
            thr = shifted_exp_thr(bg, mp, sc)
            hit = rng.random() < thr
            level_hits.append(hit)
            level_gaps.append(bg)
            if hit:
                a_last_level_hit[lv] = a_block
        
        w = compute_overage_weight(slot_gap, level_hits, level_gaps)
        a_weights.append(w)
    
    return np.array(h_weights), np.array(a_weights)


# ── Test 2: Cumulative weight divergence over N blocks ───────────────

def simulate_fork_race(num_blocks=1000, num_races=1000):
    """
    Run fork races: honest and adversary both produce N blocks.
    Compare cumulative weight.
    """
    rng = np.random.RandomState(42)
    
    honest_wins = 0
    weight_advantages = []
    
    for race in range(num_races):
        h_cum = 0.0
        a_cum = 0.0
        
        h_last_hit = [0] * NUM_LEVELS
        a_last_hit = [0] * NUM_LEVELS
        
        for b in range(1, num_blocks + 1):
            # Honest block
            h_slot_gap = max(1, rng.geometric(1.0 / EXPECTED_L0_SLOT_GAP))
            h_hits = [True]
            h_gaps = [h_slot_gap]
            for lv in range(1, NUM_LEVELS):
                bg = b - h_last_hit[lv]
                mp, sc = SHIFTED_EXP_PARAMS[lv]
                hit = rng.random() < shifted_exp_thr(bg, mp, sc)
                h_hits.append(hit)
                h_gaps.append(bg)
                if hit: h_last_hit[lv] = b
            h_cum += compute_overage_weight(h_slot_gap, h_hits, h_gaps)
            
            # Adversary block (burst)
            a_slot_gap = rng.choice([1, 1, 1, 2, 2, 3])
            a_hits = [True]
            a_gaps = [a_slot_gap]
            for lv in range(1, NUM_LEVELS):
                bg = b - a_last_hit[lv]
                mp, sc = SHIFTED_EXP_PARAMS[lv]
                hit = rng.random() < shifted_exp_thr(bg, mp, sc)
                a_hits.append(hit)
                a_gaps.append(bg)
                if hit: a_last_hit[lv] = b
            a_cum += compute_overage_weight(a_slot_gap, a_hits, a_gaps)
        
        if h_cum > a_cum:
            honest_wins += 1
        weight_advantages.append(h_cum - a_cum)
    
    return honest_wins, num_races, np.array(weight_advantages)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("NiPoPoS Work Overage Chain Weight")
    print("=" * 70)
    
    # Test 1: Per-block weight distributions
    print("\n--- Test 1: Per-block weight distribution (10k blocks each) ---")
    h_weights, a_weights = simulate_honest_adversary(10000)
    
    print(f"  Honest:    mean={np.mean(h_weights):.2f}  std={np.std(h_weights):.2f}  "
          f"p50={np.median(h_weights):.2f}  p95={np.percentile(h_weights, 95):.2f}")
    print(f"  Adversary: mean={np.mean(a_weights):.2f}  std={np.std(a_weights):.2f}  "
          f"p50={np.median(a_weights):.2f}  p95={np.percentile(a_weights, 95):.2f}")
    print(f"  Honest advantage per block: {np.mean(h_weights) - np.mean(a_weights):.2f} "
          f"({(np.mean(h_weights) / np.mean(a_weights) - 1) * 100:.1f}% more)")
    
    # Test 2: Fork races at different lengths
    print("\n--- Test 2: Fork races (honest vs burst adversary, 1000 races each) ---")
    print(f"{'N blocks':>10} {'Honest wins':>12} {'Win%':>7} {'MeanAdv':>10} {'P5':>8} {'P50':>8} {'P95':>8}")
    print("-" * 65)
    
    race_results = []
    for n in [10, 20, 50, 100, 200, 500]:
        wins, total, advantages = simulate_fork_race(n, 1000)
        race_results.append((n, wins, total, advantages))
        print(f"{n:>10} {wins:>12}/{total} {wins/total*100:>6.1f}% {np.mean(advantages):>10.1f} "
              f"{np.percentile(advantages, 5):>8.1f} {np.median(advantages):>8.1f} "
              f"{np.percentile(advantages, 95):>8.1f}")
    
    # Test 3: What if adversary forges at honest-like gaps?
    print("\n--- Test 3: Adversary at different gap strategies (50 blocks, 1000 races) ---")
    rng = np.random.RandomState(123)
    
    gap_strategies = [
        ("burst (1-2)", [1, 1, 1, 2, 2, 3]),
        ("fast (2-4)", [2, 2, 3, 3, 4]),
        ("moderate (4-7)", [4, 5, 6, 7]),
        ("honest-like (5-10)", [5, 6, 7, 8, 9, 10]),
        ("slow (8-15)", [8, 10, 12, 15]),
    ]
    
    print(f"{'Strategy':<20} {'Honest wins':>12} {'Win%':>7} {'MeanAdv':>10}")
    print("-" * 55)
    
    for name, gap_choices in gap_strategies:
        wins = 0
        advantages = []
        for race in range(1000):
            h_cum = 0.0; a_cum = 0.0
            h_last = [0]*NUM_LEVELS; a_last = [0]*NUM_LEVELS
            
            for b in range(1, 51):
                # Honest
                hsg = max(1, rng.geometric(1.0 / EXPECTED_L0_SLOT_GAP))
                hh = [True]; hg = [hsg]
                for lv in range(1, NUM_LEVELS):
                    bg = b - h_last[lv]
                    hit = rng.random() < shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
                    hh.append(hit); hg.append(bg)
                    if hit: h_last[lv] = b
                h_cum += compute_overage_weight(hsg, hh, hg)
                
                # Adversary
                asg = rng.choice(gap_choices)
                ah = [True]; ag = [asg]
                for lv in range(1, NUM_LEVELS):
                    bg = b - a_last[lv]
                    hit = rng.random() < shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
                    ah.append(hit); ag.append(bg)
                    if hit: a_last[lv] = b
                a_cum += compute_overage_weight(asg, ah, ag)
            
            if h_cum > a_cum: wins += 1
            advantages.append(h_cum - a_cum)
        
        print(f"{name:<20} {wins:>12}/1000 {wins/10:>6.1f}% {np.mean(advantages):>10.1f}")
    
    # ── Plots ────────────────────────────────────────────────────────
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('NiPoPoS Work Overage: Honest vs Adversary', fontsize=14)
    
    # Plot 1: Per-block weight distributions
    ax = axes[0][0]
    bins = np.linspace(0, max(np.percentile(h_weights, 99), np.percentile(a_weights, 99)), 60)
    ax.hist(h_weights, bins=bins, alpha=0.6, density=True, color='steelblue', label='Honest (gap~7)')
    ax.hist(a_weights, bins=bins, alpha=0.6, density=True, color='coral', label='Adversary (burst)')
    ax.axvline(np.mean(h_weights), color='blue', linestyle='--', label=f'Honest mean={np.mean(h_weights):.1f}')
    ax.axvline(np.mean(a_weights), color='red', linestyle='--', label=f'Adv mean={np.mean(a_weights):.1f}')
    ax.set_xlabel('Block Weight')
    ax.set_ylabel('Density')
    ax.set_title('Per-Block Weight Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Honest win % vs race length
    ax = axes[0][1]
    ns = [n for n, _, _, _ in race_results]
    win_pcts = [w/t*100 for _, w, t, _ in race_results]
    ax.plot(ns, win_pcts, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Race Length (blocks)')
    ax.set_ylabel('Honest Win %')
    ax.set_title('Honest Win Rate vs Fork Length')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    
    # Plot 3: Cumulative weight advantage over blocks (single race)
    ax = axes[1][0]
    rng2 = np.random.RandomState(7)
    h_cum_trace = []; a_cum_trace = []
    h_c = 0; a_c = 0
    hl = [0]*NUM_LEVELS; al = [0]*NUM_LEVELS
    for b in range(1, 501):
        hsg = max(1, rng2.geometric(1.0/7.0))
        hh=[True]; hg=[hsg]
        for lv in range(1,NUM_LEVELS):
            bg=b-hl[lv]; hit=rng2.random()<shifted_exp_thr(bg,*SHIFTED_EXP_PARAMS[lv])
            hh.append(hit); hg.append(bg)
            if hit: hl[lv]=b
        h_c += compute_overage_weight(hsg, hh, hg)
        h_cum_trace.append(h_c)
        
        asg = rng2.choice([1,1,1,2,2,3])
        ah=[True]; ag=[asg]
        for lv in range(1,NUM_LEVELS):
            bg=b-al[lv]; hit=rng2.random()<shifted_exp_thr(bg,*SHIFTED_EXP_PARAMS[lv])
            ah.append(hit); ag.append(bg)
            if hit: al[lv]=b
        a_c += compute_overage_weight(asg, ah, ag)
        a_cum_trace.append(a_c)
    
    blocks = range(1, 501)
    ax.plot(blocks, h_cum_trace, color='steelblue', linewidth=1.5, label='Honest')
    ax.plot(blocks, a_cum_trace, color='coral', linewidth=1.5, label='Adversary (burst)')
    ax.fill_between(blocks, h_cum_trace, a_cum_trace, alpha=0.2, color='green')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Cumulative Weight')
    ax.set_title('Cumulative Weight Over Time (single race)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Weight advantage distribution at N=100
    ax = axes[1][1]
    _, _, _, advs_100 = race_results[3]  # N=100
    ax.hist(advs_100, bins=50, density=True, alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break even')
    ax.axvline(np.mean(advs_100), color='blue', linestyle='--', label=f'Mean={np.mean(advs_100):.0f}')
    ax.set_xlabel('Honest Weight Advantage (positive = honest heavier)')
    ax.set_ylabel('Density')
    ax.set_title('Weight Advantage Distribution (100-block races)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig_overage_weight.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_overage_weight.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_overage_weight.png/pdf")
