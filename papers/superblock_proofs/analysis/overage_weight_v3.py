#!/usr/bin/env python3
"""
NiPoPoS Work Overage v3: Slot-gap-gated super-level thresholds.

Two changes from v2:
  1. block_weight = threshold(slot_gap)^α × (1 + Σ 2^μ × hit_μ)
  2. Super-level hit probability ALSO depends on L0 slot gap:
     
     super_threshold(block_gap_μ, slot_gap_L0) =
         shifted_exp(block_gap_μ) × min(1, slot_gap_L0 / cutoff)
     
     At slot_gap=2: super thresholds scaled by 2/15 = 0.13 (87% reduction)
     At slot_gap=7: scaled by 7/15 = 0.47 (53% reduction)  
     At slot_gap≥15: full threshold (no reduction)

This means:
  - Burst-forging gets CRUSHED on both L0 weight AND super-level hits
  - The kink at moderate gaps (4-7) should narrow because adversary gets
    fewer super hits there too, not just lower L0 threshold
  - Honest chain at natural gaps (5-10) gets partial scaling → fewer super 
    hits than max, but honest's L0 weight is healthy
"""
import numpy as np
import hashlib
import time

np.random.seed(42)

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

LEVEL_WEIGHTS = [2.0 ** lv for lv in range(NUM_LEVELS)]

def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=CUTOFF):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** 1.0

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def slot_gap_scaling(slot_gap, cutoff=CUTOFF):
    """Scale factor for super-level thresholds based on L0 slot gap."""
    return min(1.0, slot_gap / cutoff)


def block_weight(slot_gap, level_hits, alpha=2.0):
    thr = snowplow_thr(slot_gap)
    base = thr ** alpha
    super_mult = 1.0
    for lv in range(1, NUM_LEVELS):
        if level_hits[lv]:
            super_mult += LEVEL_WEIGHTS[lv]
    return base * super_mult


def simulate_race_v3(num_blocks, honest_gap_mean, adv_gap_choices, alpha, use_gating, rng):
    """
    Fork race with optional slot-gap-gated super thresholds.
    use_gating=True: super thresholds scaled by slot_gap/cutoff
    use_gating=False: super thresholds at full rate (v2 behavior)
    """
    h_cum = 0.0; a_cum = 0.0
    h_last = [0]*NUM_LEVELS; a_last = [0]*NUM_LEVELS
    h_super_hits = [0]*NUM_LEVELS; a_super_hits = [0]*NUM_LEVELS
    
    for b in range(1, num_blocks + 1):
        # Honest
        hsg = max(1, rng.geometric(1.0 / honest_gap_mean))
        h_scale = slot_gap_scaling(hsg) if use_gating else 1.0
        hh = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - h_last[lv]
            base_thr = shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
            hit = rng.random() < (base_thr * h_scale)
            hh.append(hit)
            if hit:
                h_last[lv] = b
                h_super_hits[lv] += 1
        h_cum += block_weight(hsg, hh, alpha)
        
        # Adversary
        asg = rng.choice(adv_gap_choices)
        a_scale = slot_gap_scaling(asg) if use_gating else 1.0
        ah = [True]
        for lv in range(1, NUM_LEVELS):
            bg = b - a_last[lv]
            base_thr = shifted_exp_thr(bg, *SHIFTED_EXP_PARAMS[lv])
            hit = rng.random() < (base_thr * a_scale)
            ah.append(hit)
            if hit:
                a_last[lv] = b
                a_super_hits[lv] += 1
        a_cum += block_weight(asg, ah, alpha)
    
    return h_cum, a_cum, h_super_hits, a_super_hits


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    NUM_RACES = 5000
    ALPHA = 2.0
    
    print("NiPoPoS Work Overage v3: Slot-Gap-Gated Super Thresholds")
    print("=" * 80)
    
    strategies = [
        ("burst (1-2)",      [1, 1, 1, 2, 2, 3]),
        ("fast (2-4)",       [2, 2, 3, 3, 4]),
        ("moderate (4-7)",   [4, 5, 6, 7]),
        ("honest-like (5-9)",[5, 6, 7, 8, 9]),
        ("slow (8-15)",      [8, 10, 12, 15]),
    ]
    
    # ── Compare v2 (no gating) vs v3 (with gating) ──────────────────
    
    for use_gating, version in [(False, "v2 (no gating)"), (True, "v3 (slot-gap gated)")]:
        print(f"\n--- {version}, α={ALPHA}, 50 blocks, {NUM_RACES} races ---")
        print(f"{'Strategy':<20} {'HonWin%':>8} {'Ratio':>7} {'H_L1':>5} {'A_L1':>5} {'H_L3':>5} {'A_L3':>5} {'H_L5':>5} {'A_L5':>5}")
        print("-" * 75)
        
        for name, gaps in strategies:
            rng = np.random.RandomState(42)
            wins = 0; h_tot = 0; a_tot = 0
            h_sh_sum = [0]*NUM_LEVELS; a_sh_sum = [0]*NUM_LEVELS
            
            for _ in range(NUM_RACES):
                hc, ac, hsh, ash = simulate_race_v3(50, 7.0, gaps, ALPHA, use_gating, rng)
                if hc > ac: wins += 1
                h_tot += hc; a_tot += ac
                for lv in range(NUM_LEVELS):
                    h_sh_sum[lv] += hsh[lv]
                    a_sh_sum[lv] += ash[lv]
            
            ratio = (h_tot/NUM_RACES) / max(a_tot/NUM_RACES, 1e-10)
            # Average super hits per race
            h_avg = [h_sh_sum[lv]/NUM_RACES for lv in range(NUM_LEVELS)]
            a_avg = [a_sh_sum[lv]/NUM_RACES for lv in range(NUM_LEVELS)]
            
            print(f"{name:<20} {wins/NUM_RACES*100:>7.1f}% {ratio:>7.2f}x "
                  f"{h_avg[1]:>5.1f} {a_avg[1]:>5.1f} {h_avg[3]:>5.1f} {a_avg[3]:>5.1f} "
                  f"{h_avg[5]:>5.1f} {a_avg[5]:>5.1f}")
    
    # ── Detailed race length sweep with v3 ───────────────────────────
    
    print(f"\n--- v3 race length sweep (burst adversary, α={ALPHA}, {NUM_RACES} races) ---")
    print(f"{'N':>5} {'HonWin%':>8} {'Ratio':>7} {'MeanAdv':>10}")
    print("-" * 35)
    
    race_results = []
    for n in [5, 10, 20, 50, 100, 200]:
        rng = np.random.RandomState(42)
        wins = 0; h_tot = 0; a_tot = 0; advs = []
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race_v3(n, 7.0, [1,1,1,2,2,3], ALPHA, True, rng)
            if hc > ac: wins += 1
            h_tot += hc; a_tot += ac
            advs.append(hc - ac)
        ratio = (h_tot/NUM_RACES) / max(a_tot/NUM_RACES, 1e-10)
        race_results.append((n, wins/NUM_RACES*100, ratio, np.mean(advs)))
        print(f"{n:>5} {wins/NUM_RACES*100:>7.1f}% {ratio:>7.2f}x {np.mean(advs):>10.4f}")
    
    # ── α sweep with v3 gating ───────────────────────────────────────
    
    print(f"\n--- v3 α sweep (50 blocks, burst adversary, {NUM_RACES} races) ---")
    print(f"{'α':>5} {'HonWin%':>8} {'Ratio':>7}")
    print("-" * 25)
    
    alpha_results = []
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rng = np.random.RandomState(42)
        wins = 0; h_tot = 0; a_tot = 0
        for _ in range(NUM_RACES):
            hc, ac, _, _ = simulate_race_v3(50, 7.0, [1,1,1,2,2,3], alpha, True, rng)
            if hc > ac: wins += 1
            h_tot += hc; a_tot += ac
        ratio = (h_tot/NUM_RACES) / max(a_tot/NUM_RACES, 1e-10)
        alpha_results.append((alpha, wins/NUM_RACES*100, ratio))
        print(f"{alpha:>5.1f} {wins/NUM_RACES*100:>7.1f}% {ratio:>7.2f}x")
    
    # ── The critical test: moderate adversary under v3 ───────────────
    
    print(f"\n--- CRITICAL: Moderate adversary (gap 4-7) under v3 vs v2 ---")
    for use_gating, ver in [(False, "v2"), (True, "v3")]:
        rng = np.random.RandomState(42)
        wins = 0; h_tot = 0; a_tot = 0
        for _ in range(10000):  # More races for precision
            hc, ac, _, _ = simulate_race_v3(50, 7.0, [4,5,6,7], ALPHA, use_gating, rng)
            if hc > ac: wins += 1
            h_tot += hc; a_tot += ac
        ratio = (h_tot/10000) / max(a_tot/10000, 1e-10)
        print(f"  {ver}: honest wins {wins/100:.1f}%, ratio {ratio:.3f}x")
    
    # ── Plots ────────────────────────────────────────────────────────
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('NiPoPoS v3: Slot-Gap-Gated Super Thresholds', fontsize=14)
    
    # Plot 1: v2 vs v3 strategy comparison
    ax = axes[0][0]
    v2_wins = []; v3_wins = []
    for name, gaps in strategies:
        for ug, store in [(False, v2_wins), (True, v3_wins)]:
            rng = np.random.RandomState(42)
            w = sum(1 for _ in range(2000) 
                    if simulate_race_v3(50, 7.0, gaps, ALPHA, ug, rng)[0] > 
                       simulate_race_v3(50, 7.0, gaps, ALPHA, ug, rng)[1])
        # Re-run cleanly
    # Just use pre-computed values from main runs above
    v2_data = {}; v3_data = {}
    for ug, store in [(False, v2_data), (True, v3_data)]:
        for name, gaps in strategies:
            rng = np.random.RandomState(42)
            wins = sum(1 for _ in range(NUM_RACES)
                       if (lambda: (rng2 := np.random.RandomState(rng.randint(0, 2**31)),
                                    simulate_race_v3(50, 7.0, gaps, ALPHA, ug, rng)[0] >
                                    simulate_race_v3(50, 7.0, gaps, ALPHA, ug, rng)[1]))()[-1])
            # This is getting messy, just recompute simply
            pass
    
    # Simpler approach for plots
    strat_names = [n for n, _ in strategies]
    
    v2_pcts = []
    v3_pcts = []
    for name, gaps in strategies:
        for ug, store in [(False, v2_pcts), (True, v3_pcts)]:
            rng = np.random.RandomState(42)
            w = 0
            for _ in range(NUM_RACES):
                hc, ac, _, _ = simulate_race_v3(50, 7.0, gaps, ALPHA, ug, rng)
                if hc > ac: w += 1
            store.append(w / NUM_RACES * 100)
    
    x = np.arange(len(strat_names))
    width = 0.35
    ax.bar(x - width/2, v2_pcts, width, label='v2 (no gating)', color='steelblue')
    ax.bar(x + width/2, v3_pcts, width, label='v3 (slot-gap gated)', color='coral')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Honest Win %')
    ax.set_title(f'Strategy Comparison: v2 vs v3 (α={ALPHA})')
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=15, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Race length (v3)
    ax = axes[0][1]
    ns = [n for n, _, _, _ in race_results]
    wpcts = [w for _, w, _, _ in race_results]
    ax.plot(ns, wpcts, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Race Length (blocks)')
    ax.set_ylabel('Honest Win %')
    ax.set_title(f'v3: Honest Win Rate vs Fork Length (burst adv)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    
    # Plot 3: α sweep (v3)
    ax = axes[1][0]
    alphas = [a for a, _, _ in alpha_results]
    apcts = [w for _, w, _ in alpha_results]
    arats = [r for _, _, r in alpha_results]
    ax.plot(alphas, apcts, 'go-', linewidth=2, markersize=8, label='Honest win %')
    ax2 = ax.twinx()
    ax2.plot(alphas, arats, 'bs--', linewidth=1.5, markersize=6, label='Weight ratio')
    ax.set_xlabel('α')
    ax.set_ylabel('Honest Win %', color='green')
    ax2.set_ylabel('Weight Ratio', color='blue')
    ax.set_title('v3: α Sweep (burst adversary)')
    ax.legend(loc='center left')
    ax2.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Super hit comparison v2 vs v3 at moderate gaps
    ax = axes[1][1]
    levels = list(range(1, 8))
    h_hits_v3 = []; a_hits_v3 = []
    rng = np.random.RandomState(42)
    h_sh = [0]*NUM_LEVELS; a_sh = [0]*NUM_LEVELS
    for _ in range(NUM_RACES):
        _, _, hsh, ash = simulate_race_v3(50, 7.0, [4,5,6,7], ALPHA, True, rng)
        for lv in range(NUM_LEVELS):
            h_sh[lv] += hsh[lv]; a_sh[lv] += ash[lv]
    h_avg = [h_sh[lv]/NUM_RACES for lv in levels]
    a_avg = [a_sh[lv]/NUM_RACES for lv in levels]
    
    x = np.arange(len(levels))
    ax.bar(x - width/2, h_avg, width, label='Honest (gap~7)', color='steelblue')
    ax.bar(x + width/2, a_avg, width, label='Adversary (gap 4-7)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{lv}' for lv in levels])
    ax.set_ylabel('Avg Super Hits per 50-block Race')
    ax.set_title('v3: Super Hit Rates (moderate adversary)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/fig_overage_v3.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_overage_v3.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_overage_v3.png/pdf")
