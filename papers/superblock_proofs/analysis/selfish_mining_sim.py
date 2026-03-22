#!/usr/bin/env python3
"""
NiPoPoS Selfish Mining Experiment.

Model: Adversary with α stake fraction attempts private chain attack.
Every K honest blocks, the adversary tries to build a competing fork:
  - Adversary starts from the honest chain tip
  - Burst-forges blocks as fast as possible (every eligible slot)
  - Gap between adversary blocks ≈ 1/α_effective slots (determined by LDD)
  - Because adversary blocks come fast, their base-block gaps are SMALL
  - Small gaps → low shifted exponential threshold → fewer super-level hits
  - Honest chain grows normally with natural gaps → accumulates superblocks

We run many fork races and measure:
  1. What fraction of races does adversary win on LENGTH alone?
  2. Of those length ties, how many does superblock weight resolve for honest?
  3. Net adversary success rate under each rule
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

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dhash(seed, slot, level):
    h = hashlib.sha256(f"{seed}:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


def fork_race(
    race_length: int,       # Number of honest blocks in the race window
    honest_avg_gap: float,  # Average slot gap between honest blocks (~7)
    adversary_avg_gap: float,  # Average slot gap for adversary blocks (fast)
    adv_stake_frac: float,
    seed: int = 0,
):
    """
    Simulate a single fork race.
    
    Both honest and adversary produce `race_length` blocks.
    Honest blocks have natural gaps → earn superblock credit.
    Adversary blocks have small gaps → earn less/no superblock credit.
    
    Returns: (honest_super_heights, adversary_super_heights, honest_slots, adv_slots)
    """
    rng = np.random.RandomState(seed)
    
    # Honest chain: geometric gaps centered on honest_avg_gap
    h_gaps = rng.geometric(1.0 / honest_avg_gap, size=race_length)
    h_gaps = np.maximum(h_gaps, 1)  # At least 1 slot
    h_slots = np.cumsum(h_gaps)
    
    # Adversary chain: geometric gaps centered on adversary_avg_gap
    a_gaps = rng.geometric(1.0 / adversary_avg_gap, size=race_length)
    a_gaps = np.maximum(a_gaps, 1)
    a_slots = np.cumsum(a_gaps)
    
    # Compute super levels for honest chain (base-block gaps)
    h_super = [0] * NUM_LEVELS
    h_super[0] = race_length
    h_last_hit = [0] * NUM_LEVELS
    for i in range(race_length):
        block_num = i + 1
        for lv in range(1, NUM_LEVELS):
            bg = block_num - h_last_hit[lv]
            mp, sc = SHIFTED_EXP_PARAMS[lv]
            thr = shifted_exp_thr(bg, mp, sc)
            if dhash(seed + 1000, int(h_slots[i]), lv) < thr:
                h_super[lv] += 1
                h_last_hit[lv] = block_num
    
    # Compute super levels for adversary chain
    a_super = [0] * NUM_LEVELS
    a_super[0] = race_length
    a_last_hit = [0] * NUM_LEVELS
    for i in range(race_length):
        block_num = i + 1
        for lv in range(1, NUM_LEVELS):
            bg = block_num - a_last_hit[lv]
            mp, sc = SHIFTED_EXP_PARAMS[lv]
            thr = shifted_exp_thr(bg, mp, sc)
            if dhash(seed + 2000, int(a_slots[i]), lv) < thr:
                a_super[lv] += 1
                a_last_hit[lv] = block_num
    
    return h_super, a_super, h_slots, a_slots


def compare_lexicographic(h_super, a_super):
    """Lexicographic comparison highest level first. Returns >0 if honest wins."""
    for lv in range(NUM_LEVELS - 1, 0, -1):
        if h_super[lv] != a_super[lv]:
            return h_super[lv] - a_super[lv]
    return 0


def run_experiment(
    num_races: int,
    race_length: int,
    honest_avg_gap: float,
    adversary_avg_gap: float,
    adv_stake_frac: float,
):
    """Run many fork races and collect statistics."""
    
    honest_wins_super = 0
    adversary_wins_super = 0
    ties = 0
    
    honest_super_advantage = []  # Per-level advantage across races
    
    for race in range(num_races):
        h_super, a_super, h_slots, a_slots = fork_race(
            race_length, honest_avg_gap, adversary_avg_gap, adv_stake_frac, seed=race
        )
        
        cmp = compare_lexicographic(h_super, a_super)
        if cmp > 0:
            honest_wins_super += 1
        elif cmp < 0:
            adversary_wins_super += 1
        else:
            ties += 1
        
        # Per-level advantage
        advantage = [h_super[lv] - a_super[lv] for lv in range(NUM_LEVELS)]
        honest_super_advantage.append(advantage)
    
    avg_advantage = np.mean(honest_super_advantage, axis=0)
    
    return {
        "honest_wins": honest_wins_super,
        "adversary_wins": adversary_wins_super,
        "ties": ties,
        "honest_win_pct": honest_wins_super / num_races * 100,
        "avg_advantage": avg_advantage,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    NUM_RACES = 10_000
    
    print(f"NiPoPoS Selfish Mining Experiment — {NUM_RACES:,} fork races each")
    print("=" * 80)
    print(f"Setup: Honest and adversary each produce N blocks in a fork race.")
    print(f"Honest has natural gaps (~7 slots). Adversary burst-forges.")
    print(f"Both chains have equal LENGTH → superblock weight decides the winner.")
    print()
    
    # Test different adversary speeds and race lengths
    adversary_gaps = [1, 2, 3, 4, 5]  # 1 = max burst, 5 = moderate
    race_lengths = [10, 20, 50, 100, 200]
    honest_gap = 7.0
    
    # ── Experiment 1: Vary adversary gap at fixed race length ────────
    
    print("--- Experiment 1: Adversary gap vs superblock advantage (race=50 blocks) ---")
    print(f"{'AdvGap':>7} {'HonestWin%':>11} {'AdvWin%':>9} {'Ties%':>7} {'L1adv':>7} {'L3adv':>7} {'L5adv':>7} {'L7adv':>7}")
    print("-" * 70)
    
    exp1_results = []
    for ag in adversary_gaps:
        r = run_experiment(NUM_RACES, 50, honest_gap, float(ag), 0.30)
        exp1_results.append((ag, r))
        adv = r["avg_advantage"]
        print(f"{ag:>7d} {r['honest_win_pct']:>10.1f}% {r['adversary_wins']/NUM_RACES*100:>8.1f}% "
              f"{r['ties']/NUM_RACES*100:>6.1f}% {adv[1]:>6.1f} {adv[3]:>6.1f} {adv[5]:>6.1f} {adv[7]:>6.1f}")
    
    # ── Experiment 2: Vary race length at fixed adversary gap ────────
    
    print(f"\n--- Experiment 2: Race length vs superblock advantage (adversary gap=1) ---")
    print(f"{'RaceLen':>8} {'HonestWin%':>11} {'AdvWin%':>9} {'Ties%':>7} {'L1adv':>7} {'L3adv':>7} {'L5adv':>7}")
    print("-" * 60)
    
    exp2_results = []
    for rl in race_lengths:
        r = run_experiment(NUM_RACES, rl, honest_gap, 1.0, 0.30)
        exp2_results.append((rl, r))
        adv = r["avg_advantage"]
        print(f"{rl:>8d} {r['honest_win_pct']:>10.1f}% {r['adversary_wins']/NUM_RACES*100:>8.1f}% "
              f"{r['ties']/NUM_RACES*100:>6.1f}% {adv[1]:>6.1f} {adv[3]:>6.1f} {adv[5]:>6.1f}")
    
    # ── Experiment 3: Honest gap matters (higher f_effective) ────────
    
    print(f"\n--- Experiment 3: Honest gap (f_effective proxy) vs advantage (adv gap=2, race=50) ---")
    honest_gaps = [3.0, 5.0, 7.0, 10.0, 15.0]
    print(f"{'HonGap':>7} {'~f_eff':>7} {'HonestWin%':>11} {'L1adv':>7} {'L3adv':>7} {'L5adv':>7}")
    print("-" * 55)
    
    exp3_results = []
    for hg in honest_gaps:
        r = run_experiment(NUM_RACES, 50, hg, 2.0, 0.30)
        exp3_results.append((hg, r))
        adv = r["avg_advantage"]
        f_eff = 1.0/hg * 100
        print(f"{hg:>7.1f} {f_eff:>6.0f}% {r['honest_win_pct']:>10.1f}% {adv[1]:>6.1f} {adv[3]:>6.1f} {adv[5]:>6.1f}")
    
    # ── Plots ────────────────────────────────────────────────────────
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'NiPoPoS Selfish Mining: Superblock Weight Advantage ({NUM_RACES:,} races)', fontsize=14)
    
    # Plot 1: Adversary gap vs honest win %
    ax = axes[0]
    gaps_x = [ag for ag, _ in exp1_results]
    wins_y = [r["honest_win_pct"] for _, r in exp1_results]
    ax.plot(gaps_x, wins_y, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (no advantage)')
    ax.set_xlabel('Adversary Avg Gap (blocks)')
    ax.set_ylabel('Honest Wins (%)')
    ax.set_title('Honest Win Rate vs Adversary Speed\n(50 block race)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    
    # Plot 2: Race length vs per-level advantage at adv_gap=1
    ax = axes[1]
    for lv in [1, 3, 5, 7]:
        advs = [r["avg_advantage"][lv] for _, r in exp2_results]
        ax.plot(race_lengths, advs, 'o-', linewidth=2, markersize=6, label=f'L{lv}')
    ax.set_xlabel('Race Length (blocks)')
    ax.set_ylabel('Honest Superblock Advantage')
    ax.set_title('Superblock Advantage Grows with Race Length\n(adversary gap=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Honest gap vs honest win %
    ax = axes[2]
    hgaps_x = [hg for hg, _ in exp3_results]
    wins_y = [r["honest_win_pct"] for _, r in exp3_results]
    f_eff_labels = [f'{1/hg*100:.0f}%' for hg in hgaps_x]
    ax.plot(hgaps_x, wins_y, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (no advantage)')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(hgaps_x)
    ax2.set_xticklabels(f_eff_labels)
    ax2.set_xlabel('~f_effective')
    ax.set_xlabel('Honest Avg Gap (slots)')
    ax.set_ylabel('Honest Wins (%)')
    ax.set_title('Honest Win Rate at Different f_effective\n(adversary gap=2, race=50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    
    plt.tight_layout()
    plt.savefig('figures/fig_selfish_mining.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_selfish_mining.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_selfish_mining.png/pdf")
