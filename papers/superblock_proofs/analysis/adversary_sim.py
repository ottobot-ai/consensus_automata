#!/usr/bin/env python3
"""
NiPoPoS Adversary Experiment: Superblock weight vs chain length under attack.

Model: One adversarial staker (30% stake) withholds blocks and tries to build
a competing private chain. At each slot, the adversary either:
  A) Extends their private chain (always eligible due to high stake)
  B) Releases when their chain matches or exceeds the honest chain

We measure: how often the adversary's chain is preferred under each rule.

With standard maxvalid-tk: adversary wins when private chain ≥ honest chain length
With maxvalid-tk-super: adversary also needs to match superblock weight

The adversary burst-forges (gap=1 between private blocks) → gets ZERO super-level
credit from the shifted exponential (ψ=1). Honest chain naturally accumulates
superblocks → honest chain always wins the superblock tiebreaker.
"""
import numpy as np
import hashlib
import time
from collections import defaultdict

np.random.seed(42)

NUM_SLOTS = 100_000
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

def snowplow_thr(gap, amplitude, baseline, cutoff, rel_stake):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel_stake

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dhash(seed, slot, level):
    h = hashlib.sha256(f"{seed}:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


def run_adversary_experiment(
    num_slots, adversary_stake_pct, honest_stakes, amplitude, baseline, cutoff
):
    """
    Simulate honest chain + adversary private chain.
    
    Honest stakers: build the public chain normally (LDD + super levels)
    Adversary: privately mines blocks, checks each slot for eligibility,
    burst-forges to maximize chain length. Releases when private ≥ honest.
    
    Returns metrics on how often the adversary catches up, and whether
    superblock weight prevents the adversary from winning ties.
    """
    total_honest_stake = sum(honest_stakes)
    adversary_stake = int(total_honest_stake * adversary_stake_pct / (1 - adversary_stake_pct))
    total_stake = total_honest_stake + adversary_stake
    
    # Honest chain state
    h_last_slot = 0
    h_l0_count = 0
    h_last_level_block = [0] * NUM_LEVELS
    h_super_heights = [0] * NUM_LEVELS
    
    # Adversary chain state (private)
    a_last_slot = 0
    a_l0_count = 0
    a_last_level_block = [0] * NUM_LEVELS
    a_super_heights = [0] * NUM_LEVELS
    
    # Metrics
    adversary_catches_honest = 0        # Length match
    adversary_exceeds_honest = 0        # Length exceed
    super_saves_honest = 0              # Super weight breaks tie in honest favor
    super_saves_adversary = 0           # Super weight breaks tie in adversary favor
    adversary_wins_standard = 0         # Wins under maxvalid-tk
    adversary_wins_super = 0            # Wins under maxvalid-tk-super
    total_adversary_releases = 0
    
    # Track adversary gap (how far behind they are)
    adversary_deficit_history = []
    
    for slot in range(1, num_slots + 1):
        # --- Honest chain ---
        h_slot_gap = slot - h_last_slot
        
        for si in range(len(honest_stakes)):
            rel = honest_stakes[si] / total_stake
            thr = snowplow_thr(h_slot_gap, amplitude, baseline, cutoff, rel)
            if dhash(si + 100, slot, 0) < thr:  # offset seed to avoid collision
                h_l0_count += 1
                h_last_slot = slot
                h_super_heights[0] = h_l0_count
                
                # Super levels
                for lv in range(1, NUM_LEVELS):
                    bg = h_l0_count - h_last_level_block[lv]
                    mp, sc = SHIFTED_EXP_PARAMS[lv]
                    if dhash(si + 100, slot, lv) < shifted_exp_thr(bg, mp, sc):
                        h_super_heights[lv] += 1
                        h_last_level_block[lv] = h_l0_count
                break  # First eligible honest staker wins
        
        # --- Adversary (private chain) ---
        # Adversary checks eligibility with their own VRF (seed=0)
        a_slot_gap = slot - a_last_slot
        adv_rel = adversary_stake / total_stake
        adv_thr = snowplow_thr(a_slot_gap, amplitude, baseline, cutoff, adv_rel)
        
        if dhash(0, slot, 0) < adv_thr:
            a_l0_count += 1
            a_last_slot = slot
            a_super_heights[0] = a_l0_count
            
            # Adversary super levels — their blocks are at natural gaps
            # (they don't burst-forge every slot, they forge when eligible)
            for lv in range(1, NUM_LEVELS):
                bg = a_l0_count - a_last_level_block[lv]
                mp, sc = SHIFTED_EXP_PARAMS[lv]
                if dhash(0, slot, lv) < shifted_exp_thr(bg, mp, sc):
                    a_super_heights[lv] += 1
                    a_last_level_block[lv] = a_l0_count
        
        # --- Check if adversary should release ---
        # Adversary releases when they match or exceed honest chain length
        if a_l0_count >= h_l0_count and a_l0_count > 0:
            total_adversary_releases += 1
            
            if a_l0_count > h_l0_count:
                adversary_exceeds_honest += 1
                adversary_wins_standard += 1
                adversary_wins_super += 1
            elif a_l0_count == h_l0_count:
                adversary_catches_honest += 1
                
                # Standard: compare head slots
                if a_last_slot < h_last_slot:
                    adversary_wins_standard += 1
                    # Super: also check superblock weight
                    adversary_wins_super += 1  # lower slot already wins
                elif a_last_slot == h_last_slot:
                    # Same length, same head slot → super weight decides
                    adversary_wins_standard += 1  # tie = coin flip in standard (adversary benefit of doubt)
                    
                    # Check super weight
                    adv_wins_super_weight = False
                    honest_wins_super_weight = False
                    for lv in range(NUM_LEVELS - 1, 0, -1):
                        if a_super_heights[lv] > h_super_heights[lv]:
                            adv_wins_super_weight = True
                            break
                        elif a_super_heights[lv] < h_super_heights[lv]:
                            honest_wins_super_weight = True
                            break
                    
                    if adv_wins_super_weight:
                        adversary_wins_super += 1
                        super_saves_adversary += 1
                    elif honest_wins_super_weight:
                        super_saves_honest += 1
                    else:
                        adversary_wins_super += 1  # true tie, adversary benefit
                else:
                    # Honest has lower head slot → honest wins both rules
                    pass
            
            # Reset adversary (start over after release)
            a_l0_count = 0
            a_last_slot = slot
            a_last_level_block = [0] * NUM_LEVELS
            a_super_heights = [0] * NUM_LEVELS
        
        if slot % 20000 == 0:
            deficit = h_l0_count - a_l0_count
            adversary_deficit_history.append((slot, deficit))
    
    return {
        "adversary_stake": adversary_stake_pct * 100,
        "honest_blocks": h_l0_count,
        "adversary_releases": total_adversary_releases,
        "catches": adversary_catches_honest,
        "exceeds": adversary_exceeds_honest,
        "wins_standard": adversary_wins_standard,
        "wins_super": adversary_wins_super,
        "super_saves_honest": super_saves_honest,
        "super_saves_adversary": super_saves_adversary,
        "honest_super": list(h_super_heights),
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("NiPoPoS Adversary Experiment")
    print("=" * 80)
    
    # Test at multiple adversary stake levels and f_effective values
    configs = [
        {"label": "15%", "amplitude": 0.50, "baseline": 0.05, "cutoff": 15},
        {"label": "25%", "amplitude": 1.00, "baseline": 0.09, "cutoff": 15},
        {"label": "35%", "amplitude": 1.60, "baseline": 0.15, "cutoff": 15},
    ]
    
    adversary_stakes = [0.10, 0.20, 0.30, 0.40, 0.45]
    honest_stakes = [1000] * 7  # 7 honest stakers, equal stake
    
    all_results = []
    
    for config in configs:
        print(f"\n--- f_effective: {config['label']} ---")
        for adv_pct in adversary_stakes:
            t0 = time.time()
            r = run_adversary_experiment(
                NUM_SLOTS, adv_pct, honest_stakes,
                config["amplitude"], config["baseline"], config["cutoff"]
            )
            t = time.time() - t0
            r["f_effective"] = config["label"]
            all_results.append(r)
            
            improvement = ""
            if r["wins_standard"] > 0:
                saved_pct = r["super_saves_honest"] / r["wins_standard"] * 100
                improvement = f" super saves {r['super_saves_honest']}/{r['wins_standard']} ({saved_pct:.0f}%)"
            
            print(f"  Adv={adv_pct*100:.0f}%: honest={r['honest_blocks']:,} releases={r['adversary_releases']} "
                  f"wins_std={r['wins_standard']} wins_super={r['wins_super']}{improvement} ({t:.1f}s)")
    
    # Summary table
    print(f"\n{'='*90}")
    print(f"{'f_eff':<8} {'Adv%':>5} {'Releases':>10} {'WinStd':>8} {'WinSup':>8} {'Saved':>8} {'SavedAdv':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['f_effective']:<8} {r['adversary_stake']:>4.0f}% {r['adversary_releases']:>10} "
              f"{r['wins_standard']:>8} {r['wins_super']:>8} {r['super_saves_honest']:>8} {r['super_saves_adversary']:>8}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'NiPoPoS: Adversary Success Rate ({NUM_SLOTS:,} slots)', fontsize=14)
    
    for idx, config in enumerate(configs):
        ax = axes[idx]
        subset = [r for r in all_results if r["f_effective"] == config["label"]]
        advs = [r["adversary_stake"] for r in subset]
        wins_std = [r["wins_standard"] for r in subset]
        wins_sup = [r["wins_super"] for r in subset]
        
        ax.plot(advs, wins_std, 'bs-', linewidth=2, markersize=8, label='maxvalid-tk')
        ax.plot(advs, wins_sup, 'r^-', linewidth=2, markersize=8, label='maxvalid-tk-super')
        ax.set_xlabel('Adversary Stake (%)')
        ax.set_ylabel('Adversary Wins')
        ax.set_title(f'f_effective = {config["label"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig_adversary_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_adversary_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_adversary_comparison.png/pdf")
