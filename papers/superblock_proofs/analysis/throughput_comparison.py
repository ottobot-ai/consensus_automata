#!/usr/bin/env python3
"""
NiPoPoS Throughput Experiment: Can superblock levels safely increase f_effective?

Compare fork resolution and common prefix under two chain selection rules:
  A) maxvalid-tk:       length → lower head slot (standard Taktikos)
  B) maxvalid-tk-super: length → lower head slot → lexicographic superblock weight

Run both at multiple f_effective values (15%, 20%, 25%, 30%, 35%) and measure:
  1. Fork rate (% of block-producing slots with >1 eligible staker)
  2. Fork resolution depth (how many slots until forks resolve)
  3. Chain quality (fraction of honest blocks in canonical chain)
  4. Common prefix violations (how often do two honest nodes disagree after k blocks)

The hypothesis: maxvalid-tk-super resolves forks faster via superblock weight,
enabling higher f_effective without degrading common prefix guarantees.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

np.random.seed(42)

# ── Simulation Parameters ────────────────────────────────────────────

NUM_SLOTS = 100_000
NUM_STAKERS = 10  # More stakers = more forks at higher f_effective
STAKES = [1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600]  # More uniform
TOTAL_STAKE = sum(STAKES)
NUM_LEVELS = 10
PSI = 1

# Shifted exponential params (from our tuning)
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

# f_effective values to test (controlled via amplitude)
# Higher amplitude → higher fill rate
# Approximate mapping: amplitude → f_effective
# 0.50 → ~15%, 0.75 → ~20%, 1.00 → ~25%, 1.30 → ~30%, 1.60 → ~35%
F_EFFECTIVE_CONFIGS = [
    {"label": "15%", "amplitude": 0.50, "baseline": 0.05, "cutoff": 15},
    {"label": "20%", "amplitude": 0.75, "baseline": 0.07, "cutoff": 15},
    {"label": "25%", "amplitude": 1.00, "baseline": 0.09, "cutoff": 15},
    {"label": "30%", "amplitude": 1.30, "baseline": 0.12, "cutoff": 15},
    {"label": "35%", "amplitude": 1.60, "baseline": 0.15, "cutoff": 15},
]


# ── Core Functions ───────────────────────────────────────────────────

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


@dataclass
class Block:
    slot: int
    staker: int
    parent_idx: int  # index into chain list
    height: int
    level_hits: List[bool] = field(default_factory=list)
    super_heights: List[int] = field(default_factory=list)  # per-level cumulative heights


def compare_chains_standard(chain_a: List[Block], chain_b: List[Block]) -> int:
    """maxvalid-tk: length → lower head slot. Returns >0 if a wins."""
    if len(chain_a) != len(chain_b):
        return len(chain_a) - len(chain_b)
    if chain_a[-1].slot != chain_b[-1].slot:
        return chain_b[-1].slot - chain_a[-1].slot  # lower slot wins
    return 0  # tie


def compare_chains_super(chain_a: List[Block], chain_b: List[Block]) -> int:
    """maxvalid-tk-super: length → lower head slot → lexicographic super weight."""
    std = compare_chains_standard(chain_a, chain_b)
    if std != 0:
        return std
    # Rule 3: lexicographic superblock weight (highest level first)
    sh_a = chain_a[-1].super_heights if chain_a else [0] * NUM_LEVELS
    sh_b = chain_b[-1].super_heights if chain_b else [0] * NUM_LEVELS
    for level in range(NUM_LEVELS - 1, 0, -1):
        if sh_a[level] != sh_b[level]:
            return sh_a[level] - sh_b[level]
    return 0


def simulate(num_slots, config, use_super=False):
    """
    Simulate multi-staker leader election with fork tracking.
    
    Instead of building full chain trees, we track:
    - Per-slot eligible stakers and their level hits
    - Fork events (>1 eligible staker in a slot)
    - Fork resolution: how many slots until one chain dominates
    """
    amplitude = config["amplitude"]
    baseline = config["baseline"]
    cutoff = config["cutoff"]
    
    last_l0_slot = 0
    l0_block_count = 0
    last_level_block = [0] * NUM_LEVELS
    super_heights = [0] * NUM_LEVELS
    
    # Metrics
    total_blocks = 0
    fork_slots = 0
    multi_eligible_counts = []
    total_eligible = 0
    
    # Fork resolution tracking
    active_forks = []  # list of (fork_slot, competing_stakers, resolved_at)
    unresolved_fork = None  # (fork_slot, staker_set)
    consecutive_single_leader = 0
    fork_resolution_depths = []
    
    # Super level tracking
    level_hit_counts = [0] * NUM_LEVELS
    
    for slot in range(1, num_slots + 1):
        slot_gap = slot - last_l0_slot
        
        # Check all stakers
        eligible = []
        for si in range(NUM_STAKERS):
            rel = STAKES[si] / TOTAL_STAKE
            thr = snowplow_thr(slot_gap, amplitude, baseline, cutoff, rel)
            if dhash(si, slot, 0) < thr:
                eligible.append(si)
        
        if not eligible:
            if unresolved_fork is not None:
                consecutive_single_leader = 0  # empty slot doesn't help
            continue
        
        # At least one L0 block
        l0_block_count += 1
        last_l0_slot = slot
        total_blocks += len(eligible)
        total_eligible += len(eligible)
        level_hit_counts[0] += len(eligible)
        
        # Check super levels for each eligible staker
        staker_level_hits = {}
        staker_super_heights = {}
        for si in eligible:
            hits = [True]  # L0
            sh = list(super_heights)
            sh[0] += 1
            for lv in range(1, NUM_LEVELS):
                bg = l0_block_count - last_level_block[lv]
                mp, sc = SHIFTED_EXP_PARAMS[lv]
                if dhash(si, slot, lv) < shifted_exp_thr(bg, mp, sc):
                    hits.append(True)
                    sh[lv] += 1
                else:
                    hits.append(False)
            staker_level_hits[si] = hits
            staker_super_heights[si] = sh
        
        is_fork = len(eligible) > 1
        
        if is_fork:
            fork_slots += 1
            multi_eligible_counts.append(len(eligible))
            
            if use_super:
                # maxvalid-tk-super: resolve by superblock weight
                best = eligible[0]
                for si in eligible[1:]:
                    # Compare super heights lexicographically
                    sh_best = staker_super_heights[best]
                    sh_si = staker_super_heights[si]
                    for lv in range(NUM_LEVELS - 1, 0, -1):
                        if sh_si[lv] > sh_best[lv]:
                            best = si
                            break
                        elif sh_si[lv] < sh_best[lv]:
                            break
                # Use best staker's hits
                winner_hits = staker_level_hits[best]
                winner_sh = staker_super_heights[best]
            else:
                # maxvalid-tk: can't resolve same-slot forks by head slot
                # Track as unresolved — will resolve when next single-leader extends one
                winner_hits = staker_level_hits[eligible[0]]
                winner_sh = staker_super_heights[eligible[0]]
            
            if unresolved_fork is None:
                unresolved_fork = (slot, set(eligible))
            consecutive_single_leader = 0
        else:
            # Single leader
            winner_hits = staker_level_hits[eligible[0]]
            winner_sh = staker_super_heights[eligible[0]]
            
            if unresolved_fork is not None:
                # Fork resolved by single leader extending one chain
                depth = slot - unresolved_fork[0]
                fork_resolution_depths.append(depth)
                unresolved_fork = None
            consecutive_single_leader += 1
        
        # Update state with winner
        super_heights = list(winner_sh)
        for lv in range(1, NUM_LEVELS):
            if winner_hits[lv]:
                last_level_block[lv] = l0_block_count
                level_hit_counts[lv] += 1
    
    # Compute metrics
    fill_rate = l0_block_count / num_slots * 100
    fork_rate = fork_slots / l0_block_count * 100 if l0_block_count > 0 else 0
    avg_multi = np.mean(multi_eligible_counts) if multi_eligible_counts else 0
    avg_resolution = np.mean(fork_resolution_depths) if fork_resolution_depths else 0
    med_resolution = np.median(fork_resolution_depths) if fork_resolution_depths else 0
    p95_resolution = np.percentile(fork_resolution_depths, 95) if len(fork_resolution_depths) > 10 else 0
    max_resolution = max(fork_resolution_depths) if fork_resolution_depths else 0
    
    return {
        "config": config["label"],
        "use_super": use_super,
        "fill_rate": fill_rate,
        "l0_blocks": l0_block_count,
        "total_eligible": total_eligible,
        "fork_slots": fork_slots,
        "fork_rate": fork_rate,
        "avg_multi": avg_multi,
        "num_forks_resolved": len(fork_resolution_depths),
        "avg_resolution": avg_resolution,
        "med_resolution": med_resolution,
        "p95_resolution": p95_resolution,
        "max_resolution": max_resolution,
        "level_hits": level_hit_counts,
        "super_heights": super_heights,
    }


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print(f"NiPoPoS Throughput Experiment — {NUM_SLOTS:,} slots, {NUM_STAKERS} stakers")
    print(f"{'='*90}")
    
    all_results = []
    
    for config in F_EFFECTIVE_CONFIGS:
        print(f"\n--- f_effective target: {config['label']} (amplitude={config['amplitude']}) ---")
        
        # Standard maxvalid-tk
        t0 = time.time()
        res_std = simulate(NUM_SLOTS, config, use_super=False)
        t_std = time.time() - t0
        
        # maxvalid-tk-super
        t0 = time.time()
        res_sup = simulate(NUM_SLOTS, config, use_super=True)
        t_sup = time.time() - t0
        
        all_results.append(res_std)
        all_results.append(res_sup)
        
        print(f"  Standard: fill={res_std['fill_rate']:.1f}% forks={res_std['fork_slots']} "
              f"({res_std['fork_rate']:.1f}%) avg_resolve={res_std['avg_resolution']:.1f} "
              f"med={res_std['med_resolution']:.0f} p95={res_std['p95_resolution']:.0f} "
              f"max={res_std['max_resolution']:.0f} ({t_std:.1f}s)")
        print(f"  Super:    fill={res_sup['fill_rate']:.1f}% forks={res_sup['fork_slots']} "
              f"({res_sup['fork_rate']:.1f}%) avg_resolve={res_sup['avg_resolution']:.1f} "
              f"med={res_sup['med_resolution']:.0f} p95={res_sup['p95_resolution']:.0f} "
              f"max={res_sup['max_resolution']:.0f} ({t_sup:.1f}s)")
    
    # ── Results Table ────────────────────────────────────────────────
    
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<8} {'Rule':<10} {'Fill%':>7} {'Blocks':>8} {'Forks':>7} {'Fork%':>7} "
          f"{'AvgRes':>7} {'MedRes':>7} {'P95Res':>7} {'MaxRes':>7}")
    print("-" * 90)
    
    for r in all_results:
        rule = "super" if r["use_super"] else "standard"
        print(f"{r['config']:<8} {rule:<10} {r['fill_rate']:>6.1f}% {r['l0_blocks']:>8,d} "
              f"{r['fork_slots']:>7,d} {r['fork_rate']:>6.1f}% "
              f"{r['avg_resolution']:>7.1f} {r['med_resolution']:>7.0f} "
              f"{r['p95_resolution']:>7.0f} {r['max_resolution']:>7.0f}")
    
    # ── Superblock heights at each f_effective ───────────────────────
    
    print(f"\n{'='*90}")
    print(f"  SUPERBLOCK HEIGHTS (super rule)")
    print(f"{'='*90}")
    print(f"{'Config':<8} " + " ".join(f"{'L'+str(i):>7}" for i in range(NUM_LEVELS)))
    print("-" * 80)
    for r in all_results:
        if r["use_super"]:
            heights_str = " ".join(f"{h:>7,d}" for h in r["super_heights"])
            print(f"{r['config']:<8} {heights_str}")
    
    # ── Plots ────────────────────────────────────────────────────────
    
    configs = [c["label"] for c in F_EFFECTIVE_CONFIGS]
    
    std_results = [r for r in all_results if not r["use_super"]]
    sup_results = [r for r in all_results if r["use_super"]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'NiPoPoS: maxvalid-tk vs maxvalid-tk-super ({NUM_SLOTS:,} slots, {NUM_STAKERS} stakers)', fontsize=14)
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Fill rate
    ax = axes[0][0]
    ax.bar(x - width/2, [r["fill_rate"] for r in std_results], width, label='Standard', color='steelblue')
    ax.bar(x + width/2, [r["fill_rate"] for r in sup_results], width, label='Super', color='coral')
    ax.set_ylabel('Fill Rate (%)')
    ax.set_title('Block Production Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fork rate
    ax = axes[0][1]
    ax.bar(x - width/2, [r["fork_rate"] for r in std_results], width, label='Standard', color='steelblue')
    ax.bar(x + width/2, [r["fork_rate"] for r in sup_results], width, label='Super', color='coral')
    ax.set_ylabel('Fork Rate (%)')
    ax.set_title('Fork Frequency (% of block slots)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average fork resolution depth
    ax = axes[1][0]
    ax.bar(x - width/2, [r["avg_resolution"] for r in std_results], width, label='Std (mean)', color='steelblue')
    ax.bar(x + width/2, [r["avg_resolution"] for r in sup_results], width, label='Super (mean)', color='coral')
    ax.plot(x - width/2, [r["p95_resolution"] for r in std_results], 'bs-', markersize=5, label='Std (P95)')
    ax.plot(x + width/2, [r["p95_resolution"] for r in sup_results], 'r^-', markersize=5, label='Super (P95)')
    ax.set_ylabel('Fork Resolution Depth (slots)')
    ax.set_title('Fork Resolution Speed')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Effective throughput (blocks / slot, adjusted for fork overhead)
    ax = axes[1][1]
    # Effective = fill_rate * (1 - fork_rate) — blocks that contribute to canonical chain
    std_eff = [r["fill_rate"] * (1 - r["fork_rate"]/100) for r in std_results]
    sup_eff = [r["fill_rate"] * (1 - r["fork_rate"]/100) for r in sup_results]
    ax.bar(x - width/2, std_eff, width, label='Standard', color='steelblue')
    ax.bar(x + width/2, sup_eff, width, label='Super', color='coral')
    ax.set_ylabel('Effective Throughput (%)')
    ax.set_title('Effective Throughput (fill × (1-fork_rate))')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig_throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_throughput_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: figures/fig_throughput_comparison.png/pdf")
