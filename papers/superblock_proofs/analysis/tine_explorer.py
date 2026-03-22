#!/usr/bin/env python3
"""
NiPoPoS Tine Explorer: Full fork-tree state space exploration.

Instead of honest-vs-adversary, explore ALL possible chain extensions:

Each slot:
  1. For each active tine tip, check all stakers for eligibility
  2. Each eligible (tip, staker) pair spawns a new tine
  3. New tine inherits parent's subchain state, updated for the new block
  4. Apply chain selection to find the "best" tine
  5. Prune tines whose tips are > k blocks behind best tip

Measure under both rules:
  - maxvalid-tk: prune by length → head slot
  - maxvalid-tk-super: prune by length → head slot → superblock weight

The key metric: depth of viable tines (how many slots back can an
alternative chain still compete?). Shallower = more secure = can
increase f_effective.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────

NUM_LEVELS = 10
PSI = 1
PRUNE_DEPTH = 50  # k: prune tines this far behind best

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


# ── Tine Data Structure ──────────────────────────────────────────────

@dataclass
class Tine:
    """A single chain extension (tip of a potential fork)."""
    id: int
    parent_id: int          # -1 for genesis
    slot: int               # Slot of this tine's tip block
    height: int             # Chain length (L0 blocks)
    staker: int             # Staker who produced the tip
    super_heights: List[int]  # Per-level cumulative heights
    last_level_hit: List[int]  # Per-level: L0 block# of last hit
    last_l0_slot: int       # Slot of this tine's last L0 block
    birth_slot: int         # When this tine was created (for age tracking)
    
    def score_standard(self) -> Tuple:
        """maxvalid-tk score: (height, -slot) — higher is better."""
        return (self.height, -self.slot)
    
    def score_super(self) -> Tuple:
        """maxvalid-tk-super score: (height, -slot, super_weight) — higher is better."""
        # Lexicographic super weight: highest level first
        super_w = tuple(self.super_heights[lv] for lv in range(NUM_LEVELS - 1, 0, -1))
        return (self.height, -self.slot, super_w)


# ── Simulation ───────────────────────────────────────────────────────

def run_tine_exploration(
    num_slots: int,
    stakes: List[int],
    amplitude: float,
    baseline: float,
    cutoff: int,
    prune_depth: int,
    use_super: bool,
):
    total_stake = sum(stakes)
    num_stakers = len(stakes)
    
    tine_counter = 0
    
    # Genesis tine
    genesis = Tine(
        id=tine_counter, parent_id=-1, slot=0, height=0, staker=-1,
        super_heights=[0]*NUM_LEVELS, last_level_hit=[0]*NUM_LEVELS,
        last_l0_slot=0, birth_slot=0
    )
    tine_counter += 1
    
    # Active tines: id → Tine
    active_tines: Dict[int, Tine] = {genesis.id: genesis}
    
    # Metrics
    max_active_tines = 0
    tines_created_total = 1
    tines_pruned_total = 0
    viable_depth_at_slot = []  # (slot, max_depth_of_viable_alternative)
    fork_count = 0
    
    best_tine_id = genesis.id
    
    for slot in range(1, num_slots + 1):
        new_tines = []
        
        # For each active tine tip, check all stakers for L0 eligibility
        for tine in list(active_tines.values()):
            slot_gap = slot - tine.last_l0_slot
            
            for si in range(num_stakers):
                rel = stakes[si] / total_stake
                thr = snowplow_thr(slot_gap, amplitude, baseline, cutoff, rel)
                
                if dhash(si, slot, 0) < thr:
                    # Eligible! Spawn new tine
                    new_height = tine.height + 1
                    new_super = list(tine.super_heights)
                    new_super[0] = new_height
                    new_last_hit = list(tine.last_level_hit)
                    
                    # Check super levels
                    for lv in range(1, NUM_LEVELS):
                        bg = new_height - tine.last_level_hit[lv]
                        mp, sc = SHIFTED_EXP_PARAMS[lv]
                        if dhash(si, slot, lv) < shifted_exp_thr(bg, mp, sc):
                            new_super[lv] += 1
                            new_last_hit[lv] = new_height
                    
                    new_tine = Tine(
                        id=tine_counter, parent_id=tine.id, slot=slot,
                        height=new_height, staker=si,
                        super_heights=new_super, last_level_hit=new_last_hit,
                        last_l0_slot=slot, birth_slot=slot
                    )
                    tine_counter += 1
                    new_tines.append(new_tine)
        
        # Add new tines
        for t in new_tines:
            active_tines[t.id] = t
        tines_created_total += len(new_tines)
        
        if len(new_tines) > 1:
            fork_count += 1
        
        # Find best tine
        if active_tines:
            if use_super:
                best_tine_id = max(active_tines.keys(), key=lambda tid: active_tines[tid].score_super())
            else:
                best_tine_id = max(active_tines.keys(), key=lambda tid: active_tines[tid].score_standard())
            
            best = active_tines[best_tine_id]
            
            # Prune tines that are too far behind
            to_prune = []
            for tid, tine in active_tines.items():
                if best.height - tine.height > prune_depth:
                    to_prune.append(tid)
                # Also prune old tines that haven't grown (parent tines superseded by children)
                # A tine is superseded if any of its children exist
                # Simple approach: prune tines older than prune_depth slots with no recent growth
                if slot - tine.slot > prune_depth * 7:  # ~7 slots per block
                    to_prune.append(tid)
            
            for tid in set(to_prune):
                if tid != best_tine_id:  # Never prune best
                    del active_tines[tid]
                    tines_pruned_total += 1
            
            # Compute viable fork depth: how far back is the deepest alternative?
            if len(active_tines) > 1:
                other_heights = [t.height for tid, t in active_tines.items() if tid != best_tine_id]
                if other_heights:
                    depth = best.height - min(other_heights)
                    viable_depth_at_slot.append((slot, depth, len(active_tines)))
        
        max_active_tines = max(max_active_tines, len(active_tines))
        
        if slot % 1000 == 0:
            best = active_tines.get(best_tine_id)
            n_active = len(active_tines)
            sh = best.super_heights if best else [0]*NUM_LEVELS
            print(f"  Slot {slot:>6d}: active={n_active:>4d} best_h={best.height if best else 0:>5d} "
                  f"super={sh[:6]} created={tines_created_total:,} pruned={tines_pruned_total:,}")
    
    # Compute depth statistics
    depths = [d for _, d, _ in viable_depth_at_slot]
    active_counts = [c for _, _, c in viable_depth_at_slot]
    
    return {
        "use_super": use_super,
        "total_slots": num_slots,
        "tines_created": tines_created_total,
        "tines_pruned": tines_pruned_total,
        "max_active": max_active_tines,
        "fork_slots": fork_count,
        "best_height": active_tines[best_tine_id].height if best_tine_id in active_tines else 0,
        "best_super": active_tines[best_tine_id].super_heights if best_tine_id in active_tines else [0]*NUM_LEVELS,
        "depth_mean": np.mean(depths) if depths else 0,
        "depth_median": np.median(depths) if depths else 0,
        "depth_p95": np.percentile(depths, 95) if len(depths) > 10 else 0,
        "depth_max": max(depths) if depths else 0,
        "active_mean": np.mean(active_counts) if active_counts else 0,
        "viable_depth_history": viable_depth_at_slot,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    NUM_SLOTS = 10_000  # Start small, tine space can explode
    stakes = [1000] * 5  # 5 equal stakers
    
    configs = [
        {"label": "15%", "amplitude": 0.50, "baseline": 0.05},
        {"label": "25%", "amplitude": 1.00, "baseline": 0.09},
        {"label": "35%", "amplitude": 1.60, "baseline": 0.15},
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"f_effective ~ {config['label']} (amplitude={config['amplitude']})")
        print(f"{'='*70}")
        
        for use_super in [False, True]:
            rule = "super" if use_super else "standard"
            print(f"\n  --- {rule} ---")
            t0 = time.time()
            r = run_tine_exploration(
                NUM_SLOTS, stakes,
                config["amplitude"], config["baseline"], 15,
                PRUNE_DEPTH, use_super
            )
            r["f_effective"] = config["label"]
            r["elapsed"] = time.time() - t0
            all_results.append(r)
            
            print(f"  Result: max_active={r['max_active']} fork_slots={r['fork_slots']} "
                  f"depth_mean={r['depth_mean']:.1f} depth_p95={r['depth_p95']:.0f} "
                  f"depth_max={r['depth_max']:.0f} ({r['elapsed']:.1f}s)")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"{'f_eff':<8} {'Rule':<10} {'Forks':>7} {'MaxAct':>8} {'DepthMean':>10} {'DepthP95':>9} {'DepthMax':>9}")
    print("-" * 65)
    for r in all_results:
        rule = "super" if r["use_super"] else "standard"
        print(f"{r['f_effective']:<8} {rule:<10} {r['fork_slots']:>7} {r['max_active']:>8} "
              f"{r['depth_mean']:>10.2f} {r['depth_p95']:>9.0f} {r['depth_max']:>9.0f}")
    
    # Plot: viable fork depth over time for each config
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Viable Fork Depth Over Time ({NUM_SLOTS:,} slots, k={PRUNE_DEPTH})', fontsize=14)
    
    for idx, config in enumerate(configs):
        ax = axes[idx]
        for r in all_results:
            if r["f_effective"] == config["label"]:
                rule = "super" if r["use_super"] else "standard"
                color = 'coral' if r["use_super"] else 'steelblue'
                history = r["viable_depth_history"]
                if history:
                    slots_h = [s for s, d, _ in history]
                    depths_h = [d for s, d, _ in history]
                    # Rolling average for readability
                    window = min(100, len(depths_h) // 5) if len(depths_h) > 5 else 1
                    if window > 1:
                        rolling = np.convolve(depths_h, np.ones(window)/window, mode='valid')
                        ax.plot(slots_h[:len(rolling)], rolling, color=color, alpha=0.8,
                                linewidth=1.5, label=f'{rule} (rolling avg)')
                    else:
                        ax.plot(slots_h, depths_h, color=color, alpha=0.5, linewidth=0.5, label=rule)
        
        ax.set_xlabel('Slot')
        ax.set_ylabel('Viable Fork Depth (blocks behind best)')
        ax.set_title(f'f_effective ~ {config["label"]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig_tine_depth.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_tine_depth.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_tine_depth.png/pdf")
