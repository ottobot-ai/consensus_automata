#!/usr/bin/env python3
"""
NiPoPoS Tine Explorer v2: Single-staker full decision tree.

One staker with 100% stake explores all possible chain extensions.
Each slot: if eligible, spawn a child tine. Parent stays active.
Parent pruned when depth > k behind best tip.

Compare:
  A) Heaviest by LENGTH (standard maxvalid-tk)
  B) Heaviest by CUMULATIVE WEIGHT (length + super-level bonuses)

The key metric: depth at which alternative tines become inviable.
If cumulative weight converges faster → smaller k needed.
"""
import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
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

# Weight bonus per super level hit. Level μ hit adds 2^μ to block weight.
# Base block (L0 only) = 1. L0+L1 = 1+2 = 3. L0+L1+L3 = 1+2+8 = 11.
LEVEL_WEIGHTS = [1 << lv for lv in range(NUM_LEVELS)]  # [1, 2, 4, 8, ..., 512]


def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=15, rel_stake=1.0):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel_stake

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dhash(slot, level):
    h = hashlib.sha256(f"staker0:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


@dataclass
class Tine:
    id: int
    parent_id: int
    slot: int               # Slot of tip block
    height: int             # Chain length
    cum_weight: float       # Cumulative weight (sum of block weights)
    super_heights: List[int]
    last_level_hit: List[int]  # L0 block# of last hit per level
    last_l0_slot: int

    def score_length(self) -> Tuple:
        return (self.height, -self.slot)

    def score_weight(self) -> Tuple:
        return (self.cum_weight, self.height, -self.slot)


def run_explorer(
    num_slots: int,
    prune_depth: int,
    amplitude: float = 0.5,
    baseline: float = 0.05,
    cutoff: int = 15,
):
    tine_id = 0
    genesis = Tine(
        id=tine_id, parent_id=-1, slot=0, height=0, cum_weight=0.0,
        super_heights=[0]*NUM_LEVELS, last_level_hit=[0]*NUM_LEVELS,
        last_l0_slot=0,
    )
    tine_id += 1

    active: Dict[int, Tine] = {genesis.id: genesis}

    # Metrics per slot
    metrics_length = []  # (slot, num_active, max_depth_behind_best)
    metrics_weight = []

    best_by_length = genesis.id
    best_by_weight = genesis.id

    total_created = 1
    total_pruned_length = 0
    total_pruned_weight = 0

    t0 = time.time()

    for slot in range(1, num_slots + 1):
        new_tines = []

        for tine in list(active.values()):
            slot_gap = slot - tine.last_l0_slot
            thr = snowplow_thr(slot_gap, amplitude, baseline, cutoff)

            if dhash(slot, 0) < thr:
                # Eligible — spawn child
                new_height = tine.height + 1
                new_super = list(tine.super_heights)
                new_super[0] = new_height
                new_last_hit = list(tine.last_level_hit)

                block_weight = LEVEL_WEIGHTS[0]  # Base weight = 1

                for lv in range(1, NUM_LEVELS):
                    bg = new_height - tine.last_level_hit[lv]
                    mp, sc = SHIFTED_EXP_PARAMS[lv]
                    if dhash(slot, lv) < shifted_exp_thr(bg, mp, sc):
                        new_super[lv] += 1
                        new_last_hit[lv] = new_height
                        block_weight += LEVEL_WEIGHTS[lv]

                child = Tine(
                    id=tine_id, parent_id=tine.id, slot=slot,
                    height=new_height, cum_weight=tine.cum_weight + block_weight,
                    super_heights=new_super, last_level_hit=new_last_hit,
                    last_l0_slot=slot,
                )
                tine_id += 1
                new_tines.append(child)

        for t in new_tines:
            active[t.id] = t
        total_created += len(new_tines)

        if not active:
            continue

        # Find best by each rule
        best_by_length = max(active.keys(), key=lambda tid: active[tid].score_length())
        best_by_weight = max(active.keys(), key=lambda tid: active[tid].score_weight())

        best_l = active[best_by_length]
        best_w = active[best_by_weight]

        # Prune by LENGTH rule: remove tines > prune_depth behind best height
        prune_l = [tid for tid, t in active.items()
                   if best_l.height - t.height > prune_depth and tid != best_by_length]
        for tid in prune_l:
            del active[tid]
            total_pruned_length += 1

        # Compute metrics BEFORE pruning by weight (we want to compare both on same tine set)
        # Actually, we need separate active sets. Let's track depth metrics differently.

        # Depth of deepest viable alternative (by length rule)
        if len(active) > 1:
            other_heights_l = [t.height for tid, t in active.items() if tid != best_by_length]
            max_depth_l = best_l.height - min(other_heights_l) if other_heights_l else 0
        else:
            max_depth_l = 0
        metrics_length.append((slot, len(active), max_depth_l))

        # For weight: how many tines would survive if we pruned by weight gap instead?
        # A tine is inviable if its cum_weight is more than weight_threshold behind best
        # Use equivalent: tines whose weight can't catch up even with prune_depth more blocks
        # Max weight per block ~ 1 + 2 + 4 + ... + 512 = 1023
        max_block_weight = sum(LEVEL_WEIGHTS)
        weight_gap_survivors = sum(
            1 for tid, t in active.items()
            if best_w.cum_weight - t.cum_weight <= prune_depth * max_block_weight
        )

        # Weight-based depth: find the tine closest in weight that differs in tip
        if len(active) > 1:
            other_weights = [t.cum_weight for tid, t in active.items() if tid != best_by_weight]
            if other_weights:
                weight_gap = best_w.cum_weight - max(other_weights)
                # Normalize to "equivalent blocks": weight_gap / avg_block_weight
                avg_bw = best_w.cum_weight / best_w.height if best_w.height > 0 else 1
                equiv_depth = weight_gap / avg_bw if avg_bw > 0 else 0
            else:
                weight_gap = 0
                equiv_depth = 0
        else:
            weight_gap = 0
            equiv_depth = 0
        metrics_weight.append((slot, weight_gap_survivors, equiv_depth))

        if slot % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Slot {slot:>5d}: active={len(active):>5d} best_h={best_l.height:>4d} "
                  f"best_w={best_w.cum_weight:>.0f} depth_l={max_depth_l:>2d} "
                  f"equiv_depth_w={equiv_depth:>.1f} ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Compute summary stats
    depths_l = [d for _, _, d in metrics_length if d > 0]
    equiv_depths_w = [d for _, _, d in metrics_weight if d > 0]

    return {
        "num_slots": num_slots,
        "prune_depth": prune_depth,
        "total_created": total_created,
        "total_pruned": total_pruned_length,
        "elapsed": elapsed,
        "best_height": active[best_by_length].height if best_by_length in active else 0,
        "best_weight": active[best_by_weight].cum_weight if best_by_weight in active else 0,
        # Length-based depth stats
        "depth_l_mean": np.mean(depths_l) if depths_l else 0,
        "depth_l_p50": np.median(depths_l) if depths_l else 0,
        "depth_l_p95": np.percentile(depths_l, 95) if len(depths_l) > 10 else 0,
        "depth_l_max": max(depths_l) if depths_l else 0,
        # Weight-based equivalent depth stats
        "depth_w_mean": np.mean(equiv_depths_w) if equiv_depths_w else 0,
        "depth_w_p50": np.median(equiv_depths_w) if equiv_depths_w else 0,
        "depth_w_p95": np.percentile(equiv_depths_w, 95) if len(equiv_depths_w) > 10 else 0,
        "depth_w_max": max(equiv_depths_w) if equiv_depths_w else 0,
        # Raw histories for plotting
        "metrics_length": metrics_length,
        "metrics_weight": metrics_weight,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configs = [
        {"label": "15%", "amplitude": 0.50, "baseline": 0.05},
        {"label": "25%", "amplitude": 1.00, "baseline": 0.09},
        {"label": "35%", "amplitude": 1.60, "baseline": 0.15},
    ]

    K = 20  # Prune depth
    NUM_SLOTS = 5000  # Start small to verify tractability

    print(f"NiPoPoS Tine Explorer v2 — Single staker, {NUM_SLOTS:,} slots, k={K}")
    print(f"Cumulative weight: base=1, L1=+2, L2=+4, ..., L9=+512")
    print(f"{'='*80}")

    all_results = []

    for config in configs:
        print(f"\n--- f_effective ~ {config['label']} ---")
        r = run_explorer(NUM_SLOTS, K, config["amplitude"], config["baseline"])
        r["f_effective"] = config["label"]
        all_results.append(r)

        print(f"\n  Summary:")
        print(f"    Best height: {r['best_height']:,}  Best weight: {r['best_weight']:,.0f}")
        print(f"    Tines created: {r['total_created']:,}  Pruned: {r['total_pruned']:,}")
        print(f"    Fork depth (LENGTH):  mean={r['depth_l_mean']:.1f} p50={r['depth_l_p50']:.0f} p95={r['depth_l_p95']:.0f} max={r['depth_l_max']:.0f}")
        print(f"    Fork depth (WEIGHT):  mean={r['depth_w_mean']:.1f} p50={r['depth_w_p50']:.0f} p95={r['depth_w_p95']:.0f} max={r['depth_w_max']:.0f}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'f_eff':<8} {'Blocks':>8} {'Weight':>10} {'DepthL_p95':>11} {'DepthW_p95':>11} {'Ratio':>7}")
    print("-" * 60)
    for r in all_results:
        ratio = r['depth_l_p95'] / r['depth_w_p95'] if r['depth_w_p95'] > 0 else float('inf')
        print(f"{r['f_effective']:<8} {r['best_height']:>8,} {r['best_weight']:>10,.0f} "
              f"{r['depth_l_p95']:>11.1f} {r['depth_w_p95']:>11.1f} {ratio:>7.2f}x")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Tine Viable Depth: Length vs Cumulative Weight ({NUM_SLOTS:,} slots, k={K})', fontsize=13)

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        # Length-based depth
        ml = r["metrics_length"]
        slots_l = [s for s, _, d in ml if d > 0]
        depths_l = [d for s, _, d in ml if d > 0]

        # Weight-based equiv depth
        mw = r["metrics_weight"]
        slots_w = [s for s, _, d in mw if d > 0]
        depths_w = [d for s, _, d in mw if d > 0]

        if depths_l:
            window = max(1, len(depths_l) // 50)
            roll_l = np.convolve(depths_l, np.ones(window)/window, mode='valid')
            ax.plot(slots_l[:len(roll_l)], roll_l, color='steelblue', alpha=0.8, label='Length depth')

        if depths_w:
            window = max(1, len(depths_w) // 50)
            roll_w = np.convolve(depths_w, np.ones(window)/window, mode='valid')
            ax.plot(slots_w[:len(roll_w)], roll_w, color='coral', alpha=0.8, label='Weight equiv depth')

        ax.set_xlabel('Slot')
        ax.set_ylabel('Viable Fork Depth')
        ax.set_title(f'f_effective ~ {r["f_effective"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_tine_depth_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_tine_depth_v2.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/fig_tine_depth_v2.png/pdf")
