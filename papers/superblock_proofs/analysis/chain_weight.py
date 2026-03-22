#!/usr/bin/env python3
"""
NiPoPoS Cumulative Chain Weight Model.

Implement and compare chain selection strategies:
  A) LENGTH: standard maxvalid-tk (longest chain, lower head slot)
  B) WEIGHT: cumulative weight using 2^μ per super-level hit
  C) WEIGHT_STEEP: cumulative weight using 3^μ (faster convergence)

Weight per block = Σ 2^μ (or c^μ) for each level μ hit.

Expected weight per honest block (2^μ):
  E[w] = 1 + 0.5×2 + 0.25×4 + ... = 10 (one per level)

Run as a single-staker tine explorer with capped active tines.
Compare chain divergence rate under each weighting scheme.
"""
import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

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

def snowplow_thr(gap, amplitude=0.5, baseline=0.05, cutoff=15):
    if gap <= 0: return 0.0
    diff = amplitude * gap / cutoff if gap < cutoff else baseline
    diff = min(diff, 1.0)
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** 1.0  # 100% stake

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return min(1.0, mp * (1.0 - np.exp(-(gap - PSI) / sc)))

def dhash(slot, level, salt=0):
    h = hashlib.sha256(f"staker{salt}:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


@dataclass
class Tine:
    id: int
    parent_id: int
    slot: int
    height: int
    cum_weight: float
    super_heights: List[int]
    last_level_hit: List[int]
    last_l0_slot: int

    def copy_state(self):
        return list(self.super_heights), list(self.last_level_hit)


def compute_block_weight(level_hits: List[bool], scheme: str = "exp2") -> float:
    """Compute weight for a single block given its level hits."""
    w = 0.0
    for lv, hit in enumerate(level_hits):
        if hit:
            if scheme == "exp2":
                w += 2.0 ** lv       # 1, 2, 4, 8, ..., 512
            elif scheme == "exp3":
                w += 3.0 ** lv       # 1, 3, 9, 27, ..., 19683
            elif scheme == "linear":
                w += (lv + 1)        # 1, 2, 3, ..., 10
            elif scheme == "flat":
                w += 1.0             # All blocks worth 1 (= length)
    return w


def run_capped_explorer(
    num_slots: int,
    max_active: int,
    amplitude: float,
    baseline: float,
    cutoff: int,
    weight_scheme: str,
    prune_depth: int = 30,
):
    """
    Single-staker tine explorer with capped active tines.
    
    At each eligible slot, each active tine spawns a child.
    Parents stay active until depth > prune_depth behind best.
    If active count exceeds max_active, prune lowest-weight tines.
    """
    tine_id = 0
    genesis = Tine(
        id=tine_id, parent_id=-1, slot=0, height=0, cum_weight=0.0,
        super_heights=[0]*NUM_LEVELS, last_level_hit=[0]*NUM_LEVELS,
        last_l0_slot=0,
    )
    tine_id += 1
    active: Dict[int, Tine] = {genesis.id: genesis}

    # Track weight divergence between top-2 tines over time
    weight_gaps = []  # (slot, weight_gap_between_top2, height_gap_between_top2)
    active_counts = []
    best_height_history = []

    for slot in range(1, num_slots + 1):
        # Check eligibility for this slot (single staker, 100% stake)
        # The VRF output is the same for all tines at this slot
        is_eligible = dhash(slot, 0) < snowplow_thr(slot - 0, amplitude, baseline, cutoff)

        # But each tine has a different slot gap...
        # Actually: eligibility depends on slot gap FROM THAT TINE'S TIP.
        # Different tines have different last_l0_slot → different gaps → different thresholds.
        # This is where the tree branches!

        new_tines = []
        for tine in list(active.values()):
            slot_gap = slot - tine.last_l0_slot
            thr = snowplow_thr(slot_gap, amplitude, baseline, cutoff)

            if dhash(slot, 0) < thr:
                # Eligible from this tine — spawn child
                new_height = tine.height + 1
                new_super, new_last_hit = tine.copy_state()
                new_super[0] = new_height

                level_hits = [True]  # L0
                for lv in range(1, NUM_LEVELS):
                    bg = new_height - tine.last_level_hit[lv]
                    mp, sc = SHIFTED_EXP_PARAMS[lv]
                    if dhash(slot, lv) < shifted_exp_thr(bg, mp, sc):
                        level_hits.append(True)
                        new_super[lv] += 1
                        new_last_hit[lv] = new_height
                    else:
                        level_hits.append(False)

                bw = compute_block_weight(level_hits, weight_scheme)

                child = Tine(
                    id=tine_id, parent_id=tine.id, slot=slot,
                    height=new_height, cum_weight=tine.cum_weight + bw,
                    super_heights=new_super, last_level_hit=new_last_hit,
                    last_l0_slot=slot,
                )
                tine_id += 1
                new_tines.append(child)

        for t in new_tines:
            active[t.id] = t

        if not active:
            continue

        # Find best tine by weight
        best_id = max(active.keys(), key=lambda tid: (active[tid].cum_weight, active[tid].height, -active[tid].slot))
        best = active[best_id]

        # Prune: depth behind best
        to_prune = [tid for tid, t in active.items()
                    if best.height - t.height > prune_depth and tid != best_id]
        for tid in to_prune:
            del active[tid]

        # Cap active tines by weight (keep top max_active)
        if len(active) > max_active:
            sorted_tines = sorted(active.keys(),
                                  key=lambda tid: (active[tid].cum_weight, active[tid].height),
                                  reverse=True)
            for tid in sorted_tines[max_active:]:
                if tid != best_id:
                    del active[tid]

        # Track divergence
        if len(active) >= 2:
            sorted_by_weight = sorted(active.values(), key=lambda t: t.cum_weight, reverse=True)
            top1, top2 = sorted_by_weight[0], sorted_by_weight[1]
            weight_gaps.append((slot, top1.cum_weight - top2.cum_weight, top1.height - top2.height))

        active_counts.append(len(active))
        best_height_history.append(best.height)

        if slot % 2000 == 0:
            print(f"  Slot {slot:>6d}: active={len(active):>5d} best_h={best.height:>5d} "
                  f"best_w={best.cum_weight:>10.0f} tines_total={tine_id:,}")

    return {
        "weight_scheme": weight_scheme,
        "weight_gaps": weight_gaps,
        "active_counts": active_counts,
        "best_height": best.height if active else 0,
        "best_weight": best.cum_weight if active else 0,
        "final_active": len(active),
        "total_tines": tine_id,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    NUM_SLOTS = 10_000
    MAX_ACTIVE = 500
    PRUNE_DEPTH = 30

    schemes = [
        ("flat",    "Length Only (w=1 per block)"),
        ("exp2",    "Exponential 2^μ"),
        ("exp3",    "Exponential 3^μ"),
    ]

    amplitude, baseline, cutoff = 0.50, 0.05, 15  # Standard ~15%

    print(f"NiPoPoS Chain Weight Comparison — {NUM_SLOTS:,} slots, max_active={MAX_ACTIVE}")
    print(f"{'='*70}")

    all_results = []
    for scheme, label in schemes:
        print(f"\n--- {label} ---")
        t0 = time.time()
        r = run_capped_explorer(NUM_SLOTS, MAX_ACTIVE, amplitude, baseline, cutoff, scheme, PRUNE_DEPTH)
        r["label"] = label
        r["elapsed"] = time.time() - t0
        all_results.append(r)

        wg = r["weight_gaps"]
        if wg:
            w_gaps = [g for _, g, _ in wg]
            h_gaps = [g for _, _, g in wg]
            print(f"  Weight gap (top2): mean={np.mean(w_gaps):.1f} p50={np.median(w_gaps):.0f} "
                  f"p95={np.percentile(w_gaps, 95):.0f}")
            print(f"  Height gap (top2): mean={np.mean(h_gaps):.1f} p50={np.median(h_gaps):.0f} "
                  f"p95={np.percentile(h_gaps, 95):.0f}")
        print(f"  Best: height={r['best_height']:,} weight={r['best_weight']:,.0f} "
              f"active={r['final_active']} total_tines={r['total_tines']:,} ({r['elapsed']:.1f}s)")

    # Plot: weight gap divergence over time for each scheme
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Weight Gap Between Top-2 Tines ({NUM_SLOTS:,} slots)', fontsize=14)

    for idx, (r, (scheme, label)) in enumerate(zip(all_results, schemes)):
        ax = axes[idx]
        wg = r["weight_gaps"]
        if wg:
            slots = [s for s, _, _ in wg]
            gaps = [g for _, g, _ in wg]
            # Rolling average
            window = max(1, len(gaps) // 100)
            rolling = np.convolve(gaps, np.ones(window)/window, mode='valid')
            ax.plot(slots[:len(rolling)], rolling, linewidth=1.5)
            ax.set_xlabel('Slot')
            ax.set_ylabel('Weight Gap (top1 - top2)')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_weight_divergence.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_weight_divergence.pdf', bbox_inches='tight')
    plt.close()

    # Plot 2: height gap vs weight gap scatter
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['steelblue', 'coral', 'green']
    for idx, (r, (scheme, label)) in enumerate(zip(all_results, schemes)):
        wg = r["weight_gaps"]
        if wg:
            h_gaps = [g for _, _, g in wg]
            w_gaps = [g for _, g, _ in wg]
            ax.scatter(h_gaps, w_gaps, alpha=0.1, s=5, color=colors[idx], label=label)

    ax.set_xlabel('Height Gap (top1 - top2)')
    ax.set_ylabel('Weight Gap (top1 - top2)')
    ax.set_title('Height vs Weight Gap Between Competing Tines')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('figures/fig_height_vs_weight.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_height_vs_weight.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nSaved: figures/fig_weight_divergence.png/pdf, fig_height_vs_weight.png/pdf")
