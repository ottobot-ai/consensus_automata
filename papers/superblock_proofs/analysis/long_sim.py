#!/usr/bin/env python3
"""
Long simulation (1M slots) to validate per-level LDD curves at all 10 levels.
Uses the optimized shifted exponential parameters from curve_optimization_v4.py.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict

np.random.seed(42)

NUM_LEVELS = 10
NUM_SLOTS = 1_000_000

# L0 snowplow parameters (slot-based gap)
L0_PSI = 0
L0_GAMMA = 15
L0_FA = 0.5
L0_FB = 0.05

# Shifted exponential configs from optimization (psi=1 for all)
# Format: (max_prob, scale)
# These were derived analytically + tuned by simulation
SHIFTED_EXP_PARAMS = {
    1: (0.7615, 1.544),   # L1: target 50% of base blocks
    2: (0.5094, 2.088),   # L2: target 25%
    3: (0.2989, 2.754),   # L3: target 12.5%
    4: (0.1621, 3.612),   # L4: target 6.25%
    5: (0.0841, 4.734),   # L5: target 3.125%
    6: (0.0428, 6.213),   # L6: target 1.5625%
    7: (0.0217, 8.145),   # L7: target 0.78125%
    8: (0.0109, 10.689),  # L8: target 0.390625%
    9: (0.0055, 14.014),  # L9: target 0.1953125%
}

PSI = 1  # Dormant period for all super levels


def snowplow_threshold(gap: int, psi: int, gamma: int, fA: float, fB: float, rel_stake: float = 1.0) -> float:
    if gap < psi:
        return 0.0
    elif gap < gamma:
        difficulty = fA * (gap - psi) / (gamma - psi)
    else:
        difficulty = fB
    
    if difficulty <= 0:
        return 0.0
    return 1.0 - (1.0 - difficulty) ** rel_stake


def shifted_exp_threshold(gap: int, max_prob: float, scale: float) -> float:
    if gap < PSI:
        return 0.0
    shifted = gap - PSI
    return max_prob * (1.0 - np.exp(-shifted / scale))


def domain_hash(seed: int, slot: int, level: int) -> float:
    """Deterministic pseudo-random test value for (seed, slot, level)."""
    data = f"{seed}:{slot}:TEST-{level}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


def run_simulation(num_slots: int, num_stakers: int = 5):
    """Run the full multi-level simulation."""
    
    # Stake distribution (same as Scala sim)
    stakes = [3000, 2500, 2000, 1500, 1000]
    total_stake = sum(stakes)
    staker_seeds = list(range(num_stakers))  # Deterministic seeds
    
    # Per-level tracking
    l0_block_count = 0
    level_hit_counts = [0] * NUM_LEVELS
    last_level_hit_block = [0] * NUM_LEVELS  # In L0 block units
    last_l0_slot = 0
    
    # For gap distribution analysis
    level_gap_at_hit = [[] for _ in range(NUM_LEVELS)]
    
    # Slot-by-slot
    t0 = time.time()
    for slot in range(1, num_slots + 1):
        if slot % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  Slot {slot:>8d} / {num_slots} ({elapsed:.1f}s) — L0={l0_block_count} heights={[level_hit_counts[i] for i in range(NUM_LEVELS)]}")
        
        slot_gap = slot - last_l0_slot
        
        # Check L0 for each staker
        l0_eligible = []
        for s_idx in range(num_stakers):
            rel_stake = stakes[s_idx] / total_stake
            threshold = snowplow_threshold(slot_gap, L0_PSI, L0_GAMMA, L0_FA, L0_FB, rel_stake)
            test_val = domain_hash(staker_seeds[s_idx], slot, 0)
            if test_val < threshold:
                l0_eligible.append(s_idx)
        
        if not l0_eligible:
            continue
        
        # L0 hit — update
        l0_block_count += 1
        last_l0_slot = slot
        level_hit_counts[0] += 1
        level_gap_at_hit[0].append(slot_gap)
        
        # Pick first eligible staker for super-level tests
        staker = l0_eligible[0]
        
        # Test L1-L9 (gated behind L0)
        for level in range(1, NUM_LEVELS):
            block_gap = l0_block_count - last_level_hit_block[level]
            max_prob, scale = SHIFTED_EXP_PARAMS[level]
            threshold = shifted_exp_threshold(block_gap, max_prob, scale)
            test_val = domain_hash(staker_seeds[staker], slot, level)
            
            if test_val < threshold:
                level_hit_counts[level] += 1
                last_level_hit_block[level] = l0_block_count
                level_gap_at_hit[level].append(block_gap)
    
    elapsed = time.time() - t0
    return {
        'num_slots': num_slots,
        'l0_blocks': l0_block_count,
        'level_hits': level_hit_counts,
        'level_gaps': level_gap_at_hit,
        'elapsed': elapsed,
    }


def print_results(results: Dict):
    num_slots = results['num_slots']
    l0_blocks = results['l0_blocks']
    level_hits = results['level_hits']
    level_gaps = results['level_gaps']
    
    print(f"\n{'='*80}")
    print(f"  TAKTIKOS SUPERBLOCK SIMULATION — {num_slots:,} slots")
    print(f"{'='*80}")
    print(f"L0 base blocks: {l0_blocks:,} ({l0_blocks/num_slots*100:.1f}% fill rate)")
    print(f"Elapsed: {results['elapsed']:.1f}s")
    print()
    
    print(f"{'Level':<8} {'Hits':>8} {'Rate':>8} {'Target':>8} {'Cond%':>8} {'TargCond':>8} {'Error':>8} {'AvgGap':>8} {'MedGap':>8} {'MaxGap':>8}")
    print("-" * 80)
    
    for level in range(NUM_LEVELS):
        hits = level_hits[level]
        rate = hits / num_slots * 100
        cond = hits / l0_blocks * 100 if l0_blocks > 0 else 0
        target_cond = 100.0 / (2 ** level)
        target_rate = l0_blocks / num_slots * 100 / (2 ** level)
        error = abs(cond - target_cond) / target_cond * 100 if target_cond > 0 else 0
        
        gaps = level_gaps[level]
        avg_gap = np.mean(gaps) if gaps else 0
        med_gap = np.median(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        
        print(f"L{level:<7d} {hits:>8,d} {rate:>7.2f}% {target_rate:>7.2f}% {cond:>7.1f}% {target_cond:>7.1f}% {error:>7.1f}% {avg_gap:>7.1f} {med_gap:>7.0f} {max_gap:>7d}")
    
    print()
    print("Height ratios (Lμ / L{μ-1}):")
    for level in range(1, NUM_LEVELS):
        if level_hits[level - 1] > 0:
            ratio = level_hits[level] / level_hits[level - 1]
            print(f"  L{level}/L{level-1} = {ratio:.3f} (target: 0.500)")
    
    print()
    print("Gap distributions for higher levels:")
    for level in [4, 5, 6, 7, 8, 9]:
        gaps = level_gaps[level]
        if len(gaps) >= 5:
            p25 = np.percentile(gaps, 25)
            p75 = np.percentile(gaps, 75)
            p95 = np.percentile(gaps, 95)
            print(f"  L{level}: n={len(gaps):>5d}  P25={p25:.0f}  P50={np.median(gaps):.0f}  P75={p75:.0f}  P95={p95:.0f}  Max={max(gaps)}")
        elif gaps:
            print(f"  L{level}: n={len(gaps):>5d}  gaps={gaps}")
        else:
            print(f"  L{level}: n=0 (no hits)")


if __name__ == "__main__":
    print("Running 1M slot simulation...")
    results = run_simulation(NUM_SLOTS)
    print_results(results)
