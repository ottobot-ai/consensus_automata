#!/usr/bin/env python3
"""
Retune shifted exponential parameters using 1M slot simulation feedback.

The v4 params showed:
  L1: 32.7% cond (target 50%) — way too sparse
  L2: 24.4% (target 25%) — close
  L3-L9: 10-50% too generous (ratios ~0.55-0.75 instead of 0.500)

Strategy: Use binary search on max_prob for each level, validating with
100k slot mini-simulations, then verify the full set at 1M slots.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass

np.random.seed(42)

PSI = 1  # Dormant period

# L0 params
L0_PSI, L0_GAMMA, L0_FA, L0_FB = 0, 15, 0.5, 0.05

def snowplow_threshold(gap, psi, gamma, fA, fB, rel_stake=1.0):
    if gap < psi: return 0.0
    diff = fA * (gap - psi) / (gamma - psi) if gap < gamma else fB
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel_stake

def shifted_exp_threshold(gap, max_prob, scale):
    if gap < PSI: return 0.0
    return max_prob * (1.0 - np.exp(-(gap - PSI) / scale))

def domain_hash(seed, slot, level):
    data = f"{seed}:{slot}:TEST-{level}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)

def simulate_level(num_slots, level, max_prob, scale, stakes=[3000,2500,2000,1500,1000]):
    """Simulate just one level to measure its conditional hit rate."""
    total_stake = sum(stakes)
    l0_count = 0
    level_count = 0
    last_level_block = 0
    last_l0_slot = 0
    
    for slot in range(1, num_slots + 1):
        slot_gap = slot - last_l0_slot
        
        # L0 check (pick first eligible staker)
        l0_hit = False
        staker = 0
        for s_idx in range(len(stakes)):
            rel = stakes[s_idx] / total_stake
            thr = snowplow_threshold(slot_gap, L0_PSI, L0_GAMMA, L0_FA, L0_FB, rel)
            if domain_hash(s_idx, slot, 0) < thr:
                l0_hit = True
                staker = s_idx
                break
        
        if not l0_hit:
            continue
        
        l0_count += 1
        last_l0_slot = slot
        
        # Level check
        block_gap = l0_count - last_level_block
        thr = shifted_exp_threshold(block_gap, max_prob, scale)
        if domain_hash(staker, slot, level) < thr:
            level_count += 1
            last_level_block = l0_count
    
    cond_rate = level_count / l0_count if l0_count > 0 else 0
    return cond_rate, l0_count, level_count

def find_params(level, target_cond, scale_guess, num_slots=200000):
    """Binary search on max_prob to hit target conditional rate."""
    
    # First find a good scale by trying a few
    best_err = float('inf')
    best_params = None
    
    for scale in [scale_guess * 0.5, scale_guess * 0.75, scale_guess, scale_guess * 1.5, scale_guess * 2.0]:
        lo, hi = 0.001, 0.999
        for _ in range(20):  # Binary search iterations
            mid = (lo + hi) / 2
            cond, l0, lev = simulate_level(num_slots, level, mid, scale)
            if cond < target_cond:
                lo = mid
            else:
                hi = mid
        
        # Final measurement
        mp = (lo + hi) / 2
        cond, l0, lev = simulate_level(num_slots, level, mp, scale)
        err = abs(cond - target_cond)
        
        if err < best_err:
            best_err = err
            best_params = (mp, scale, cond)
            print(f"  L{level} scale={scale:.2f} max_prob={mp:.4f} → cond={cond:.3f} (target={target_cond:.3f}) err={err:.4f}")
    
    return best_params

def full_simulation(num_slots, params_dict):
    """Run full 10-level simulation with given params."""
    stakes = [3000, 2500, 2000, 1500, 1000]
    total_stake = sum(stakes)
    
    l0_count = 0
    level_hits = [0] * 10
    last_level_block = [0] * 10
    last_l0_slot = 0
    
    t0 = time.time()
    for slot in range(1, num_slots + 1):
        if slot % 200000 == 0:
            print(f"  {slot:>8d} / {num_slots} ({time.time()-t0:.1f}s)")
        
        slot_gap = slot - last_l0_slot
        
        l0_hit = False
        staker = 0
        for s_idx in range(len(stakes)):
            rel = stakes[s_idx] / total_stake
            thr = snowplow_threshold(slot_gap, L0_PSI, L0_GAMMA, L0_FA, L0_FB, rel)
            if domain_hash(s_idx, slot, 0) < thr:
                l0_hit = True
                staker = s_idx
                break
        
        if not l0_hit:
            continue
        
        l0_count += 1
        last_l0_slot = slot
        level_hits[0] += 1
        
        for level in range(1, 10):
            block_gap = l0_count - last_level_block[level]
            mp, sc = params_dict[level]
            thr = shifted_exp_threshold(block_gap, mp, sc)
            if domain_hash(staker, slot, level) < thr:
                level_hits[level] += 1
                last_level_block[level] = l0_count
    
    return l0_count, level_hits


if __name__ == "__main__":
    targets = {i: 1.0 / (2**i) for i in range(1, 10)}
    
    # Scale guesses — roughly proportional to target gap
    scale_guesses = {1: 1.5, 2: 3, 3: 5, 4: 8, 5: 15, 6: 25, 7: 45, 8: 80, 9: 150}
    
    print("=== Phase 1: Per-Level Parameter Search (200k slots each) ===\n")
    tuned = {}
    for level in range(1, 10):
        print(f"Tuning L{level} (target cond={targets[level]:.4f})...")
        mp, sc, cond = find_params(level, targets[level], scale_guesses[level])
        tuned[level] = (mp, sc)
        print(f"  → BEST: max_prob={mp:.4f}, scale={sc:.2f}, cond={cond:.4f}\n")
    
    print("\n=== Phase 2: Full 10-Level Validation (1M slots) ===\n")
    print("Tuned parameters:")
    for level in range(1, 10):
        mp, sc = tuned[level]
        print(f"  L{level}: max_prob={mp:.4f}, scale={sc:.2f}")
    
    print("\nRunning 1M slot simulation...")
    l0_count, level_hits = full_simulation(1_000_000, tuned)
    
    print(f"\n{'='*70}")
    print(f"  RESULTS — 1,000,000 slots, {l0_count:,} base blocks")
    print(f"{'='*70}")
    print(f"{'Level':<6} {'Hits':>8} {'Cond%':>8} {'Target%':>8} {'Error%':>8} {'Ratio':>8}")
    print("-" * 50)
    
    for level in range(10):
        hits = level_hits[level]
        cond = hits / l0_count * 100
        target = 100.0 / (2**level)
        err = abs(cond - target) / target * 100
        ratio = hits / level_hits[level-1] if level > 0 and level_hits[level-1] > 0 else 1.0
        print(f"L{level:<5d} {hits:>8,d} {cond:>7.1f}% {target:>7.1f}% {err:>7.1f}% {ratio:>7.3f}")
    
    print(f"\n{'='*70}")
    print("\nScala-ready parameters:")
    print("val ShiftedExpParams: Vector[(Double, Double)] = Vector(")
    print(f"  (1.0, 0.0),     // L0: LDD snowplow (not used)")
    for level in range(1, 10):
        mp, sc = tuned[level]
        tgt = 1.0 / (2**level)
        print(f"  ({mp:.6f}, {sc:.6f}),  // L{level}: target {tgt:.6f}")
    print(")")
