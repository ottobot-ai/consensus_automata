#!/usr/bin/env python3
"""
Fast retuning: 50k slots per level search, then 1M validation.
Uses analytical expected rate to narrow the search first.
"""
import numpy as np
import hashlib
import time

np.random.seed(42)
PSI = 1
L0_PSI, L0_GAMMA, L0_FA, L0_FB = 0, 15, 0.5, 0.05

def snowplow_thr(gap, rel_stake=1.0):
    if gap < L0_PSI: return 0.0
    diff = L0_FA * gap / L0_GAMMA if gap < L0_GAMMA else L0_FB
    return 0.0 if diff <= 0 else 1.0 - (1.0 - diff) ** rel_stake

def shifted_exp_thr(gap, mp, sc):
    if gap < PSI: return 0.0
    return mp * (1.0 - np.exp(-(gap - PSI) / sc))

def dhash(seed, slot, level):
    h = hashlib.sha256(f"{seed}:{slot}:TEST-{level}".encode()).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)

def analytical_rate(mp, sc, max_gap=500):
    """Expected conditional hit rate via renewal theory."""
    surv = 1.0
    eg = 0.0
    tp = 0.0
    for g in range(PSI, max_gap):
        t = shifted_exp_thr(g, mp, sc)
        p = surv * t
        eg += g * p
        tp += p
        surv *= (1 - t)
        if surv < 1e-15: break
    return 1.0 / (eg / tp) if tp > 0 and eg > 0 else 0

def sim_level(num_slots, level, mp, sc):
    """Fast simulation of one level."""
    stakes = [3000, 2500, 2000, 1500, 1000]
    ts = 10000
    l0 = 0; lev = 0; llb = 0; lls = 0
    for slot in range(1, num_slots + 1):
        sg = slot - lls
        hit = False; st = 0
        for si in range(5):
            if dhash(si, slot, 0) < snowplow_thr(sg, stakes[si]/ts):
                hit = True; st = si; break
        if not hit: continue
        l0 += 1; lls = slot
        bg = l0 - llb
        if dhash(st, slot, level) < shifted_exp_thr(bg, mp, sc):
            lev += 1; llb = l0
    return lev / l0 if l0 > 0 else 0

print("=== Fast Retune: Analytical + 50k Sim Validation ===\n")

tuned = {}
for level in range(1, 10):
    target = 1.0 / (2**level)
    print(f"L{level} target cond={target:.4f}...")
    
    # Analytical search over scale values
    best = (None, None, 1e9)
    for sc in np.linspace(0.5, max(3, level * 8), 20):
        # Binary search on max_prob
        lo, hi = 0.001, 0.999
        for _ in range(30):
            mid = (lo + hi) / 2
            rate = analytical_rate(mid, sc)
            if rate < target: lo = mid
            else: hi = mid
        mp = (lo + hi) / 2
        rate = analytical_rate(mp, sc)
        err = abs(rate - target)
        if err < best[2]:
            best = (mp, sc, err)
    
    mp, sc, _ = best
    # Validate with 50k sim
    cond = sim_level(50000, level, mp, sc)
    
    # Adjust if off
    for _ in range(3):
        if abs(cond - target) > 0.01 * target:
            ratio = target / cond if cond > 0 else 2
            mp *= min(max(ratio, 0.5), 2.0)
            cond = sim_level(50000, level, mp, sc)
    
    tuned[level] = (mp, sc)
    print(f"  → mp={mp:.6f}, sc={sc:.3f}, sim_cond={cond:.4f} (target={target:.4f})\n")

# Full validation
print("=== Full Validation: 1M slots ===\n")
stakes = [3000, 2500, 2000, 1500, 1000]
ts = 10000
l0 = 0; lls = 0
lvl_hits = [0]*10; lvl_last = [0]*10

t0 = time.time()
for slot in range(1, 1_000_001):
    if slot % 200000 == 0:
        print(f"  {slot:>8d} ({time.time()-t0:.1f}s) L0={l0} hits={lvl_hits}")
    sg = slot - lls
    hit = False; st = 0
    for si in range(5):
        if dhash(si, slot, 0) < snowplow_thr(sg, stakes[si]/ts):
            hit = True; st = si; break
    if not hit: continue
    l0 += 1; lls = slot; lvl_hits[0] += 1
    for lv in range(1, 10):
        bg = l0 - lvl_last[lv]
        mp, sc = tuned[lv]
        if dhash(st, slot, lv) < shifted_exp_thr(bg, mp, sc):
            lvl_hits[lv] += 1; lvl_last[lv] = l0

print(f"\n{'='*70}")
print(f"  1,000,000 slots — {l0:,} base blocks ({time.time()-t0:.1f}s)")
print(f"{'='*70}")
print(f"{'Lvl':<5} {'Hits':>8} {'Cond%':>8} {'Tgt%':>8} {'Err%':>8} {'Ratio':>8}")
print("-"*45)
for lv in range(10):
    h = lvl_hits[lv]
    c = h/l0*100
    t = 100.0/(2**lv)
    e = abs(c-t)/t*100
    r = h/lvl_hits[lv-1] if lv>0 and lvl_hits[lv-1]>0 else 1.0
    print(f"L{lv:<4d} {h:>8,d} {c:>7.1f}% {t:>7.1f}% {e:>7.1f}% {r:>7.3f}")

print(f"\n\nScala params:")
for lv in range(1, 10):
    mp, sc = tuned[lv]
    print(f"  L{lv}: maxProb={mp:.6f}, scale={sc:.6f}")
