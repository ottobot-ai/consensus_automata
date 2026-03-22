#!/usr/bin/env python3
"""
Per-Level LDD Curve Optimization for Taktikos Superblocks — VERSION 4

This version uses DIRECT SIMULATION for parameter tuning, matching
the exact simulation dynamics rather than approximating with Monte Carlo.

KEY INSIGHT: The higher levels (L4-L9) are too sparse for the optimization
to converge properly. Instead of optimizing, we'll use ANALYTICAL DERIVATION
based on the renewal theory of hit processes.

For shifted exponential with dormant period:
  P(gap) = max_prob * (1 - exp(-(gap - psi) / scale))  for gap >= psi
         = 0                                            for gap < psi

The expected rate r satisfies:
  r = sum_{g=psi}^{∞} P(g) * Pr(gap = g)

where Pr(gap = g) is the probability that the gap is exactly g given
the process dynamics. After a hit at gap G, the next gap is 1 with prob 1
(next base block), then 2, etc. until a hit occurs.

This is a geometric-like process but with the threshold function applied.

Author: CodeBot 🔍  
Date: 2026-03-22
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import hashlib
import os

np.random.seed(42)

NUM_LEVELS = 10
NUM_SLOTS = 200_000  # Larger for better statistics
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# L0 parameters
L0_PSI = 0
L0_GAMMA = 15
L0_FA = 0.5
L0_FB = 0.05


@dataclass
class SnowplowConfig:
    level: int
    psi: int
    gamma: int
    fA: float
    fB: float


@dataclass  
class ShiftedExpConfig:
    level: int
    psi: int
    max_prob: float
    scale: float


def snowplow_threshold(gap: int, config: SnowplowConfig) -> float:
    if gap < config.psi:
        return 0.0
    elif gap < config.gamma:
        return config.fA * (gap - config.psi) / (config.gamma - config.psi)
    else:
        return config.fB


def shifted_exp_threshold(gap: int, config: ShiftedExpConfig) -> float:
    if gap < config.psi:
        return 0.0
    shifted = gap - config.psi
    return config.max_prob * (1.0 - np.exp(-shifted / config.scale))


def domain_test(seed: int, slot: int, level: int) -> float:
    data = f"{seed}:{slot}:TEST-{level}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


# -----------------------------------------------------------------------------
# Analytical Rate Computation
# -----------------------------------------------------------------------------

def compute_expected_rate_analytical(config: ShiftedExpConfig, max_gap: int = 2000) -> float:
    """
    Compute the expected hit rate analytically using renewal theory.
    
    For a shifted exponential threshold, the probability of hitting at gap g
    (given we haven't hit at gaps psi, psi+1, ..., g-1) is:
    
      P(hit at g | no hit before) = P(g) * prod_{j=psi}^{g-1} (1 - P(j))
    
    The expected gap E[G] = sum_{g=psi}^{∞} g * P(hit at g | no hit before)
    
    Rate = 1 / E[G]
    """
    psi = config.psi
    
    # Compute probability of hitting at each gap
    hit_probs = np.zeros(max_gap + 1)
    survival = 1.0  # Probability of not having hit yet
    
    for g in range(psi, max_gap + 1):
        threshold = shifted_exp_threshold(g, config)
        hit_probs[g] = survival * threshold
        survival *= (1 - threshold)
        
        if survival < 1e-12:
            break
    
    # Expected gap
    expected_gap = sum(g * hit_probs[g] for g in range(psi, max_gap + 1))
    
    # Normalize (in case survival didn't reach 0)
    total_prob = sum(hit_probs)
    if total_prob > 0:
        expected_gap /= total_prob
    
    if expected_gap > 0:
        return 1.0 / expected_gap
    else:
        return 0.0


def find_params_for_rate(target_rate: float, psi: int = 1) -> ShiftedExpConfig:
    """
    Find shifted exponential parameters that achieve the target rate.
    
    Uses binary search over scale and max_prob.
    """
    
    def objective(params):
        max_prob, scale = params
        config = ShiftedExpConfig(level=0, psi=psi, max_prob=max_prob, scale=scale)
        achieved = compute_expected_rate_analytical(config)
        return (achieved - target_rate) ** 2
    
    # Initial guess based on heuristics
    mean_gap = 1.0 / target_rate
    x0 = [min(0.9, target_rate * 2), mean_gap / 2]
    
    bounds = [
        (target_rate * 0.5, 0.99),  # max_prob
        (0.1, 1000.0),              # scale
    ]
    
    result = optimize.minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 1000}
    )
    
    return ShiftedExpConfig(
        level=0, psi=psi, max_prob=result.x[0], scale=result.x[1]
    )


# -----------------------------------------------------------------------------
# Full Simulation
# -----------------------------------------------------------------------------

def simulate(
    num_slots: int,
    l0_config: SnowplowConfig,
    exp_configs: List[ShiftedExpConfig],
) -> Dict:
    """Full simulation with L0 snowplow and L1-L9 shifted exponential."""
    last_base_count = [0] * NUM_LEVELS
    level_counts = [0] * NUM_LEVELS
    gap_at_hit = [[] for _ in range(NUM_LEVELS)]
    
    slot_gap = 0
    
    for slot in range(num_slots):
        slot_gap += 1
        
        l0_threshold = snowplow_threshold(slot_gap, l0_config)
        l0_test = domain_test(42, slot, 0)
        
        if l0_threshold > l0_test:
            level_counts[0] += 1
            gap_at_hit[0].append(slot_gap)
            slot_gap = 0
            
            base_count = level_counts[0]
            
            for level in range(1, NUM_LEVELS):
                config = exp_configs[level - 1]
                gap = base_count - last_base_count[level]
                
                threshold = shifted_exp_threshold(gap, config)
                test = domain_test(42, slot, level)
                
                if threshold > test:
                    level_counts[level] += 1
                    gap_at_hit[level].append(gap)
                    last_base_count[level] = base_count
    
    results = {
        "total_slots": num_slots,
        "level_counts": level_counts.copy(),
        "conditional_rates": [],
        "gap_distributions": gap_at_hit,
    }
    
    results["conditional_rates"].append(level_counts[0] / num_slots)
    for level in range(1, NUM_LEVELS):
        cond_rate = level_counts[level] / level_counts[0] if level_counts[0] > 0 else 0
        results["conditional_rates"].append(cond_rate)
    
    return results


# -----------------------------------------------------------------------------
# Iterative Refinement
# -----------------------------------------------------------------------------

def refine_params_via_simulation(
    l0_config: SnowplowConfig,
    initial_configs: List[ShiftedExpConfig],
    target_rates: List[float],
    num_slots: int = 50000,
    iterations: int = 5
) -> List[ShiftedExpConfig]:
    """
    Refine parameters by running actual simulations and adjusting.
    
    This accounts for the VRF hash distribution and simulation dynamics.
    """
    configs = [c for c in initial_configs]  # Copy
    
    for iteration in range(iterations):
        print(f"\n  Iteration {iteration + 1}/{iterations}:")
        
        # Simulate
        results = simulate(num_slots, l0_config, configs)
        
        # Adjust each level
        for level in range(1, NUM_LEVELS):
            achieved = results["conditional_rates"][level]
            target = target_rates[level]
            
            if target == 0:
                continue
                
            ratio = achieved / target
            config = configs[level - 1]
            
            # Adjust max_prob based on ratio
            # If achieved > target, decrease max_prob
            # If achieved < target, increase max_prob
            
            adjustment = 1.0 / ratio if ratio > 0 else 2.0
            adjustment = max(0.5, min(2.0, adjustment))  # Limit adjustment
            
            new_max_prob = config.max_prob * adjustment
            new_max_prob = max(target, min(0.99, new_max_prob))
            
            # Also adjust scale slightly
            new_scale = config.scale
            if ratio > 1.2:
                new_scale *= 0.9  # Smaller scale = steeper curve = fewer hits
            elif ratio < 0.8:
                new_scale *= 1.1  # Larger scale = gentler curve = more hits
            
            new_scale = max(0.1, min(500, new_scale))
            
            configs[level - 1] = ShiftedExpConfig(
                level=level, psi=config.psi,
                max_prob=new_max_prob, scale=new_scale
            )
            
            status = "✓" if abs(ratio - 1) < 0.2 else "→"
            print(f"    L{level}: achieved={achieved:.5f}, target={target:.5f}, "
                  f"ratio={ratio:.2f} {status}")
    
    return configs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Per-Level LDD Curve Optimization — VERSION 4")
    print("Using ANALYTICAL + SIMULATION REFINEMENT")
    print("=" * 80)
    print()
    
    target_rates = [0.14] + [1.0 / (2 ** i) for i in range(1, NUM_LEVELS)]
    
    print("Target conditional rates (P(Lμ | L0)):")
    for i in range(NUM_LEVELS):
        spacing = 1.0 / target_rates[i] if target_rates[i] > 0 else float('inf')
        unit = "slots" if i == 0 else "base blocks"
        print(f"  L{i}: {target_rates[i]:.6f} (~1 per {spacing:.0f} {unit})")
    
    l0_config = SnowplowConfig(
        level=0, psi=L0_PSI, gamma=L0_GAMMA, fA=L0_FA, fB=L0_FB
    )
    
    print()
    print("-" * 80)
    print("Step 1: Analytical parameter derivation")
    print("-" * 80)
    
    initial_configs = []
    for level in range(1, NUM_LEVELS):
        target = target_rates[level]
        config = find_params_for_rate(target, psi=1)
        config = ShiftedExpConfig(
            level=level, psi=config.psi,
            max_prob=config.max_prob, scale=config.scale
        )
        initial_configs.append(config)
        
        # Verify analytical rate
        analytical_rate = compute_expected_rate_analytical(config)
        error = abs(analytical_rate - target) / target * 100
        print(f"  L{level}: max_prob={config.max_prob:.4f}, scale={config.scale:.2f}, "
              f"analytical_rate={analytical_rate:.5f} (err={error:.1f}%)")
    
    print()
    print("-" * 80)
    print("Step 2: Simulation refinement")
    print("-" * 80)
    
    refined_configs = refine_params_via_simulation(
        l0_config, initial_configs, target_rates,
        num_slots=50000, iterations=5
    )
    
    print()
    print("-" * 80)
    print(f"Step 3: Final validation ({NUM_SLOTS:,} slots)")
    print("-" * 80)
    
    results = simulate(NUM_SLOTS, l0_config, refined_configs)
    
    print("\nFinal Results:")
    print("  Level | Count  | Cond Rate | Target  | Error  | Status")
    print("  ------|--------|-----------|---------|--------|--------")
    
    errors = []
    for i, count in enumerate(results['level_counts']):
        cond = results['conditional_rates'][i]
        target = target_rates[i]
        error = abs(cond - target) / target * 100 if target > 0 else 0
        errors.append(error)
        
        status = "✓" if error < 20 else "~" if error < 40 else "✗"
        print(f"  L{i}    | {count:6,} | {cond:.5f}   | {target:.5f} | {error:5.1f}% | {status}")
    
    mean_error = np.mean(errors)
    print(f"\n  Mean error: {mean_error:.1f}%")
    
    # Burst resistance
    print("\nBurst Resistance (threshold at gap=1):")
    total_burst = snowplow_threshold(1, l0_config)
    print(f"  L0: {total_burst:.6f}")
    for config in refined_configs:
        t = shifted_exp_threshold(1, config)
        print(f"  L{config.level}: {t:.6f}")
        total_burst += t
    print(f"  TOTAL expected hits per 10-block burst: {total_burst * 10:.4f}")
    
    # Generate figures
    print()
    print("-" * 80)
    print("Generating figures...")
    print("-" * 80)
    
    # Curve plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # L0
    ax = axes[0]
    gaps = np.arange(0, 50)
    thresholds = [snowplow_threshold(g, l0_config) for g in gaps]
    ax.plot(gaps, thresholds, 'b-', linewidth=2)
    ax.fill_between(gaps, thresholds, alpha=0.3)
    ax.set_title('L0 (Snowplow)')
    ax.set_xlabel('Gap (slots)')
    ax.set_ylabel('Threshold')
    ax.grid(True, alpha=0.3)
    
    # L1-L9
    for i, config in enumerate(refined_configs):
        ax = axes[i + 1]
        level = config.level
        target = target_rates[level]
        max_gap = min(500, int(5 / target))
        gaps = np.arange(0, max_gap + 1)
        thresholds = [shifted_exp_threshold(g, config) for g in gaps]
        
        ax.plot(gaps, thresholds, 'b-', linewidth=2)
        ax.axhline(y=config.max_prob, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=config.psi, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(gaps, thresholds, alpha=0.3)
        
        ax.set_title(f'L{level}')
        ax.set_xlabel('Gap (base blocks)')
        ax.set_ylabel('Threshold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "v4_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/v4_curves.png")
    
    # Rate comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_LEVELS)
    width = 0.35
    
    achieved = results["conditional_rates"]
    ax.bar(x - width/2, target_rates, width, label='Target', alpha=0.7)
    ax.bar(x + width/2, achieved, width, label='Achieved', alpha=0.7)
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Conditional Rate')
    ax.set_title('Target vs Achieved Rates (V4)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(NUM_LEVELS)])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "v4_rates.png"), dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/v4_rates.png")
    
    # Summary
    print()
    print("=" * 80)
    print("FINAL PARAMETERS (for Scala implementation)")
    print("=" * 80)
    print()
    print("```scala")
    print("object SuperLevelParams {")
    print("  // L0: Standard Taktikos snowplow")
    print(f"  val L0 = SnowplowConfig(psi = {l0_config.psi}, gamma = {l0_config.gamma}, "
          f"fA = {l0_config.fA}, fB = {l0_config.fB})")
    print()
    print("  // L1-L9: Shifted exponential on base-block gaps")
    print("  // threshold(gap) = max_prob * (1 - exp(-(gap - psi) / scale)) if gap >= psi else 0")
    print("  val superLevels: Vector[ShiftedExpConfig] = Vector(")
    for config in refined_configs:
        print(f"    ShiftedExpConfig(level = {config.level}, psi = {config.psi}, "
              f"maxProb = {config.max_prob:.6f}, scale = {config.scale:.4f}),")
    print("  )")
    print("}")
    print("```")
    
    return {
        "l0_config": l0_config,
        "configs": refined_configs,
        "results": results,
    }


if __name__ == "__main__":
    results = main()
