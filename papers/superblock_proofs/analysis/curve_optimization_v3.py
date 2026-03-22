#!/usr/bin/env python3
"""
Per-Level LDD Curve Optimization for Taktikos Superblocks — VERSION 3

This version introduces SHIFTED EXPONENTIAL curves:
  P(gap) = max_prob * (1 - exp(-(gap - psi) / scale))  for gap >= psi
         = 0                                            for gap < psi

The shift parameter psi provides burst resistance (P=0 when gap < psi)
while the exponential form provides smooth, monotonic threshold increase.

KEY INSIGHT: We need to account for the GAP DISTRIBUTION when fitting curves.
After an L_μ hit, the gap until next test follows a geometric distribution
(each base block has probability p of being followed by the next).

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
NUM_SLOTS = 100_000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# L0 parameters (Taktikos paper)
L0_PSI = 0
L0_GAMMA = 15
L0_FA = 0.5
L0_FB = 0.05


@dataclass
class SnowplowConfig:
    """Standard snowplow LDD config."""
    level: int
    psi: int
    gamma: int
    fA: float
    fB: float


@dataclass
class ShiftedExpConfig:
    """Shifted exponential config with dormant period."""
    level: int
    psi: int        # Dormant period (threshold=0 for gap < psi)
    max_prob: float # Asymptotic probability
    scale: float    # Decay rate


def snowplow_threshold(gap: int, config: SnowplowConfig) -> float:
    """Standard Taktikos snowplow threshold."""
    if gap < config.psi:
        return 0.0
    elif gap < config.gamma:
        return config.fA * (gap - config.psi) / (config.gamma - config.psi)
    else:
        return config.fB


def shifted_exp_threshold(gap: int, config: ShiftedExpConfig) -> float:
    """
    Shifted exponential threshold with dormant period.
    
    P(gap) = max_prob * (1 - exp(-(gap - psi) / scale))  for gap >= psi
           = 0                                            for gap < psi
    
    Properties:
    - P(gap < psi) = 0 (burst resistance via dormant period)
    - P(psi) = 0 (continuity)
    - P(∞) → max_prob (bounded)
    - Smooth, monotonically increasing for gap >= psi
    """
    if gap < config.psi:
        return 0.0
    shifted = gap - config.psi
    return config.max_prob * (1.0 - np.exp(-shifted / config.scale))


def domain_test(seed: int, slot: int, level: int) -> float:
    """Domain-separated test value in [0, 1)."""
    data = f"{seed}:{slot}:TEST-{level}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


# -----------------------------------------------------------------------------
# Simulation with Shifted Exponential
# -----------------------------------------------------------------------------

def simulate_shifted_exp(
    num_slots: int,
    l0_config: SnowplowConfig,
    exp_configs: List[ShiftedExpConfig],
) -> Dict:
    """Simulate using shifted exponential for super levels."""
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
# Analytical Parameter Derivation
# -----------------------------------------------------------------------------

def derive_shifted_exp_params(target_rates: List[float]) -> List[ShiftedExpConfig]:
    """
    Derive shifted exponential parameters analytically.
    
    For target rate r (per base block) with shifted exponential:
    - Mean gap before hit ≈ 1/r base blocks
    - Set psi = 1 for burst resistance
    - After a hit, the gap distribution is geometric (each base block
      has probability r of scoring a hit)
    
    The expected value of the threshold over the gap distribution should equal r:
      E[P(gap)] = r
    
    For shifted exponential with geometric gap:
      E[max_prob * (1 - exp(-(gap - psi) / scale))] = r
    
    This requires numerical solving.
    """
    configs = []
    
    for level in range(1, NUM_LEVELS):
        target = target_rates[level]
        
        # Set psi = 1 for burst resistance
        psi = 1
        
        # For high-frequency levels (L1-L3), we need higher max_prob
        # For low-frequency levels (L4-L9), lower max_prob suffices
        
        # Heuristic: mean gap ≈ 1/target
        # Scale should be ~mean_gap to make curve reach ~63% of max at mean gap
        mean_gap = 1.0 / target
        
        # Set max_prob and scale based on solving:
        # target ≈ integral from psi to ∞ of P(gap) * (1-target)^(gap-psi) * target dgap
        # This is complex; use a simpler heuristic and tune
        
        # For shifted exponential with geometric arrival, empirically:
        # - max_prob ≈ target * 2.5 works for L1-L3
        # - max_prob ≈ target * 4.0 works for L4-L9 (need more boost due to sparsity)
        
        if level <= 3:
            max_prob = min(0.99, target * 2.5)
            scale = max(0.5, mean_gap / 3)
        else:
            max_prob = min(0.99, target * 4.0)
            scale = max(1.0, mean_gap / 4)
        
        configs.append(ShiftedExpConfig(
            level=level, psi=psi, max_prob=max_prob, scale=scale
        ))
    
    return configs


def optimize_shifted_exp_level(
    level: int,
    target_rate: float,
    iterations: int = 200
) -> Tuple[ShiftedExpConfig, float]:
    """
    Numerically optimize shifted exponential parameters for a single level.
    
    Uses Monte Carlo to estimate achieved rate for each parameter set.
    """
    
    def objective(params):
        psi_f, max_prob, scale = params
        psi = max(1, int(psi_f))
        
        config = ShiftedExpConfig(
            level=level, psi=psi, max_prob=max_prob, scale=scale
        )
        
        # Monte Carlo simulation
        hits = 0
        total_base = 20000  # More samples for accuracy
        last_hit = 0
        
        for base_idx in range(1, total_base + 1):
            gap = base_idx - last_hit
            threshold = shifted_exp_threshold(gap, config)
            test = np.random.random()
            
            if threshold > test:
                hits += 1
                last_hit = base_idx
        
        achieved_rate = hits / total_base
        error = (achieved_rate - target_rate) ** 2
        
        # Penalty for threshold > 0 at gap < psi (shouldn't happen with our formula)
        burst_penalty = 0.0
        for g in range(psi):
            if shifted_exp_threshold(g, config) > 0.001:
                burst_penalty += 10.0
        
        return error + burst_penalty
    
    mean_gap = 1.0 / target_rate
    
    # Initial guess
    if level <= 3:
        x0 = [1.0, min(0.9, target_rate * 2.5), max(0.5, mean_gap / 3)]
    else:
        x0 = [1.0, min(0.9, target_rate * 4.0), max(1.0, mean_gap / 4)]
    
    bounds = [
        (1.0, 3.0),           # psi
        (target_rate, 0.99),  # max_prob must be >= target
        (0.1, 500.0),         # scale
    ]
    
    result = optimize.minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': iterations}
    )
    
    psi = max(1, int(result.x[0]))
    config = ShiftedExpConfig(
        level=level, psi=psi, max_prob=result.x[1], scale=result.x[2]
    )
    
    return config, result.fun


def optimize_all_shifted_exp(target_rates: List[float]) -> List[ShiftedExpConfig]:
    """Optimize parameters for all super levels."""
    configs = []
    
    print("\nOptimizing shifted exponential parameters...")
    for level in range(1, NUM_LEVELS):
        config, loss = optimize_shifted_exp_level(level, target_rates[level])
        configs.append(config)
        print(f"  L{level}: ψ={config.psi}, max_prob={config.max_prob:.4f}, "
              f"scale={config.scale:.2f} (loss={loss:.6f})")
    
    return configs


# -----------------------------------------------------------------------------
# Adversary Analysis
# -----------------------------------------------------------------------------

def adversary_burst_analysis(
    l0_config: SnowplowConfig,
    exp_configs: List[ShiftedExpConfig],
    burst_size: int = 10
):
    """Analyze what happens when adversary produces blocks at gap=1."""
    print(f"\nAdversary burst analysis ({burst_size} blocks at gap=1):")
    print("  Level | Threshold at gap=1 | Expected Hits")
    print("  ------|-------------------|---------------")
    
    total = 0.0
    
    # L0
    t0 = snowplow_threshold(1, l0_config)
    e0 = burst_size * t0
    total += e0
    print(f"  L0    | {t0:.6f}           | {e0:.3f}")
    
    # L1-L9
    for config in exp_configs:
        t = shifted_exp_threshold(1, config)
        e = burst_size * t
        total += e
        print(f"  L{config.level}    | {t:.6f}           | {e:.3f}")
    
    print(f"  ------|-------------------|---------------")
    print(f"  TOTAL |                   | {total:.3f}")
    
    # Check threshold at gap=0 (should be 0 for all)
    print("\n  Threshold at gap=0 (should all be 0):")
    for config in exp_configs:
        t = shifted_exp_threshold(0, config)
        status = "✓" if t == 0 else "✗"
        print(f"    L{config.level}: {t:.6f} {status}")
    
    return total


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_shifted_exp_curves(
    l0_config: SnowplowConfig,
    exp_configs: List[ShiftedExpConfig],
    filename: str
):
    """Plot threshold curves for all levels."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # L0 (snowplow)
    ax = axes[0]
    gaps = np.arange(0, l0_config.gamma * 3 + 1)
    thresholds = [snowplow_threshold(g, l0_config) for g in gaps]
    ax.plot(gaps, thresholds, 'b-', linewidth=2)
    ax.axvline(x=l0_config.gamma, color='g', linestyle='--', alpha=0.5)
    ax.fill_between(gaps, thresholds, alpha=0.3)
    ax.set_title('L0 (Snowplow on slots)')
    ax.set_xlabel('Gap (slots)')
    ax.set_ylabel('Threshold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # L1-L9 (shifted exponential)
    for i, config in enumerate(exp_configs):
        ax = axes[i + 1]
        level = config.level
        
        # Plot range depends on target rate
        target = 1.0 / (2 ** level)
        max_gap = min(300, int(5 / target))
        gaps = np.arange(0, max_gap + 1)
        thresholds = [shifted_exp_threshold(g, config) for g in gaps]
        
        ax.plot(gaps, thresholds, 'b-', linewidth=2)
        ax.axvline(x=config.psi, color='r', linestyle='--', alpha=0.5, 
                   label=f'ψ={config.psi}')
        ax.axhline(y=config.max_prob, color='orange', linestyle=':', alpha=0.5,
                   label=f'max={config.max_prob:.3f}')
        ax.fill_between(gaps, thresholds, alpha=0.3)
        
        ax.set_title(f'L{level} (Shifted Exp)')
        ax.set_xlabel('Gap (base blocks)')
        ax.set_ylabel('Threshold')
        ax.set_ylim(-0.05, min(1.05, config.max_prob * 1.3))
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(
    results: Dict,
    target_rates: List[float],
    filename: str
):
    """Plot achieved vs target rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(NUM_LEVELS)
    achieved = results["conditional_rates"]
    
    # Absolute comparison
    width = 0.35
    ax1.bar(x - width/2, target_rates, width, label='Target', alpha=0.7)
    ax1.bar(x + width/2, achieved, width, label='Achieved', alpha=0.7)
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Conditional Rate')
    ax1.set_title('Target vs Achieved Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in range(NUM_LEVELS)])
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Error percentage
    errors = []
    for i in range(NUM_LEVELS):
        if target_rates[i] > 0:
            err = abs(achieved[i] - target_rates[i]) / target_rates[i] * 100
        else:
            err = 0
        errors.append(err)
    
    colors = ['green' if e < 20 else 'orange' if e < 50 else 'red' for e in errors]
    ax2.bar(x, errors, color=colors)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% threshold')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Rate Error by Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i}' for i in range(NUM_LEVELS)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_gap_distributions(results: Dict, filename: str):
    """Plot gap distributions at hit time."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for level in range(NUM_LEVELS):
        ax = axes[level]
        gaps = results["gap_distributions"][level]
        
        if len(gaps) > 10:
            # Histogram
            bins = min(50, max(10, len(set(gaps))))
            ax.hist(gaps, bins=bins, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(gaps), color='r', linestyle='--',
                       label=f'Mean={np.mean(gaps):.1f}')
            ax.axvline(x=np.median(gaps), color='g', linestyle=':',
                       label=f'Median={np.median(gaps):.1f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f'n={len(gaps)}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
        
        unit = "slots" if level == 0 else "base blocks"
        ax.set_title(f'L{level} (n={len(gaps)})')
        ax.set_xlabel(f'Gap ({unit})')
        ax.set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Per-Level LDD Curve Optimization — VERSION 3")
    print("Using SHIFTED EXPONENTIAL curves with dormant period")
    print("=" * 80)
    print()
    
    # Target rates: P(Lμ | L0) = 1/2^μ
    # Note: L0 is ~14% of slots, so L0's "conditional" rate is its absolute rate
    target_rates = [0.14] + [1.0 / (2 ** i) for i in range(1, NUM_LEVELS)]
    
    print("Target conditional rates (P(Lμ | L0)):")
    for i in range(NUM_LEVELS):
        if i == 0:
            print(f"  L{i}: {target_rates[i]:.6f} (absolute rate, vs slots)")
        else:
            spacing = 1.0 / target_rates[i]
            print(f"  L{i}: {target_rates[i]:.6f} (~1 per {spacing:.0f} base blocks)")
    
    # L0 config (standard snowplow)
    l0_config = SnowplowConfig(
        level=0, psi=L0_PSI, gamma=L0_GAMMA, fA=L0_FA, fB=L0_FB
    )
    
    print()
    print("-" * 80)
    print("Optimizing shifted exponential parameters for L1-L9...")
    print("-" * 80)
    
    exp_configs = optimize_all_shifted_exp(target_rates)
    
    print()
    print("-" * 80)
    print(f"Simulating {NUM_SLOTS:,} slots...")
    print("-" * 80)
    
    results = simulate_shifted_exp(NUM_SLOTS, l0_config, exp_configs)
    
    print("\nResults:")
    print("  Level | Count  | Cond Rate | Target  | Error  | Status")
    print("  ------|--------|-----------|---------|--------|--------")
    
    total_error = 0
    for i, count in enumerate(results['level_counts']):
        cond = results['conditional_rates'][i]
        target = target_rates[i]
        error = abs(cond - target) / target * 100 if target > 0 else 0
        total_error += error
        
        status = "✓" if error < 20 else "~" if error < 50 else "✗"
        print(f"  L{i}    | {count:6,} | {cond:.5f}   | {target:.5f} | {error:5.1f}% | {status}")
    
    mean_error = total_error / NUM_LEVELS
    print(f"\n  Mean error across all levels: {mean_error:.1f}%")
    
    adversary_burst_analysis(l0_config, exp_configs)
    
    print()
    print("-" * 80)
    print("Generating figures...")
    print("-" * 80)
    
    plot_shifted_exp_curves(l0_config, exp_configs,
                           os.path.join(FIGURES_DIR, "v3_shifted_exp_curves.png"))
    plot_comparison(results, target_rates,
                   os.path.join(FIGURES_DIR, "v3_comparison.png"))
    plot_gap_distributions(results,
                          os.path.join(FIGURES_DIR, "v3_gap_distributions.png"))
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Check burst resistance
    burst_total = sum(shifted_exp_threshold(1, c) for c in exp_configs)
    burst_total += snowplow_threshold(1, l0_config)
    
    print(f"\nBurst resistance (threshold at gap=1):")
    print(f"  Expected super-level hits per 10-block burst: {burst_total * 10:.3f}")
    
    if burst_total < 0.1:
        print("  Status: ✓ EXCELLENT burst resistance")
    elif burst_total < 0.5:
        print("  Status: ~ ACCEPTABLE burst resistance")
    else:
        print("  Status: ✗ POOR burst resistance")
    
    print(f"\nRate accuracy (mean error): {mean_error:.1f}%")
    if mean_error < 20:
        print("  Status: ✓ EXCELLENT rate accuracy")
    elif mean_error < 40:
        print("  Status: ~ ACCEPTABLE rate accuracy")
    else:
        print("  Status: ✗ NEEDS IMPROVEMENT")
    
    print("\nFinal parameters (for Scala implementation):")
    print("```scala")
    print("object SuperLevelParams {")
    print("  // L0: Standard Taktikos snowplow on slot gaps")
    print(f"  val L0_psi = {l0_config.psi}")
    print(f"  val L0_gamma = {l0_config.gamma}")
    print(f"  val L0_fA = {l0_config.fA}")
    print(f"  val L0_fB = {l0_config.fB}")
    print()
    print("  // L1-L9: Shifted exponential on base-block gaps")
    print("  // P(gap) = max_prob * (1 - exp(-(gap - psi) / scale)) for gap >= psi")
    print("  //        = 0                                          for gap < psi")
    print("  val superLevelParams: Vector[(Int, Double, Double)] = Vector(")
    for config in exp_configs:
        print(f"    ({config.psi}, {config.max_prob:.6f}, {config.scale:.4f}),  // L{config.level}")
    print("  )")
    print("}")
    print("```")
    
    return {
        "l0_config": l0_config,
        "exp_configs": exp_configs,
        "results": results,
    }


if __name__ == "__main__":
    results = main()
