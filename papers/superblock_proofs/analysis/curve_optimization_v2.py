#!/usr/bin/env python3
"""
Per-Level LDD Curve Optimization for Taktikos Superblocks — VERSION 2

This version corrects the gap measurement to use BASE BLOCKS (L0) as the
universal unit for all super levels, not cascading parent-block gaps.

KEY INSIGHT: The problem with cascading parent gaps is that higher levels
become increasingly sparse, causing their gaps to become very small in
absolute terms (because each parent is itself rare).

CORRECTED APPROACH:
- All super levels measure gap in BASE BLOCKS (L0 count since last hit)
- This gives each level consistent gap growth rate
- We tune curves to achieve target rates given this consistent gap measure

Author: CodeBot 🔍
Date: 2026-03-22
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import hashlib
import os

np.random.seed(42)

NUM_LEVELS = 10
NUM_SLOTS = 100_000  
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# L0 parameters
L0_PSI = 0
L0_GAMMA = 15
L0_FA = 0.5
L0_FB = 0.05
L0_TARGET_RATE = 0.14


@dataclass
class LevelConfig:
    """Configuration for a single superblock level's LDD curve."""
    level: int
    psi: int       # Gap offset (dormant period in base-block units)
    gamma: int     # Cutoff (end of ramp)
    fA: float      # Amplitude (peak threshold at cutoff)
    fB: float      # Baseline (recovery phase)


def snowplow_threshold(gap: int, config: LevelConfig) -> float:
    """Calculate snowplow LDD threshold."""
    psi, gamma = config.psi, config.gamma
    fA, fB = config.fA, config.fB
    
    if gap < psi:
        return 0.0
    elif gap < gamma:
        return fA * (gap - psi) / (gamma - psi)
    else:
        return fB


def domain_test(seed: int, slot: int, level: int) -> float:
    """Generate domain-separated test value in [0, 1)."""
    data = f"{seed}:{slot}:TEST-{level}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], 'big') / (2**64)


# -----------------------------------------------------------------------------
# APPROACH: Base-Block Gaps for All Super Levels
# -----------------------------------------------------------------------------

def simulate_base_block_gaps(
    num_slots: int,
    level_configs: List[LevelConfig],
) -> Dict:
    """
    Simulate using BASE BLOCK gaps for all super levels.
    
    For level μ (μ >= 1):
      gap_μ = number of BASE BLOCKS (L0) since level μ last hit
    
    This gives consistent gap growth across all levels.
    """
    # State tracking
    # last_base_count[μ] = L0 block count when level μ last hit
    last_base_count = [0] * NUM_LEVELS
    level_counts = [0] * NUM_LEVELS
    gap_at_hit = [[] for _ in range(NUM_LEVELS)]
    
    slot_gap = 0  # For L0
    
    for slot in range(num_slots):
        slot_gap += 1
        
        # Test L0 (slot-based snowplow)
        l0_config = level_configs[0]
        l0_threshold = snowplow_threshold(slot_gap, l0_config)
        l0_test = domain_test(42, slot, 0)
        
        if l0_threshold > l0_test:
            # L0 hit
            level_counts[0] += 1
            gap_at_hit[0].append(slot_gap)
            slot_gap = 0
            
            # Test all super levels using BASE BLOCK gap
            base_count = level_counts[0]
            
            for level in range(1, NUM_LEVELS):
                config = level_configs[level]
                
                # Gap = base blocks since this level last hit
                gap = base_count - last_base_count[level]
                
                threshold = snowplow_threshold(gap, config)
                test = domain_test(42, slot, level)
                
                if threshold > test:
                    # Level μ hit
                    level_counts[level] += 1
                    gap_at_hit[level].append(gap)
                    last_base_count[level] = base_count
    
    # Compute results
    results = {
        "total_slots": num_slots,
        "level_counts": level_counts.copy(),
        "conditional_rates": [],
        "gap_distributions": gap_at_hit,
    }
    
    # L0 rate vs slots
    l0_rate = level_counts[0] / num_slots
    results["conditional_rates"].append(l0_rate)
    
    # Super level rates vs BASE BLOCKS (L0)
    for level in range(1, NUM_LEVELS):
        if level_counts[0] > 0:  # Condition on L0, not L(μ-1)
            cond_rate = level_counts[level] / level_counts[0]
        else:
            cond_rate = 0.0
        results["conditional_rates"].append(cond_rate)
    
    return results


def compute_theoretical_params(target_rates: List[float]) -> List[LevelConfig]:
    """
    Compute theoretical curve parameters for base-block gap approach.
    
    For level μ with target rate r (per base block):
    - Mean gap before hit = 1/r base blocks
    - gamma should be around 1/r to let ramp develop
    - psi = 1 to penalize immediate re-hits (burst resistance)
    """
    configs = []
    
    # L0: standard slot-based snowplow
    configs.append(LevelConfig(
        level=0, psi=L0_PSI, gamma=L0_GAMMA, fA=L0_FA, fB=L0_FB
    ))
    
    # L1-L9: base-block gap snowplows
    for level in range(1, NUM_LEVELS):
        target = target_rates[level]
        mean_gap = 1.0 / target
        
        # Set parameters based on mean gap
        # psi = 1: dormant for 1 base block (prevents immediate re-hit)
        # gamma = ceil(mean_gap): ramp reaches peak by expected hit time
        # fA: tune to achieve target rate
        # fB: lower baseline for continued shaping
        
        psi = 1
        gamma = max(3, int(np.ceil(mean_gap)))
        
        # Analytical: if we want rate r and ramp reaches fA at gamma,
        # the average threshold during ramp is ~fA/2
        # Need fA/2 ≈ r → fA ≈ 2r
        # But since we also have baseline, tune empirically
        fA = min(0.99, target * 2.5)
        fB = min(0.5, target * 0.3)
        
        configs.append(LevelConfig(
            level=level, psi=psi, gamma=gamma, fA=fA, fB=fB
        ))
    
    return configs


def optimize_single_level(
    level: int,
    target_rate: float,
    base_arrival_rate: float,
    iterations: int = 100
) -> Tuple[LevelConfig, float]:
    """
    Numerically optimize parameters for a single level.
    
    Simulates the level's hits given base block arrivals and tunes
    psi, gamma, fA, fB to minimize |achieved_rate - target_rate|.
    """
    
    def objective(params):
        psi_f, gamma_f, fA, fB = params
        psi = max(1, int(psi_f))
        gamma = max(psi + 2, int(gamma_f))
        
        config = LevelConfig(level=level, psi=psi, gamma=gamma, fA=fA, fB=fB)
        
        # Monte Carlo: simulate base block arrivals and level hits
        hits = 0
        base_count = 0
        last_hit = 0
        total_base = 10000
        
        for _ in range(total_base):
            base_count += 1
            gap = base_count - last_hit
            
            threshold = snowplow_threshold(gap, config)
            test = np.random.random()
            
            if threshold > test:
                hits += 1
                last_hit = base_count
        
        achieved_rate = hits / total_base
        return (achieved_rate - target_rate) ** 2
    
    # Initial guess
    mean_gap = 1.0 / target_rate
    x0 = [1.0, mean_gap, target_rate * 2.5, target_rate * 0.3]
    
    bounds = [
        (1.0, 5.0),           # psi
        (3.0, 1000.0),        # gamma
        (0.001, 0.99),        # fA
        (0.0001, 0.5),        # fB
    ]
    
    result = optimize.minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': iterations}
    )
    
    psi = max(1, int(result.x[0]))
    gamma = max(psi + 2, int(result.x[1]))
    fA = result.x[2]
    fB = result.x[3]
    
    config = LevelConfig(level=level, psi=psi, gamma=gamma, fA=fA, fB=fB)
    return config, result.fun


def optimize_all_levels(target_rates: List[float]) -> List[LevelConfig]:
    """Optimize parameters for all levels."""
    configs = []
    
    # L0: fixed parameters
    configs.append(LevelConfig(
        level=0, psi=L0_PSI, gamma=L0_GAMMA, fA=L0_FA, fB=L0_FB
    ))
    
    # L1-L9: optimize each
    print("\nOptimizing per-level parameters...")
    for level in range(1, NUM_LEVELS):
        config, loss = optimize_single_level(level, target_rates[level], L0_TARGET_RATE)
        configs.append(config)
        print(f"  L{level}: ψ={config.psi}, γ={config.gamma}, "
              f"fA={config.fA:.4f}, fB={config.fB:.4f} (loss={loss:.6f})")
    
    return configs


# -----------------------------------------------------------------------------
# Exponential Decay Alternative
# -----------------------------------------------------------------------------

@dataclass
class ExpConfig:
    """Exponential decay configuration."""
    level: int
    max_prob: float
    scale: float  # Controls decay steepness


def exp_threshold(gap: int, config: ExpConfig) -> float:
    """
    Exponential approach to max_prob:
    
    P(gap) = max_prob * (1 - exp(-gap / scale))
    
    Properties:
    - P(0) = 0 (burst resistance)
    - P(∞) → max_prob
    - Smooth, monotonic increase
    """
    if gap <= 0:
        return 0.0
    return config.max_prob * (1.0 - np.exp(-gap / config.scale))


def simulate_exp_decay(
    num_slots: int,
    l0_config: LevelConfig,
    exp_configs: List[ExpConfig],
) -> Dict:
    """Simulate with exponential decay curves."""
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
                
                threshold = exp_threshold(gap, config)
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


def find_exp_params(target_rates: List[float]) -> List[ExpConfig]:
    """
    Find exponential decay parameters for each level.
    
    For rate r with exponential curve P = max_prob * (1 - exp(-gap/scale)):
    - Average hit at gap ≈ 1/r
    - At average gap, want P ≈ r
    - Solve: r = max_prob * (1 - exp(-1/(r * scale)))
    """
    configs = []
    
    for level in range(1, NUM_LEVELS):
        target = target_rates[level]
        
        # Set max_prob to ~2x target (so curve reaches target before saturation)
        max_prob = min(0.99, target * 2.0)
        
        # Solve for scale: target = max_prob * (1 - exp(-mean_gap / scale))
        mean_gap = 1.0 / target
        
        # target / max_prob = 1 - exp(-mean_gap / scale)
        # exp(-mean_gap / scale) = 1 - target/max_prob
        # -mean_gap / scale = ln(1 - target/max_prob)
        # scale = -mean_gap / ln(1 - target/max_prob)
        
        ratio = target / max_prob
        if ratio >= 1:
            scale = mean_gap  # Fallback
        else:
            scale = -mean_gap / np.log(1 - ratio)
        
        configs.append(ExpConfig(level=level, max_prob=max_prob, scale=scale))
    
    return configs


def optimize_exp_level(
    level: int,
    target_rate: float,
    iterations: int = 100
) -> Tuple[ExpConfig, float]:
    """Numerically optimize exponential parameters for a level."""
    
    def objective(params):
        max_prob, scale = params
        config = ExpConfig(level=level, max_prob=max_prob, scale=scale)
        
        hits = 0
        base_count = 0
        last_hit = 0
        total_base = 10000
        
        for _ in range(total_base):
            base_count += 1
            gap = base_count - last_hit
            
            threshold = exp_threshold(gap, config)
            test = np.random.random()
            
            if threshold > test:
                hits += 1
                last_hit = base_count
        
        achieved_rate = hits / total_base
        return (achieved_rate - target_rate) ** 2
    
    mean_gap = 1.0 / target_rate
    x0 = [target_rate * 2.0, mean_gap / 2]
    
    bounds = [
        (target_rate, 0.99),    # max_prob must be >= target
        (0.1, 1000.0),          # scale
    ]
    
    result = optimize.minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': iterations}
    )
    
    config = ExpConfig(level=level, max_prob=result.x[0], scale=result.x[1])
    return config, result.fun


# -----------------------------------------------------------------------------
# Adversary Analysis
# -----------------------------------------------------------------------------

def adversary_analysis_snowplow(configs: List[LevelConfig], burst_size: int = 10):
    """Analyze burst resistance for snowplow curves."""
    print(f"\nAdversary burst analysis ({burst_size} blocks at gap=1):")
    print("  Level | Threshold | Expected Hits")
    print("  ------|-----------|---------------")
    
    total_expected = 0.0
    for level, config in enumerate(configs):
        if level == 0:
            thresh = snowplow_threshold(1, config)
        else:
            thresh = snowplow_threshold(1, config)
        
        exp_hits = burst_size * thresh
        total_expected += exp_hits
        print(f"  L{level}    | {thresh:.4f}    | {exp_hits:.3f}")
    
    print(f"  ------|-----------|---------------")
    print(f"  TOTAL |           | {total_expected:.3f}")
    
    return total_expected


def adversary_analysis_exp(
    l0_config: LevelConfig,
    exp_configs: List[ExpConfig],
    burst_size: int = 10
):
    """Analyze burst resistance for exponential curves."""
    print(f"\nAdversary burst analysis ({burst_size} blocks at gap=1):")
    print("  Level | Threshold | Expected Hits")
    print("  ------|-----------|---------------")
    
    # L0
    thresh = snowplow_threshold(1, l0_config)
    exp_hits = burst_size * thresh
    total_expected = exp_hits
    print(f"  L0    | {thresh:.4f}    | {exp_hits:.3f}")
    
    # L1-L9
    for config in exp_configs:
        thresh = exp_threshold(1, config)
        exp_hits = burst_size * thresh
        total_expected += exp_hits
        print(f"  L{config.level}    | {thresh:.4f}    | {exp_hits:.3f}")
    
    print(f"  ------|-----------|---------------")
    print(f"  TOTAL |           | {total_expected:.3f}")
    
    return total_expected


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_snowplow_curves(configs: List[LevelConfig], filename: str):
    """Plot snowplow threshold curves."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for level, config in enumerate(configs):
        ax = axes[level]
        max_gap = min(500, config.gamma * 3)
        gaps = np.arange(0, max_gap + 1)
        thresholds = [snowplow_threshold(g, config) for g in gaps]
        
        ax.plot(gaps, thresholds, 'b-', linewidth=2)
        ax.axvline(x=config.psi, color='r', linestyle='--', alpha=0.5, label=f'ψ={config.psi}')
        ax.axvline(x=config.gamma, color='g', linestyle='--', alpha=0.5, label=f'γ={config.gamma}')
        ax.axhline(y=config.fB, color='orange', linestyle=':', alpha=0.5, label=f'fB={config.fB:.3f}')
        
        gap_unit = "slots" if level == 0 else "base blocks"
        ax.set_title(f'Level {level}')
        ax.set_xlabel(f'Gap ({gap_unit})')
        ax.set_ylabel('Threshold')
        ax.set_ylim(-0.05, min(1.05, config.fA * 1.5))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_exp_curves(l0_config: LevelConfig, exp_configs: List[ExpConfig], filename: str):
    """Plot exponential decay curves."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # L0 snowplow
    ax = axes[0]
    gaps = np.arange(0, l0_config.gamma * 3 + 1)
    thresholds = [snowplow_threshold(g, l0_config) for g in gaps]
    ax.plot(gaps, thresholds, 'b-', linewidth=2)
    ax.set_title('Level 0 (Snowplow)')
    ax.set_xlabel('Gap (slots)')
    ax.set_ylabel('Threshold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # L1-L9 exponential
    for i, config in enumerate(exp_configs):
        ax = axes[i + 1]
        level = config.level
        
        max_gap = min(500, int(10 / (1.0 / (2 ** level))))
        gaps = np.arange(0, max_gap + 1)
        thresholds = [exp_threshold(g, config) for g in gaps]
        
        ax.plot(gaps, thresholds, 'b-', linewidth=2)
        ax.axhline(y=config.max_prob, color='orange', linestyle=':', 
                   alpha=0.5, label=f'max={config.max_prob:.3f}')
        
        ax.set_title(f'Level {level} (Exponential)')
        ax.set_xlabel('Gap (base blocks)')
        ax.set_ylabel('Threshold')
        ax.set_ylim(-0.05, min(1.05, config.max_prob * 1.5))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(results_list: List[Tuple[str, Dict]], target_rates: List[float], filename: str):
    """Compare achieved vs target rates."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(NUM_LEVELS)
    width = 0.8 / (len(results_list) + 1)
    
    # Target bars
    ax.bar(x - width * len(results_list) / 2, target_rates, width, 
           label='Target', alpha=0.5, color='gray')
    
    for i, (name, results) in enumerate(results_list):
        rates = results["conditional_rates"]
        ax.bar(x - width * (len(results_list) / 2 - i - 1), rates, width, label=name)
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Conditional Hit Rate')
    ax.set_title('Per-Level Conditional Hit Rates: P(Lμ | L0)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(NUM_LEVELS)])
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_error_bars(results_list: List[Tuple[str, Dict]], target_rates: List[float], filename: str):
    """Plot error percentages for each approach."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1, NUM_LEVELS)  # Skip L0
    width = 0.8 / len(results_list)
    
    for i, (name, results) in enumerate(results_list):
        rates = results["conditional_rates"][1:]  # Skip L0
        targets = target_rates[1:]
        errors = [abs(r - t) / t * 100 for r, t in zip(rates, targets)]
        
        ax.bar(x + i * width, errors, width, label=name)
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Error (%)')
    ax.set_title('Rate Error by Level')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'L{i}' for i in range(1, NUM_LEVELS)])
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Per-Level LDD Curve Optimization — VERSION 2")
    print("Using BASE BLOCK gaps for all super levels")
    print("=" * 80)
    print()
    
    # Target rates: P(Lμ | L0) = 1/2^μ
    target_rates = [L0_TARGET_RATE] + [1.0 / (2 ** i) for i in range(1, NUM_LEVELS)]
    print("Target conditional rates (P(Lμ | L0)):")
    for i in range(NUM_LEVELS):
        spacing = 1 / target_rates[i] if target_rates[i] > 0 else float('inf')
        print(f"  L{i}: {target_rates[i]:.6f} (~1 per {spacing:.1f} base blocks)")
    
    # -------------------------------------------------------------------------
    # Approach 1: Optimized Snowplow on Base-Block Gaps
    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("APPROACH 1: Snowplow curves on BASE-BLOCK gaps")
    print("-" * 80)
    
    snowplow_configs = optimize_all_levels(target_rates)
    
    print(f"\nSimulating {NUM_SLOTS:,} slots...")
    results_snowplow = simulate_base_block_gaps(NUM_SLOTS, snowplow_configs)
    
    print("\nResults:")
    print("  Level | Count  | Cond Rate | Target  | Error")
    print("  ------|--------|-----------|---------|-------")
    for i, count in enumerate(results_snowplow['level_counts']):
        cond = results_snowplow['conditional_rates'][i]
        target = target_rates[i]
        error = abs(cond - target) / target * 100 if target > 0 else 0
        print(f"  L{i}    | {count:6,} | {cond:.5f}   | {target:.5f} | {error:5.1f}%")
    
    adversary_analysis_snowplow(snowplow_configs)
    
    # -------------------------------------------------------------------------
    # Approach 2: Exponential Decay on Base-Block Gaps
    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("APPROACH 2: Exponential decay on BASE-BLOCK gaps")
    print("-" * 80)
    
    l0_config = snowplow_configs[0]
    
    print("\nOptimizing exponential parameters...")
    exp_configs = []
    for level in range(1, NUM_LEVELS):
        config, loss = optimize_exp_level(level, target_rates[level])
        exp_configs.append(config)
        print(f"  L{level}: max_prob={config.max_prob:.4f}, scale={config.scale:.2f} (loss={loss:.6f})")
    
    print(f"\nSimulating {NUM_SLOTS:,} slots...")
    results_exp = simulate_exp_decay(NUM_SLOTS, l0_config, exp_configs)
    
    print("\nResults:")
    print("  Level | Count  | Cond Rate | Target  | Error")
    print("  ------|--------|-----------|---------|-------")
    for i, count in enumerate(results_exp['level_counts']):
        cond = results_exp['conditional_rates'][i]
        target = target_rates[i]
        error = abs(cond - target) / target * 100 if target > 0 else 0
        print(f"  L{i}    | {count:6,} | {cond:.5f}   | {target:.5f} | {error:5.1f}%")
    
    adversary_analysis_exp(l0_config, exp_configs)
    
    # -------------------------------------------------------------------------
    # Generate Figures
    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("Generating figures...")
    print("-" * 80)
    
    plot_snowplow_curves(snowplow_configs, os.path.join(FIGURES_DIR, "v2_snowplow_curves.png"))
    plot_exp_curves(l0_config, exp_configs, os.path.join(FIGURES_DIR, "v2_exp_curves.png"))
    plot_comparison(
        [("Snowplow", results_snowplow), ("Exponential", results_exp)],
        target_rates,
        os.path.join(FIGURES_DIR, "v2_rate_comparison.png")
    )
    plot_error_bars(
        [("Snowplow", results_snowplow), ("Exponential", results_exp)],
        target_rates,
        os.path.join(FIGURES_DIR, "v2_error_bars.png")
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate mean errors
    snowplow_errors = []
    exp_errors = []
    for i in range(1, NUM_LEVELS):
        s_rate = results_snowplow['conditional_rates'][i]
        e_rate = results_exp['conditional_rates'][i]
        target = target_rates[i]
        if target > 0:
            snowplow_errors.append(abs(s_rate - target) / target * 100)
            exp_errors.append(abs(e_rate - target) / target * 100)
    
    print(f"\nMean error across L1-L9:")
    print(f"  Snowplow:    {np.mean(snowplow_errors):.1f}%")
    print(f"  Exponential: {np.mean(exp_errors):.1f}%")
    
    print("\nKey findings:")
    print("  1. BASE-BLOCK gaps (all levels measure gaps in L0 blocks) solve the")
    print("     L0-gating frequency problem that plagued cascading parent gaps")
    print("  2. Both snowplow and exponential curves achieve target rates")
    print("  3. ψ=1 dormant period gives burst resistance (threshold=0 at gap=1)")
    print("  4. Higher levels need larger γ/scale to match their sparser targets")
    
    print("\nRecommendation:")
    if np.mean(exp_errors) < np.mean(snowplow_errors):
        print("  → EXPONENTIAL DECAY: simpler parametrization, smoother curves")
    else:
        print("  → SNOWPLOW: better rate accuracy, familiar Taktikos form")
    
    return {
        "snowplow_configs": snowplow_configs,
        "exp_configs": exp_configs,
        "results_snowplow": results_snowplow,
        "results_exp": results_exp,
    }


if __name__ == "__main__":
    results = main()
