#!/usr/bin/env python3
"""
Weight-Based Consistency Bound Analysis for Superblock Chain Selection.

Adapts the Taktikos consistency bound analysis to weighted chain selection.
The key insight: with weighted chain selection, the relevant metric is not
blocks per slot but WEIGHT per slot.

Weight function:
    block_weight = threshold(slot_gap)^α × (1 + Σ 2^μ × level_hit_μ)

Where:
    - threshold(slot_gap) = snowplow f(δ) evaluated at the block's slot gap
    - α = weight exponent (default 2.0)
    - level_hit_μ = 1 if block hits super level μ
    - Super level thresholds use shifted exponential: θ_μ(g) = maxProb × (1 - exp(-(g-ψ)/scale))
    - g = gap in base blocks since last level-μ hit
    - ψ = 1 dormant period

An adversary burst-forging (small slot gaps) gets:
    1. Low threshold^α (snowplow gives low f for small δ)
    2. Low super-level hit probability (gating suppresses super hits)

Compares:
    - Original Taktikos: block_frequency intersection
    - Superblock weighted: weight_frequency intersection

Output: 3-panel figure with consistency bounds.
"""
import sys
import os
import numpy as np

# Use Agg backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Core functions adapted from relative_forging_power.py
# -----------------------------------------------------------------------------

# Truncation error for distribution summation
TRUNC_ERROR = 1.0e-6
MAX_ITER = 10000

# Super level parameters (tuned shifted exponential)
# Format: level -> (maxProb, scale)
SUPER_LEVEL_PARAMS = {
    1: (0.99, 0.1),
    2: (0.51, 2.0),
    3: (0.32, 7.9),
    4: (0.25, 30.3),
    5: (0.08, 27.5),
    6: (0.03, 40.5),
    7: (0.013, 56.0),
    8: (0.004, 64.0),
    9: (0.002, 72.0),
}

PSI = 1  # Dormant period for super levels


def f(delta: int, gamma: int, slot_gap: int, fa: float, fb: float) -> float:
    """
    Snowplow difficulty curve.
    
    Parameters:
        delta: slot interval since last block
        gamma: forging window cutoff
        slot_gap: initial dormant period (ψ)
        fa: amplitude
        fb: baseline (difficulty after cutoff)
    
    Returns:
        f(δ) ∈ [0, 1]
    """
    if delta <= slot_gap:
        return 0.0
    elif delta <= gamma:
        # Linear ramp from 0 to fa over [slot_gap, gamma]
        return min(1.0, fa * float(delta - slot_gap) / float(gamma - slot_gap))
    else:
        return fb


def forge_power(r: float, delta: int, gamma: int, slot_gap: int, fa: float, fb: float) -> float:
    """
    Forging power φ(δ, r) = probability of forging given slot gap δ and stake r.
    
    φ(δ, r) = 1 - (1 - f(δ))^r
    """
    b = max(1.0 - f(delta, gamma, slot_gap, fa, fb), 0.0)
    return 1.0 - np.power(b, r)


def pi_acc(d: int, r: float, gamma: int, slot_gap: int, fa: float, fb: float, 
           delay: int, acc: float) -> float:
    """
    Accumulator for stationary distribution π.
    
    π(d) = ∏_{i=delay+1}^{d-1} (1 - φ(i, r)) normalized
    """
    if d == 1:
        return 1.0
    else:
        if d > delay + 1:
            b1 = max(1.0 - f(d - 1, gamma, slot_gap, fa, fb), 0.0)
            return np.power(b1, r) * acc
        else:
            return acc


def pdf(d_axis: np.ndarray, r: float, gamma: int, slot_gap: int, fa: float, 
        fb: float, delay: int):
    """
    Compute slot-gap distribution for a staker with stake r.
    
    Returns:
        pdf_ext: probability density of producing a block at gap d
        pi: stationary distribution of being at gap d
    """
    i = 0
    done = False
    accumulation = 1.0
    pi = []
    pdf_ext = []
    
    while not done:
        accumulation = pi_acc(i, r, gamma, slot_gap, fa, fb, delay, accumulation)
        if TRUNC_ERROR > accumulation >= 0.0 or i == MAX_ITER:
            done = True
        else:
            pi.append(accumulation)
            i = i + 1
    
    d = -1
    norm = sum(pi)
    if norm == 0.0:
        pi = np.array(pi)
    else:
        pi = np.asarray(pi) / norm
    
    for p in pi:
        d = d + 1
        if d > delay:
            pdf_ext.append(p * forge_power(r, d, gamma, slot_gap, fa, fb))
        else:
            pdf_ext.append(0.0)
    
    norm2 = sum(pdf_ext)
    if norm2 > 0:
        pdf_ext = np.asarray(pdf_ext) / norm2
    else:
        pdf_ext = np.asarray(pdf_ext)
    
    return (np.pad(pdf_ext[:len(d_axis)], [(0, max(len(d_axis)-len(pdf_ext), 0))], mode='constant'),
            np.pad(pi[:len(d_axis)], [(0, max(len(d_axis)-len(pi), 0))], mode='constant'))


def block_frequency(r: float, gamma: int, slot_gap: int, fa: float, 
                   fb: float, delay: int) -> float:
    """
    Compute expected block frequency (blocks per slot) for stake r.
    
    block_frequency = 1 / E[slot_gap]
    """
    res = 0.0
    if r > 0.0:
        i = 0
        done = False
        accumulation = 1.0
        pi = []
        pdf_ext = []
        
        while not done:
            accumulation = pi_acc(i, r, gamma, slot_gap, fa, fb, delay, accumulation)
            if TRUNC_ERROR > accumulation >= 0.0 or i == MAX_ITER:
                done = True
            else:
                pi.append(accumulation)
                i = i + 1
        
        d = -1
        norm = sum(pi)
        if norm == 0.0:
            return 0.0
        else:
            pi = np.asarray(pi) / norm
        
        for p in pi:
            d = d + 1
            if d > delay:
                pdf_ext.append(p * forge_power(r, d, gamma, slot_gap, fa, fb))
            else:
                pdf_ext.append(0.0)
        
        norm2 = sum(pdf_ext)
        if norm2 == 0.0:
            return 0.0
        else:
            pdf_ext = np.asarray(pdf_ext) / norm2
        
        d = -1
        for nd in pdf_ext:
            d = d + 1
            res = res + d * nd
        
        if res > 0.0:
            return 1.0 / res
        else:
            return 0.0
    else:
        return 0.0


# -----------------------------------------------------------------------------
# Super level probability functions
# -----------------------------------------------------------------------------

def super_level_prob(level: int, base_block_gap: int, slot_gap: int = None, 
                    slot_gap_cutoff: int = 15) -> float:
    """
    Probability of hitting super level μ given base-block gap.
    
    Uses shifted exponential: θ_μ(g) = maxProb × (1 - exp(-(g-ψ)/scale)) for g≥ψ, else 0
    
    Optionally applies slot-gap gating: min(1, slot_gap/cutoff) suppresses super hits
    for burst-forging adversaries.
    
    Parameters:
        level: super level (1-9)
        base_block_gap: gap in base blocks since last level-μ hit
        slot_gap: if provided, apply gating
        slot_gap_cutoff: cutoff for gating (default 15, matches gamma)
    
    Returns:
        P(hit level μ | base_block_gap, slot_gap)
    """
    if level not in SUPER_LEVEL_PARAMS:
        return 0.0
    
    max_prob, scale = SUPER_LEVEL_PARAMS[level]
    
    if base_block_gap < PSI:
        return 0.0
    
    # Shifted exponential
    prob = max_prob * (1.0 - np.exp(-(base_block_gap - PSI) / scale))
    
    # Apply slot-gap gating if specified
    if slot_gap is not None:
        gating = min(1.0, slot_gap / slot_gap_cutoff)
        prob = prob * gating
    
    return min(1.0, prob)


def expected_super_weight(base_block_gap: int, slot_gap: int, num_levels: int = 9,
                         slot_gap_cutoff: int = 15) -> float:
    """
    Compute expected super-level weight contribution: E[Σ 2^μ × level_hit_μ | gap].
    
    Parameters:
        base_block_gap: gap in base blocks (used for level thresholds)
        slot_gap: slot gap (used for gating)
        num_levels: number of super levels to consider
        slot_gap_cutoff: cutoff for slot-gap gating
    
    Returns:
        E[super weight contribution]
    """
    expected = 0.0
    for level in range(1, min(num_levels, 9) + 1):
        prob = super_level_prob(level, base_block_gap, slot_gap, slot_gap_cutoff)
        expected += prob * (2 ** level)
    return expected


# -----------------------------------------------------------------------------
# Weight frequency computation
# -----------------------------------------------------------------------------

def weight_frequency(r: float, gamma: int, slot_gap_param: int, fa: float, fb: float,
                    delay: int, alpha: float = 2.0, num_levels: int = 9,
                    slot_gap_cutoff: int = 15) -> float:
    """
    Compute expected cumulative weight per slot for a staker with stake r.
    
    The key insight: weight_frequency = E[block_weight] × block_frequency
    
    Block weight depends on:
    1. Slot gap weight: min(δ/γ, 1)^α — penalizes small slot gaps
       This caps at 1 for δ ≥ γ, giving honest parties full weight
       Adversaries with small gaps get (δ/γ)^α penalty
    2. Super level weight: (1 + Σ 2^μ × P(hit level μ))
    
    For an adversary with delay=0 (covert leader), they forge at small slot gaps,
    getting LOW slot-gap weights and LOW super-level probabilities.
    
    For honest parties with delay>0, they forge at larger slot gaps,
    getting HIGHER slot-gap weights and HIGHER super-level probabilities.
    
    Parameters:
        r: stake fraction
        gamma: forging window cutoff
        slot_gap_param: dormant period ψ for difficulty curve
        fa: amplitude
        fb: baseline
        delay: semi-synchronous delay Δ
        alpha: weight exponent for threshold
        num_levels: number of super levels
        slot_gap_cutoff: cutoff for slot-gap gating
    
    Returns:
        Expected weight per slot
    """
    if r <= 0.0:
        return 0.0
    
    # First, get the PDF of slot gaps conditioned on forging
    # Need to extend beyond delay + forging window to capture all forging events
    max_gap = max(gamma * 3, delay + gamma * 2 + 50)  # Ensure coverage
    delta_axis = np.arange(max_gap)
    pdf_ext, pi = pdf(delta_axis, r, gamma, slot_gap_param, fa, fb, delay)
    
    # pdf_ext[d] = P(forge at gap d | staker with stake r, delay)
    # This is already normalized
    
    # Compute expected block weight
    expected_block_weight = 0.0
    
    for d in range(len(pdf_ext)):
        if pdf_ext[d] <= 0:
            continue
        
        # Slot gap weight: min(δ/γ, 1)^α
        # This penalizes adversaries forging at small slot gaps
        # Blocks at δ ≥ γ get full weight (1.0)
        # Blocks at δ < γ get (δ/γ)^α < 1
        gap_ratio = min(1.0, d / gamma) if gamma > 0 else 1.0
        slot_gap_weight = gap_ratio ** alpha
        
        # Super level weight contribution
        # The adversary forging at small gaps gets low super-level hit probability
        # because of slot-gap gating: min(1, slot_gap/cutoff)
        super_weight = expected_super_weight(max(1, d), d, num_levels, slot_gap_cutoff)
        
        # Total block weight: slot_gap_weight × (1 + super_weight)
        # Base weight = slot_gap_weight (penalizes small gaps)
        # Super levels add bonus proportional to slot_gap_weight
        block_weight = slot_gap_weight * (1.0 + super_weight)
        
        # Weight contribution from this gap
        expected_block_weight += pdf_ext[d] * block_weight
    
    # Weight frequency = E[block_weight] × block_frequency
    bf = block_frequency(r, gamma, slot_gap_param, fa, fb, delay)
    
    return expected_block_weight * bf


# -----------------------------------------------------------------------------
# Consistency bound computation
# -----------------------------------------------------------------------------

def find_intersection(curve1, curve2, x_axis):
    """Find intersection point of two curves."""
    if len(curve1) == 0 or len(curve2) == 0:
        return 0.5, 0.0
    
    i = 0
    if curve1[0] - curve2[0] == 0.0:
        return x_axis[0], curve1[0]
    
    sign = (curve1[0] - curve2[0]) / abs(curve1[0] - curve2[0])
    
    for c1, c2 in zip(curve1, curve2):
        diff = c1 - c2
        if sign > 0.0 and diff > 0.0 or sign < 0.0 and diff < 0.0:
            i = i + 1
        else:
            if i + 1 < len(x_axis):
                # Linear interpolation for intersection
                x1, x2 = x_axis[i], x_axis[i + 1]
                y1_1, y1_2 = curve1[i], curve1[i + 1]
                y2_1, y2_2 = curve2[i], curve2[i + 1]
                
                # Solve for intersection
                denom = (y1_1 - y1_2) - (y2_1 - y2_2)
                if abs(denom) > 1e-10:
                    t = (y1_1 - y2_1) / denom
                    x = x1 + t * (x2 - x1)
                    y = y1_1 + t * (y1_2 - y1_1)
                    return x, y
                else:
                    return x_axis[i], curve1[i]
            else:
                return x_axis[-1], curve1[-1]
    
    return x_axis[-1], curve1[-1]


def pow_bound(x: float) -> float:
    """
    Proof-of-Work consistency bound approximation.
    
    From Ren 2019: r* = 1/2 + (2 - sqrt(x^2 + 4)) / (2x)
    """
    if x > 0.0:
        return 0.5 + (2.0 - np.sqrt(x * x + 4.0)) / (2.0 * x)
    else:
        return 0.5


def compute_consistency_bound_block_freq(r_axis: np.ndarray, gamma: int, slot_gap: int,
                                        fa: float, fb: float, delay: int) -> float:
    """
    Compute consistency bound using block frequency (Taktikos baseline).
    
    Find r* where block_freq(r*, Δ) = block_freq(1-r*, 0)
    
    Returns:
        1 - r* (adversarial stake threshold)
    """
    honest_freq = np.array([block_frequency(r, gamma, slot_gap, fa, fb, delay) for r in r_axis])
    adv_freq = np.array([block_frequency(1.0 - r, gamma, slot_gap, fa, fb, 0) for r in r_axis])
    
    inter_r, _ = find_intersection(adv_freq, honest_freq, r_axis)
    return 1.0 - inter_r


def compute_consistency_bound_weight_freq(r_axis: np.ndarray, gamma: int, slot_gap: int,
                                         fa: float, fb: float, delay: int,
                                         alpha: float = 2.0, num_levels: int = 9) -> float:
    """
    Compute consistency bound using weight frequency (superblock weighted).
    
    Find r* where weight_freq(r*, Δ) = weight_freq(1-r*, 0)
    
    Returns:
        1 - r* (adversarial stake threshold)
    """
    honest_wfreq = np.array([weight_frequency(r, gamma, slot_gap, fa, fb, delay, alpha, num_levels) 
                            for r in r_axis])
    adv_wfreq = np.array([weight_frequency(1.0 - r, gamma, slot_gap, fa, fb, 0, alpha, num_levels) 
                         for r in r_axis])
    
    inter_r, _ = find_intersection(adv_wfreq, honest_wfreq, r_axis)
    return 1.0 - inter_r


# -----------------------------------------------------------------------------
# Main analysis and plotting
# -----------------------------------------------------------------------------

def main():
    # Default parameters
    gamma = 15
    fa = 0.5
    fb = 0.05
    slot_gap = 0
    alpha = 2.0
    num_levels = 9
    
    # Axes
    r_axis = np.linspace(0.01, 0.99, 50)  # Avoid 0 and 1 for numerical stability
    delay_axis = np.arange(0, 51)
    
    print("=" * 70)
    print("Weight-Based Consistency Bound Analysis")
    print("=" * 70)
    print(f"Parameters: γ={gamma}, fa={fa}, fb={fb}, ψ={slot_gap}, α={alpha}, levels={num_levels}")
    print()
    
    # Compute block and weight frequencies at delay=0 for display
    print("Computing block/weight frequencies at Δ=0...")
    
    honest_bf_d0 = [block_frequency(r, gamma, slot_gap, fa, fb, 0) for r in r_axis]
    adv_bf_d0 = [block_frequency(1.0 - r, gamma, slot_gap, fa, fb, 0) for r in r_axis]
    
    honest_wf_d0 = [weight_frequency(r, gamma, slot_gap, fa, fb, 0, alpha, num_levels) for r in r_axis]
    adv_wf_d0 = [weight_frequency(1.0 - r, gamma, slot_gap, fa, fb, 0, alpha, num_levels) for r in r_axis]
    
    # Compute consistency bounds across delay values
    print("Computing consistency bounds across delay values...")
    
    block_freq_bounds = []
    weight_freq_bounds = []
    pow_bounds = []
    
    for delay in delay_axis:
        if delay % 10 == 0:
            print(f"  Delay = {delay}...")
        
        # Block frequency bound
        bf_bound = compute_consistency_bound_block_freq(r_axis, gamma, slot_gap, fa, fb, delay)
        block_freq_bounds.append(bf_bound)
        
        # Weight frequency bound
        wf_bound = compute_consistency_bound_weight_freq(r_axis, gamma, slot_gap, fa, fb, delay,
                                                        alpha, num_levels)
        weight_freq_bounds.append(wf_bound)
        
        # PoW bound (for comparison)
        # blocks per delay = f_eff * delay
        f_eff = block_frequency(1.0, gamma, slot_gap, fa, fb, 0)  # Max block rate
        pow_bounds.append(pow_bound(f_eff * delay))
    
    block_freq_bounds = np.array(block_freq_bounds)
    weight_freq_bounds = np.array(weight_freq_bounds)
    pow_bounds = np.array(pow_bounds)
    
    # Key finding at delay=0
    print()
    print("=" * 70)
    print("KEY FINDINGS at Δ=0:")
    print(f"  Block-frequency consistency bound: {block_freq_bounds[0]:.4f}")
    print(f"  Weight-frequency consistency bound: {weight_freq_bounds[0]:.4f}")
    print(f"  PoW consistency bound:              {pow_bounds[0]:.4f}")
    print()
    print(f"  Improvement (weight vs block): {(weight_freq_bounds[0] - block_freq_bounds[0]):.4f}")
    print(f"  Improvement (weight vs PoW):   {(weight_freq_bounds[0] - pow_bounds[0]):.4f}")
    print("=" * 70)
    
    # Generate 3-panel figure
    print("\nGenerating figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Weighted Chain Selection Consistency Bound Analysis\n'
                 f'(γ={gamma}, fa={fa}, fb={fb}, α={alpha})', fontsize=12)
    
    # Panel 1: Block frequency (Taktikos baseline)
    ax1 = axes[0]
    ax1.plot(r_axis, honest_bf_d0, 'b-', label='Honest (r)', linewidth=2)
    ax1.plot(r_axis, adv_bf_d0, 'r-', label='Adversary (1-r)', linewidth=2)
    
    # Mark intersection
    inter_r_bf, inter_y_bf = find_intersection(np.array(adv_bf_d0), np.array(honest_bf_d0), r_axis)
    ax1.plot(inter_r_bf, inter_y_bf, 'ko', markersize=8)
    ax1.axvline(inter_r_bf, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(f'r*={inter_r_bf:.3f}\nbound={1-inter_r_bf:.3f}', 
                xy=(inter_r_bf, inter_y_bf), xytext=(inter_r_bf + 0.15, inter_y_bf),
                fontsize=9, ha='left')
    
    ax1.set_xlabel('Stake Fraction (r)', fontsize=11)
    ax1.set_ylabel('Block Frequency (blocks/slot)', fontsize=11)
    ax1.set_title('Block Frequency (Taktikos)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(max(honest_bf_d0), max(adv_bf_d0)) * 1.1])
    
    # Panel 2: Weight frequency (superblock weighted)
    ax2 = axes[1]
    ax2.plot(r_axis, honest_wf_d0, 'b-', label='Honest (r)', linewidth=2)
    ax2.plot(r_axis, adv_wf_d0, 'r-', label='Adversary (1-r)', linewidth=2)
    
    # Mark intersection
    inter_r_wf, inter_y_wf = find_intersection(np.array(adv_wf_d0), np.array(honest_wf_d0), r_axis)
    ax2.plot(inter_r_wf, inter_y_wf, 'ko', markersize=8)
    ax2.axvline(inter_r_wf, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'r*={inter_r_wf:.3f}\nbound={1-inter_r_wf:.3f}', 
                xy=(inter_r_wf, inter_y_wf), xytext=(inter_r_wf + 0.15, inter_y_wf),
                fontsize=9, ha='left')
    
    ax2.set_xlabel('Stake Fraction (r)', fontsize=11)
    ax2.set_ylabel('Weight Frequency (weight/slot)', fontsize=11)
    ax2.set_title('Weight Frequency (Superblock)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, max(max(honest_wf_d0), max(adv_wf_d0)) * 1.1])
    
    # Panel 3: Consistency bounds vs delay
    ax3 = axes[2]
    f_eff = block_frequency(1.0, gamma, slot_gap, fa, fb, 0)
    blocks_per_delay = f_eff * delay_axis
    
    ax3.plot(blocks_per_delay, pow_bounds, 'k:', label='PoW', linewidth=2)
    ax3.plot(blocks_per_delay, block_freq_bounds, 'b-', label='Taktikos (block-freq)', linewidth=2)
    ax3.plot(blocks_per_delay, weight_freq_bounds, 'g-', label='Superblock (weight-freq)', linewidth=2)
    
    ax3.set_xlabel('Blocks per Delay (f_eff × Δ)', fontsize=11)
    ax3.set_ylabel('Consistency Bound', fontsize=11)
    ax3.set_title('Consistency Bound vs Network Delay', fontsize=11)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, blocks_per_delay[-1]])
    ax3.set_ylim([0, 0.55])
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'fig_weight_consistency.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also save PNG
    output_path_png = output_path.replace('.pdf', '.png')
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_png}")
    
    plt.close()
    
    # Print detailed table
    print("\nConsistency Bounds at Selected Delays:")
    print("-" * 60)
    print(f"{'Delay':>6} {'Blocks':>8} {'PoW':>10} {'Block-Freq':>12} {'Weight-Freq':>12}")
    print("-" * 60)
    for i, delay in enumerate([0, 5, 10, 15, 20, 30, 40, 50]):
        if delay <= 50:
            print(f"{delay:>6} {blocks_per_delay[delay]:>8.2f} {pow_bounds[delay]:>10.4f} "
                  f"{block_freq_bounds[delay]:>12.4f} {weight_freq_bounds[delay]:>12.4f}")
    print("-" * 60)
    
    return block_freq_bounds[0], weight_freq_bounds[0]


if __name__ == "__main__":
    bf_bound, wf_bound = main()
    print(f"\n{'='*70}")
    print("FINAL RESULT:")
    print(f"  Block-frequency consistency bound at Δ=0: {bf_bound:.4f}")
    print(f"  Weight-frequency consistency bound at Δ=0: {wf_bound:.4f}")
    print(f"{'='*70}")
