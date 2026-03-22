"""
Generate figures for the Taktikos Superblocks paper.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Load data
with open("../data/simulation_data.json") as f:
    data = json.load(f)

milestones = {int(k): v for k, v in data["milestones"].items()}
final_heights = data["final_heights"]
total_slots = data["total_slots"]

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ===================================================================
# Figure 1: LDD Snowplow Curve
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

psi = 0
gamma = 15
fA = 0.5
fB = 0.05

delta = np.arange(0, 30, 0.1)
f_delta = np.where(delta < psi, 0,
          np.where(delta < gamma, fA * (delta - psi) / (gamma - psi), fB))

ax.plot(delta, f_delta, 'b-', linewidth=2)
ax.axvline(x=gamma, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=fB, color='red', linestyle=':', alpha=0.5, label=f'$f_B = {fB}$')
ax.axhline(y=fA, color='green', linestyle=':', alpha=0.5, label=f'$f_A = {fA}$')

ax.fill_between(delta[delta < gamma], f_delta[delta < gamma], alpha=0.1, color='blue')
ax.fill_between(delta[delta >= gamma], f_delta[delta >= gamma], alpha=0.1, color='red')

ax.annotate('Ramp\n(forging window)', xy=(7, 0.15), fontsize=9, ha='center')
ax.annotate('Recovery\n(baseline)', xy=(22, 0.08), fontsize=9, ha='center')
ax.annotate(r'$\gamma$', xy=(gamma, -0.03), fontsize=11, ha='center')

ax.set_xlabel(r'Slot interval $\delta$')
ax.set_ylabel(r'Difficulty $f(\delta)$')
ax.set_title('Taktikos Snowplow Difficulty Curve (Level 0)')
ax.legend(loc='upper left')
ax.set_xlim(0, 29)
ax.set_ylim(-0.02, 0.55)
ax.grid(True, alpha=0.3)

plt.savefig('fig1_snowplow.png')
plt.savefig('fig1_snowplow.pdf')
plt.close()
print("Fig 1: Snowplow curve saved")

# ===================================================================
# Figure 2: Superblock Density Decay (bar chart)
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

levels = list(range(10))
hits = final_heights
expected = [total_slots * (1.0 / (2**i)) * 0.143 for i in range(10)]  # ~14.3% base rate

bars = ax.bar(levels, hits, color='steelblue', alpha=0.8, label='Observed hits')
ax.plot(levels, expected, 'ro-', linewidth=1.5, markersize=6, label='Expected ($N \\cdot f_0 / 2^\\mu$)')

for i, (h, e) in enumerate(zip(hits, expected)):
    if h > 0:
        ax.text(i, h + 15, str(h), ha='center', fontsize=8)

ax.set_xlabel('Superblock Level $\\mu$')
ax.set_ylabel('Number of Hits (10,000 slots)')
ax.set_title('Superblock Hit Rate by Level')
ax.set_xticks(levels)
ax.set_xticklabels([f'L{i}' for i in levels])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')
ax.set_ylim(0.5, 3000)

plt.savefig('fig2_density_decay.png')
plt.savefig('fig2_density_decay.pdf')
plt.close()
print("Fig 2: Density decay saved")

# ===================================================================
# Figure 3: Subchain Growth Over Time
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

slots = sorted(milestones.keys())
colors = plt.cm.viridis(np.linspace(0, 0.9, 10))

for level in range(10):
    heights = [milestones[s][level] for s in slots]
    if max(heights) > 0:
        label = f'L{level} (1/{2**level})' if level > 0 else 'L0 (base)'
        ax.plot(slots, heights, '-o', color=colors[level], linewidth=2,
                markersize=4, label=label)

ax.set_xlabel('Slot')
ax.set_ylabel('Subchain Height')
ax.set_title('Subchain Growth Over Time (10,000 slots)')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.savefig('fig3_subchain_growth.png')
plt.savefig('fig3_subchain_growth.pdf')
plt.close()
print("Fig 3: Subchain growth saved")

# ===================================================================
# Figure 4: Light Client Sync Protocol (conceptual)
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Light Client Synchronization Protocol', fontsize=13, pad=20)

# Draw chain levels
y_positions = [40, 32, 24, 16, 8]
level_labels = ['L0 (base)', 'L1 (1/2)', 'L2 (1/4)', 'L3 (1/8)', 'L4 (1/16)']
block_counts = [20, 10, 5, 3, 1]
colors_l = ['#4477AA', '#66AADD', '#88CCEE', '#AADDCC', '#44BB99']

for i, (y, label, n, c) in enumerate(zip(y_positions, level_labels, block_counts, colors_l)):
    ax.text(-2, y, label, fontsize=8, ha='right', va='center')
    spacing = 90 / max(n, 1)
    for j in range(n):
        x = 5 + j * spacing
        rect = FancyBboxPatch((x, y-1.5), 3, 3, boxstyle="round,pad=0.3",
                               facecolor=c, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        if j > 0:
            ax.annotate('', xy=(x, y), xytext=(x - spacing + 3, y),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Arrow showing sync direction
ax.annotate('', xy=(92, 45), xytext=(5, 45),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(48, 47, 'Sync direction (genesis → tip)', ha='center', fontsize=9)

# Labels
ax.text(50, 2, 'Light client downloads L4 skeleton first,\nthen fills in lower levels near tip',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('fig4_light_client.png')
plt.savefig('fig4_light_client.pdf')
plt.close()
print("Fig 4: Light client protocol saved")

# ===================================================================
# Figure 5: Block Header Structure
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')
ax.set_title('Block Header with Subchain State Vector', fontsize=13, pad=10)

# Main block header box
header = FancyBboxPatch((5, 25), 90, 30, boxstyle="round,pad=0.5",
                         facecolor='lightyellow', edgecolor='black', linewidth=1.5)
ax.add_patch(header)

# Fields
fields = [
    ('slot', '455'),
    ('parentHash', '0x8a7f...'),
    ('stakerVK', '0xd4d3...'),
    ('vrfProof', 'π₄₅₅'),
    ('stateRoot', '0x1b25...'),
]

y_start = 50
for i, (name, val) in enumerate(fields):
    x = 10 + (i % 3) * 30
    y = y_start - (i // 3) * 8
    ax.text(x, y, f'{name}: {val}', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

# Subchain state vector
ax.text(10, 33, 'subchainState:', fontsize=9, fontweight='bold')

levels_data = [
    ('L0', '455', '145', '0x61e1...', True),
    ('L1', '455', '73', '0x61e1...', True),
    ('L2', '450', '38', '0x0388...', False),
    ('L3', '420', '18', '0x6adc...', False),
    ('L4', '380', '10', '0x8048...', False),
]

for i, (level, slot, height, tip, hit) in enumerate(levels_data):
    x = 10 + i * 17
    color = '#44BB99' if hit else '#DDDDDD'
    rect = FancyBboxPatch((x, 27), 15, 5, boxstyle="round,pad=0.2",
                           facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x + 7.5, 29.5, f'{level}: ({slot},{height})', fontsize=6,
            ha='center', va='center', family='monospace')

# Legend
hit_patch = mpatches.Patch(facecolor='#44BB99', edgecolor='black', label='Hit this block')
miss_patch = mpatches.Patch(facecolor='#DDDDDD', edgecolor='black', label='Carried forward')
ax.legend(handles=[hit_patch, miss_patch], loc='lower right', fontsize=8)

# Annotation
ax.text(50, 22, 'Block at slot 455 hits L[0,1]. L2-L4 carry forward from parent.',
        ha='center', fontsize=9, style='italic')

plt.savefig('fig5_header_structure.png')
plt.savefig('fig5_header_structure.pdf')
plt.close()
print("Fig 5: Header structure saved")

# ===================================================================
# Figure 6: Adversary Cost Analysis
# ===================================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

levels = np.arange(0, 10)
honest_rate = 1434  # L0 blocks per 10k slots
adversary_cost_multiplier = 2.0 ** levels

ax.semilogy(levels, adversary_cost_multiplier, 'rs-', linewidth=2, markersize=8,
            label='Relative adversary cost')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Superblock Level $\\mu$')
ax.set_ylabel('Relative Cost to Forge Competing Superchain')
ax.set_title('Adversary Cost Scaling by Superblock Level')
ax.set_xticks(levels)
ax.set_xticklabels([f'L{i}' for i in levels])
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('fig6_adversary_cost.png')
plt.savefig('fig6_adversary_cost.pdf')
plt.close()
print("Fig 6: Adversary cost saved")

print("\nAll figures generated!")
