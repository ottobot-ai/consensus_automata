"""
Extract simulation data from a Taktikos run for paper figures.
Run the simulation first and pipe output here, or use pre-collected data.
"""

# Pre-collected from our 10k slot run
# Heights at each 1000-slot milestone
milestones = {
    1000:  [144, 70, 33, 16, 9, 4, 2, 1, 0, 0],
    2000:  [294, 148, 68, 33, 25, 8, 2, 1, 2, 0],
    3000:  [439, 230, 106, 45, 34, 16, 4, 1, 2, 0],
    4000:  [588, 297, 136, 70, 39, 18, 7, 1, 2, 0],
    5000:  [735, 372, 161, 79, 48, 27, 8, 3, 3, 0],
    6000:  [895, 459, 210, 106, 59, 29, 8, 5, 4, 0],
    7000:  [1045, 531, 255, 128, 68, 34, 9, 6, 4, 0],
    8000:  [1181, 603, 295, 143, 80, 35, 13, 8, 5, 0],
    9000:  [1297, 656, 317, 163, 87, 42, 17, 8, 6, 0],
    10000: [1434, 730, 351, 174, 99, 48, 19, 8, 6, 0],
}

# Final state
final_heights = milestones[10000]
level_names = [f"L{i}" for i in range(10)]
target_gaps = [7 * (2**i) for i in range(10)]
conditional_probs = [1.0 / (2**i) for i in range(10)]

import json
with open("simulation_data.json", "w") as f:
    json.dump({
        "milestones": {str(k): v for k, v in milestones.items()},
        "final_heights": final_heights,
        "level_names": level_names,
        "target_gaps": target_gaps,
        "conditional_probs": conditional_probs,
        "total_slots": 10000,
        "config": {
            "num_stakers": 5,
            "stake_distribution": [3000, 2500, 2000, 1500, 1000],
            "ldd_cutoff": 15,
            "amplitude": 0.5,
            "baseline": 0.05,
            "slots_per_epoch": 100
        }
    }, f, indent=2)

print("Data written to simulation_data.json")
