import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

POINT_COLORS = [
    'red', 'crimson', 'tomato', 'blue', 'royalblue', 'deepskyblue',
    'gold', 'orange', 'darkorange', 'purple', 'mediumpurple', 'orchid',
    'brown', 'chocolate', 'sienna', 'hotpink', 'deeppink'
]

def plot_image_distribution(N, csv_file, step_deg=1, seed=None):
    if seed is not None:
        random.seed(seed)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    df = pd.read_csv(csv_file)
    point_names = df.iloc[:, 0].astype(str).values
    azimuth     = df.iloc[:, 4].astype(float).values
    elevation   = df.iloc[:, 5].astype(float).values

    distances = 90.0 - elevation
    angles_deg = azimuth.astype(float) % 360.0
    delta = 360.0 / float(N)

    def count_points_in_regions(start_angle_deg):
        counts = np.zeros(N, dtype=int)
        for ang in angles_deg:
            adjusted = (ang - start_angle_deg) % 360.0
            region = int(np.floor(adjusted / delta))
            if region == N:
                region = 0
            counts[region] += 1
        return counts

    best_start_angle, best_std = 0.0, float('inf')
    best_region_counts = None
    for start in np.arange(0.0, 360.0, step_deg):
        rc = count_points_in_regions(start)
        std = np.std(rc)
        if std < best_std:
            best_std = std
            best_start_angle = float(start)
            best_region_counts = rc

    print(f"最佳起始角度: {best_start_angle:.0f}°")
    print(f"每个区间的点数分布: {best_region_counts}")

    def region_color_by_count(cnt):
        palette = ['honeydew', 'palegreen', 'lightgreen', 'mediumseagreen',
                   'seagreen', 'forestgreen', 'green', 'darkgreen']
        idx = int(cnt)
        return palette[min(idx, len(palette)-1)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    r_max = float(np.max(distances)) if len(distances) else 1.0

    for i in range(N):
        start_deg = (best_start_angle + i * delta) % 360.0
        color = region_color_by_count(best_region_counts[i])
        ax.bar(
            np.deg2rad(start_deg),
            [r_max],
            width=np.deg2rad(delta),
            bottom=0.0,
            color=color,
            alpha=0.6,
            align='edge'
        )

    for i, ang in enumerate(angles_deg):
        point_color = random.choice(POINT_COLORS)
        ax.scatter(np.deg2rad(ang), distances[i], s=80, color=point_color)
        ax.text(np.deg2rad(ang), distances[i], point_names[i], fontsize=12, ha='right')

    tick_degs = (best_start_angle + np.arange(N) * delta) % 360.0
    ax.set_xticks(np.deg2rad(tick_degs))
    xtl = [f"{int(t)%360}–{int((t+delta)%360)}°" for t in tick_degs]
    ax.set_xticklabels(xtl, fontsize=12)

    ax.tick_params(axis='both', labelsize=12)
    ax.set_rlabel_position(0)
    ax.grid(True)
    ax.set_title(f'Image Distribution After Interval Adjustment (N={N})', fontsize=16, fontweight='bold')
    plt.show()

    return best_start_angle, best_region_counts
