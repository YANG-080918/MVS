import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations, product
import random

def select_best_combination(ad, csv_file):
    df = pd.read_csv(csv_file)
    df['TLCTime'] = pd.to_datetime(df['TLCTime'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    point_names = df['Name']
    in_track_angles = df['meanInTrackViewAngle']
    cross_track_angles = df['meanCrossTrackViewAngle']
    mean_sun_az = df['meanSunAz']
    mean_sun_el = df['meanSunEl']
    mean_sat_az = df['meanSatAz']
    mean_sat_el = df['meanSatEl']
    dates = df['TLCTime']

    angles = np.arctan2(cross_track_angles, in_track_angles)
    distances = np.sqrt(in_track_angles**2 + cross_track_angles**2)
    angles_deg = np.degrees(angles)
    angles_deg = np.mod(angles_deg, 360)

    def count_points_in_regions(start_angle, angles_deg, point_names):
        region_counts = np.zeros(ad)
        region_points = {i: [] for i in range(ad)}
        for idx, angle in enumerate(angles_deg):
            adjusted_angle = (angle - start_angle) % 360
            region = int(adjusted_angle // (360 // ad))
            region_counts[region] += 1
            region_points[region].append(point_names.iloc[idx])
        return region_counts, region_points

    best_start_angle = 0
    best_std_dev = float('inf')
    best_region_counts = None
    best_region_points = None

    for start_angle in range(0, 360, 1):
        region_counts, region_points = count_points_in_regions(start_angle, angles_deg, point_names)
        std_dev = np.std(region_counts)
        if std_dev < best_std_dev:
            best_std_dev = std_dev
            best_start_angle = start_angle
            best_region_counts = region_counts
            best_region_points = region_points

    print(f"每个区间的点数分布: {best_region_counts}")

    def calculate_view_vector(in_track_angle, cross_track_angle):
        in_track_rad = np.radians(in_track_angle)
        cross_track_rad = np.radians(cross_track_angle)
        v_x = np.cos(in_track_rad) * np.cos(cross_track_rad)
        v_y = np.sin(in_track_rad) * np.cos(cross_track_rad)
        v_z = np.sin(cross_track_rad)
        return np.array([v_x, v_y, v_z])

    def calculate_intersection_angle(angle1, angle2):
        v1 = calculate_view_vector(angle1[0], angle1[1])
        v2 = calculate_view_vector(angle2[0], angle2[1])
        dot_product = np.dot(v1, v2)
        intersection_angle_rad = np.arccos(dot_product)
        intersection_angle_deg = np.degrees(intersection_angle_rad)
        return intersection_angle_deg

    def calculate_sun_diff(angle):
        sun_diff = []
        for angle1, angle2 in combinations(angle, 2):
            sundiff = np.sqrt((angle1[0] - angle2[0])**2 + (angle1[1] - angle2[1])**2)
            sun_diff.append(sundiff)
        return np.mean(sun_diff)

    def calculate_time_diff_std(dates):
        time_diffs = []
        for date1, date2 in combinations(dates, 2):
            diff = abs((date1 - date2).days)
            if 182 < diff < 365:
                actual_diff = 365 - diff
            elif diff <= 182:
                actual_diff = diff
            elif 182 < (diff % 365) < 365:
                actual_diff = 365 - (diff % 365)
            elif (diff % 365) <= 182:
                actual_diff = diff % 365
            time_diffs.append(actual_diff)
        return np.mean(time_diffs)

    def calculate_sunEI_average(selected_data):
        sunEI_sum = sum([90 - value for value in selected_data])
        sunEI_average = sunEI_sum / len(selected_data)
        return sunEI_average

    def select_optimal_points(best_region_points, num_regions=ad):
        count1 = []
        count2 = []
        time_count = []
        sunEI_count = []
        sundiff_count = []
        selected_points_count = []
        region_combinations = list(product(*[best_region_points[i] for i in range(num_regions) if len(best_region_points[i]) > 0]))
        print(len(region_combinations))
        for selected_points in region_combinations:
            selected_data = df.loc[df['Name'].isin(selected_points), ['meanInTrackViewAngle', 'meanCrossTrackViewAngle']]
            selected_angles = list(zip(selected_data['meanInTrackViewAngle'], selected_data['meanCrossTrackViewAngle']))
            selected_data1 = df.loc[df['Name'].isin(selected_points), ['meanSunAz', 'meanSunEl']]
            selected_sun_angles = list(zip(selected_data1['meanSunAz'], selected_data1['meanSunEl']))
            selected_dates = df[df['Name'].isin(selected_points)]['TLCTime'].tolist()
            selected_sunEI = df[df['Name'].isin(selected_points)]['meanSunEl'].tolist()
            selected_sunAZ = df[df['Name'].isin(selected_points)]['meanSunAz'].tolist()
            intersection_angles = [calculate_intersection_angle(a1, a2) for a1, a2 in combinations(selected_angles, 2)]
            sun_diff = calculate_sun_diff(selected_sun_angles)
            count_5_35 = sum(5 <= angle <= 35 for angle in intersection_angles)
            count_15_25 = sum(15 <= angle <= 25 for angle in intersection_angles)
            time_diff_std = calculate_time_diff_std(selected_dates)
            sunEI_average = calculate_sunEI_average(selected_sunEI)
            count1.append(count_5_35)
            count2.append(count_15_25)
            time_count.append(time_diff_std)
            sunEI_count.append(sunEI_average)
            sundiff_count.append(sun_diff)
            selected_points_count.append(selected_points)
        return count1, count2, time_count, sunEI_count, sundiff_count, selected_points_count

    def optimize_combinations(combinations_data):
        N1 = round(0.05 * len(combinations_data['sun_diff']))
        N2 = round(0.2 * N1)
        w1, w2 = 0.6, 0.4
        combinations_data['angle_score'] = w1 * combinations_data['count_5_35'] + w2 * combinations_data['count_15_25']
        top_N1_combinations = combinations_data.nlargest(N1, 'angle_score')
        w11, w22 = 0.6, 0.4
        top_N1_combinations['sun_score'] = w11 * top_N1_combinations['sun_diff'] + w22 * top_N1_combinations['sunEI_average']
        top_N1_combinations['time_score'] = top_N1_combinations['time_diff_std']
        top_N2_combinations_sun = top_N1_combinations.nsmallest(N2, 'sun_score')
        top_N2_combinations_time = top_N1_combinations.nsmallest(N2, 'time_score')
        return top_N2_combinations_sun, top_N2_combinations_time, top_N1_combinations

    count1, count2, time_count, sunEI_count, sundiff_count, selected_points_count = select_optimal_points(best_region_points, num_regions=ad)
    combinations_data = pd.DataFrame({
        'selected_points': selected_points_count,
        'count_5_35': count1,
        'count_15_25': count2,
        'sunEI_average': sunEI_count,
        'sun_diff': sundiff_count,
        'time_diff_std': time_count
    })

    top_N2_combinations_sun, top_N2_combinations_time, top_N1_combinations = optimize_combinations(combinations_data)
    common_rows = pd.merge(top_N2_combinations_sun, top_N2_combinations_time,
                           on=['selected_points', 'count_5_35', 'count_15_25', 'sunEI_average', 'sun_diff', 'time_diff_std'])
    df1 = pd.DataFrame(common_rows)

    w11, w22 = 0.6, 0.4
    if not df1.empty and len(df1) == 1:
        best_combination = df1.iloc[0]['selected_points']
    elif not df1.empty:
        df1['count_sun'] = w11 * df1['sun_diff'] + w22 * df1['sunEI_average']
        min_count_sun = df1['count_sun'].min()
        min_count_rows = df1[df1['count_sun'] == min_count_sun]
        best_combination = min_count_rows.iloc[0]['selected_points']
    else:
        top_N1_combinations['count_sun'] = w11 * top_N1_combinations['sun_diff'] + w22 * top_N1_combinations['sunEI_average']
        min_count_sun1 = top_N1_combinations['count_sun'].min()
        min_count_rows1 = top_N1_combinations[top_N1_combinations['count_sun'] == min_count_sun1]
        best_combination = min_count_rows1.iloc[0]['selected_points']

    print("Selected Points:", best_combination)
    return best_combination
