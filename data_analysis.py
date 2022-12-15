import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 

csv_root = "/home/chohome/Master_research/LGSVL/data_analysis/scenario_0"

gan_csv_list = glob.glob(os.path.join(csv_root + "/gan" + "/*.csv"))
nogan_csv_list = glob.glob(os.path.join(csv_root + "/nogan" + "/*.csv"))
expert_path = os.path.join(csv_root + "/expert" + "/ver1.csv")

fig, ax = plt.subplots(figsize=(100, 100))
ax.set_aspect("equal")
# x軸に補助目盛線を設定
ax.grid(which = "major", axis = "x", color = "black", alpha = 0.3,
        linestyle = "--", linewidth = 1)
# y軸に目盛線を設定
ax.grid(which = "major", axis = "y", color = "black", alpha = 0.3,
        linestyle = "--", linewidth = 1)

# エキスパートデータ
expert_route = pd.read_csv(expert_path, header=None, skiprows=1).to_numpy()
degree = 2
d_rad = np.radians(degree)
expert_route[:, 0] = (expert_route[:, 0] - expert_route[0, 0]) * np.cos(d_rad) - (expert_route[:, 1] - expert_route[0, 1]) * np.sin(d_rad) + expert_route[0, 0]
expert_route[:, 1] = (expert_route[:, 0] - expert_route[0, 0]) * np.sin(d_rad) + (expert_route[:, 1] - expert_route[0, 1]) * np.cos(d_rad) + expert_route[0, 1]


plt.scatter(expert_route[:, 0], expert_route[:, 1], s=50, marker="x", linewidth=3, c="red")

global_gan_array = np.zeros(0)
for gan_csv in gan_csv_list:
    gan_array = pd.read_csv(gan_csv, header=None, skiprows=1).to_numpy()
    gan_points = np.zeros(0)
    for waypoint in expert_route:
        error_distance = np.sum(np.abs(gan_array[:, 1:3] - waypoint[:2]), axis=1)
        nearest_waypoints = np.unravel_index(np.argmin(error_distance), error_distance.shape) # 最小値の座標を取得
        gan_points = np.append(gan_points, gan_array[nearest_waypoints, 1:3])
    gan_array = gan_points.reshape(-1, 2)

    global_gan_array = np.append(global_gan_array, gan_array)

global_gan_array = global_gan_array.reshape(-1, expert_route.shape[0], 2)
mean_global_gan_array = np.mean(global_gan_array, axis=0)
print("mean_global_gan_array : \n", mean_global_gan_array.shape)
plt.scatter(mean_global_gan_array[:, 0], mean_global_gan_array[:, 1], s=50, marker="x", linewidth=3, c="green")

global_nogan_array = np.zeros(0)
for nogan_csv in nogan_csv_list:
    nogan_array = pd.read_csv(nogan_csv, header=None, skiprows=1).to_numpy()
    error_coord = np.where(nogan_array[:, 2] > -8.0)
    nogan_array = np.delete(nogan_array, error_coord, 0)

    nogan_points = np.zeros(0)
    for waypoint in expert_route:
        error_distance = np.sum(np.abs(nogan_array[:, 1:3] - waypoint[:2]), axis=1)
        nearest_waypoints = np.unravel_index(np.argmin(error_distance), error_distance.shape) # 最小値の座標を取得
        nogan_points = np.append(nogan_points, nogan_array[nearest_waypoints, 1:3])
    nogan_array = nogan_points.reshape(-1, 2)
    global_nogan_array = np.append(global_nogan_array, nogan_array)

global_nogan_array = global_nogan_array.reshape(-1, expert_route.shape[0], 2)
mean_global_nogan_array = np.mean(global_nogan_array, axis=0)
print("mean_global_nogan_array : ", mean_global_nogan_array.shape)
plt.scatter(mean_global_nogan_array[:, 0], mean_global_nogan_array[:, 1], s=50, marker="x", linewidth=3, c="blue")

plt.show()


    
