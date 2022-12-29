import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
import glob 
from tqdm import tqdm

csv_root = "/home/chohome/Master_research/LGSVL/data_analysis/scenario_5"
gan_npy_list = glob.glob(os.path.join(csv_root + "/gan/image" + "/ep62/*.npy"))
nogan_npy_list = glob.glob(os.path.join(csv_root + "/nogan/image" + "/ep72/*.npy"))
print("gan_npy_list : ", gan_npy_list)



frames = []
for npy_path in tqdm(nogan_npy_list):
    fig= plt.figure(figsize=(101, 101))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=90, azim=0)
    # x軸に補助目盛線を設定
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.3,
            linestyle = "--", linewidth = 1)
    # y軸に目盛線を設定
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.3,
            linestyle = "--", linewidth = 1)

    ax.set_xlabel("x", size = 16)
    ax.set_ylabel("y", size = 16)
    ax.set_zlabel("value", size = 16)
    # npy_path = gan_npy_list[56]
#     print("npy_path : ", npy_path)
    npy_data = np.load(npy_path, allow_pickle=True).item()
#     print("npy_data : ", npy_data)
    pot_all = npy_data["pot_all"]
    pot_all = np.clip(pot_all, 199.5, 201)
    route_grid = npy_data["route"]
    obstacle_grid = npy_data["obst_grid"]
#     print("pot_all : ", pot_all.shape)

    x = np.arange(-15, 15, 0.3)
    y = np.arange(-10, 20, 0.3)
#     print("x shape : ", x.shape)
    X, Y = np.meshgrid(x, y)
    Z = pot_all

    for grid in route_grid:
        if grid[0] != 0 and grid[1] != 0 and grid[0] != 100 and grid[1] != 100:
            grid_y = 100 - grid[0]
            x_pos, y_pos = x[int(grid[1])], y[int(grid_y)]
            z_value = pot_all[int(grid[1]), int(grid_y)] + 1
            ax.scatter(x_pos, y_pos, z_value, s=1000, c="red")
    
    for grid in obstacle_grid:
        if grid[0] != 0 and grid[1] != 0 and grid[0] != 100 and grid[1] != 100:
            grid_y = 100 - grid[0]
            x_pos, y_pos = x[int(grid[1])], y[int(grid_y)]
            z_value = pot_all[int(grid[1]), int(grid_y)] + 1
            ax.scatter(x_pos, y_pos, z_value, s=1000, c="blue")

    ax.plot_surface(X, Y, Z, cmap="summer", alpha=0.7)

    frame = ax.contour(X, Y, Z, colors="black", offset=10)
    plt.savefig(npy_path[:-4] + ".png")
    print("save png ", npy_path[:-4], ".png")
#     frames.append(frame.collections)

# ani = ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
# ani.save("sample.gif", writer="pillow")

# plt.show()

    


