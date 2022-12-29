import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
import glob 
from tqdm import tqdm

csv_root = "/home/chohome/Master_research/LGSVL/data_analysis/scenario_0"
gan_csv_list = glob.glob(os.path.join(csv_root + "/full_data/" + "/scenario_0_gan/*.csv"))
gan_csv_list.sort()
nogan_csv_list = glob.glob(os.path.join(csv_root +  "/full_data/" + "/scenario_0_nogan/*.csv"))
nogan_csv_list.sort()
# print("gan_csv_list", gan_csv_list)

sum_reward = []
sum_error2dist = []
sum_collision = []
sum_goal_dist = []
success_error2dist = []

dict_pandas = pd.DataFrame({"episode" : [0], 
                            "sum_reward" :[0], 
                            "sum_error2dist" : [0], 
                            "sum_collision" : [0], 
                            "sum_goal_dist" : [0]})
actor_pandas = pd.DataFrame({})

for i, csv_path in enumerate(tqdm(gan_csv_list)):
    if ("failure" in csv_path) or ("success" in csv_path):
        csv_data = pd.read_csv(csv_path, header=None, skiprows=1).to_numpy()
        # objectとの距離の計算
        # print("csv_data[:, 4] : ", csv_data[1:, 4])
        obj_dist = np.min(csv_data[1:, 4])
        sum_error2dist.append([i, obj_dist])
        if "success" in csv_path:
            success_error2dist.append(obj_dist)
        # rewardの平均を計算
        reward = np.mean(csv_data[:, 5])
        sum_reward.append([i, reward])
        # goalまでの距離を計算
        goal_dist = np.mean(csv_data[:, 17])
        sum_goal_dist.append([i, goal_dist])
        # collisionフラグの計算
        collision = np.sum(csv_data[:, 16])
        sum_collision.append([i, collision])
    else:
        csv_data = pd.read_csv(csv_path, header=None, skiprows=1).to_numpy()
        actor_loss = csv_data[:, -1]
        
        actor_loss_compensate = []
        for i, loss in enumerate(actor_loss):
            if i == 0:
                actor_loss_compensate.append(None)
            elif loss != 0.0:
                 actor_loss_compensate.append(loss)
            elif loss == 0.0:
                actor_loss_compensate.append(actor_loss_compensate[i-1])
        actor_pandas["actor_loss"] = actor_loss_compensate

    dict_pandas = dict_pandas.append({"episode" : i,
                                    "sum_reward" : reward, 
                                    "min_error2dist" : obj_dist, 
                                    "sum_collision" : collision, 
                                    "sum_goal_dist" : goal_dist}, ignore_index=True) 

sum_reward = np.array(sum_reward)
sum_error2dist = np.array(sum_error2dist)
success_error2dist = np.array(success_error2dist)
sum_collision = np.array(sum_collision)
sum_goal_dist = np.array(sum_goal_dist)

mean_object_distance = np.mean(success_error2dist)
dict_pandas["mean_object_distance"] = mean_object_distance
dict_pandas.to_csv(csv_root + "/summarize_gan.csv")
actor_pandas.to_csv(csv_root + "/summarize_gan_actor_loss.csv")
print("save to : ", csv_root + "/summarize_gan.csv")

# ------------------------------------------------------------------------------------------------------------------------------------
dict_pandas = pd.DataFrame({"episode" : [0], 
                            "sum_reward" :[0], 
                            "sum_error2dist" : [0], 
                            "sum_collision" : [0], 
                            "sum_goal_dist" : [0]})
actor_pandas = pd.DataFrame({})
sum_reward = []
sum_error2dist = []
sum_collision = []
sum_goal_dist = []
success_error2dist = []

for i, csv_path in enumerate(tqdm(nogan_csv_list[:-1])):
    if ("failure" in csv_path) or ("success" in csv_path):
        csv_data = pd.read_csv(csv_path, header=None, skiprows=1).to_numpy()
        # objectとの距離の計算
        obj_dist = np.min(csv_data[1:, 4])
        sum_error2dist.append([i, obj_dist])
        if "success" in csv_path:
            success_error2dist.append(obj_dist)
        # rewardの平均を計算
        reward = np.mean(csv_data[:, 5])
        sum_reward.append([i, reward])
        # goalまでの距離を計算
        goal_dist = np.mean(csv_data[:, 17])
        sum_goal_dist.append([i, goal_dist])
        # collisionフラグの計算
        collision = np.sum(csv_data[:, 16])
        sum_collision.append([i, collision])
    else:
        csv_data = pd.read_csv(csv_path, header=None, skiprows=1).to_numpy()
        actor_loss = csv_data[:, -1]
        
        actor_loss_compensate = []
        for i, loss in enumerate(actor_loss):
            if i == 0:
                actor_loss_compensate.append(None)
            elif loss != 0.0:
                 actor_loss_compensate.append(loss)
            elif loss == 0.0:
                actor_loss_compensate.append(actor_loss_compensate[i-1])
        actor_pandas["actor_loss"] = actor_loss_compensate

    dict_pandas = dict_pandas.append({"episode" : i,
                                    "sum_reward" : reward, 
                                    "min_error2dist" : obj_dist, 
                                    "sum_collision" : collision, 
                                    "sum_goal_dist" : goal_dist}, ignore_index=True)

sum_reward = np.array(sum_reward)
sum_error2dist = np.array(sum_error2dist)
success_error2dist = np.array(success_error2dist)
sum_collision = np.array(sum_collision)
sum_goal_dist = np.array(sum_goal_dist)

mean_object_distance = np.mean(success_error2dist)
dict_pandas["mean_object_distance"] = mean_object_distance

dict_pandas.to_csv(csv_root + "/summarize_nogan.csv")
actor_pandas.to_csv(csv_root + "/summarize_nogan_actor_loss.csv")
print("save to : ", csv_root + "/summarize_nogan.csv")

fig_actor_loss = plt.figure()
fig_critic_loss = plt.figure()
fig_discriminator_loss = plt.figure()
fig_sum_reward = plt.figure()
fig_sum_error2dist = plt.figure()
fig_sum_collision = plt.figure()
fig_sum_goal_dist = plt.figure()


ax4 = fig_sum_reward.add_subplot(7, 1, 4)
# ax4.set_aspect("equal")
ax4.plot(sum_reward[:, 0], sum_reward[:, 1], c = "blue")
ax4.set_xlabel("episode", size = 16)
ax4.set_ylabel("reward", size = 16)

ax5 = fig_sum_goal_dist.add_subplot(7, 1, 5)
# ax5.set_aspect("equal")
ax5.plot(sum_goal_dist[:, 0], sum_goal_dist[:, 1], c = "blue")
ax5.set_xlabel("episode", size = 16)
ax5.set_ylabel("distance [m]", size = 16)

ax6 = fig_sum_error2dist.add_subplot(7, 1, 6)
# ax6.set_aspect("equal")
ax6.plot(sum_error2dist[:, 0], sum_error2dist[:, 1], c = "blue")
ax6.set_xlabel("episode", size = 16)
ax6.set_ylabel("distance [m]", size = 16)

ax7 = fig_sum_collision.add_subplot(7, 1, 7)
# ax7.set_aspect("equal")
ax7.plot(sum_collision[:, 0], sum_collision[:, 1], c = "blue")
ax7.set_xlabel("episode", size = 16)
ax7.set_ylabel("reward", size = 16)



plt.tight_layout()
plt.show()


