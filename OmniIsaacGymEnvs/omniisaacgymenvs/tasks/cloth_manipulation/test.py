import torch
import math
import numpy as np


if __name__=="__main__":
    reward = np.array([1,2,3,4])
    ppo_device = np.array([0, 0, 0, 0])
    torch.from_numpy(reward).to(ppo_device)
    print("reward = ", reward)
    print("-ppo_device = ", ppo_device)