import torch

# 加载 .pth 文件
model_path = '/home/ruiqiang/workspace/isaac_sim_cloth/success_training_experiment/traning_with_0.02_3/FrankaClothManipulation.pth'
checkpoint = torch.load(model_path)
print(checkpoint.keys())

# 提取 epoch 和 loss
epoch = checkpoint.get('epoch', None)
model = checkpoint.get('model', None)
frame = checkpoint.get('frame', None)
optimizer = checkpoint.get('optimizer', None)
last_mean_rewards = checkpoint.get('last_mean_rewards', None)
env_state = checkpoint.get('env_state', None)
scaler = checkpoint.get('scaler', None)

# 检查并打印 epoch 和 loss
if epoch is not None:
    print(f"Epoch: {epoch}")
else:
    print("Epoch not found in the checkpoint.")

# if model is not None:
#     print(f"model: {model}")

if frame is not None:
    print(f"frame: {frame}")

if scaler is not None:
    print(f"scaler: {scaler}")

# if optimizer is not None:
#     print(f"optimizer: {optimizer}")

if env_state is not None:
    print(f"env_state: {env_state}")

if last_mean_rewards is not None:
    print(f"last_mean_rewards: {last_mean_rewards}")