# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from datetime import datetime

import numpy as np
import torch
# from omni.isaac.gym.vec_env import VecEnvBase


# VecEnv Wrapper for RL training
class VecEnvRealWorld():
    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.device)
        self._rew = self._rew.to(self._task.device)
        # self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.device)
        self._resets = self._resets.to(self._task.device)
        self._extras = self._extras

    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True, rendering_dt=1.0 / 60.0) -> None:
        # super().set_task(task, backend, sim_params, init_sim, rendering_dt)
        self.observation_space = task.observation_space
        self.action_space = task.action_space
        self._task = task

        # self.num_states = task.num_states
        # self.state_space = task.state_space

    def step(self, actions):

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device)

        # print("----------------------start-----------------------------------------")
        self._task.pre_physics_step(actions)

        # if (self.sim_frame_count + self._task.control_frequency_inv) % self._task.rendering_interval == 0:
        #     for _ in range(self._task.control_frequency_inv - 1):
        #         self._world.step(render=False)
        #         self.sim_frame_count += 1
        #     self._world.step(render=self._render)
        #     self.sim_frame_count += 1
        # else:
        #     for _ in range(self._task.control_frequency_inv):
        #         self._world.step(render=False)
        #         self.sim_frame_count += 1
        is_failed = False

        self._obs, self._rew, self._resets, self._extras, is_failed = self._task.post_physics_step()
        while is_failed:
            self._obs, self._rew, self._resets, self._extras, is_failed = self._task.post_physics_step()

        # self._states = self._task.get_states()
        self._process_data()

        # obs_dict = {"obs": self._obs, "states": self._states}
        obs_dict = {"obs": self._obs}

        # print("----------------------end-----------------------------------------")

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self, seed=None, options=None):
        """Resets the task and applies default zero actions to recompute observations and states."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((1, self._task.num_actions))
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict
