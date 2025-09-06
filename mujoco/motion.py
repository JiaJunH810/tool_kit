import os, argparse

import torch
from torch import Tensor
import mujoco, mujoco_viewer
from tqdm import tqdm
import joblib
import numpy as np

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    assert(len(t.shape) == len(q0.shape) - 1)
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = torch.where(neg_mask.unsqueeze(-1), -q1, q1)
    
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    t = t.unsqueeze(-1)
    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

class MotionLib:
    def __init__(self, motion_path):
        motion = joblib.load(motion_path)
        for key in list(motion.keys()):
            motion_data = motion[key]
        
        self.fps = int(motion_data['fps'])
        dt = 1.0 / self.fps
        self.num_frames = torch.tensor(len(motion_data['root_rot']))
        
        self.motion_len = dt * (self.num_frames - 1)
        
        self.root_pos = motion_data['root_trans_offset']
        
        self.dof_pos = motion_data['dof']
        
        self.root_rot = motion_data['root_rot']
    
    def calc_motion_frame(self, times):
        num_frames = self.num_frames
        phase = times / self.motion_len
        phase = torch.clip(phase, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()
        blend = blend.cpu()
        
        root_pos0 = self.root_pos[frame_idx0]
        root_pos1 = self.root_pos[frame_idx1]

        root_rot0 = torch.from_numpy(self.root_rot[frame_idx0])
        root_rot1 = torch.from_numpy(self.root_rot[frame_idx1])

        dof_pos0 = self.dof_pos[frame_idx0]
        dof_pos1 = self.dof_pos[frame_idx1]

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        return root_pos, root_rot, dof_pos

class MotionViewEnv:
    def __init__(self, motion_path, model_path):
        self.device = device
        
        self.motion_file_name = os.path.basename(motion_path)
        
        self.motion_lib = MotionLib(motion_path=motion_path)
        self.motion_len = self.motion_lib.motion_len
                
        self.sim_duration = 10*self.motion_len
        
        self.sim_dt = 0.02
        self.sim_decimation = 1
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
    def run(self):
        
        for i in tqdm(range(int(self.sim_duration / self.control_dt)), desc="Running simulation..."):
            curr_time = i * self.control_dt
            motion_time = torch.tensor(curr_time, dtype=torch.float, device=self.device) % self.motion_len
            root_pos, root_rot, dof_pos = self.motion_lib.calc_motion_frame(motion_time)
            self.data.qpos[:3] = root_pos.cpu().numpy()
            self.data.qpos[3:7] = root_rot.cpu().numpy()[[3, 0, 1, 2]]
            self.data.qpos[7:] = dof_pos.cpu().numpy()
            
            mujoco.mj_forward(self.model, self.data)
            
            self.viewer.render()
        
        self.viewer.close()


if __name__ == "__main__":
    motion_path = "mujoco/pkl/combine.pkl"
    model_path = "mujoco/g1_23dof.xml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_env = MotionViewEnv(motion_path, model_path)
    motion_env.run()
    