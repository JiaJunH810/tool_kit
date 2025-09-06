import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys


def process_motion_sequences(seq1, seq2):
    # 获取seq1的最后一帧 作为第二序列的起点
    last_trans_s1 = seq1['root_trans_offset'][-1]
    last_rot_s1 = R.from_quat(seq1['root_rot'][-1])

    # 获取seq2的第一帧 作为对齐的帧
    first_trans_s2 = seq2['root_trans_offset'][0]
    first_rot_s2 = R.from_quat(seq2['root_rot'][0])

    # 对齐偏移量
    # 只对齐偏航角yaw 作为前进方向 防止姿势扭曲
    _, _, yaw1 = last_rot_s1.as_euler('xyz')
    _, _, yaw2 = first_rot_s2.as_euler('xyz')

    # 创建只包含偏航分量的旋转
    yaw_rot1 = R.from_euler('z', yaw1)
    yaw_rot2 = R.from_euler('z', yaw2)

    # 计算seq1和seq2在偏航yaw方面的旋转量
    delta_yaw_rot = yaw_rot1 * yaw_rot2.inv()

    new_trans = list(seq1['root_trans_offset'])
    new_rot = list(seq1['root_rot'])
    new_dof = list(seq1['dof'])

    for i in range(len(seq2['root_rot'])):

        curr_rot_s2 = R.from_quat(seq2['root_rot'][i])
        curr_trans_s2 = seq2['root_trans_offset'][i]

        delta_trans_world = curr_trans_s2 - first_trans_s2
        # delta_yaw_rot是yaw2->yaw1的旋转，所以这里相当于直接获得了旋转后的全局坐标系
        inc_trans = delta_yaw_rot.apply(delta_trans_world)

        final_trans = last_trans_s1 + inc_trans
        final_rot = delta_yaw_rot * curr_rot_s2

        new_trans.append(final_trans)
        new_rot.append(final_rot.as_quat())
        new_dof.append(seq2['dof'][i])
    
    return new_trans, new_rot, new_dof

def main(pkl_path1, pkl_path2):
    motion1 = joblib.load(pkl_path1)
    motion2 = joblib.load(pkl_path2)

    motion = motion1.copy()
    k = list(motion.keys())[0]

    motion_data1 = motion1[list(motion1.keys())[0]]
    motion_data2 = motion2[list(motion2.keys())[0]]

    # 处理运动序列
    new_trans, new_rots, new_dofs = process_motion_sequences(motion_data1, motion_data2)

    motion[k]['root_trans_offset'] = np.array(new_trans)
    motion[k]['root_rot'] = np.array(new_rots)
    motion[k]['dof'] = np.array(new_dofs)
    print(motion[k]['dof'].shape)
    joblib.dump(motion, "/home/ubuntu/projects/tool_kit/mujoco/pkl/combine.pkl")

if __name__ == "__main__":
    pkl_path1 = "/home/ubuntu/projects/tool_kit/mujoco/pkl/slowwalk_05_01_high_inter.pkl"
    pkl_path2 = "/home/ubuntu/projects/tool_kit/mujoco/pkl/Walk-Bendover_walkback.pkl"

    main(pkl_path1, pkl_path2)