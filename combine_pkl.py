import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys

def obtain_pose_aa(root_rot, dof_pos, dof_axis):
    pose_aa = np.zeros((1 + len(dof_pos), 3))
    pose_aa[0] = R.from_quat(root_rot).as_euler('xyz', degrees=False)
    pose_aa[1:] = dof_pos.reshape(-1, 1) * dof_axis
    return pose_aa

def make_interpolation(last_trans, first_trans, last_rot, first_rot, last_dof, first_dof, num_frame=30):

    make_trans = np.linspace(last_trans, first_trans, num=num_frame)

    key_times = [0, 1]      # 创建时间点
    key_rots = R.from_quat([last_rot, first_rot])   # 起始旋转和终止旋转
    slerp = Slerp(key_times, key_rots)      # 创建插值器
    interp_rots = slerp(np.linspace(0, 1, num_frame))   # 通过np.linspace获取插值中的时间点
    make_rot = interp_rots.as_quat()

    make_dof = np.linspace(last_dof, first_dof, num=num_frame)

    return make_trans, make_rot, make_dof

def process_motion_sequences(seq1, seq2, dof_axis):
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

    new_trans = []
    new_rot = []
    new_dof = []
    new_pose_aa = []

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
        new_pose_aa.append(obtain_pose_aa(final_rot.as_quat(), seq2['dof'][i], dof_axis))
    
    inter_trans, inter_rot, inter_dof = make_interpolation(seq1['root_trans_offset'][-1], new_trans[0],
                                        seq1['root_rot'][-1], new_rot[0], seq1['dof'][-1], new_dof[0], 30)
    
    new_trans = list(seq1['root_trans_offset']) + list(inter_trans) + new_trans
    new_rot = list(seq1['root_rot']) + list(inter_rot) + new_rot
    new_dof = list(seq1['dof']) + list(inter_dof) + new_dof
    return new_trans, new_rot, new_dof

def main(pkl_lists, dof_axis_path):
    dof_axis = np.load(dof_axis_path)

    motion = joblib.load(pkl_lists[0])
    k = list(motion.keys())[0]

    for i in range(1, len(pkl_lists)):
        motion1 = motion.copy()
        motion2 = joblib.load(pkl_lists[i])
        motion_data1 = motion1[list(motion1.keys())[0]]
        motion_data2 = motion2[list(motion2.keys())[0]]

        # 处理运动序列
        new_trans, new_rots, new_dofs = process_motion_sequences(motion_data1, motion_data2, dof_axis)

        motion[k]['root_trans_offset'] = np.array(new_trans)
        motion[k]['root_rot'] = np.array(new_rots)
        motion[k]['dof'] = np.array(new_dofs)

    print(motion[k]['dof'].shape)
    joblib.dump(motion, "/home/ubuntu/projects/tool_kit/assets/pkl/combine.pkl")

if __name__ == "__main__":
    pkl_lists = [
        "/home/ubuntu/projects/tool_kit/assets/pkl/slowwalk_05_01_high_inter.pkl",
        "/home/ubuntu/projects/tool_kit/assets/pkl/Walk-Bendover_walkback.pkl"
    ]
    dof_axis_path = "interpolation/g1_23dof_axis.npy"
    main(pkl_lists, dof_axis_path)