import joblib

pkl_path = "/home/ubuntu/projects/tool_kit/assets/pkl/cut_squat.pkl"
is_PBHC=False

motion = joblib.load(pkl_path)
if is_PBHC:
    motion = motion[list(motion.keys())[0]]

print(motion.keys())
print(motion['root_pos'].shape)