"""常量定义：官方 VBench-I2V 10 维与输入目录约定。"""

DIMS_10 = [
    "i2v_subject",
    "i2v_background",
    "camera_motion",
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

I2V_NATIVE_DIMS = {"i2v_subject", "i2v_background", "camera_motion"}
STANDARD_VBENCH_DIMS = set(DIMS_10) - I2V_NATIVE_DIMS

# VBench-I2V case input 的目录约定。
SB_GROUP = "i2v_subject_background"
CAM_GROUP = "i2v_camera_only"

MODE_TO_SB_DIR = {
    "bf16": "videos_bf16_sb",
    "quant": "videos_quant_sb",
}

MODE_TO_CAM_DIR = {
    "bf16": "videos_bf16_camera",
    "quant": "videos_quant_camera",
}
