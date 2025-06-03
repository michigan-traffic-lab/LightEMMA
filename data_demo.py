import os
import re
import ast
import argparse
import datetime
import random
import numpy as np
import json
from nuscenes import NuScenes

from utils import *
from vlm import ModelHandler
from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version="v1.0-trainval", dataroot="data/v2x-seq-nuscenes/vehicle-side", verbose=True)
scene = nusc.scene[0]
scene_token = scene["token"]
scene_name = scene["name"]
scene_description = scene["description"]
scene_log_token = scene["log_token"]
curr_sample_token = scene["first_sample_token"]
last_sample_token = scene["last_sample_token"]

camera_params = []
front_camera_images = []
ego_positions = []
ego_rotations = []
ego_headings = []
timestamps = []
sample_tokens = []
while curr_sample_token:
    sample = nusc.get("sample", curr_sample_token)
    sample_tokens.append(curr_sample_token)
    
    cam_front_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    front_camera_image = os.path.join(nusc.dataroot, cam_front_data["filename"].replace("velodyne", "image").replace("bin", "jpg"))
    front_camera_images.append(front_camera_image)
    
    # Get the camera parameters
    camera_params.append(
        nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
    )
    
    # Get ego vehicle state
    ego_state = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
    ego_positions.append(tuple(ego_state["translation"]))
    ego_rotations.append(ego_state["rotation"])
    ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
    timestamps.append(ego_state["timestamp"])
    
    # Move to next sample or exit loop if at the end
    curr_sample_token = (
        sample["next"] if curr_sample_token != last_sample_token else None
    )

print(len(front_camera_images))
print(front_camera_images[0])