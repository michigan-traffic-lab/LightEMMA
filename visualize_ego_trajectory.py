#!/usr/bin/env python3
"""
Ego Vehicle Trajectory Visualization Script

This script demonstrates how to visualize ego vehicle trajectories in world coordinates
using the EgoTrajectoryVisualizer class from predict.py.

Usage:
    python visualize_ego_trajectory.py --dataroot /path/to/nuscenes --scene_name scene-0001
    python visualize_ego_trajectory.py --dataroot /path/to/nuscenes --scene_idx 0
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from utils import quaternion_to_yaw
from predict import EgoTrajectoryVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize ego vehicle trajectory in world coordinates')
    parser.add_argument('--dataroot', type=str, default='data/v2x-seq-nuscenes/vehicle-side',
                       help='Path to the NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--scene_idx', type=int, default=None,
                       help='Index of the scene to visualize')
    parser.add_argument('--scene_name', type=str, default=None,
                       help='Name of the scene to visualize')
    parser.add_argument('--output_dir', type=str, default='ego_trajectory_output',
                       help='Directory to save the output')
    parser.add_argument('--show_plot', action='store_true',
                       help='Display the plot')
    parser.add_argument('--show_orientation_arrows', action='store_true', default=True,
                       help='Show orientation arrows on trajectory')
    parser.add_argument('--show_ego_axes', action='store_true', default=True,
                       help='Show ego coordinate axes on trajectory')
    parser.add_argument('--ego_axes_interval', type=int, default=10,
                       help='Interval for showing ego axes (every N trajectory points)')
    
    return parser.parse_args()

def get_ego_trajectory_from_scene(nusc, scene):
    """
    Extract ego vehicle trajectory from a NuScenes scene
    
    Args:
        nusc: NuScenes instance
        scene: Scene dictionary
        
    Returns:
        ego_positions: List of ego positions [(x, y), ...]
        ego_headings: List of ego headings [yaw1, yaw2, ...]
        timestamps: List of timestamps
    """
    ego_positions = []
    ego_headings = []
    timestamps = []
    
    # Get first sample token
    sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    
    # Iterate through all samples in the scene
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Get ego pose from LiDAR sensor
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Store ego position and heading
        ego_positions.append(tuple(ego_pose['translation'][0:2]))  # (x, y)
        ego_headings.append(quaternion_to_yaw(ego_pose['rotation']))
        timestamps.append(ego_pose['timestamp'])
        
        # Move to next sample or exit loop if at the end
        sample_token = sample['next'] if sample_token != last_sample_token else None
    
    print(f"Extracted {len(ego_positions)} trajectory points from scene '{scene['name']}'")
    return ego_positions, ego_headings, timestamps

def main():
    args = parse_args()
    
    # Initialize NuScenes
    print(f"Loading NuScenes dataset from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Select scene to visualize
    if args.scene_name:
        # Find scene by name
        selected_scene = None
        for scene in nusc.scene:
            if scene['name'] == args.scene_name:
                selected_scene = scene
                break
        
        if selected_scene is None:
            print(f"Scene '{args.scene_name}' not found in dataset")
            return
    elif args.scene_idx is not None:
        # Select scene by index
        if 0 <= args.scene_idx < len(nusc.scene):
            selected_scene = nusc.scene[args.scene_idx]
        else:
            print(f"Scene index {args.scene_idx} out of range (0-{len(nusc.scene)-1})")
            return
    else:
        # Default to first scene
        selected_scene = nusc.scene[0]
        print(f"No scene specified, using first scene: {selected_scene['name']}")
    
    scene_name = selected_scene['name']
    print(f"Visualizing trajectory for scene: {scene_name}")
    print(f"Scene description: {selected_scene['description']}")
    
    # Extract ego trajectory from scene
    ego_positions, ego_headings, timestamps = get_ego_trajectory_from_scene(nusc, selected_scene)
    
    if len(ego_positions) < 2:
        print(f"Insufficient trajectory points ({len(ego_positions)}) for visualization")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trajectory visualizer
    trajectory_visualizer = EgoTrajectoryVisualizer()
    
    # Create visualization save path
    viz_save_path = os.path.join(args.output_dir, f"{scene_name}_ego_trajectory.png")
    
    # Create trajectory visualization
    print("Creating trajectory visualization...")
    fig, ax = trajectory_visualizer.visualize_scene_trajectory(
        ego_positions=ego_positions,
        ego_headings=ego_headings,
        scene_name=scene_name,
        save_path=viz_save_path,
        show_plot=args.show_plot,
        show_orientation_arrows=args.show_orientation_arrows,
        show_ego_axes=args.show_ego_axes,
        ego_axes_interval=args.ego_axes_interval
    )
    
    if fig is not None:
        print(f"Trajectory visualization saved to: {viz_save_path}")
        
        # Print trajectory statistics
        distance = trajectory_visualizer._calculate_trajectory_distance(ego_positions)
        duration = (timestamps[-1] - timestamps[0]) / 1e6  # Convert to seconds
        avg_speed = distance / duration if duration > 0 else 0
        
        print(f"\nTrajectory Statistics:")
        print(f"  Total points: {len(ego_positions)}")
        print(f"  Total distance: {distance:.1f} meters")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Average speed: {avg_speed * 3.6:.1f} km/h")
        
        # Print coordinate bounds
        x_coords = [pos[0] for pos in ego_positions]
        y_coords = [pos[1] for pos in ego_positions]
        print(f"  X range: [{min(x_coords):.1f}, {max(x_coords):.1f}] meters")
        print(f"  Y range: [{min(y_coords):.1f}, {max(y_coords):.1f}] meters")
    else:
        print("Failed to create trajectory visualization")

if __name__ == "__main__":
    main() 