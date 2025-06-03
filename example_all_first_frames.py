#!/usr/bin/env python3
"""
Example script to visualize the first frame of all scenes in a NuScenes dataset.

This script demonstrates how to use the enhanced BEVVisualizer to batch process
all scenes and generate visualizations for their first frames.

Usage examples:
1. Generate BEV visualizations for all first frames:
   python example_all_first_frames.py --dataroot /path/to/nuscenes --vis_types bev

2. Generate multiple visualization types:
   python example_all_first_frames.py --dataroot /path/to/nuscenes --vis_types bev image map bev_with_map

3. Use custom point cloud range:
   python example_all_first_frames.py --dataroot /path/to/nuscenes --range -50 -50 50 50 --vis_types bev
"""

import os
import sys
import argparse
from nuscenes.nuscenes import NuScenes
from visualize_bev import BEVVisualizer

def main():
    parser = argparse.ArgumentParser(
        description='Visualize the first frame of all scenes in a NuScenes dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to the NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--range', type=float, nargs=4, default=[-100, -100, 100, 100],
                       help='Point cloud range: xmin ymin xmax ymax')
    parser.add_argument('--vis_types', type=str, nargs='+', 
                       default=['bev'], choices=['bev', 'image', 'map', 'bev_with_map'],
                       help='Types of visualizations to generate')
    parser.add_argument('--output_dir', type=str, default='all_first_frames_output',
                       help='Directory to save all visualizations')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 10],
                       help='Figure size for plots (width height)')
    
    args = parser.parse_args()
    
    # Validate dataroot
    if not os.path.exists(args.dataroot):
        print(f"Error: Dataroot path does not exist: {args.dataroot}")
        sys.exit(1)
    
    print("=" * 60)
    print("NuScenes All First Frames Visualization")
    print("=" * 60)
    print(f"Dataroot: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Point cloud range: {args.range}")
    print(f"Visualization types: {args.vis_types}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figure size: {args.figsize}")
    print("=" * 60)
    
    try:
        # Initialize NuScenes
        print("Loading NuScenes dataset...")
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        print(f"✓ Dataset loaded successfully with {len(nusc.scene)} scenes")
        
        # Initialize visualizer
        print("Initializing BEV visualizer...")
        visualizer = BEVVisualizer(
            nusc, 
            scene_idx=0,  # This will be ignored for all_first_frames
            figsize=tuple(args.figsize), 
            point_cloud_range=args.range
        )
        print("✓ Visualizer initialized successfully")
        
        # Process all first frames
        print("\nStarting batch processing of all first frames...")
        visualizer.visualize_all_first_frames(
            output_dir=args.output_dir, 
            visualization_types=args.vis_types
        )
        
        print("\n" + "=" * 60)
        print("✓ All first frames processed successfully!")
        print(f"✓ Results saved to: {os.path.abspath(args.output_dir)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 