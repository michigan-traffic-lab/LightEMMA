#!/usr/bin/env python3
"""
Example script to visualize all frames of all scenes in a NuScenes dataset.

This script demonstrates how to use the enhanced BEVVisualizer to batch process
all frames from all scenes and generate comprehensive visualizations.

Output structure:
- Each scene gets its own folder: scene_XXX_scene_name/
- Images are saved directly in scene folders with naming: frame_XXXX_type.png
- No nested frame subfolders are created

Usage examples:
1. Generate BEV visualizations for all frames (with frame skipping):
   python example_all_frames.py --dataroot /path/to/nuscenes --vis_types bev --skip_frames 5

2. Generate multiple visualization types with frame limit:
   python example_all_frames.py --dataroot /path/to/nuscenes --vis_types bev image --max_frames_per_scene 20

3. Process every frame (warning: very time consuming):
   python example_all_frames.py --dataroot /path/to/nuscenes --vis_types image --skip_frames 1

4. Quick test with limited frames:
   python example_all_frames.py --dataroot /path/to/nuscenes --vis_types bev --max_frames_per_scene 5 --skip_frames 2
"""

import os
import sys
import argparse
from nuscenes.nuscenes import NuScenes
from visualize_bev import BEVVisualizer

def estimate_processing_time(nusc, vis_types, max_frames_per_scene, skip_frames):
    """
    Estimate total processing time based on dataset size and parameters
    """
    total_scenes = len(nusc.scene)
    
    # Estimate average frames per scene (NuScenes typically has ~40 frames per scene)
    avg_frames_per_scene = 40
    
    if max_frames_per_scene:
        avg_frames_per_scene = min(avg_frames_per_scene, max_frames_per_scene)
    
    avg_frames_per_scene = avg_frames_per_scene // skip_frames
    
    total_frames = total_scenes * avg_frames_per_scene
    
    # Estimate processing time per frame based on visualization types
    time_per_frame = {
        'bev': 2.0,
        'image': 1.5,
        'map': 3.0,
        'bev_with_map': 4.0
    }
    
    estimated_time_per_frame = sum(time_per_frame.get(vt, 2.0) for vt in vis_types)
    total_estimated_time = total_frames * estimated_time_per_frame
    
    return total_frames, total_estimated_time

def main():
    parser = argparse.ArgumentParser(
        description='Visualize all frames of all scenes in a NuScenes dataset',
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
                       default=['image'], choices=['bev', 'image'],
                       help='Types of visualizations to generate')
    parser.add_argument('--output_dir', type=str, default='all_frames_output',
                       help='Directory to save all visualizations')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 10],
                       help='Figure size for plots (width height)')
    parser.add_argument('--max_frames_per_scene', type=int, default=None,
                       help='Maximum number of frames to process per scene (None for all)')
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='Skip every N frames (1=all frames, 5=every 5th frame, etc.)')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt for large processing jobs')
    
    args = parser.parse_args()
    
    # Validate dataroot
    if not os.path.exists(args.dataroot):
        print(f"Error: Dataroot path does not exist: {args.dataroot}")
        sys.exit(1)
    
    # Validate skip_frames
    if args.skip_frames < 1:
        print("Error: skip_frames must be >= 1")
        sys.exit(1)
    
    print("=" * 70)
    print("NuScenes All Frames Visualization")
    print("=" * 70)
    print(f"Dataroot: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Point cloud range: {args.range}")
    print(f"Visualization types: {args.vis_types}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figure size: {args.figsize}")
    print(f"Max frames per scene: {args.max_frames_per_scene or 'No limit'}")
    print(f"Skip frames: {args.skip_frames} (process every {args.skip_frames} frame(s))")
    print("=" * 70)
    
    try:
        # Initialize NuScenes
        print("Loading NuScenes dataset...")
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        print(f"✓ Dataset loaded successfully with {len(nusc.scene)} scenes")
        
        # Estimate processing time and get user confirmation
        total_frames, estimated_time = estimate_processing_time(
            nusc, args.vis_types, args.max_frames_per_scene, args.skip_frames
        )
        
        print(f"\nProcessing Estimation:")
        print(f"  Estimated total frames to process: {total_frames:,}")
        print(f"  Estimated processing time: {estimated_time/3600:.1f} hours ({estimated_time/60:.1f} minutes)")
        print(f"  Estimated storage space: {total_frames * len(args.vis_types) * 2:.1f} MB")
        
        if not args.confirm and estimated_time > 300:  # More than 5 minutes
            print(f"\n⚠️  This is a large processing job that may take {estimated_time/3600:.1f} hours!")
            print("   Consider using --skip_frames to reduce processing time.")
            print("   Or use --max_frames_per_scene to limit frames per scene.")
            response = input("\nDo you want to continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Processing cancelled.")
                sys.exit(0)
        
        # Initialize visualizer
        print("\nInitializing BEV visualizer...")
        visualizer = BEVVisualizer(
            nusc, 
            scene_idx=0,  # This will be ignored for all_frames
            figsize=tuple(args.figsize), 
            point_cloud_range=args.range
        )
        print("✓ Visualizer initialized successfully")
        
        # Process all frames
        print("\nStarting batch processing of all frames...")
        print("Note: This may take a very long time depending on your settings.")
        print("You can interrupt with Ctrl+C if needed.\n")
        
        visualizer.visualize_all_frames(
            output_dir=args.output_dir, 
            visualization_types=args.vis_types,
            max_frames_per_scene=args.max_frames_per_scene,
            skip_frames=args.skip_frames
        )
        
        print("\n" + "=" * 70)
        print("✓ All frames processed successfully!")
        print(f"✓ Results saved to: {os.path.abspath(args.output_dir)}")
        print("✓ Check the summary report for detailed statistics.")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
        print("Partial results may be available in the output directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 