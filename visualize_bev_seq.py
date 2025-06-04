import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from utils import quaternion_to_yaw
from pypcd import pypcd

class BEVSequenceVisualizer:
    def __init__(self, nusc, scene_idx=0, figsize=(12, 12), 
                 point_cloud_range=[-100, -100, 100, 100], resolution=0.2):
        """
        Initialize the Bird's Eye View sequence visualizer for trajectory visualization
        
        Args:
            nusc: NuScenes instance
            scene_idx: Index of the scene to visualize
            figsize: Figure size for the plot
            point_cloud_range: [xmin, ymin, xmax, ymax] range for BEV display
            resolution: Resolution of the BEV grid in meters
        """
        self.nusc = nusc
        self.scene = nusc.scene[scene_idx]
        self.point_cloud_range = point_cloud_range
        self.resolution = resolution
        self.figsize = figsize
        print(f"Initializing sequence visualizer for scene {scene_idx}")
        print(f"Point cloud range: {self.point_cloud_range}")
        
        # Define colors for trajectory visualization
        self.trajectory_colors = {
            'ego_trajectory': '#FF0000',  # Red for ego vehicle trajectory
            'ego_current': '#FF4500',     # Orange red for current position
            'ego_start': '#00FF00',       # Green for start position
            'ego_end': '#0000FF',         # Blue for end position
            'trajectory_line': '#FF6B6B', # Light red for trajectory line
        }
    
    def get_point_cloud(self, sample_token):
        """
        Get LiDAR point cloud for a sample and transform to world coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            points: Nx4 array of point cloud points in world coordinates (x, y, z, intensity)
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        
        # Check if the original file exists, if not try .pcd extension
        if not os.path.exists(lidar_path):
            lidar_path = lidar_path.replace('.bin', '.pcd')
        
        # Check file extension to determine loading method
        file_extension = os.path.splitext(lidar_path)[1].lower()
        if file_extension == '.pcd':
            # Load PCD file using pypcd
            points = self._load_pcd_with_pypcd(lidar_path)
            # Convert to LidarPointCloud format for transformation
            pc = LidarPointCloud(points.T)  # Transpose to 4xN format
        elif file_extension == '.bin':
            # Load binary file using NuScenes LidarPointCloud
            pc = LidarPointCloud.from_file(lidar_path)
        else:
            raise ValueError(f"Unsupported point cloud file format: {file_extension}")
        
        # Get calibration and pose information
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Transform from sensor to ego vehicle coordinates
        pc.rotate(Quaternion(calibrated_sensor['rotation']).rotation_matrix)
        pc.translate(np.array(calibrated_sensor['translation']))
        
        # Transform from ego vehicle to world coordinates
        pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix)
        pc.translate(np.array(ego_pose['translation']))
        
        return pc.points.T  # Nx4 array (x, y, z, intensity) in world coordinates
    
    def _load_pcd_with_pypcd(self, pcd_path):
        """
        Load point cloud from PCD file using pypcd library
        
        Args:
            pcd_path: Path to the PCD file
            
        Returns:
            points: Nx4 array of point cloud points (x, y, z, intensity)
        """
        # Load PCD file using pypcd
        pc = pypcd.PointCloud.from_path(pcd_path)
        
        # Extract point data
        pc_data = pc.pc_data
        
        # Get x, y, z coordinates
        x = pc_data['x'].astype(np.float32)
        y = pc_data['y'].astype(np.float32)
        z = pc_data['z'].astype(np.float32)
        
        # Try to get intensity information
        intensity = None
        
        # Check for common intensity field names
        intensity_fields = ['intensity', 'i', 'reflectance', 'remission', 'ring']
        for field in intensity_fields:
            if field in pc_data.dtype.names:
                intensity = pc_data[field].astype(np.float32)
                print(f"Found intensity field: {field}")
                break
        
        # If no intensity field found, check for RGB and convert to intensity
        if intensity is None:
            rgb_fields = ['rgb', 'rgba']
            for field in rgb_fields:
                if field in pc_data.dtype.names:
                    # Convert RGB to intensity (grayscale)
                    rgb_data = pc_data[field]
                    if rgb_data.dtype == np.uint32:
                        # Unpack RGB from uint32
                        r = (rgb_data >> 16) & 0xFF
                        g = (rgb_data >> 8) & 0xFF
                        b = rgb_data & 0xFF
                        intensity = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                    else:
                        intensity = rgb_data.astype(np.float32)
                    print(f"Converted {field} to intensity")
                    break
        
        # If still no intensity, use default values
        if intensity is None:
            intensity = np.ones(len(x), dtype=np.float32)
            print("No intensity field found, using default intensity values")
        
        # Combine into Nx4 array
        points = np.column_stack([x, y, z, intensity])
        
        print(f"Successfully loaded {len(points)} points from PCD file using pypcd")
        print(f"Available fields in PCD: {list(pc_data.dtype.names)}")
        return points
    
    def filter_points_in_range(self, points, point_cloud_range):
        """
        Filter points within the specified range
        
        Args:
            points: Nx4 array of point cloud points (x, y, z, intensity)
            point_cloud_range: [xmin, ymin, xmax, ymax] range for filtering
            
        Returns:
            filtered_points: Points within the specified range
        """
        x_min, y_min, x_max, y_max = point_cloud_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        return points[mask]
    
    def get_ego_trajectory(self, start_sample_idx=0, end_sample_idx=None, step=1):
        """
        Get ego vehicle trajectory for a sequence of samples
        
        Args:
            start_sample_idx: Starting sample index in the scene
            end_sample_idx: Ending sample index (if None, use all samples)
            step: Step size for sampling (1 means every sample)
            
        Returns:
            trajectory: List of dictionaries containing position, rotation, and sample info
        """
        trajectory = []
        
        # Get first sample token
        sample_token = self.scene['first_sample_token']
        
        # Skip to start sample
        for i in range(start_sample_idx):
            if not sample_token:
                break
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
        
        # Collect trajectory points
        sample_idx = start_sample_idx
        step_counter = 0
        
        while sample_token:
            # Check if we've reached the end
            if end_sample_idx is not None and sample_idx > end_sample_idx:
                break
            
            # Only collect every 'step' samples
            if step_counter % step == 0:
                sample = self.nusc.get('sample', sample_token)
                
                # Get ego pose from LiDAR sensor
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = self.nusc.get('sample_data', lidar_token)
                ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                
                # Store trajectory point
                trajectory_point = {
                    'position': ego_pose['translation'],
                    'rotation': ego_pose['rotation'],
                    'sample_token': sample_token,
                    'sample_idx': sample_idx,
                    'timestamp': sample['timestamp']
                }
                trajectory.append(trajectory_point)
            
            # Move to next sample
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            sample_idx += 1
            step_counter += 1
        
        print(f"Collected {len(trajectory)} trajectory points from sample {start_sample_idx} to {sample_idx-1}")
        return trajectory
    
    def visualize_trajectory_on_map(self, start_sample_idx=0, end_sample_idx=None, 
                                    step=1, save_path=None, show_plot=True,
                                    show_trajectory_points=True, show_orientation_arrows=True,
                                    trajectory_line_width=3, point_size=50, show_point_cloud=True,
                                    show_ego_axes=True, ego_axes_interval=5):
        """
        Visualize ego vehicle trajectory overlaid on map data with optional point cloud background
        
        Args:
            start_sample_idx: Starting sample index
            end_sample_idx: Ending sample index (if None, use all samples)
            step: Step size for sampling trajectory points
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            show_trajectory_points: Whether to show individual trajectory points
            show_orientation_arrows: Whether to show orientation arrows
            trajectory_line_width: Width of the trajectory line
            point_size: Size of trajectory points
            show_point_cloud: Whether to show point cloud background from first frame
            show_ego_axes: Whether to show ego coordinate axes
            ego_axes_interval: Interval for showing ego axes (every N points)
            
        Returns:
            fig, ax: Figure and axis objects
        """
        # Get trajectory
        trajectory = self.get_ego_trajectory(start_sample_idx, end_sample_idx, step)
        
        if not trajectory:
            print("No trajectory points found!")
            return None, None
        
        # Get scene and map info from first trajectory point
        first_sample = self.nusc.get('sample', trajectory[0]['sample_token'])
        scene = self.nusc.get('scene', first_sample['scene_token'])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_facecolor('#F8F8F8')
        
        # Extract trajectory coordinates
        traj_x = [point['position'][0] for point in trajectory]
        traj_y = [point['position'][1] for point in trajectory]
        
        # Calculate trajectory bounds and adjust point cloud range if needed
        traj_x_min, traj_x_max = min(traj_x), max(traj_x)
        traj_y_min, traj_y_max = min(traj_y), max(traj_y)
        
        # Calculate trajectory span
        traj_x_span = traj_x_max - traj_x_min
        traj_y_span = traj_y_max - traj_y_min
        
        # Add padding based on trajectory span (minimum 50m, or 20% of span)
        padding_x = max(50, traj_x_span * 0.2)
        padding_y = max(50, traj_y_span * 0.2)
        
        # Ensure minimum visualization area (at least 200m x 200m)
        min_range = 200
        if traj_x_span + 2 * padding_x < min_range:
            padding_x = (min_range - traj_x_span) / 2
        if traj_y_span + 2 * padding_y < min_range:
            padding_y = (min_range - traj_y_span) / 2
        
        # Calculate adjusted range to fully cover trajectory
        adjusted_range = [
            traj_x_min - padding_x,
            traj_y_min - padding_y,
            traj_x_max + padding_x,
            traj_y_max + padding_y
        ]
        
        print(f"Trajectory bounds: X[{traj_x_min:.1f}, {traj_x_max:.1f}], Y[{traj_y_min:.1f}, {traj_y_max:.1f}]")
        print(f"Trajectory span: X={traj_x_span:.1f}m, Y={traj_y_span:.1f}m")
        print(f"Applied padding: X={padding_x:.1f}m, Y={padding_y:.1f}m")
        print(f"Final visualization range: X[{adjusted_range[0]:.1f}, {adjusted_range[2]:.1f}], Y[{adjusted_range[1]:.1f}, {adjusted_range[3]:.1f}]")
        
        # Get point cloud from first frame if requested
        first_frame_points = None
        print(f"---------show_point_cloud: {show_point_cloud}")
        if show_point_cloud:
            try:
                print("Loading point cloud from first frame...")
                first_frame_points = self.get_point_cloud(trajectory[0]['sample_token'])
                print(f"Loaded {len(first_frame_points)} points within visualization range")
                first_frame_points = self.filter_points_in_range(first_frame_points, adjusted_range)
                print(f"Loaded {len(first_frame_points)} points within visualization range")
            except Exception as e:
                print(f"Failed to load point cloud: {e}")
                first_frame_points = None
        
        # Create simple background with grid
        print("Creating simple background visualization...")
        
        # Set clean background color
        ax.set_facecolor('#FAFAFA')  # Very light gray background
        
        # Add simple grid for reference
        grid_spacing = 100  # 100 meter grid
        x_min, y_min, x_max, y_max = adjusted_range
        
        # Adaptive grid spacing based on visualization range
        range_x = x_max - x_min
        range_y = y_max - y_min
        max_range = max(range_x, range_y)
        
        # Choose appropriate grid spacing
        if max_range <= 500:
            major_grid = 50
            minor_grid = 10
        elif max_range <= 1000:
            major_grid = 100
            minor_grid = 20
        elif max_range <= 2000:
            major_grid = 200
            minor_grid = 50
        else:
            major_grid = 500
            minor_grid = 100
        
        print(f"Using grid spacing: major={major_grid}m, minor={minor_grid}m")
        
        # Minor grid lines (thinner, more frequent)
        for x in range(int(x_min), int(x_max) + 1, minor_grid):
            ax.axvline(x=x, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=1)
        for y in range(int(y_min), int(y_max) + 1, minor_grid):
            ax.axhline(y=y, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Major grid lines (thicker, less frequent)
        for x in range(int(x_min), int(x_max) + 1, major_grid):
            ax.axvline(x=x, color='gray', linestyle='-', alpha=0.6, linewidth=0.8, zorder=1)
        for y in range(int(y_min), int(y_max) + 1, major_grid):
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.6, linewidth=0.8, zorder=1)
        
        # Add coordinate axes at origin for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        
        # Plot point cloud background if available
        if first_frame_points is not None and len(first_frame_points) > 0:
            print("Rendering point cloud background...")
            # Use height (z) for coloring, with a suitable colormap
            scatter = ax.scatter(first_frame_points[:, 0], first_frame_points[:, 1], 
                               s=1.0, c=first_frame_points[:, 2], cmap='viridis', 
                               alpha=0.6, vmin=first_frame_points[:, 2].min(), 
                               vmax=first_frame_points[:, 2].max(), zorder=3)
            
            # Add colorbar for point cloud height
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Height (m)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
        
        # Plot trajectory line
        ax.plot(traj_x, traj_y, color=self.trajectory_colors['trajectory_line'], 
               linewidth=trajectory_line_width, alpha=0.8, zorder=5, 
               label=f'Ego Trajectory ({len(trajectory)} points)')
        
        # Plot trajectory points if requested
        if show_trajectory_points:
            ax.scatter(traj_x, traj_y, c=self.trajectory_colors['ego_trajectory'], 
                      s=point_size, alpha=0.7, zorder=6, edgecolors='white', linewidth=1)
        
        # Mark start and end positions
        if len(trajectory) > 0:
            # Start position
            start_x, start_y = trajectory[0]['position'][0], trajectory[0]['position'][1]
            print(f"Start position: {start_x}, {start_y}")
            ax.plot(start_x, start_y, 'o', color=self.trajectory_colors['ego_start'], 
                   markersize=12, markeredgecolor='white', markeredgewidth=2, 
                   label='Start Position', zorder=8)
            
            # End position
            if len(trajectory) > 1:
                end_x, end_y = trajectory[-1]['position'][0], trajectory[-1]['position'][1]
                print(f"End position: {end_x}, {end_y}")
                ax.plot(end_x, end_y, 's', color=self.trajectory_colors['ego_end'], 
                       markersize=12, markeredgecolor='white', markeredgewidth=2, 
                       label='End Position', zorder=8)
        
        # Show orientation arrows if requested
        if show_orientation_arrows and len(trajectory) > 1:
            # Show arrows at regular intervals
            arrow_step = max(1, len(trajectory) // 10)  # Show about 10 arrows
            for i in range(0, len(trajectory), arrow_step):
                point = trajectory[i]
                x, y = point['position'][0], point['position'][1]
                yaw = quaternion_to_yaw(point['rotation'])
                
                arrow_length = 8
                dx = arrow_length * np.cos(yaw)
                dy = arrow_length * np.sin(yaw)
                
                ax.arrow(x, y, dx, dy, head_width=3, head_length=2, 
                        fc=self.trajectory_colors['ego_current'], 
                        ec='white', linewidth=1, alpha=0.8, zorder=7)
        
        # Show ego coordinate axes if requested
        if show_ego_axes and len(trajectory) > 0:
            # Show ego axes at regular intervals
            axes_step = max(1, ego_axes_interval)
            for i in range(0, len(trajectory), axes_step):
                point = trajectory[i]
                self.draw_ego_coordinate_axes(
                    ax, point['position'], point['rotation'],
                    axis_length=12, line_width=2, alpha=0.8, zorder=9
                )
        
        # Set axis limits
        ax.set_xlim(adjusted_range[0], adjusted_range[2])
        ax.set_ylim(adjusted_range[1], adjusted_range[3])
        
        # Set axis labels and title
        ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
        
        title = f'Ego Vehicle Trajectory\nScene: {self.scene["name"]}'
        if end_sample_idx is not None:
            title += f' | Samples: {start_sample_idx}-{end_sample_idx} (step={step})'
        else:
            title += f' | Samples: {start_sample_idx}-end (step={step})'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                 fancybox=True, shadow=True)
        
        # Add simplified info text box
        duration = (trajectory[-1]['timestamp'] - trajectory[0]['timestamp']) / 1e6  # Convert to seconds
        distance = self._calculate_trajectory_distance(trajectory)
        
        # Calculate visualization area
        vis_width = adjusted_range[2] - adjusted_range[0]
        vis_height = adjusted_range[3] - adjusted_range[1]
        
        # Prepare info text
        info_text = (f"Trajectory Info:\n"
                    f"Points: {len(trajectory)}\n"
                    f"Duration: {duration:.1f}s\n"
                    f"Distance: {distance:.1f}m\n"
                    f"Avg Speed: {distance/duration*3.6:.1f} km/h\n"
                    f"View Area: {vis_width:.0f}×{vis_height:.0f}m")
        
        # Add point cloud info if available
        if first_frame_points is not None:
            info_text += f"\nLiDAR Points: {len(first_frame_points):,}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightblue', alpha=0.8), zorder=10)
        
        # Save if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Trajectory visualization saved to: {save_path}")
        
        # Show plot if required
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def _calculate_trajectory_distance(self, trajectory):
        """
        Calculate total distance traveled along the trajectory
        
        Args:
            trajectory: List of trajectory points
            
        Returns:
            total_distance: Total distance in meters
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            prev_pos = np.array(trajectory[i-1]['position'][:2])
            curr_pos = np.array(trajectory[i]['position'][:2])
            distance = np.linalg.norm(curr_pos - prev_pos)
            total_distance += distance
        
        return total_distance
    
    def create_trajectory_animation(self, start_sample_idx=0, end_sample_idx=None, 
                                   step=1, output_dir="trajectory_animation", fps=5,
                                   trail_length=20, show_point_cloud=True, show_ego_axes=True):
        """
        Create an animation showing the ego vehicle moving along its trajectory
        
        Args:
            start_sample_idx: Starting sample index
            end_sample_idx: Ending sample index
            step: Step size for sampling
            output_dir: Directory to save animation frames
            fps: Frames per second for the video
            trail_length: Number of previous positions to show as trail
            show_point_cloud: Whether to show point cloud background
            show_ego_axes: Whether to show ego coordinate axes
            
        Returns:
            video_path: Path to the created video file
        """
        import cv2
        
        # Get full trajectory
        full_trajectory = self.get_ego_trajectory(start_sample_idx, end_sample_idx, step)
        
        if not full_trajectory:
            print("No trajectory points found!")
            return None
        
        # Get point cloud from first frame for background if requested
        background_points = None
        print(f"---------show_point_cloud: {show_point_cloud}")
        if show_point_cloud:
            print(f"Loading point cloud for animation background...")
            try:
                background_points = self.get_point_cloud(full_trajectory[0]['sample_token'])
                print(f"Loaded {len(background_points)} background points")
            except Exception as e:
                print(f"Failed to load background point cloud: {e}")
                background_points = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        frame_paths = []
        
        # Create frames
        for i, current_point in enumerate(full_trajectory):
            print(f"Creating frame {i+1}/{len(full_trajectory)}")
            
            # Get trail points (previous positions)
            trail_start = max(0, i - trail_length)
            trail_points = full_trajectory[trail_start:i+1]
            
            # Create frame
            frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
            self._create_animation_frame(trail_points, current_point, frame_path, i, background_points, show_ego_axes)
            frame_paths.append(frame_path)
        
        # Create video from frames
        if frame_paths:
            video_path = os.path.join(output_dir, 'trajectory_animation.mp4')
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_paths[0])
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Add frames to video
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video.write(frame)
            
            video.release()
            print(f"Animation video saved to: {video_path}")
            return video_path
        
        return None
    
    def _create_animation_frame(self, trail_points, current_point, save_path, frame_idx, background_points, show_ego_axes):
        """
        Create a single frame for the trajectory animation
        
        Args:
            trail_points: List of previous trajectory points to show as trail
            current_point: Current position point
            save_path: Path to save the frame
            frame_idx: Frame index for title
            background_points: Point cloud background points
            show_ego_axes: Whether to show ego coordinate axes
        """
        
        # Get scene and map info
        sample = self.nusc.get('sample', current_point['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set view range around current position
        curr_x, curr_y = current_point['position'][0], current_point['position'][1]
        view_range = 100  # meters around current position
        
        ax.set_xlim(curr_x - view_range, curr_x + view_range)
        ax.set_ylim(curr_y - view_range, curr_y + view_range)
        
        # Set clean background color (same as single frame visualization)
        ax.set_facecolor('#FAFAFA')  # Very light gray background
        
        # Add adaptive grid for reference (same as single frame visualization)
        x_min, y_min = curr_x - view_range, curr_y - view_range
        x_max, y_max = curr_x + view_range, curr_y + view_range
        
        # Use same adaptive grid spacing logic as single frame
        range_x = x_max - x_min
        range_y = y_max - y_min
        max_range = max(range_x, range_y)
        
        # Choose appropriate grid spacing (same logic as single frame)
        if max_range <= 500:
            major_grid = 50
            minor_grid = 10
        elif max_range <= 1000:
            major_grid = 100
            minor_grid = 20
        elif max_range <= 2000:
            major_grid = 200
            minor_grid = 50
        else:
            major_grid = 500
            minor_grid = 100
        
        # Minor grid lines (same as single frame)
        for x in range(int(x_min), int(x_max) + 1, minor_grid):
            ax.axvline(x=x, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=1)
        for y in range(int(y_min), int(y_max) + 1, minor_grid):
            ax.axhline(y=y, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Major grid lines (same as single frame)
        for x in range(int(x_min), int(x_max) + 1, major_grid):
            ax.axvline(x=x, color='gray', linestyle='-', alpha=0.6, linewidth=0.8, zorder=1)
        for y in range(int(y_min), int(y_max) + 1, major_grid):
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.6, linewidth=0.8, zorder=1)
        
        # Add coordinate axes at origin for reference (same as single frame)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        
        # Plot point cloud background if available
        if background_points is not None and len(background_points) > 0:
            # Filter points to current view range
            view_range_bounds = [curr_x - view_range, curr_y - view_range, 
                               curr_x + view_range, curr_y + view_range]
            visible_points = self.filter_points_in_range(background_points, view_range_bounds)
            
            if len(visible_points) > 0:
                # Use height (z) for coloring, with a suitable colormap (same as single frame)
                ax.scatter(visible_points[:, 0], visible_points[:, 1], 
                          s=1.0, c=visible_points[:, 2], cmap='viridis', 
                          alpha=0.6, vmin=background_points[:, 2].min(), 
                          vmax=background_points[:, 2].max(), zorder=3)
        
        # Plot trail with fading effect
        if len(trail_points) > 1:
            trail_x = [p['position'][0] for p in trail_points]
            trail_y = [p['position'][1] for p in trail_points]
            
            # Create fading trail
            for i in range(len(trail_points) - 1):
                alpha = (i + 1) / len(trail_points) * 0.8
                ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                       color=self.trajectory_colors['trajectory_line'], 
                       linewidth=3, alpha=alpha, zorder=5)
        
        # Plot current position
        ax.plot(curr_x, curr_y, 'o', color=self.trajectory_colors['ego_current'], 
               markersize=15, markeredgecolor='white', markeredgewidth=3, zorder=8)
        
        # Plot orientation arrow
        yaw = quaternion_to_yaw(current_point['rotation'])
        arrow_length = 10
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        ax.arrow(curr_x, curr_y, dx, dy, head_width=4, head_length=3, 
                fc='red', ec='white', linewidth=2, zorder=9)
        
        # Show ego coordinate axes if requested
        if show_ego_axes:
            # Show current ego coordinate axes
            self.draw_ego_coordinate_axes(
                ax, current_point['position'], current_point['rotation'],
                axis_length=12, line_width=2, alpha=0.9, zorder=9
            )
        
        # Set labels and title (same style as single frame)
        ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
        ax.set_title(f'Ego Vehicle Trajectory Animation\nFrame {frame_idx + 1} | Position: ({curr_x:.1f}, {curr_y:.1f})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set aspect ratio to equal (same as single frame)
        ax.set_aspect('equal')
        
        # Save frame
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def draw_ego_coordinate_axes(self, ax, position, rotation, axis_length=15, 
                                line_width=3, alpha=0.9, zorder=10):
        """
        Draw ego vehicle coordinate axes (x: forward, y: left)
        
        Args:
            ax: Matplotlib axis object
            position: Ego vehicle position [x, y, z]
            rotation: Ego vehicle rotation quaternion
            axis_length: Length of coordinate axes in meters
            line_width: Width of axis lines
            alpha: Transparency of axes
            zorder: Drawing order
        """
        ego_x, ego_y = position[0], position[1]
        yaw = quaternion_to_yaw(rotation)
        
        # Calculate axis directions
        # X-axis (forward direction) - Red
        x_axis_dx = axis_length * np.cos(yaw)
        x_axis_dy = axis_length * np.sin(yaw)
        
        # Y-axis (left direction) - Green  
        y_axis_dx = axis_length * np.cos(yaw + np.pi/2)
        y_axis_dy = axis_length * np.sin(yaw + np.pi/2)
        
        # Draw X-axis (forward, red)
        ax.arrow(ego_x, ego_y, x_axis_dx, x_axis_dy, 
                head_width=axis_length*0.15, head_length=axis_length*0.1,
                fc='red', ec='darkred', linewidth=line_width, 
                alpha=alpha, zorder=zorder, label='Ego X-axis (Forward)')
        
        # Draw Y-axis (left, green)
        ax.arrow(ego_x, ego_y, y_axis_dx, y_axis_dy,
                head_width=axis_length*0.15, head_length=axis_length*0.1,
                fc='green', ec='darkgreen', linewidth=line_width,
                alpha=alpha, zorder=zorder, label='Ego Y-axis (Left)')
        
        # Add text labels
        # X-axis label
        x_label_x = ego_x + x_axis_dx * 1.2
        x_label_y = ego_y + x_axis_dy * 1.2
        ax.text(x_label_x, x_label_y, 'X', fontsize=12, fontweight='bold',
               color='red', ha='center', va='center', zorder=zorder+1)
        
        # Y-axis label  
        y_label_x = ego_x + y_axis_dx * 1.2
        y_label_y = ego_y + y_axis_dy * 1.2
        ax.text(y_label_x, y_label_y, 'Y', fontsize=12, fontweight='bold',
               color='green', ha='center', va='center', zorder=zorder+1)
        
        # Add origin point
        ax.plot(ego_x, ego_y, 'ko', markersize=8, markeredgecolor='white', 
               markeredgewidth=2, zorder=zorder+1, label='Ego Origin')

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize ego vehicle trajectory on map')
    parser.add_argument('--dataroot', type=str, default='data/v2x-seq-nuscenes/vehicle-side',
                       help='Path to the NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--scene_idx', type=int, default=0,
                       help='Index of the scene to visualize')
    parser.add_argument('--start_sample', type=int, default=0,
                       help='Starting sample index')
    parser.add_argument('--end_sample', type=int, default=None,
                       help='Ending sample index (if None, use all samples)')
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for sampling trajectory points')
    parser.add_argument('--range', type=float, nargs=4, default=[-200, -200, 200, 200],
                       help='Point cloud range: xmin ymin xmax ymax')
    parser.add_argument('--output_dir', type=str, default='trajectory_output',
                       help='Directory to save the output')
    parser.add_argument('--animate', action='store_true',
                       help='Create trajectory animation')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for animation')
    parser.add_argument('--trail_length', type=int, default=20,
                       help='Length of trajectory trail in animation')
    parser.add_argument('--show_point_cloud', action='store_true', default=True,
                       help='Show point cloud background from first frame')
    parser.add_argument('--no_point_cloud', action='store_true',
                       help='Disable point cloud background')
    parser.add_argument('--show_ego_axes', action='store_true', default=True,
                       help='Show ego vehicle coordinate axes')
    parser.add_argument('--no_ego_axes', action='store_true',
                       help='Disable ego coordinate axes')
    parser.add_argument('--ego_axes_interval', type=int, default=8,
                       help='Interval for showing ego axes (every N trajectory points)')
    
    args = parser.parse_args()
    
    print(f"Using point cloud range: {args.range}")
    
    # Initialize NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Initialize sequence visualizer
    visualizer = BEVSequenceVisualizer(nusc, scene_idx=args.scene_idx, 
                                      point_cloud_range=args.range)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.animate:
        # Create trajectory animation
        print("Creating trajectory animation...")
        video_path = visualizer.create_trajectory_animation(
            start_sample_idx=args.start_sample,
            end_sample_idx=args.end_sample,
            step=args.step,
            output_dir=args.output_dir,
            fps=args.fps,
            trail_length=args.trail_length,
            show_point_cloud=not args.no_point_cloud,
            show_ego_axes=not args.no_ego_axes
        )
        if video_path:
            print(f"Animation saved to: {video_path}")
    else:
        # Create static trajectory visualization
        output_path = os.path.join(args.output_dir, 
                                  f'trajectory_scene{args.scene_idx}_samples{args.start_sample}-{args.end_sample or "end"}.png')
        
        visualizer.visualize_trajectory_on_map(
            start_sample_idx=args.start_sample,
            end_sample_idx=args.end_sample,
            step=args.step,
            save_path=output_path,
            show_plot=True,
            show_point_cloud=not args.no_point_cloud,
            show_ego_axes=not args.no_ego_axes,
            ego_axes_interval=args.ego_axes_interval
        )
        
        print(f"Trajectory visualization saved to: {output_path}")
