import os
import re
import ast
import argparse
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from utils import *
from vlm import ModelHandler

class EgoTrajectoryVisualizer:
    """
    Visualizer for ego vehicle trajectory in world coordinates with optional point cloud background
    """
    def __init__(self, figsize=(12, 12), nusc=None):
        self.figsize = figsize
        self.nusc = nusc  # NuScenes instance for point cloud loading
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
        if self.nusc is None:
            print("Warning: NuScenes instance not provided, cannot load point cloud")
            return None
            
        try:
            sample = self.nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            
            # Load point cloud in sensor coordinates
            pc = LidarPointCloud.from_file(lidar_path)
            
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
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None
    
    def filter_points_in_range(self, points, point_cloud_range):
        """
        Filter points within the specified range
        
        Args:
            points: Nx4 array of point cloud points (x, y, z, intensity)
            point_cloud_range: [xmin, ymin, xmax, ymax] range for filtering
            
        Returns:
            filtered_points: Points within the specified range
        """
        if points is None or len(points) == 0:
            return None
            
        x_min, y_min, x_max, y_max = point_cloud_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        return points[mask]
    
    def visualize_scene_trajectory(self, ego_positions, ego_headings, scene_name, 
                                 save_path=None, show_plot=True, current_index=None,
                                 show_orientation_arrows=True, trajectory_line_width=3, 
                                 point_size=50, show_ego_axes=True, ego_axes_interval=5,
                                 show_point_cloud=False, first_sample_token=None,
                                 point_cloud_alpha=0.6, point_cloud_size=1.0):
        """
        Visualize ego vehicle trajectory for a complete scene in world coordinates with optional point cloud background
        
        Args:
            ego_positions: List of ego positions [(x, y), ...]
            ego_headings: List of ego headings [yaw1, yaw2, ...]
            scene_name: Name of the scene
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            current_index: Index of current position to highlight (optional)
            show_orientation_arrows: Whether to show orientation arrows
            trajectory_line_width: Width of the trajectory line
            point_size: Size of trajectory points
            show_ego_axes: Whether to show ego coordinate axes
            ego_axes_interval: Interval for showing ego axes (every N points)
            show_point_cloud: Whether to show point cloud background from first frame
            first_sample_token: Sample token for the first frame (required if show_point_cloud=True)
            point_cloud_alpha: Transparency of point cloud points
            point_cloud_size: Size of point cloud points
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if not ego_positions or len(ego_positions) < 2:
            print("Insufficient trajectory points for visualization!")
            return None, None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_facecolor('#FAFAFA')  # Very light gray background
        
        # Extract trajectory coordinates
        traj_x = [pos[0] for pos in ego_positions]
        traj_y = [pos[1] for pos in ego_positions]
        
        # Calculate trajectory bounds and adjust visualization range
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
        
        # Load and render point cloud background if requested
        first_frame_points = None
        if show_point_cloud and first_sample_token and self.nusc:
            try:
                print("Loading point cloud from first frame...")
                first_frame_points = self.get_point_cloud(first_sample_token)
                if first_frame_points is not None:
                    print(f"Loaded {len(first_frame_points)} total points")
                    first_frame_points = self.filter_points_in_range(first_frame_points, adjusted_range)
                    if first_frame_points is not None:
                        print(f"Filtered to {len(first_frame_points)} points within visualization range")
                    else:
                        print("No points found within visualization range")
                else:
                    print("Failed to load point cloud")
            except Exception as e:
                print(f"Error loading point cloud: {e}")
                first_frame_points = None
        
        # Create background with grid
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
        scatter = None
        if first_frame_points is not None and len(first_frame_points) > 0:
            print("Rendering point cloud background...")
            # Use height (z) for coloring, with a suitable colormap
            scatter = ax.scatter(first_frame_points[:, 0], first_frame_points[:, 1], 
                               s=point_cloud_size, c=first_frame_points[:, 2], cmap='viridis', 
                               alpha=point_cloud_alpha, vmin=first_frame_points[:, 2].min(), 
                               vmax=first_frame_points[:, 2].max(), zorder=3,
                               label=f'LiDAR Points ({len(first_frame_points):,})')
        
        # Plot trajectory line
        ax.plot(traj_x, traj_y, color=self.trajectory_colors['trajectory_line'], 
               linewidth=trajectory_line_width, alpha=0.8, zorder=5, 
               label=f'Ego Trajectory ({len(ego_positions)} points)')
        
        # Plot trajectory points
        ax.scatter(traj_x, traj_y, c=self.trajectory_colors['ego_trajectory'], 
                  s=point_size, alpha=0.7, zorder=6, edgecolors='white', linewidth=1)
        
        # Mark start and end positions
        if len(ego_positions) > 0:
            # Start position
            start_x, start_y = ego_positions[0][0], ego_positions[0][1]
            ax.plot(start_x, start_y, 'o', color=self.trajectory_colors['ego_start'], 
                   markersize=12, markeredgecolor='white', markeredgewidth=2, 
                   label='Start Position', zorder=8)
            
            # End position
            if len(ego_positions) > 1:
                end_x, end_y = ego_positions[-1][0], ego_positions[-1][1]
                ax.plot(end_x, end_y, 's', color=self.trajectory_colors['ego_end'], 
                       markersize=12, markeredgecolor='white', markeredgewidth=2, 
                       label='End Position', zorder=8)
        
        # Highlight current position if specified
        if current_index is not None and 0 <= current_index < len(ego_positions):
            curr_x, curr_y = ego_positions[current_index][0], ego_positions[current_index][1]
            ax.plot(curr_x, curr_y, 'o', color=self.trajectory_colors['ego_current'], 
                   markersize=15, markeredgecolor='white', markeredgewidth=3, 
                   label='Current Position', zorder=9)
        
        # Show orientation arrows if requested
        if show_orientation_arrows and len(ego_positions) > 1:
            # Show arrows at regular intervals
            arrow_step = max(1, len(ego_positions) // 10)  # Show about 10 arrows
            for i in range(0, len(ego_positions), arrow_step):
                if i < len(ego_headings):
                    x, y = ego_positions[i][0], ego_positions[i][1]
                    yaw = ego_headings[i]
                    
                    arrow_length = 8
                    dx = arrow_length * np.cos(yaw)
                    dy = arrow_length * np.sin(yaw)
                    
                    ax.arrow(x, y, dx, dy, head_width=3, head_length=2, 
                            fc=self.trajectory_colors['ego_current'], 
                            ec='white', linewidth=1, alpha=0.8, zorder=7)
        
        # Show ego coordinate axes if requested
        if show_ego_axes and len(ego_positions) > 0:
            # Show ego axes at regular intervals
            axes_step = max(1, ego_axes_interval)
            for i in range(0, len(ego_positions), axes_step):
                if i < len(ego_headings):
                    self.draw_ego_coordinate_axes(
                        ax, ego_positions[i], ego_headings[i],
                        axis_length=12, line_width=2, alpha=0.8, zorder=9
                    )
        
        # Set axis limits
        ax.set_xlim(adjusted_range[0], adjusted_range[2])
        ax.set_ylim(adjusted_range[1], adjusted_range[3])
        
        # Set axis labels and title
        ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
        
        title_parts = [f'Ego Vehicle Trajectory in World Coordinates']
        if show_point_cloud and first_frame_points is not None:
            title_parts.append('with LiDAR Point Cloud Background')
        title_parts.append(f'Scene: {scene_name}')
        title = '\n'.join(title_parts)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                 fancybox=True, shadow=True)
        
        # Add colorbar for point cloud height if points exist
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Height (m)', fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)
        
        # Add trajectory info text box
        distance = self._calculate_trajectory_distance(ego_positions)
        
        # Calculate visualization area
        vis_width = adjusted_range[2] - adjusted_range[0]
        vis_height = adjusted_range[3] - adjusted_range[1]
        
        # Prepare info text
        info_text = (f"Trajectory Info:\n"
                    f"Points: {len(ego_positions)}\n"
                    f"Distance: {distance:.1f}m\n"
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
    
    def draw_ego_coordinate_axes(self, ax, position, heading, axis_length=15, 
                                line_width=3, alpha=0.9, zorder=10):
        """
        Draw ego vehicle coordinate axes (x: forward, y: left) in world coordinates
        
        Args:
            ax: Matplotlib axis object
            position: Ego vehicle position (x, y)
            heading: Ego vehicle heading (yaw angle in radians)
            axis_length: Length of coordinate axes in meters
            line_width: Width of axis lines
            alpha: Transparency of axes
            zorder: Drawing order
        """
        ego_x, ego_y = position[0], position[1]
        yaw = heading
        
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
                alpha=alpha, zorder=zorder)
        
        # Draw Y-axis (left, green)
        ax.arrow(ego_x, ego_y, y_axis_dx, y_axis_dy,
                head_width=axis_length*0.15, head_length=axis_length*0.1,
                fc='green', ec='darkgreen', linewidth=line_width,
                alpha=alpha, zorder=zorder)
        
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
               markeredgewidth=2, zorder=zorder+1)
    
    def _calculate_trajectory_distance(self, positions):
        """
        Calculate total distance traveled along the trajectory
        
        Args:
            positions: List of positions [(x, y), ...]
            
        Returns:
            total_distance: Total distance in meters
        """
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            prev_pos = np.array(positions[i-1])
            curr_pos = np.array(positions[i])
            distance = np.linalg.norm(curr_pos - prev_pos)
            total_distance += distance
        
        return total_distance

    def overlay_trajectory_on_first_frame(self, first_frame_image_path, ego_positions, 
                                        ego_headings, camera_params, scene_name,
                                        save_path=None, trajectory_color=(255, 0, 0),
                                        line_width=3, point_size=8, show_arrows=True,
                                        arrow_interval=5, alpha=0.8):
        """
        Overlay the complete sequence trajectory on the first frame image
        
        Args:
            first_frame_image_path: Path to the first frame image
            ego_positions: List of ego positions [(x, y), ...] in world coordinates
            ego_headings: List of ego headings [yaw1, yaw2, ...] in radians
            camera_params: Camera parameters including rotation, translation, and intrinsic matrix
            scene_name: Name of the scene
            save_path: Path to save the visualization
            trajectory_color: RGB color for trajectory (default: red)
            line_width: Width of trajectory line
            point_size: Size of trajectory points
            show_arrows: Whether to show orientation arrows
            arrow_interval: Interval for showing arrows (every N points)
            alpha: Transparency of trajectory overlay
            
        Returns:
            success: Boolean indicating if visualization was successful
        """
        try:
            # Load the first frame image
            original_img = cv2.imread(first_frame_image_path)
            if original_img is None:
                print(f"Failed to load image from path: {first_frame_image_path}")
                return False

            img = original_img.copy()
            
            # Get first frame ego position and heading for transformation
            first_ego_pos = ego_positions[0]
            first_ego_heading = ego_headings[0]
            
            # --- Form Transformation Matrices ---
            # Ego to global transformation matrix
            T_ego_global = np.eye(4)
            T_ego_global[:3, :3] = np.array([
                [np.cos(first_ego_heading), -np.sin(first_ego_heading), 0],
                [np.sin(first_ego_heading), np.cos(first_ego_heading), 0],
                [0, 0, 1],
            ])
            T_ego_global[:3, 3] = np.array([first_ego_pos[0], first_ego_pos[1], 0])

            # Camera to ego transformation matrix
            T_cam_ego = np.eye(4)
            # T_cam_ego[:3, :3] = Quaternion(camera_params["rotation"]).rotation_matrix
            T_cam_ego[:3, :3] = np.array(camera_params["rotation"])
            T_cam_ego[:3, 3] = np.array(camera_params["translation"])

            # Camera to global transformation matrix
            T_cam_global = T_ego_global @ T_cam_ego
            T_global_cam = np.linalg.inv(T_cam_global)

            # Transform world trajectory points to camera coordinates
            points3d_world = [np.array([pos[0], pos[1], 0.0]) for pos in ego_positions]
            points3d_cam = np.array(
                [(T_global_cam @ np.append(p, 1))[:3] for p in points3d_world]
            )
            
            # Filter points that are in front of the camera
            valid = points3d_cam[:, 2] > 0
            if not valid.any():
                print("No trajectory points are visible in the camera view")
                return False
                
            # Project valid points onto the image plane
            points_valid = points3d_cam[valid]
            valid_indices = np.where(valid)[0]
            
            proj = (np.array(camera_params["camera_intrinsic"]) @ points_valid.T).T
            points2d_img = proj[:, :2] / proj[:, 2][:, np.newaxis]
            
            # Filter points that are within image bounds
            img_height, img_width = img.shape[:2]
            in_bounds = (
                (points2d_img[:, 0] >= 0) & (points2d_img[:, 0] < img_width) &
                (points2d_img[:, 1] >= 0) & (points2d_img[:, 1] < img_height)
            )
            
            if not in_bounds.any():
                print("No trajectory points are within image bounds")
                return False
            
            # Get final valid points and their indices
            final_points = points2d_img[in_bounds]
            final_indices = valid_indices[in_bounds]
            
            print(f"Projecting {len(final_points)} trajectory points onto image (with 1.5m downward offset)")
            
            # Create figure for overlay
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.set_position([0, 0, 1, 1])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)

            # Convert BGR color to RGB and normalize
            color_rgb = (trajectory_color[2]/255, trajectory_color[1]/255, trajectory_color[0]/255, alpha)
            
            if len(final_points) > 1:
                # Draw trajectory line
                x_coords = final_points[:, 0]
                y_coords = final_points[:, 1]
                ax.plot(x_coords, y_coords, color=color_rgb, linewidth=line_width, 
                       linestyle='solid', alpha=alpha, label='Ego Trajectory')
                
                # Draw trajectory points
                ax.scatter(x_coords, y_coords, c=[color_rgb], s=point_size**2, 
                          alpha=alpha, edgecolors='white', linewidth=1, zorder=6)
                
                # Draw orientation arrows if requested
                if show_arrows and len(final_indices) > 1:
                    arrow_step = max(1, arrow_interval)
                    for i in range(0, len(final_indices), arrow_step):
                        if i < len(final_indices) - 1:
                            # Get current and next point for arrow direction
                            curr_idx = final_indices[i]
                            next_idx = final_indices[min(i + 1, len(final_indices) - 1)]
                            
                            if curr_idx < len(ego_headings):
                                # Use heading for arrow direction
                                yaw = ego_headings[curr_idx]
                                
                                # Project arrow direction to image coordinates
                                # Apply same 1.5m downward offset for arrow start point
                                arrow_start_world = np.array([ego_positions[curr_idx][0], ego_positions[curr_idx][1], -1.5])
                                arrow_length_world = 5.0  # 5 meters in world coordinates
                                arrow_end_world = arrow_start_world + np.array([
                                    arrow_length_world * np.cos(yaw),
                                    arrow_length_world * np.sin(yaw),
                                    0
                                ])
                                
                                # Transform arrow end to camera coordinates
                                arrow_end_cam = (T_global_cam @ np.append(arrow_end_world, 1))[:3]
                                
                                if arrow_end_cam[2] > 0:  # Check if arrow end is in front of camera
                                    # Project to image coordinates
                                    arrow_end_proj = np.array(camera_params["camera_intrinsic"]) @ arrow_end_cam
                                    arrow_end_img = arrow_end_proj[:2] / arrow_end_proj[2]
                                    
                                    # Draw arrow
                                    start_point = final_points[i]
                                    arrow_vector = arrow_end_img - start_point
                                    arrow_norm = np.linalg.norm(arrow_vector)
                                    
                                    if arrow_norm > 10:  # Only draw if arrow is long enough in pixels
                                        # Normalize and scale arrow
                                        arrow_vector = arrow_vector / arrow_norm * 20  # 20 pixel length
                                        
                                        ax.annotate('', xy=start_point + arrow_vector, xytext=start_point,
                                                  arrowprops=dict(arrowstyle='->', color=color_rgb, 
                                                                lw=line_width, mutation_scale=15))
            
            # Add start and end markers
            if len(final_points) > 0:
                # Start point (green)
                ax.plot(final_points[0, 0], final_points[0, 1], 'o', 
                       color='green', markersize=12, markeredgecolor='white', 
                       markeredgewidth=2, label='Start', zorder=8)
                
                # End point (blue) 
                if len(final_points) > 1:
                    ax.plot(final_points[-1, 0], final_points[-1, 1], 's', 
                           color='blue', markersize=12, markeredgecolor='white', 
                           markeredgewidth=2, label='End', zorder=8)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
            
            # Add title
            ax.text(0.5, 0.02, f'Ego Trajectory Overlay - Scene: {scene_name}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            # Convert matplotlib figure to OpenCV image
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            # Remove any extra padding
            if buf.shape[0] > img.shape[0]:
                buf = buf[:img.shape[0], :, :]
            if buf.shape[1] > img.shape[1]:
                buf = buf[:, :img.shape[1], :]
                
            result_img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            # Close matplotlib figure to prevent memory leaks
            plt.close(fig)

            # Save the result
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_img)
                print(f"Trajectory overlay saved to: {save_path}")
            else:
                # Use default path
                default_path = f"trajectory_overlay_{scene_name}.png"
                cv2.imwrite(default_path, result_img)
                print(f"Trajectory overlay saved to: {default_path}")
            
            return True
            
        except Exception as e:
            print(f"Error creating trajectory overlay: {e}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: End-to-End Autonomous Driving")
    parser.add_argument("--model", type=str, default="qwen2.5-3b", 
                        help="Model to use for reasoning (default: gpt-4o, "
                        "options: gpt-4o, gpt-4.1, claude-3.7, claude-3.5, "
                        "gemini-2.5, gemini-2.0, qwen2.5-7b, qwen2.5-72b, "
                        "deepseek-vl2-16b, deepseek-vl2-28b, llama-3.2-11b, "
                        "llama-3.2-90b)")
    parser.add_argument("--model_weights", type=str, default="/home/yanglei/QWen/Qwen2.5-VL-3B-Instruct/",
                        help="Path to local model weights (for local models only)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Optional: Specific scene name to process.")
    parser.add_argument("--all_scenes", action="store_true",
                        help="Process all scenes instead of random sampling")
    parser.add_argument("--continue_dir", type=str, default=None,
                        help="Path to the directory with previously processed scene JSON files to resume processing")
    
    # Trajectory visualization arguments
    parser.add_argument("--visualize_trajectory", action="store_true",
                        help="Enable trajectory visualization for each scene")
    parser.add_argument("--viz_output_dir", type=str, default="trajectory_visualizations",
                        help="Directory to save trajectory visualizations")
    parser.add_argument("--show_plot", action="store_true",
                        help="Display trajectory plots (default: False, only save)")
    parser.add_argument("--show_orientation_arrows", action="store_true", default=True,
                        help="Show orientation arrows on trajectory")
    parser.add_argument("--show_ego_axes", action="store_true", default=True,
                        help="Show ego coordinate axes on trajectory")
    parser.add_argument("--ego_axes_interval", type=int, default=10,
                        help="Interval for showing ego axes (every N trajectory points)")
    
    # Point cloud visualization arguments
    parser.add_argument("--show_point_cloud", action="store_true",
                        help="Show LiDAR point cloud background in trajectory visualization")
    parser.add_argument("--point_cloud_alpha", type=float, default=0.6,
                        help="Transparency of point cloud points (0.0-1.0)")
    parser.add_argument("--point_cloud_size", type=float, default=1.0,
                        help="Size of point cloud points")
    
    # First frame trajectory overlay arguments
    parser.add_argument("--overlay_trajectory_on_first_frame", action="store_true",
                        help="Overlay complete sequence trajectory on first frame image")
    parser.add_argument("--overlay_output_dir", type=str, default="trajectory_overlays",
                        help="Directory to save trajectory overlay images")
    parser.add_argument("--overlay_color", type=str, default="255,0,0",
                        help="RGB color for trajectory overlay (format: R,G,B)")
    parser.add_argument("--overlay_line_width", type=int, default=3,
                        help="Width of trajectory line in overlay")
    parser.add_argument("--overlay_point_size", type=int, default=8,
                        help="Size of trajectory points in overlay")
    parser.add_argument("--overlay_show_arrows", action="store_true", default=True,
                        help="Show orientation arrows in trajectory overlay")
    parser.add_argument("--overlay_arrow_interval", type=int, default=5,
                        help="Interval for showing arrows in overlay (every N points)")
    parser.add_argument("--overlay_alpha", type=float, default=0.8,
                        help="Transparency of trajectory overlay (0.0-1.0)")
    
    return parser.parse_args()


def run_prediction():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Configure output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Use the provided directory for continuation, or create a new one
    if args.continue_dir:
        results_dir = args.continue_dir
        print(f"Continuing from existing directory: {results_dir}")
    else:
        results_dir = f"{config['data']['results']}/{args.model}_{timestamp}/output"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created new results directory: {results_dir}")
    
    # Initialize random seed for reproducibility
    random.seed(42)
    
    # Load NuScenes parameters from config
    OBS_LEN = config["prediction"]["obs_len"]
    FUT_LEN = config["prediction"]["fut_len"]
    EXT_LEN = config["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN 
    
    # Initialize model
    model_handler = ModelHandler(args.model, args.config, model_weights=args.model_weights)
    model_handler.model_instance, model_handler.processor = model_handler.initialize_model()
    print(f"Using model: {args.model}")
    
    # Initialize NuScenes dataset
    nusc = NuScenes(version=config["data"]["version"], dataroot=config["data"]["root"], verbose=True)
    
    # Select scenes to process
    if args.scene:
        # Find the specific scene by name
        selected_scenes = [scene for scene in nusc.scene if scene["name"] == args.scene]
        if not selected_scenes:
            print(f"Scene '{args.scene}' not found in dataset")
            return
    else:
        # Process all scenes if no specific scene is specified
        selected_scenes = nusc.scene
        print(f"Processing all {len(selected_scenes)} scenes")
    # Get list of already processed scenes if continuing
    processed_scene_names = []
    if args.continue_dir:
        for filename in os.listdir(args.continue_dir):
            if filename.endswith('.json'):
                processed_scene_names.append(filename.replace('.json', ''))
        print(f"Found {len(processed_scene_names)} previously processed scenes")
    
    # Process each selected scene
    for scene in selected_scenes:
        scene_name = scene["name"]

        # Skip already processed scenes when in continuation mode
        if args.continue_dir and scene_name in processed_scene_names:
            print(f"Skipping already processed scene: {scene_name}")
            continue

        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        description = scene["description"]
        
        print(f"\nProcessing scene '{scene_name}': {description}")
        
        # Create scene data structure
        scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": first_sample_token,
                "last_sample_token": last_sample_token
            },
            "frames": [],
            "metadata": {
                "model": args.model,
                "timestamp": timestamp,
                "total_frames": 0
            }
        }
        
        # Collect scene data
        camera_params = []
        front_camera_images = []
        ego_positions = []
        ego_headings = []
        timestamps = []
        sample_tokens = []
        
        curr_sample_token = first_sample_token
        
        # Retrieve all frames in the scene
        while curr_sample_token:
            sample = nusc.get("sample", curr_sample_token)
            sample_tokens.append(curr_sample_token)
            if "v2x-seq" not in nusc.dataroot:
                cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                front_camera_images.append(
                    os.path.join(nusc.dataroot, cam_front_data["filename"])
                )
            else:
                cam_front_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                front_camera_image = os.path.join(nusc.dataroot, cam_front_data["filename"].replace("velodyne", "image").replace("bin", "jpg"))
                front_camera_images.append(front_camera_image)
            
            # Get the camera parameters
            camera_param_path = os.path.join(nusc.dataroot, "v1.0-trainval/calibrated_sensor.json")
            with open(camera_param_path, 'r') as f:
                camera_param = json.load(f)[0]
            camera_params.append(camera_param)
            
            # Get ego vehicle state
            ego_state = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_positions.append(tuple(ego_state["translation"][0:2]))
            ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
            timestamps.append(ego_state["timestamp"])
            
            # Move to next sample or exit loop if at the end
            curr_sample_token = (
                sample["next"] if curr_sample_token != last_sample_token else None
            )
        
        num_frames = len(front_camera_images)
        
        # Check if we have enough frames
        if num_frames < TTL_LEN:
            print(f"Skipping '{scene_name}', insufficient frames ({num_frames} < {TTL_LEN}).")
            continue

        # Visualize trajectory if requested
        if args.visualize_trajectory and len(ego_positions) > 1:
            print(f"Creating trajectory visualization for scene '{scene_name}'...")
            
            # Initialize trajectory visualizer
            trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
            
            # Create visualization output directory
            viz_dir = args.viz_output_dir
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create visualization save path
            viz_save_path = os.path.join(viz_dir, f"{scene_name}_trajectory.png")
            
            try:
                # Create trajectory visualization
                fig, ax = trajectory_visualizer.visualize_scene_trajectory(
                    ego_positions=ego_positions,
                    ego_headings=ego_headings,
                    scene_name=scene_name,
                    save_path=viz_save_path,
                    show_plot=args.show_plot,
                    show_orientation_arrows=args.show_orientation_arrows,
                    show_ego_axes=args.show_ego_axes,
                    ego_axes_interval=args.ego_axes_interval,
                    show_point_cloud=args.show_point_cloud,
                    first_sample_token=first_sample_token,
                    point_cloud_alpha=args.point_cloud_alpha,
                    point_cloud_size=args.point_cloud_size
                )
                
                if fig is not None:
                    print(f"Trajectory visualization saved to: {viz_save_path}")
                else:
                    print(f"Failed to create trajectory visualization for scene '{scene_name}'")        
            except Exception as e:
                print(f"Error creating trajectory visualization for scene '{scene_name}': {e}")
                continue
        
        # Overlay trajectory on first frame if requested
        if args.overlay_trajectory_on_first_frame and len(ego_positions) > 1:
            print(f"Creating trajectory overlay on first frame for scene '{scene_name}'...")
            
            # Initialize trajectory visualizer if not already done
            if not args.visualize_trajectory:
                trajectory_visualizer = EgoTrajectoryVisualizer(nusc=nusc)
            
            # Create overlay output directory
            overlay_dir = args.overlay_output_dir
            os.makedirs(overlay_dir, exist_ok=True)
            
            # Parse overlay color
            try:
                color_parts = args.overlay_color.split(',')
                if len(color_parts) == 3:
                    overlay_color = tuple(int(c.strip()) for c in color_parts)
                else:
                    print(f"Invalid color format: {args.overlay_color}, using default red")
                    overlay_color = (255, 0, 0)
            except:
                print(f"Error parsing color: {args.overlay_color}, using default red")
                overlay_color = (255, 0, 0)
            
            # Create overlay save path
            overlay_save_path = os.path.join(overlay_dir, f"{scene_name}_trajectory_overlay.png")
            
            try:
                # Get first frame image path and camera parameters
                first_frame_image = front_camera_images[0]
                first_frame_camera_params = camera_params[0]
                
                # Create trajectory overlay
                success = trajectory_visualizer.overlay_trajectory_on_first_frame(
                    first_frame_image_path=first_frame_image,
                    ego_positions=ego_positions,
                    ego_headings=ego_headings,
                    camera_params=first_frame_camera_params,
                    scene_name=scene_name,
                    save_path=overlay_save_path,
                    trajectory_color=overlay_color,
                    line_width=args.overlay_line_width,
                    point_size=args.overlay_point_size,
                    show_arrows=args.overlay_show_arrows,
                    arrow_interval=args.overlay_arrow_interval,
                    alpha=args.overlay_alpha
                )
                
                if success:
                    print(f"Trajectory overlay saved to: {overlay_save_path}")
                else:
                    print(f"Failed to create trajectory overlay for scene '{scene_name}'")
            except Exception as e:
                print(f"Error creating trajectory overlay for scene '{scene_name}': {e}")
                continue
        
        # Process each frame in the scene
        for i in range(0, num_frames - TTL_LEN, 1):
            try:
                cur_index = i + OBS_LEN + 1
                frame_index = i  # The relative index in the processed subset
                
                image_path = front_camera_images[cur_index]
                print(f"Processing frame {i} from {scene_name}, image: {image_path}")
                
                # Extract image ID from filename
                match = re.search(r"(\d+)(?=\.jpg$)", image_path)
                image_id = match.group(1) if match else None
                
                sample_token = sample_tokens[cur_index]
                camera_param = camera_params[cur_index]
                
                # Get current position and heading
                cur_pos = ego_positions[cur_index]
                cur_heading = ego_headings[cur_index]

                # Get observation data (past positions and timestamps)
                obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]
                
                # Calculate past speeds and curvatures
                prev_speed = compute_speed(obs_pos, obs_time)
                prev_curvatures = compute_curvature(obs_pos)
                prev_actions = list(zip(prev_speed, prev_curvatures))
                
                # Get future positions and timestamps (ground truth)
                fut_pos = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                fut_time = timestamps[cur_index - 1 : cur_index + FUT_LEN + 1]
                
                # Calculate ground truth speeds and curvatures
                gt_speed = compute_speed(fut_pos, fut_time)
                gt_curvatures = compute_curvature(fut_pos)
                gt_actions = list(zip(gt_speed, gt_curvatures))
                
                # Remove extra indices used for speed and curvature calculation
                fut_pos = fut_pos[2:]
                
                # Define prompts for LLM inference
                scene_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "You must observe and analyze the movements of vehicles and pedestrians, "
                    "lane markings, traffic lights, and any relevant objects in the scene. "
                    "describe what you observe, but do not infer the ego's action. "
                    "generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run scene description inference
                scene_description, scene_tokens, scene_time = model_handler.get_response(
                    prompt=scene_prompt,
                    image_path=image_path
                )
                print("Scene description:", scene_description)
                
                # Generate intent prompt based on scene description
                intent_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "What was the ego's previous intent? "
                    "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
                    "Was it turning left (by how much), turning right (by how much), or following the lane? "
                    "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
                    "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
                    "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
                    "Generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run driving intent inference
                driving_intent, intent_tokens, intent_time = model_handler.get_response(
                    prompt=intent_prompt,
                    image_path=image_path
                )
                print("Driving intent:", driving_intent)
                
                # Generate waypoint prompt based on scene and intent
                waypoint_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "The high-level driving instructions are as follows: "
                    f"{driving_intent} "
                    "Predict the speed and curvature for the next 6 waypoints, with 0.5-second resolution. "
                    "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                    "Predict Exactly 6 pairs of speed and curvature, in the format:"
                    "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
                    "ONLY return the answers in the required format, do not include punctuation or text."
                )
                
                # Run waypoint prediction inference
                pred_actions_str, waypoint_tokens, waypoint_time = model_handler.get_response(
                    prompt=waypoint_prompt,
                    image_path=image_path
                )
                print("Predicted actions:", pred_actions_str)
                
                # Prepare frame data structure
                frame_data = {
                    "frame_index": frame_index,
                    "sample_token": sample_token,
                    "image_path": image_path,
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": camera_param["rotation"],
                        "translation": camera_param["translation"],
                        "camera_intrinsic": camera_param["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": fut_pos,
                        "gt_actions": gt_actions
                    },
                    "inference": {
                        "scene_prompt": format_long_text(scene_prompt),
                        "scene_description": format_long_text(scene_description),
                        "intent_prompt": format_long_text(intent_prompt),
                        "driving_intent": format_long_text(driving_intent),
                        "waypoint_prompt": format_long_text(waypoint_prompt),
                        "pred_actions_str": pred_actions_str
                    },
                    "token_usage": {
                        "scene_prompt": scene_tokens,
                        "intent_prompt": intent_tokens,
                        "waypoint_prompt": waypoint_tokens
                    },
                    "time_usage": {
                        "scene_prompt": scene_time,
                        "intent_prompt": intent_time,
                        "waypoint_prompt": waypoint_time
                    }
                }
                
                # Try to parse predicted actions and generate trajectory
                try:
                    pred_actions = ast.literal_eval(pred_actions_str)
                    if isinstance(pred_actions, list) and len(pred_actions) > 0:
                        prediction = integrate_driving_commands(pred_actions, dt=0.5)
                        frame_data["predictions"] = {
                            "pred_actions": pred_actions,
                            "trajectory": prediction
                        }
                    else:
                        frame_data["predictions"] = {
                            "pred_actions_str": pred_actions_str
                        }
                except Exception as e:
                    frame_data["predictions"] = {
                        "pred_actions_str": pred_actions_str
                    }
                
                # Add frame data to scene
                scene_data["frames"].append(frame_data)
                
            except Exception as e:
                print(f"Error processing frame {i} in {scene_name}: {e}")
                continue
        
        # Update total frames count
        scene_data["metadata"]["total_frames"] = len(scene_data["frames"])
        
        # Save scene data
        scene_file_path = f"{results_dir}/{scene_name}.json"
        save_dict_to_json(scene_data, scene_file_path)
        print(f"Scene data saved to {scene_file_path} with {len(scene_data['frames'])} frames")

if __name__ == "__main__":
    run_prediction()