import os
import json
import cv2
from pypcd import pypcd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from utils import quaternion_to_yaw
from nuscenes.map_expansion.map_api import NuScenesMap

class BEVVisualizer:
    def __init__(self, nusc, scene_idx=0, figsize=(10, 10), 
                 point_cloud_range=[-100, -100, 100, 100], resolution=0.2,
                 point_cloud_alpha=0.3, box_alpha=0.7, arrow_alpha=0.8):
        """
        Initialize the Bird's Eye View visualizer
        
        Args:
            nusc: NuScenes instance
            scene_idx: Index of the scene to visualize
            figsize: Figure size for the plot
            point_cloud_range: [xmin, ymin, xmax, ymax] range for BEV display
            resolution: Resolution of the BEV grid in meters
            point_cloud_alpha: Transparency for point cloud (0.0=transparent, 1.0=opaque)
            box_alpha: Transparency for 3D bounding boxes (0.0=transparent, 1.0=opaque)
            arrow_alpha: Transparency for direction arrows (0.0=transparent, 1.0=opaque)
        """
        self.nusc = nusc
        self.scene = nusc.scene[scene_idx]
        self.point_cloud_range = point_cloud_range
        self.resolution = resolution
        self.figsize = figsize
        self.point_cloud_alpha = point_cloud_alpha
        self.box_alpha = box_alpha
        self.arrow_alpha = arrow_alpha
        print("point_cloud_range: ", self.point_cloud_range)
        print(f"Transparency settings - Point cloud: {point_cloud_alpha}, Boxes: {box_alpha}, Arrows: {arrow_alpha}")
        
        # Define colors for different object categories
        self.category_colors = {
            'car': 'blue',
            'pedestrian': 'orange',
            'bicycle': 'magenta',
            'bus': 'cyan',
            'truck': 'red',
            'motorcycle': 'yellow',
            'default': 'gray'
        }
    
    def get_point_cloud(self, sample_token):
        """
        Get LiDAR point cloud for a sample
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            points: Nx4 array of point cloud points (x, y, z, intensity)
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        if not os.path.exists(lidar_path):
            lidar_path = lidar_path.replace('.bin', '.pcd')
        # Check file extension to determine loading method
        file_extension = os.path.splitext(lidar_path)[1].lower()
        if file_extension == '.pcd':
            # Load PCD file using pypcd
            points = self._load_pcd_with_pypcd(lidar_path)
        elif file_extension == '.bin':
            # Load binary file using NuScenes LidarPointCloud
            pc = LidarPointCloud.from_file(lidar_path)
            points = pc.points.T  # Nx4 array (x, y, z, intensity)
        else:
            raise ValueError(f"Unsupported point cloud file format: {file_extension}")
        
        return points
    
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
        
    def get_boxes(self, sample_token):
        """
        Get 3D bounding boxes for a sample and convert from world coordinates to LiDAR coordinates
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            boxes: List of box dictionaries with position, size, rotation, and category in LiDAR frame
        """
        sample = self.nusc.get('sample', sample_token)
        boxes = []
        
        # Get LiDAR pose information
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Get transformations
        ego_to_world = {
            'translation': lidar_ego_pose['translation'],
            'rotation': lidar_ego_pose['rotation']
        }
        
        lidar_to_ego = {
            'translation': lidar_calibrated_sensor['translation'],
            'rotation': lidar_calibrated_sensor['rotation']
        }
        
        # Get all annotations for this sample
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Convert world to LiDAR coordinates
            lidar_box = self.world_to_lidar_box(ann, ego_to_world, lidar_to_ego)
            
            # Get category - extract the general category type (like 'vehicle', 'pedestrian', etc.)
            category = ann['category_name']
            
            # Create box dictionary
            box = {
                'position': lidar_box['translation'],
                'size': ann['size'],  # Size is invariant to coordinate transformations
                'rotation': lidar_box['rotation'],
                'category': category
            }
            boxes.append(box)
        
        return boxes
    
    def world_to_lidar_box(self, box_in_world, ego_to_world, lidar_to_ego):
        """
        Transform a box from world coordinate to LiDAR coordinate
        
        Args:
            box_in_world: Box in world coordinate
            ego_to_world: Ego vehicle to world transformation
            lidar_to_ego: LiDAR to ego vehicle transformation
            
        Returns:
            Box in LiDAR coordinate
        """
        # Create transformation matrices
        ego_to_world_mat = self.transform_matrix(ego_to_world['translation'], ego_to_world['rotation'])
        lidar_to_ego_mat = self.transform_matrix(lidar_to_ego['translation'], lidar_to_ego['rotation'])
        
        # Compute world to lidar matrix
        world_to_ego_mat = np.linalg.inv(ego_to_world_mat)
        ego_to_lidar_mat = np.linalg.inv(lidar_to_ego_mat)
        world_to_lidar_mat = ego_to_lidar_mat @ world_to_ego_mat
        
        # Transform box position
        world_position = np.array(box_in_world['translation']).reshape(3, 1)
        world_position_hom = np.vstack([world_position, np.ones((1, 1))])
        lidar_position_hom = world_to_lidar_mat @ world_position_hom
        lidar_position = lidar_position_hom[:3].flatten().tolist()
        
        # Transform box rotation
        world_rotation = Quaternion(box_in_world['rotation'])
        world_rotation_mat = world_rotation.rotation_matrix
        world_rotation_mat_hom = np.eye(4)
        world_rotation_mat_hom[:3, :3] = world_rotation_mat
        
        # Only rotate, don't translate
        rot_world_to_lidar_mat = np.copy(world_to_lidar_mat)
        rot_world_to_lidar_mat[:3, 3] = 0
        
        lidar_rotation_mat_hom = rot_world_to_lidar_mat @ world_rotation_mat_hom
        lidar_rotation_mat = lidar_rotation_mat_hom[:3, :3]
        lidar_rotation = Quaternion(matrix=lidar_rotation_mat)
        
        return {
            'translation': lidar_position,
            'rotation': [lidar_rotation.w, lidar_rotation.x, lidar_rotation.y, lidar_rotation.z]
        }
    
    def transform_matrix(self, translation, rotation):
        """
        Create a 4x4 transformation matrix from translation and rotation
        
        Args:
            translation: Translation vector [x, y, z]
            rotation: Quaternion [w, x, y, z]
            
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = Quaternion(rotation).rotation_matrix
        transform[:3, 3] = translation
        return transform
    
    def filter_points_in_range(self, points):
        """
        Filter points within the specified range
        
        Args:
            points: Nx4 array of point cloud points (x, y, z, intensity)
            
        Returns:
            filtered_points: Points within the specified range
        """
        x_min, y_min, x_max, y_max = self.point_cloud_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        return points[mask]
    
    def filter_boxes_in_range(self, boxes):
        """
        Filter boxes that are within or partially overlapping with the specified range
        
        Args:
            boxes: List of box dictionaries with position, size, rotation, and category
            
        Returns:
            filtered_boxes: Boxes within or partially overlapping with the specified range
        """
        x_min, y_min, x_max, y_max = self.point_cloud_range
        filtered_boxes = []
        
        for box in boxes:
            x, y, _ = box['position']
            length, width, _ = box['size']
            
            # Get a conservative estimate of box bounds by considering max dimensions
            # This simple check will include boxes that might partially overlap with range
            max_dimension = max(length, width) / 2
            
            # Check if the box is completely outside the range
            if (x + max_dimension < x_min or 
                x - max_dimension > x_max or 
                y + max_dimension < y_min or 
                y - max_dimension > y_max):
                continue
            
            filtered_boxes.append(box)
            
        return filtered_boxes


    def get_image(self, sample_token):
        """
        Get the image of a sample
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        image_path = lidar_path.replace("velodyne", "image").replace("bin", "jpg")
        return image_path

    def transform_matrix_from_rotation_matrix(self, translation, rotation_matrix):
        """
        Create a 4x4 transformation matrix from translation and rotation matrix
        
        Args:
            translation: Translation vector [x, y, z]
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return transform
        
    def visualize_image(self, dataroot, sample_token, save_path=None, show_plot=True):
        """
        Visualize the image of a sample with projected 3D bounding boxes
        
        Args:
            sample_token: Token of the sample to visualize
            save_path: Path to save the visualization (if None, default path is used)
            show_plot: Whether to display the plot
        """
        # Get camera data (front camera)
        image_path = self.get_image(sample_token)
        image = cv2.imread(image_path)

        # Get 3D boxes in LiDAR coordinates
        boxes = self.get_boxes(sample_token)
        
        # Use camera parameters directly
        '''
        camera_params = {
            "rotation": np.array([[0.05365781, -0.01496656,  0.99844747],
            [-0.99855032, -0.00522108,  0.05358429],
            [ 0.00441089, -0.99987472, -0.01522545]]),
            "translation": np.array([-0.181511365002354, -0.025192293890423524, -0.2666933938012268]),
            "camera_intrinsic": np.array([[
                797.351678, 0.0, 993.586356], [
                0.0, 799.238376, 583.568721], [
                0.0, 0.0, 1.0
            ]])
        }
        '''
        # Get the camera parameters
        camera_param_path = os.path.join(dataroot, "v1.0-trainval/calibrated_sensor.json")
        with open(camera_param_path, 'r') as f:
            camera_params = json.load(f)[0]

        # Use camera intrinsic matrix directly from camera_params
        print("camera_params: ", camera_params)
        cam_intrinsic = camera_params['camera_intrinsic']
        
        # Get the sample
        sample = self.nusc.get('sample', sample_token)

        # Get transformations
        # 1. LiDAR to ego vehicle
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_to_ego_mat = self.transform_matrix(lidar_calibrated_sensor['translation'], lidar_calibrated_sensor['rotation'])
        
        # 2. Camera to ego vehicle - use camera_params directly with rotation matrix
        cam_to_ego_mat = self.transform_matrix_from_rotation_matrix(camera_params['translation'], camera_params['rotation'])
        
        # 3. Ego vehicle to camera (inverse of camera to ego)
        ego_to_cam_mat = np.linalg.inv(cam_to_ego_mat)
        
        # 4. LiDAR to camera: first transform to ego, then to camera
        lidar_to_cam_mat = ego_to_cam_mat @ lidar_to_ego_mat
        
        # Draw 3D boxes projected to 2D image
        for box in boxes:
            self._draw_box_in_image(image, box, lidar_to_cam_mat, cam_intrinsic)
        
        # Create output directory if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)
        else:
            # Use default path
            os.makedirs('bev_demo_output', exist_ok=True)
            cv2.imwrite("bev_demo_output/image_with_boxes.jpg", image)
        
        # Show the image if required
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return image
        
    def _draw_box_in_image(self, image, box, lidar_to_cam_mat, cam_intrinsic):
        """
        Draw a 3D box projected onto a 2D image
        
        Args:
            image: Image to draw on
            box: Box dictionary with position, size, rotation, and category
            lidar_to_cam_mat: Transformation matrix from LiDAR to camera
            cam_intrinsic: Camera intrinsic matrix
        """
        # Extract box parameters
        x, y, z = box['position']
        width, length, height = box['size']
        quaternion = box['rotation']
        category = box['category'].split('.')[1] if '.' in box['category'] else box['category']
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw = quaternion_to_yaw(quaternion)
        
        # Get color for category
        color_name = self.category_colors.get(category, self.category_colors['default'])
        # Convert named color to BGR tuple for OpenCV
        color_map = {
            'blue': (255, 0, 0),
            'orange': (0, 165, 255),
            'magenta': (255, 0, 255),
            'cyan': (255, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'gray': (128, 128, 128)
        }
        color = color_map.get(color_name, (128, 128, 128))  # Default to gray
        
        # Get 3D box corners in LiDAR coordinate
        corners_3d = self._get_3d_box_corners(x, y, z, length, width, height, yaw)
        
        # Transform corners from LiDAR to camera coordinate
        corners_cam = []
        for point in corners_3d:
            # Convert to homogeneous coordinates
            point_hom = np.append(point, 1)
            # Transform to camera coordinate
            point_cam = lidar_to_cam_mat @ point_hom
            corners_cam.append(point_cam[:3])
        
        # Project 3D points to 2D image plane
        corners_2d = []
        for point in corners_cam:
            # Check if point is in front of camera
            if point[2] <= 0:
                continue
            
            # Project point to image plane
            point_2d = cam_intrinsic @ (point / point[2])
            corners_2d.append((int(point_2d[0]), int(point_2d[1])))
        
        # Draw the box if we have at least 2 valid corners
        if len(corners_2d) >= 2:
            # Define the box edges (indices of connected corners)
            edges = [
                # Bottom face
                (0, 1), (1, 2), (2, 3), (3, 0),
                # Top face
                (4, 5), (5, 6), (6, 7), (7, 4),
                # Connecting top and bottom
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            
            # Draw lines for visible edges
            for i, j in edges:
                if i < len(corners_2d) and j < len(corners_2d):
                    cv2.line(image, corners_2d[i], corners_2d[j], color, 2)
            
            # Add category label
            if corners_2d:
                x_values = [p[0] for p in corners_2d]
                y_values = [p[1] for p in corners_2d]
                if x_values and y_values:
                    label_x = min(x_values)
                    label_y = min(y_values) - 10
                    cv2.putText(image, category, (label_x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def _get_3d_box_corners(self, x, y, z, length, width, height, yaw):
        """
        Get the eight corners of a 3D box
        
        Args:
            x, y, z: Center coordinates
            length, width, height: Box dimensions
            yaw: Rotation angle in radians
            
        Returns:
            corners: List of (x, y, z) coordinates for the corners
        """
        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2
        
        # Calculate corners (centered at origin, unrotated)
        corners = [
            # Bottom face
            [-half_length, -half_width, -half_height],
            [half_length, -half_width, -half_height],
            [half_length, half_width, -half_height],
            [-half_length, half_width, -half_height],
            # Top face
            [-half_length, -half_width, half_height],
            [half_length, -half_width, half_height],
            [half_length, half_width, half_height],
            [-half_length, half_width, half_height]
        ]
        
        # Rotate corners around z-axis
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotated_corners = []
        for cx, cy, cz in corners:
            # Rotate
            rx = cx * cos_yaw - cy * sin_yaw
            ry = cx * sin_yaw + cy * cos_yaw
            # Translate
            rx += x
            ry += y
            rz = cz + z
            rotated_corners.append(np.array([rx, ry, rz]))
        
        return rotated_corners

    def visualize_bev(self, sample_token, save_path=None, show_plot=True):
        """
        Visualize Bird's Eye View with point cloud and 3D boxes
        
        Args:
            sample_token: Token of the sample to visualize
            save_path: Path to save the visualization (if None, the plot is not saved)
            show_plot: Whether to display the plot
            
        Returns:
            fig, ax: Figure and axis objects
        """
        # Get point cloud and boxes
        points = self.get_point_cloud(sample_token)
        boxes = self.get_boxes(sample_token)
        
        # Filter points and boxes in range
        points = self.filter_points_in_range(points)
        boxes = self.filter_boxes_in_range(boxes)
        
        print(f"Displaying {len(boxes)} boxes within range {self.point_cloud_range}")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot point cloud
        ax.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap='viridis', alpha=self.point_cloud_alpha)
        
        # Plot boxes
        for box in boxes:
            self._plot_box(ax, box)
        
        # Set axis limits to exactly match the specified range
        ax.set_xlim(self.point_cloud_range[0], self.point_cloud_range[2])
        ax.set_ylim(self.point_cloud_range[1], self.point_cloud_range[3])
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Bird\'s Eye View ({self.point_cloud_range[0]},{self.point_cloud_range[1]}) to ({self.point_cloud_range[2]},{self.point_cloud_range[3]})')
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Z (height)')
        
        # Add legend for box categories
        handles = []
        labels = []
        for category, color in self.category_colors.items():
            if category != 'default':
                handle = plt.Rectangle((0, 0), 1, 1, color=color)
                handles.append(handle)
                labels.append(category)
        
        ax.legend(handles, labels, loc='upper right')
        
        # Save if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if required
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def _plot_box(self, ax, box):
        """
        Plot a 3D box in BEV
        
        Args:
            ax: Matplotlib axis
            box: Box dictionary with position, size, rotation, and category
        """
        # Extract box parameters
        x, y, z = box['position']
        width, length, height = box['size']
        quaternion = box['rotation']
        category = box['category']
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw = quaternion_to_yaw(quaternion)
        
        # Get color for category
        print("category: ", category)
        color = self.category_colors.get(category, self.category_colors['default'])
        
        # Calculate corners of the rectangle
        corners = self._get_corners(x, y, length, width, yaw)
        
        # Plot rectangle
        rect = Polygon(corners, fill=True, alpha=self.box_alpha, color=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Plot direction arrow
        arrow_length = length * 0.5
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        arrow = Arrow(x, y, dx, dy, width=width*0.5, color='red', alpha=self.arrow_alpha)
        ax.add_patch(arrow)
    
    def _get_corners(self, x, y, length, width, yaw):
        """
        Get the four corners of a rotated rectangle
        
        Args:
            x, y: Center coordinates
            length, width: Rectangle dimensions
            yaw: Rotation angle in radians
            
        Returns:
            corners: List of (x, y) coordinates for the corners
        """
        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2
        
        # Calculate corners (centered at origin, unrotated)
        corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ]
        
        # Rotate corners
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotated_corners = []
        for cx, cy in corners:
            # Rotate
            rx = cx * cos_yaw - cy * sin_yaw
            ry = cx * sin_yaw + cy * cos_yaw
            # Translate
            rx += x
            ry += y
            rotated_corners.append((rx, ry))
        
        return rotated_corners
    
    def visualize_all_frames(self, output_dir="all_frames", visualization_types=['bev'], 
                            max_frames_per_scene=None, skip_frames=1, fps=10):
        """
        Visualize all frames of all scenes in the dataset and save each scene as an mp4 video
        
        Args:
            output_dir: Directory to save all visualizations
            visualization_types: List of visualization types to generate
                                Options: ['bev', 'image', 'combined']
            max_frames_per_scene: Maximum number of frames to process per scene (None for all)
            skip_frames: Skip every N frames (1 means process all frames, 2 means every other frame)
            fps: Frames per second for the video
        """
        import time
        import cv2
        import tempfile
        import shutil
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        total_scenes = len(self.nusc.scene)
        print(f"Processing all frames from {total_scenes} scenes...")
        print(f"Skip frames: {skip_frames} (processing every {skip_frames} frame(s))")
        print(f"Video FPS: {fps}")
        if max_frames_per_scene:
            print(f"Max frames per scene: {max_frames_per_scene}")
        
        start_time = time.time()
        total_frames_processed = 0
        total_frames_skipped = 0
        
        for scene_idx, scene in enumerate(self.nusc.scene):
            try:
                print(f"\nProcessing scene {scene_idx + 1}/{total_scenes}: {scene['name']}")
                
                # Get all sample tokens for this scene
                sample_tokens = self._get_all_sample_tokens(scene)
                
                # Apply frame skipping
                if skip_frames > 1:
                    sample_tokens = sample_tokens[::skip_frames]
                    frames_skipped = len(self._get_all_sample_tokens(scene)) - len(sample_tokens)
                    total_frames_skipped += frames_skipped
                    print(f"  Skipped {frames_skipped} frames, processing {len(sample_tokens)} frames")
                
                # Apply max frames limit
                if max_frames_per_scene and len(sample_tokens) > max_frames_per_scene:
                    sample_tokens = sample_tokens[:max_frames_per_scene]
                    print(f"  Limited to {max_frames_per_scene} frames")
                
                scene_start_time = time.time()
                
                # Process each visualization type
                for vis_type in visualization_types:
                    print(f"  Creating {vis_type} video...")
                    
                    # Create temporary directory for frames
                    with tempfile.TemporaryDirectory() as temp_dir:
                        frame_paths = []
                        
                        # Generate all frames for this visualization type
                        for frame_idx, sample_token in enumerate(sample_tokens):
                            try:
                                frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
                                
                                if vis_type == 'bev':
                                    self.visualize_bev(sample_token, save_path=frame_path, show_plot=False)
                                elif vis_type == 'image':
                                    self.visualize_image(self.nusc.dataroot, sample_token, 
                                                       save_path=frame_path, show_plot=False)
                                elif vis_type == 'combined':
                                    # For combined visualization, we need to save the image directly
                                    combined_image = self.visualize_combined(self.nusc.dataroot, sample_token, 
                                                                           save_path=frame_path, show_plot=False)
                                else:
                                    print(f"    ✗ Unknown visualization type: {vis_type}")
                                    continue
                                
                                frame_paths.append(frame_path)
                                total_frames_processed += 1
                                
                                # Progress update every 10 frames
                                if (frame_idx + 1) % 10 == 0 or frame_idx == len(sample_tokens) - 1:
                                    scene_elapsed = time.time() - scene_start_time
                                    scene_avg_time = scene_elapsed / (frame_idx + 1)
                                    scene_remaining = scene_avg_time * (len(sample_tokens) - frame_idx - 1)
                                    
                                    print(f"    Frame {frame_idx + 1}/{len(sample_tokens)} "
                                          f"({(frame_idx + 1)/len(sample_tokens)*100:.1f}%) "
                                          f"Scene time: {scene_elapsed:.1f}s, Est. remaining: {scene_remaining:.1f}s")
                                
                            except Exception as e:
                                print(f"    ✗ Frame {frame_idx} failed: {e}")
                                continue
                        
                        # Create video from frames if we have any
                        if frame_paths:
                            video_filename = f"scene_{scene_idx:03d}_{scene['name']}_{vis_type}.mp4"
                            video_path = os.path.join(output_dir, video_filename)
                            
                            try:
                                # Read the first frame to get dimensions
                                first_frame = cv2.imread(frame_paths[0])
                                if first_frame is not None:
                                    height, width, _ = first_frame.shape
                                    
                                    # Create video writer
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                                    
                                    # Add frames to video
                                    for frame_path in frame_paths:
                                        frame = cv2.imread(frame_path)
                                        if frame is not None:
                                            video_writer.write(frame)
                                    
                                    # Release video writer
                                    video_writer.release()
                                    print(f"    ✓ {vis_type} video saved: {video_filename}")
                                else:
                                    print(f"    ✗ Failed to read first frame for {vis_type} video")
                            except Exception as e:
                                print(f"    ✗ Failed to create {vis_type} video: {e}")
                        else:
                            print(f"    ✗ No frames generated for {vis_type} video")
                
                scene_time = time.time() - scene_start_time
                print(f"  ✓ Scene completed in {scene_time:.1f}s, processed {len(sample_tokens)} frames")
                
                # Overall progress update
                elapsed_time = time.time() - start_time
                avg_time_per_scene = elapsed_time / (scene_idx + 1)
                remaining_scenes = total_scenes - (scene_idx + 1)
                estimated_remaining_time = avg_time_per_scene * remaining_scenes
                
                print(f"  Overall progress: {scene_idx + 1}/{total_scenes} scenes "
                      f"({(scene_idx + 1)/total_scenes*100:.1f}%)")
                print(f"  Total frames processed: {total_frames_processed}")
                print(f"  Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining_time:.1f}s")
                
            except Exception as e:
                print(f"  ✗ Scene {scene_idx} failed completely: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n✓ Completed processing all {total_scenes} scenes in {total_time:.1f} seconds")
        print(f"✓ Total frames processed: {total_frames_processed}")
        print(f"✓ Total frames skipped: {total_frames_skipped}")
        print(f"✓ Average time per frame: {total_time/max(total_frames_processed, 1):.2f} seconds")
        print(f"✓ All videos saved to: {output_dir}")
        
        # Generate summary report
        self._generate_all_frames_summary_report(output_dir, total_scenes, total_frames_processed, 
                                                total_frames_skipped, visualization_types, 
                                                max_frames_per_scene, skip_frames, fps)
    
    def _get_all_sample_tokens(self, scene):
        """
        Get all sample tokens for a scene
        
        Args:
            scene: Scene dictionary
            
        Returns:
            List of sample tokens
        """
        sample_tokens = []
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample_tokens.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
        
        return sample_tokens
    
    def _generate_all_frames_summary_report(self, output_dir, total_scenes, total_frames_processed, 
                                          total_frames_skipped, visualization_types, 
                                          max_frames_per_scene, skip_frames, fps):
        """
        Generate a summary report for all frames processing
        
        Args:
            output_dir: Directory where visualizations are saved
            total_scenes: Total number of scenes processed
            total_frames_processed: Total number of frames processed
            total_frames_skipped: Total number of frames skipped
            visualization_types: Types of visualizations generated
            max_frames_per_scene: Maximum frames per scene limit
            skip_frames: Frame skipping interval
            fps: Frames per second for the video
        """
        report_path = os.path.join(output_dir, "all_frames_summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== All Frames Visualization Summary Report ===\n\n")
            f.write(f"Dataset: {self.nusc.version}\n")
            f.write(f"Dataroot: {self.nusc.dataroot}\n")
            f.write(f"Total scenes processed: {total_scenes}\n")
            f.write(f"Total frames processed: {total_frames_processed}\n")
            f.write(f"Total frames skipped: {total_frames_skipped}\n")
            f.write(f"Frame skip interval: {skip_frames}\n")
            f.write(f"Max frames per scene: {max_frames_per_scene or 'No limit'}\n")
            f.write(f"Visualization types: {', '.join(visualization_types)}\n")
            f.write(f"Point cloud range: {self.point_cloud_range}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Video FPS: {fps}\n\n")
            
            f.write("Scene Details:\n")
            f.write("-" * 70 + "\n")
            
            for scene_idx, scene in enumerate(self.nusc.scene):
                scene_dir = f"scene_{scene_idx:03d}_{scene['name']}"
                f.write(f"Scene {scene_idx:03d}: {scene['name']}\n")
                f.write(f"  Description: {scene['description']}\n")
                f.write(f"  Directory: {scene_dir}\n")
                
                # Count total frames in original scene
                total_frames_in_scene = len(self._get_all_sample_tokens(scene))
                processed_frames = total_frames_in_scene // skip_frames
                if max_frames_per_scene:
                    processed_frames = min(processed_frames, max_frames_per_scene)
                
                f.write(f"  Total frames in scene: {total_frames_in_scene}\n")
                f.write(f"  Frames processed: {processed_frames}\n")
                
                # Check which video files were generated for this scene
                video_files = []
                for vis_type in visualization_types:
                    video_filename = f"scene_{scene_idx:03d}_{scene['name']}_{vis_type}.mp4"
                    video_path = os.path.join(output_dir, video_filename)
                    if os.path.exists(video_path):
                        video_files.append(video_filename)
                
                f.write(f"  Generated video files: {len(video_files)}\n")
                if video_files:
                    for video_file in video_files:
                        f.write(f"    - {video_file}\n")
                else:
                    f.write(f"  Status: Failed to process\n")
                f.write("\n")
        
        print(f"✓ All frames summary report saved to: {report_path}")

    def visualize_combined(self, dataroot, sample_token, save_path=None, show_plot=True):
        """
        Visualize combined BEV and image view side by side
        
        Args:
            dataroot: Path to the dataset root
            sample_token: Token of the sample to visualize
            save_path: Path to save the visualization (if None, default path is used)
            show_plot: Whether to display the plot
            
        Returns:
            combined_image: Combined visualization image
        """
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Generate BEV visualization
        points = self.get_point_cloud(sample_token)
        boxes = self.get_boxes(sample_token)
        
        # Filter points and boxes in range
        points = self.filter_points_in_range(points)
        boxes = self.filter_boxes_in_range(boxes)
        
        # Plot BEV on left subplot
        ax1.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap='viridis', alpha=self.point_cloud_alpha)
        
        # Plot boxes on BEV
        for box in boxes:
            self._plot_box_on_axis(ax1, box)
        
        # Set BEV axis properties
        ax1.set_xlim(self.point_cloud_range[0], self.point_cloud_range[2])
        ax1.set_ylim(self.point_cloud_range[1], self.point_cloud_range[3])
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Bird\'s Eye View\n({self.point_cloud_range[0]},{self.point_cloud_range[1]}) to ({self.point_cloud_range[2]},{self.point_cloud_range[3]})')
        ax1.set_aspect('equal')
        ax1.grid(True)
        
        # Add colorbar for BEV
        if len(ax1.collections) > 0:
            cbar = plt.colorbar(ax1.collections[0], ax=ax1)
            cbar.set_label('Z (height)')
        
        # Generate image visualization
        image_path = self.get_image(sample_token)
        image = cv2.imread(image_path)
        
        # Get camera parameters and project boxes
        camera_param_path = os.path.join(dataroot, "v1.0-trainval/calibrated_sensor.json")
        with open(camera_param_path, 'r') as f:
            camera_params = json.load(f)[0]
        
        cam_intrinsic = camera_params['camera_intrinsic']
        
        # Get the sample and transformations
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_to_ego_mat = self.transform_matrix(lidar_calibrated_sensor['translation'], lidar_calibrated_sensor['rotation'])
        
        cam_to_ego_mat = self.transform_matrix_from_rotation_matrix(camera_params['translation'], camera_params['rotation'])
        ego_to_cam_mat = np.linalg.inv(cam_to_ego_mat)
        lidar_to_cam_mat = ego_to_cam_mat @ lidar_to_ego_mat
        
        # Draw 3D boxes projected to 2D image
        image_with_boxes = image.copy()
        for box in boxes:
            self._draw_box_in_image(image_with_boxes, box, lidar_to_cam_mat, cam_intrinsic)
        
        # Display image on right subplot
        ax2.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        ax2.set_title('Camera View with 3D Boxes')
        ax2.axis('off')
        
        # Add legend for box categories (shared between both views)
        handles = []
        labels = []
        for category, color in self.category_colors.items():
            if category != 'default':
                handle = plt.Rectangle((0, 0), 1, 1, color=color)
                handles.append(handle)
                labels.append(category)
        
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for legend
        
        # Save if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            # Use default path
            os.makedirs('bev_demo_output', exist_ok=True)
            plt.savefig("bev_demo_output/combined_visualization.png", dpi=300, bbox_inches='tight')
        
        # Show plot if required
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        # Convert matplotlib figure to image array for video creation
        fig.canvas.draw()
        combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        return combined_image
    
    def _plot_box_on_axis(self, ax, box):
        """
        Plot a 3D box in BEV on a specific axis (helper method for combined visualization)
        
        Args:
            ax: Matplotlib axis
            box: Box dictionary with position, size, rotation, and category
        """
        # Extract box parameters
        x, y, z = box['position']
        width, length, height = box['size']
        quaternion = box['rotation']
        category = box['category']
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw = quaternion_to_yaw(quaternion)
        
        # Get color for category
        color = self.category_colors.get(category, self.category_colors['default'])
        
        # Calculate corners of the rectangle
        corners = self._get_corners(x, y, length, width, yaw)
        
        # Plot rectangle
        rect = Polygon(corners, fill=True, alpha=self.box_alpha, color=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Plot direction arrow
        arrow_length = length * 0.5
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        arrow = Arrow(x, y, dx, dy, width=width*0.5, color='red', alpha=self.arrow_alpha)
        ax.add_patch(arrow)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Bird\'s Eye View of LiDAR data with 3D boxes')
    parser.add_argument('--dataroot', type=str, default='data/v2x-seq-nuscenes/cooperative',
                       help='Path to the NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--scene_idx', type=int, default=2,
                       help='Index of the scene to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to visualize (if not animating)')
    parser.add_argument('--range', type=float, nargs=4, default=[0, -50, 100, 50],
                       help='Point cloud range: xmin ymin xmax ymax')
    parser.add_argument('--all_frames', action='store_true',
                       help='Visualize all frames of all scenes in the dataset')
    parser.add_argument('--max_frames_per_scene', type=int, default=None,
                       help='Maximum number of frames to process per scene (for --all_frames)')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Skip every N frames (1=all frames, 2=every other frame, etc.)')
    parser.add_argument('--vis_types', type=str, nargs='+', 
                       default=['combined'], choices=['bev', 'image', 'combined'],
                       help='Types of visualizations to generate (for --all_frames)')
    parser.add_argument('--output_dir', type=str, default='bev_output',
                       help='Directory to save the output')
    parser.add_argument('--fps', type=int, default=3,
                       help='Frames per second for video output (for --all_frames)')
    parser.add_argument('--point_cloud_alpha', type=float, default=0.3,
                       help='Transparency for point cloud (0.0=transparent, 1.0=opaque)')
    parser.add_argument('--box_alpha', type=float, default=0.7,
                       help='Transparency for 3D bounding boxes (0.0=transparent, 1.0=opaque)')
    parser.add_argument('--arrow_alpha', type=float, default=0.8,
                       help='Transparency for direction arrows (0.0=transparent, 1.0=opaque)')
    
    args = parser.parse_args()
    
    print(f"Using point cloud range: {args.range}")
    print(f"Using transparency settings - Point cloud: {args.point_cloud_alpha}, Boxes: {args.box_alpha}, Arrows: {args.arrow_alpha}")
    
    # Initialize NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Initialize visualizer
    visualizer = BEVVisualizer(nusc, scene_idx=args.scene_idx, 
                              point_cloud_range=args.range,
                              point_cloud_alpha=args.point_cloud_alpha,
                              box_alpha=args.box_alpha,
                              arrow_alpha=args.arrow_alpha)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all_frames:
        # Visualize all frames of all scenes
        visualizer.visualize_all_frames(output_dir=args.output_dir, visualization_types=args.vis_types, 
                                        max_frames_per_scene=args.max_frames_per_scene, skip_frames=args.skip_frames, fps=args.fps)
    else:
        # Visualize a single sample
        scene = nusc.scene[args.scene_idx]
        sample_token = scene['first_sample_token']
        
        # Get the sample at the specified index
        for i in range(args.sample_idx):
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
            if not sample_token:
                print(f"Warning: Requested sample index {args.sample_idx} exceeds scene length. Using last sample.")
                break
        
        # Visualize based on specified types
        for vis_type in args.vis_types:
            if vis_type == 'bev':
                output_path = os.path.join(args.output_dir, f'bev_scene{args.scene_idx}_sample{args.sample_idx}.png')
                visualizer.visualize_bev(sample_token, save_path=output_path)
                print(f"BEV visualization saved to {output_path}")
            elif vis_type == 'image':
                output_path_image = os.path.join(args.output_dir, f'image_scene{args.scene_idx}_sample{args.sample_idx}.png')
                visualizer.visualize_image(args.dataroot, sample_token, save_path=output_path_image)
                print(f"Image visualization saved to {output_path_image}")
            elif vis_type == 'combined':
                output_path_combined = os.path.join(args.output_dir, f'combined_scene{args.scene_idx}_sample{args.sample_idx}.png')
                visualizer.visualize_combined(args.dataroot, sample_token, save_path=output_path_combined)
                print(f"Combined visualization saved to {output_path_combined}")
            else:
                print(f"Unknown visualization type: {vis_type}") 