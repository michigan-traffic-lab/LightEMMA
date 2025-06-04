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
from matplotlib.gridspec import GridSpec

class BEVVisualizer:
    def __init__(self, nusc, scene_idx=0, figsize=(10, 10), 
                 point_cloud_range=[-100, -100, 100, 100], resolution=0.2,
                 point_cloud_alpha=0.3, box_alpha=0.7, arrow_alpha=0.8,
                 cooperative_data_root=None, show_vehicle_position=True,
                 infra_point_color='cyan', vehicle_point_color='orange'):
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
            cooperative_data_root: Root path for cooperative data (containing i2v_pair.json and vehicle-side data)
            show_vehicle_position: Whether to show vehicle position in BEV visualization
            infra_point_color: Color for infrastructure point cloud (default: 'cyan')
            vehicle_point_color: Color for vehicle point cloud (default: 'orange')
        """
        self.nusc = nusc
        self.scene = nusc.scene[scene_idx]
        self.point_cloud_range = point_cloud_range
        self.resolution = resolution
        self.figsize = figsize
        self.point_cloud_alpha = point_cloud_alpha
        self.box_alpha = box_alpha
        self.arrow_alpha = arrow_alpha
        self.cooperative_data_root = cooperative_data_root
        self.show_vehicle_position = show_vehicle_position
        self.infra_point_color = infra_point_color
        self.vehicle_point_color = vehicle_point_color
        
        # Load i2v pair mapping if cooperative data root is provided
        self.i2v_pairs = {}
        if cooperative_data_root and show_vehicle_position:
            self._load_i2v_pairs()
        
        print("point_cloud_range: ", self.point_cloud_range)
        print(f"Transparency settings - Point cloud: {point_cloud_alpha}, Boxes: {box_alpha}, Arrows: {arrow_alpha}")
        print(f"Point cloud colors - Infrastructure: {infra_point_color}, Vehicle: {vehicle_point_color}")
        if show_vehicle_position:
            print(f"Vehicle position visualization: Enabled")
            print(f"Cooperative data root: {cooperative_data_root}")
        
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
    
    def _load_i2v_pairs(self):
        """
        Load infrastructure to vehicle frame pairs from i2v_pair.json
        """
        if not self.cooperative_data_root:
            return
            
        i2v_pair_path = os.path.join(self.cooperative_data_root, "i2v_pair.json")
        if os.path.exists(i2v_pair_path):
            try:
                with open(i2v_pair_path, 'r') as f:
                    self.i2v_pairs = json.load(f)
                print(f"Loaded i2v pairs from {i2v_pair_path}")
                print(f"Found {len(self.i2v_pairs)} infrastructure sequences")
            except Exception as e:
                print(f"Error loading i2v pairs: {e}")
                self.i2v_pairs = {}
        else:
            print(f"Warning: i2v_pair.json not found at {i2v_pair_path}")

    def _get_vehicle_world_position(self, vehicle_frame):
        """
        Get vehicle position in world coordinates
        
        Args:
            vehicle_sequence: Vehicle sequence ID
            vehicle_frame: Vehicle frame ID
            
        Returns:
            dict: Vehicle pose with translation and rotation in world coordinates
                  Returns None if file not found
        """
        if not self.cooperative_data_root:
            return None
            
        vehicle_pose_path = os.path.join(
            self.cooperative_data_root, 
            "vehicle-side", 
            "calib", 
            "novatel_to_world", 
            f"{vehicle_frame}.json"
        )
        
        if os.path.exists(vehicle_pose_path):
            try:
                with open(vehicle_pose_path, 'r') as f:
                    vehicle_pose = json.load(f)
                return vehicle_pose
            except Exception as e:
                print(f"Error loading vehicle pose from {vehicle_pose_path}: {e}")
                return None
        else:
            print(f"Vehicle pose file not found: {vehicle_pose_path}")
            return None

    def _world_to_infrastructure_coordinates(self, world_position, world_rotation, sample_token):
        """
        Transform vehicle position from world coordinates to infrastructure LiDAR coordinates
        
        Args:
            world_position: Vehicle position in world coordinates [x, y, z]
            world_rotation: Vehicle rotation matrix in world coordinates (3x3)
            sample_token: Infrastructure sample token to get transformation
            
        Returns:
            dict: Vehicle position and rotation in infrastructure LiDAR coordinates
        """
        # Get infrastructure sample and pose information
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Get transformations for infrastructure
        ego_to_world = {
            'translation': lidar_ego_pose['translation'],
            'rotation': lidar_ego_pose['rotation']
        }
        
        lidar_to_ego = {
            'translation': lidar_calibrated_sensor['translation'],
            'rotation': lidar_calibrated_sensor['rotation']
        }
        
        # Create transformation matrices
        ego_to_world_mat = self.transform_matrix(ego_to_world['translation'], ego_to_world['rotation'])
        lidar_to_ego_mat = self.transform_matrix(lidar_to_ego['translation'], lidar_to_ego['rotation'])
        
        # Compute world to infrastructure lidar matrix
        world_to_ego_mat = np.linalg.inv(ego_to_world_mat)
        ego_to_lidar_mat = np.linalg.inv(lidar_to_ego_mat)
        world_to_lidar_mat = ego_to_lidar_mat @ world_to_ego_mat
        
        # Transform vehicle position
        if isinstance(world_position[0], list):
            # Handle format [[x], [y], [z]]
            world_pos = np.array([world_position[0][0], world_position[1][0], world_position[2][0]])
        else:
            # Handle format [x, y, z]
            world_pos = np.array(world_position)
        
        world_pos_hom = np.append(world_pos, 1)
        lidar_pos_hom = world_to_lidar_mat @ world_pos_hom
        lidar_position = lidar_pos_hom[:3]
        
        # Transform vehicle rotation
        world_rotation_mat_hom = np.eye(4)
        world_rotation_mat_hom[:3, :3] = world_rotation
        
        # Only rotate, don't translate
        rot_world_to_lidar_mat = np.copy(world_to_lidar_mat)
        rot_world_to_lidar_mat[:3, 3] = 0
        
        lidar_rotation_mat_hom = rot_world_to_lidar_mat @ world_rotation_mat_hom
        lidar_rotation_mat = lidar_rotation_mat_hom[:3, :3]
        
        return {
            'position': lidar_position,
            'rotation_matrix': lidar_rotation_mat
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

    def get_vehicle_image(self, vehicle_frame):
        """
        Get the vehicle-side image of a given vehicle frame
        
        Args:
            vehicle_frame: Vehicle frame ID
            
        Returns:
            image_path: Path to the vehicle-side image file, None if not found
        """
        if not self.cooperative_data_root:
            return None
            
        # Try different possible image file extensions and naming patterns
        possible_paths = [
            os.path.join(self.cooperative_data_root, "vehicle-side", "image", f"{vehicle_frame}.jpg"),
            os.path.join(self.cooperative_data_root, "vehicle-side", "image", f"{vehicle_frame}.png"),
            os.path.join(self.cooperative_data_root, "vehicle-side", "images", f"{vehicle_frame}.jpg"),
            os.path.join(self.cooperative_data_root, "vehicle-side", "images", f"{vehicle_frame}.png"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        print(f"Vehicle image file not found for frame {vehicle_frame}")
        print(f"Searched paths: {possible_paths}")
        return None

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

    def _get_paired_vehicle_frame(self, sample_token):
        # Get vehicle_sequence from scene
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        infrastructure_sequence = scene['token']
        infrastructure_frame = sample_token
        
        # Check if this vehicle frame has a paired infrastructure frame
        if (infrastructure_sequence in self.i2v_pairs and 
            infrastructure_frame in self.i2v_pairs[infrastructure_sequence]):
            
            infrastructure_info = self.i2v_pairs[infrastructure_sequence][infrastructure_frame]
            vehicle_frame = infrastructure_info['vehicle_frame']
            return {
                'vehicle_sequence': infrastructure_sequence,
                'vehicle_frame': vehicle_frame
            }
        
        return None

    def _get_vehicle_position(self, sample_token):
        """
        Get vehicle LiDAR position in both world and infrastructure-side coordinates for a given infrastructure sample
        
        Args:
            sample_token: Infrastructure sample token
            
        Returns:
            dict: Vehicle LiDAR position and rotation in both coordinate systems, None if not found
                Format: {
                    'world': {'position': [...], 'rotation': [...], 'vehicle_frame': '...'},
                    'infrastructure': {'position': [...], 'rotation': [...], 'vehicle_frame': '...'},
                    'vehicle_frame': '...'
                }
        """
        # Get vehicle_sequence from scene
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        infrastructure_sequence = scene['token']
        infrastructure_frame = sample_token
        
        # Check if this vehicle frame has a paired infrastructure frame
        if (infrastructure_sequence in self.i2v_pairs and 
            infrastructure_frame in self.i2v_pairs[infrastructure_sequence]):
            
            infrastructure_info = self.i2v_pairs[infrastructure_sequence][infrastructure_frame]
            vehicle_frame = infrastructure_info['vehicle_frame']
            
            # Load novatel_to_world calibration
            novatel_to_world_path = f"data/v2x-seq-nuscenes/cooperative/vehicle-side/calib/novatel_to_world/{vehicle_frame}.json"
            
            # Load lidar_to_novatel calibration
            lidar_to_novatel_path = f"data/v2x-seq-nuscenes/cooperative/vehicle-side/calib/lidar_to_novatel/{vehicle_frame}.json"
            
            try:
                # Load novatel to world transformation
                with open(novatel_to_world_path, 'r') as f:
                    novatel_to_world_data = json.load(f)
                
                # Load lidar to novatel transformation
                with open(lidar_to_novatel_path, 'r') as f:
                    lidar_to_novatel_data = json.load(f)

                # Extract novatel position and rotation in world coordinates
                novatel_world_position = novatel_to_world_data['translation']
                novatel_world_rotation = novatel_to_world_data['rotation']
                
                # Extract lidar to novatel transformation
                lidar_to_novatel_translation = lidar_to_novatel_data['transform']['translation']
                lidar_to_novatel_rotation = lidar_to_novatel_data['transform']['rotation']
                
                # Create transformation matrices
                # 1. Novatel to world transformation matrix
                novatel_to_world_mat = self.transform_matrix_from_rotation_matrix(
                    [novatel_world_position[0][0], novatel_world_position[1][0], novatel_world_position[2][0]], 
                    np.array(novatel_world_rotation)
                )
                
                # 2. LiDAR to novatel transformation matrix
                lidar_to_novatel_mat = self.transform_matrix_from_rotation_matrix(
                    [lidar_to_novatel_translation[0][0], lidar_to_novatel_translation[1][0], lidar_to_novatel_translation[2][0]],
                    np.array(lidar_to_novatel_rotation)
                )
                
                # 3. Compute LiDAR to world transformation: LiDAR -> Novatel -> World
                lidar_to_world_mat = novatel_to_world_mat @ lidar_to_novatel_mat
                
                # Extract LiDAR position and rotation in world coordinates
                lidar_world_position = [[lidar_to_world_mat[0, 3]], 
                                        [lidar_to_world_mat[1, 3]], 
                                        [lidar_to_world_mat[2, 3]]]
                lidar_world_rotation = lidar_to_world_mat[:3, :3].tolist()
                
                # Get infrastructure pose information for coordinate transformation
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = self.nusc.get('sample_data', lidar_token)
                lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                '''
                lidar_ego_pose['translation'] = [lidar_ego_pose['translation'][0] + infrastructure_info['system_error_offset']['delta_x'],
                                                lidar_ego_pose['translation'][1] + infrastructure_info['system_error_offset']['delta_y'],
                                                lidar_ego_pose['translation'][2]]
                '''
                # Create transformation matrices for infrastructure coordinates
                # 1. LiDAR position in world coordinates (4x1 homogeneous)
                lidar_world_pos = np.array([[lidar_world_position[0][0]], 
                                          [lidar_world_position[1][0]], 
                                          [lidar_world_position[2][0]], 
                                          [1.0]])
                
                # 2. Infrastructure ego to world transformation
                ego_to_world_mat = self.transform_matrix(lidar_ego_pose['translation'], 
                                                    lidar_ego_pose['rotation'])
                
                # 3. LiDAR to ego transformation  
                lidar_to_ego_mat = self.transform_matrix(lidar_calibrated_sensor['translation'], lidar_calibrated_sensor['rotation'])

                # 4. Compute world to Infrastructure LiDAR transformation
                world_to_ego_mat = np.linalg.inv(ego_to_world_mat)
                ego_to_lidar_mat = np.linalg.inv(lidar_to_ego_mat)
                world_to_lidar_mat = ego_to_lidar_mat @ world_to_ego_mat
                
                # 5. Transform vehicle LiDAR position from world to infrastructure LiDAR coordinates
                lidar_infra_pos_hom = world_to_lidar_mat @ lidar_world_pos
                lidar_infra_position = lidar_infra_pos_hom[:3].flatten().tolist()
                
                # 6. Transform vehicle LiDAR rotation from world to infrastructure coordinates
                lidar_world_rotation_mat = np.array(lidar_world_rotation)
                
                # For rotation transformation, we only need the rotation part of the transformation
                world_to_lidar_rot = world_to_lidar_mat[:3, :3]
                lidar_infra_rotation_mat = world_to_lidar_rot @ lidar_world_rotation_mat
                
                return {
                    'world': {
                        'position': lidar_world_position,
                        'rotation': lidar_world_rotation,
                        'vehicle_frame': vehicle_frame
                    },
                    'infrastructure': {
                        'position': [[lidar_infra_position[0]], 
                                [lidar_infra_position[1]], 
                                [lidar_infra_position[2]]],
                        'rotation': lidar_infra_rotation_mat.tolist(),
                        'vehicle_frame': vehicle_frame
                    },
                    'system_error_offset': infrastructure_info['system_error_offset'],
                    'vehicle_frame': vehicle_frame
                }
                
            except Exception as e:
                print(f"Warning: Could not load calibration data for frame {vehicle_frame}: {e}")
                return None
        
        return None
    
    def _plot_vehicle_position(self, ax, infra_pos, coordinate_system='infrastructure'):
        """
        Plot infrastructure position as cross arrows in BEV
        
        Args:
            ax: Matplotlib axis
            infra_pos: Infrastructure position dictionary with both coordinate systems
            coordinate_system: 'world' or 'vehicle' to choose which coordinate system to use
        """
        if infra_pos is None:
            return
        
        # Select coordinate system
        if coordinate_system not in infra_pos:
            print(f"Warning: {coordinate_system} coordinate system not available in infrastructure position data")
            return
            
        pos_data = infra_pos[coordinate_system]
        
        # Extract position (note: translation is nested list format)
        x, y = pos_data['position'][0][0], pos_data['position'][1][0]
        
        # Define cross arrow parameters
        arrow_length = 10  # meters
        arrow_width = 2    # meters
        
        # Plot cross arrows (X and Y axes)
        # X-axis arrow (red)
        arrow_x = Arrow(x - arrow_length/2, y, arrow_length, 0, 
                       width=arrow_width, color='red', alpha=0.8, zorder=10)
        ax.add_patch(arrow_x)
        
        # Y-axis arrow (green)  
        arrow_y = Arrow(x, y - arrow_length/2, 0, arrow_length,
                       width=arrow_width, color='green', alpha=0.8, zorder=10)
        ax.add_patch(arrow_y)
        
        # Add infrastructure marker (circle)
        circle = plt.Circle((x, y), radius=3, color='purple', alpha=0.8, zorder=11)
        ax.add_patch(circle)
        
        # Add text label with coordinate system info
        coord_label = coordinate_system.upper()
        ax.text(x + 5, y + 5, f'INFRA-{coord_label}\n{infra_pos["vehicle_frame"]}', 
                fontsize=8, color='purple', fontweight='bold', zorder=12)

    def _transform_vehicle_points_to_infrastructure(self, vehicle_points, sample_token):
        """
        Transform vehicle point cloud from vehicle coordinate to infrastructure coordinate
        
        Args:
            vehicle_points: Nx4 array of vehicle point cloud points
            sample_token: Infrastructure sample token to get transformation info
            
        Returns:
            transformed_points: Nx4 array of transformed point cloud points in infrastructure coordinates
        """
        if vehicle_points is None:
            return None
        
        try:
            # Get vehicle position information
            veh_pos = self._get_vehicle_position(sample_token)
            if veh_pos is None:
                return None
            
            # Get infrastructure sample information for transformation
            sample = self.nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            '''
            lidar_ego_pose['translation'] = [lidar_ego_pose['translation'][0] + veh_pos['system_error_offset']['delta_x'],
                                            lidar_ego_pose['translation'][1] + veh_pos['system_error_offset']['delta_y'],
                                            lidar_ego_pose['translation'][2]]
            '''
            
            # Get vehicle transformation information
            veh_world_position = np.array(veh_pos['world']['position']).flatten()
            veh_world_rotation = np.array(veh_pos['world']['rotation'])
            
            # Create transformation matrices
            # 1. Vehicle to world transformation
            veh_to_world_mat = np.eye(4)
            veh_to_world_mat[:3, :3] = veh_world_rotation
            veh_to_world_mat[:3, 3] = veh_world_position
            
            # 2. Infrastructure ego to world transformation
            ego_to_world_mat = self.transform_matrix(lidar_ego_pose['translation'], 
                                                   lidar_ego_pose['rotation'])
            
            # 3. LiDAR to ego transformation  
            lidar_to_ego_mat = self.transform_matrix(lidar_calibrated_sensor['translation'],
                                                   lidar_calibrated_sensor['rotation'])
            
            # 4. Compute vehicle to infrastructure LiDAR transformation
            # veh_local -> veh_world -> ego_world -> ego_local -> lidar_local
            world_to_ego_mat = np.linalg.inv(ego_to_world_mat)
            ego_to_lidar_mat = np.linalg.inv(lidar_to_ego_mat)
            
            # Full transformation chain
            veh_to_infrastructure_mat = ego_to_lidar_mat @ world_to_ego_mat @ veh_to_world_mat
            
            # Transform points
            # Add homogeneous coordinate
            veh_points_hom = np.column_stack([vehicle_points[:, :3], np.ones(len(vehicle_points))])
            
            # Apply transformation
            transformed_points_hom = (veh_to_infrastructure_mat @ veh_points_hom.T).T
            
            # Extract 3D coordinates and keep intensity
            transformed_points = np.column_stack([
                transformed_points_hom[:, :3],  # x, y, z
                vehicle_points[:, 3]              # intensity
            ])
            
            return transformed_points
            
        except Exception as e:
            print(f"Error transforming vehicle points: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_bev(self, sample_token, save_path=None, show_plot=True):
        """
        Visualize Bird's Eye View with point cloud, 3D boxes, and vehicle position
        Now includes vehicle-side point cloud data transformed to infrastructure coordinates
        
        Args:
            sample_token: Token of the sample to visualize
            save_path: Path to save the visualization (if None, the plot is not saved)
            show_plot: Whether to display the plot
            
        Returns:
            fig, ax: Figure and axis objects
        """
        # Get infrastructure point cloud and boxes
        infra_points = self.get_point_cloud(sample_token)
        infra_boxes = self.get_boxes(sample_token)
        
        # Filter infrastructure points and boxes in range
        infra_points = self.filter_points_in_range(infra_points)
        infra_boxes = self.filter_boxes_in_range(infra_boxes)
                
        # Get vehicle position and point cloud if cooperative visualization is enabled
        veh_position = None
        vehicle_points_transformed = None
        
        if self.show_vehicle_position and self.cooperative_data_root:
            # Get paired vehicle frame information
            vehicle_info = self._get_paired_vehicle_frame(sample_token)
            
            if vehicle_info is not None:
                vehicle_sequence = vehicle_info['vehicle_sequence']
                vehicle_frame = vehicle_info['vehicle_frame']
                print(f"Found paired vehicle frame: sequence={vehicle_sequence}, frame={vehicle_frame}")
                
                # Get vehicle position for visualization
                veh_position = self._get_vehicle_position(sample_token)
                # Load vehicle-side point cloud
                vehicle_points = self._load_vehicle_point_cloud(vehicle_frame)
                
                if vehicle_points is not None:
                    vehicle_points_transformed = self._transform_vehicle_points_to_infrastructure(vehicle_points, sample_token)
                else:
                    print("Failed to load vehicle-side point cloud")
            else:
                print("No paired vehicle frame found for this infrastructure sample")
        
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot infrastructure point cloud (in solid cyan color)
        scatter_infra = ax.scatter(infra_points[:, 0], infra_points[:, 1], s=0.5, 
                                  c=self.infra_point_color, 
                                  alpha=self.point_cloud_alpha, label='Infrastructure LiDAR')
        
        # Plot vehicle point cloud if available (in solid orange color)
        if vehicle_points_transformed is not None:
            scatter_vehicle = ax.scatter(vehicle_points_transformed[:, 0], vehicle_points_transformed[:, 1], 
                                       s=0.5, c=self.vehicle_point_color, 
                                       alpha=self.point_cloud_alpha * 0.8, label='Vehicle LiDAR')
        
        # Plot boxes
        for box in infra_boxes:
            self._plot_box(ax, box)
        
        # Plot vehicle position if available
        if veh_position:
            self._plot_vehicle_position(ax, veh_position)
        
        # Set axis limits to exactly match the specified range
        ax.set_xlim(self.point_cloud_range[0], self.point_cloud_range[2])
        ax.set_ylim(self.point_cloud_range[1], self.point_cloud_range[3])
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = f'Bird\'s Eye View ({self.point_cloud_range[0]},{self.point_cloud_range[1]}) to ({self.point_cloud_range[2]},{self.point_cloud_range[3]})'
        if self.show_vehicle_position and veh_position:
            title += ' with Vehicle Position'
        if vehicle_points_transformed is not None:
            title += ' + Vehicle LiDAR'
        ax.set_title(title)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True)
        
        # Add colorbar for infrastructure points
        # cbar = plt.colorbar(scatter_infra, ax=ax)
        # cbar.set_label('Z (height)')
        
        # Add legend for point clouds and other elements
        handles = []
        labels = []
        
        # Add point cloud legend entries
        if hasattr(scatter_infra, 'legend_elements'):
            infra_handle = plt.Line2D([0], [0], marker='o', color=self.infra_point_color, markersize=4, 
                                    linestyle='None', alpha=self.point_cloud_alpha)
            handles.append(infra_handle)
            labels.append('Infrastructure LiDAR')
        
        if vehicle_points_transformed is not None:
            vehicle_handle = plt.Line2D([0], [0], marker='o', color=self.vehicle_point_color, markersize=4, 
                                      linestyle='None', alpha=self.point_cloud_alpha * 0.8)
            handles.append(vehicle_handle)
            labels.append('Vehicle LiDAR')
        
        # Add vehicle position to legend if shown
        if self.show_vehicle_position and veh_position:
            vehicle_pos_handle = plt.Line2D([0], [0], marker='o', color='purple', markersize=8, 
                                          linestyle='None', markeredgecolor='darkred')
            handles.append(vehicle_pos_handle)
            labels.append('Vehicle Position')
        
        # Add legend if we have any handles
        if handles:
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

    def _get_vehicle_point_cloud_path(self, vehicle_frame):
        """
        Get vehicle-side point cloud file path
        
        Args:
            vehicle_sequence: Vehicle sequence ID  
            vehicle_frame: Vehicle frame ID
            
        Returns:
            str: Path to vehicle point cloud file, None if not found
        """
        if not self.cooperative_data_root:
            return None
            
        # Try different possible file extensions and naming patterns
        possible_paths = [
            os.path.join(self.cooperative_data_root, "vehicle-side", "velodyne", f"{vehicle_frame}.pcd"),
            os.path.join(self.cooperative_data_root, "vehicle-side", "velodyne", f"{vehicle_frame}.bin"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None

    def _load_vehicle_point_cloud(self, vehicle_frame):
        """
        Load vehicle-side point cloud data
        
        Args:
            vehicle_sequence: Vehicle sequence ID
            vehicle_frame: Vehicle frame ID
            
        Returns:
            points: Nx4 array of point cloud points (x, y, z, intensity), None if not found
        """
        pcd_path = self._get_vehicle_point_cloud_path(vehicle_frame)
        if not pcd_path:
            return None
            
        try:
            # Check file extension to determine loading method
            file_extension = os.path.splitext(pcd_path)[1].lower()
            
            if file_extension == '.pcd':
                # Load PCD file using pypcd
                points = self._load_pcd_with_pypcd(pcd_path)
            elif file_extension == '.bin':
                # Load binary file using NuScenes LidarPointCloud
                pc = LidarPointCloud.from_file(pcd_path)
                points = pc.points.T  # Nx4 array (x, y, z, intensity)
            else:
                print(f"Unsupported vehicle point cloud file format: {file_extension}")
                return None
                
            return points
            
        except Exception as e:
            print(f"Error loading vehicle point cloud from {pcd_path}: {e}")
            return None

    def _get_vehicle_to_infrastructure_transform(self, sample_token, vehicle_frame):
        """
        Get transformation matrix from vehicle-side coordinates to infrastructure-side coordinates
        
        Args:
            sample_token: Infrastructure sample token
            vehicle_sequence: Vehicle sequence ID
            vehicle_frame: Vehicle frame ID
            
        Returns:
            transform_matrix: 4x4 transformation matrix, None if transformation cannot be computed
        """
        # Get vehicle world position
        vehicle_world_pose = self._get_vehicle_world_position(vehicle_frame)
        if not vehicle_world_pose:
            return None
            
        # Get infrastructure world position
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Get transformations for infrastructure
        ego_to_world = {
            'translation': lidar_ego_pose['translation'],
            'rotation': lidar_ego_pose['rotation']
        }
        
        lidar_to_ego = {
            'translation': lidar_calibrated_sensor['translation'],
            'rotation': lidar_calibrated_sensor['rotation']
        }
        
        # Create transformation matrices
        ego_to_world_mat = self.transform_matrix(ego_to_world['translation'], ego_to_world['rotation'])
        lidar_to_ego_mat = self.transform_matrix(lidar_to_ego['translation'], lidar_to_ego['rotation'])
        
        # Infrastructure lidar to world transformation
        infra_lidar_to_world_mat = ego_to_world_mat @ lidar_to_ego_mat
        
        # Vehicle lidar to world transformation
        vehicle_translation = vehicle_world_pose['translation']
        vehicle_rotation_mat = vehicle_world_pose['rotation']
        
        # Handle different formats of vehicle translation
        if isinstance(vehicle_translation[0], list):
            # Format: [[x], [y], [z]]
            vehicle_trans = [vehicle_translation[0][0], vehicle_translation[1][0], vehicle_translation[2][0]]
        else:
            # Format: [x, y, z]
            vehicle_trans = vehicle_translation
            
        # Create vehicle to world transformation matrix
        vehicle_to_world_mat = np.eye(4)
        vehicle_to_world_mat[:3, :3] = vehicle_rotation_mat
        vehicle_to_world_mat[:3, 3] = vehicle_trans
        
        # Compute vehicle to infrastructure lidar transformation
        world_to_infra_lidar_mat = np.linalg.inv(infra_lidar_to_world_mat)
        vehicle_to_infra_lidar_mat = world_to_infra_lidar_mat @ vehicle_to_world_mat
        
        return vehicle_to_infra_lidar_mat

    def _apply_vehicle_to_infrastructure_transform(self, vehicle_points, transform_matrix):
        """
        Transform vehicle-side point cloud to infrastructure coordinates using a pre-computed transformation matrix
        
        Args:
            vehicle_points: Nx4 array of vehicle point cloud points (x, y, z, intensity)
            transform_matrix: 4x4 transformation matrix from vehicle to infrastructure
            
        Returns:
            transformed_points: Nx4 array of transformed points in infrastructure coordinates
        """
        if vehicle_points is None or transform_matrix is None:
            return None
            
        # Convert to homogeneous coordinates
        points_hom = np.column_stack([vehicle_points[:, :3], np.ones(len(vehicle_points))])
        
        # Transform points
        transformed_points_hom = (transform_matrix @ points_hom.T).T
        
        # Convert back to 3D coordinates and keep intensity
        transformed_points = np.column_stack([
            transformed_points_hom[:, :3], 
            vehicle_points[:, 3]  # Keep original intensity
        ])
        
        return transformed_points

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
                    
                    # Create fixed directory for frames instead of temporary directory
                    frames_dir = os.path.join(output_dir, "frames", f"scene_{scene_idx:03d}_{scene['name']}", vis_type)
                    os.makedirs(frames_dir, exist_ok=True)
                    frame_paths = []
                    
                    # Generate all frames for this visualization type
                    for frame_idx, sample_token in enumerate(sample_tokens):
                        try:
                            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
                            
                            # Generate frame based on visualization type
                            if vis_type == 'bev':
                                self.visualize_bev(sample_token, save_path=frame_path, show_plot=False)
                            elif vis_type == 'image':
                                self.visualize_image(self.nusc.dataroot, sample_token, 
                                                   save_path=frame_path, show_plot=False)
                            elif vis_type == 'combined':
                                # For combined visualization, get image array directly and save
                                combined_image = self.visualize_combined(self.nusc.dataroot, sample_token, 
                                                                       save_path=frame_path, show_plot=False)
                                # Verify the image was saved properly
                                if not os.path.exists(frame_path):
                                    print(f"    Warning: Frame {frame_idx} was not saved properly, using direct array")
                                    # Save the array directly as backup
                                    cv2.imwrite(frame_path, combined_image)
                            else:
                                print(f"    ✗ Unknown visualization type: {vis_type}")
                                continue
                            
                            # Verify frame was created successfully
                            if os.path.exists(frame_path):
                                frame_paths.append(frame_path)
                                total_frames_processed += 1
                            else:
                                print(f"    Warning: Frame {frame_idx} file was not created: {frame_path}")
                                continue
                            
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
                            import traceback
                            traceback.print_exc()
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
                                print(f"Frame dimensions: {width}x{height}")
                                
                                # Create video writer with better codec and quality settings
                                # Try different codecs in order of preference
                                codecs_to_try = [
                                    ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # H.264 codec
                                    ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # Alternative H.264
                                    ('H264', cv2.VideoWriter_fourcc(*'H264'))   # Fallback codec
                                ]
                                
                                video_writer = None
                                for codec_name, codec in codecs_to_try:
                                    try:
                                        temp_path = os.path.join(output_dir, f"temp_{codec_name}_{video_filename}")
                                        video_writer = cv2.VideoWriter(
                                            temp_path,
                                            codec,
                                            fps,
                                            (width, height),
                                            True  # isColor
                                        )
                                        if video_writer.isOpened():
                                            print(f"Using codec: {codec_name}")
                                            break
                                        video_writer.release()
                                    except Exception as e:
                                        print(f"Failed to initialize codec {codec_name}: {e}")
                                        if video_writer:
                                            video_writer.release()
                                        if os.path.exists(temp_path):
                                            os.remove(temp_path)
                                
                                if video_writer is None or not video_writer.isOpened():
                                    raise Exception("Failed to initialize any video codec")
                                
                                # Add frames to video
                                for frame_path in frame_paths:
                                    frame = cv2.imread(frame_path)
                                    if frame is not None:
                                        if frame.shape != (height, width, 3):
                                            print(f"Warning: Frame size mismatch. Expected {(height, width, 3)}, got {frame.shape}")
                                            frame = cv2.resize(frame, (width, height))
                                        video_writer.write(frame)
                                
                                # Release video writer
                                video_writer.release()
                                
                                # Move temp file to final location if successful
                                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                    import shutil
                                    shutil.move(temp_path, video_path)
                                    print(f"    ✓ {vis_type} video saved: {video_filename}")
                                    print(f"    ✓ Video size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
                                    print(f"    ✓ Frames saved to: {frames_dir}")
                                else:
                                    raise Exception("Video file was not created successfully")
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
        Visualize combined BEV, infrastructure image and vehicle image view
        Left: BEV view
        Right: Infrastructure image (top) and Vehicle image (bottom) stacked vertically
        
        Args:
            dataroot: Path to the dataset root
            sample_token: Token of the sample to visualize
            save_path: Path to save the visualization (if None, default path is used)
            show_plot: Whether to display the plot
            
        Returns:
            combined_image: Combined visualization image
        """
        # Get paired vehicle frame information first
        vehicle_info = None
        vehicle_frame = None
        vehicle_image_path = None
        
        if self.show_vehicle_position and self.cooperative_data_root:
            vehicle_info = self._get_paired_vehicle_frame(sample_token)
            if vehicle_info is not None:
                vehicle_frame = vehicle_info['vehicle_frame']
                vehicle_image_path = self.get_vehicle_image(vehicle_frame)
        
        # Create figure with custom layout using GridSpec
        # Left side: BEV view (spans 2 rows)
        # Right side: Infrastructure image (top) and Vehicle image (bottom)
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # BEV subplot spans both rows on the left
        ax_bev = fig.add_subplot(gs[:, 0])
        
        # Infrastructure image on top right
        ax_infra = fig.add_subplot(gs[0, 1])
        
        # Vehicle image on bottom right (if available)
        ax_vehicle = fig.add_subplot(gs[1, 1])
        
        # Generate BEV visualization - EXACTLY AS visualize_bev
        infra_points = self.get_point_cloud(sample_token)
        infra_boxes = self.get_boxes(sample_token)
        
        # Filter infrastructure points and boxes in range
        infra_points = self.filter_points_in_range(infra_points)
        infra_boxes = self.filter_boxes_in_range(infra_boxes)
                
        # Get vehicle position and point cloud if cooperative visualization is enabled
        veh_position = None
        vehicle_points_transformed = None
        
        if self.show_vehicle_position and self.cooperative_data_root:
            if vehicle_info is not None:
                vehicle_sequence = vehicle_info['vehicle_sequence']
                vehicle_frame = vehicle_info['vehicle_frame']
                
                # Get vehicle position for visualization
                veh_position = self._get_vehicle_position(sample_token)
                # Load vehicle-side point cloud
                vehicle_points = self._load_vehicle_point_cloud(vehicle_frame)
                
                if vehicle_points is not None:
                    vehicle_points_transformed = self._transform_vehicle_points_to_infrastructure(vehicle_points, sample_token)
                else:
                    print("Failed to load vehicle-side point cloud")
            else:
                print("No paired vehicle frame found for this infrastructure sample")
                
        # Plot BEV on left subplot - EXACTLY AS visualize_bev
        scatter_infra = ax_bev.scatter(infra_points[:, 0], infra_points[:, 1], s=0.5, 
                                      c=self.infra_point_color, 
                                      alpha=self.point_cloud_alpha, label='Infrastructure LiDAR')
        
        # Plot vehicle point cloud if available (in solid orange color)
        if vehicle_points_transformed is not None:
            scatter_vehicle = ax_bev.scatter(vehicle_points_transformed[:, 0], vehicle_points_transformed[:, 1], 
                                           s=0.5, c=self.vehicle_point_color, 
                                           alpha=self.point_cloud_alpha * 0.8, label='Vehicle LiDAR')
        
        # Plot boxes on BEV
        for box in infra_boxes:
            self._plot_box(ax_bev, box)
        
        # Plot vehicle position if available
        if veh_position:
            self._plot_vehicle_position(ax_bev, veh_position)
        
        # Set BEV axis properties - EXACTLY AS visualize_bev
        ax_bev.set_xlim(self.point_cloud_range[0], self.point_cloud_range[2])
        ax_bev.set_ylim(self.point_cloud_range[1], self.point_cloud_range[3])
        ax_bev.set_xlabel('X (m)')
        ax_bev.set_ylabel('Y (m)')
        title = f'Bird\'s Eye View ({self.point_cloud_range[0]},{self.point_cloud_range[1]}) to ({self.point_cloud_range[2]},{self.point_cloud_range[3]})'
        if self.show_vehicle_position and veh_position:
            title += ' with Vehicle Position'
        if vehicle_points_transformed is not None:
            title += ' + Vehicle LiDAR'
        ax_bev.set_title(title)
        ax_bev.set_aspect('equal')
        ax_bev.grid(True)
        
        # Add legend for point clouds and other elements - EXACTLY AS visualize_bev
        handles = []
        labels = []
        
        # Add point cloud legend entries
        if hasattr(scatter_infra, 'legend_elements'):
            infra_handle = plt.Line2D([0], [0], marker='o', color=self.infra_point_color, markersize=4, 
                                    linestyle='None', alpha=self.point_cloud_alpha)
            handles.append(infra_handle)
            labels.append('Infrastructure LiDAR')
        
        if vehicle_points_transformed is not None:
            vehicle_handle = plt.Line2D([0], [0], marker='o', color=self.vehicle_point_color, markersize=4, 
                                      linestyle='None', alpha=self.point_cloud_alpha * 0.8)
            handles.append(vehicle_handle)
            labels.append('Vehicle LiDAR')
        
        # Add vehicle position to legend if shown
        if self.show_vehicle_position and veh_position:
            vehicle_pos_handle = plt.Line2D([0], [0], marker='o', color='purple', markersize=8, 
                                          linestyle='None', markeredgecolor='darkred')
            handles.append(vehicle_pos_handle)
            labels.append('Vehicle Position')
        
        # Add legend if we have any handles
        if handles:
            ax_bev.legend(handles, labels, loc='upper right')

        # Generate infrastructure image visualization
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
        for box in infra_boxes:
            self._draw_box_in_image(image_with_boxes, box, lidar_to_cam_mat, cam_intrinsic)
        
        # Display infrastructure image on top right subplot
        ax_infra.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        ax_infra.set_title('Infrastructure Camera View with 3D Boxes')
        ax_infra.axis('off')
        
        # Display vehicle image on bottom right subplot
        if vehicle_image_path and os.path.exists(vehicle_image_path):
            vehicle_image = cv2.imread(vehicle_image_path)
            if vehicle_image is not None:
                ax_vehicle.imshow(cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB))
                ax_vehicle.set_title(f'Vehicle Camera View (Frame: {vehicle_frame})')
                ax_vehicle.axis('off')
            else:
                ax_vehicle.text(0.5, 0.5, 'Vehicle image\nnot available', ha='center', va='center', 
                               transform=ax_vehicle.transAxes, fontsize=16)
                ax_vehicle.set_title('Vehicle Camera View')
                ax_vehicle.axis('off')
        else:
            # No vehicle image available
            ax_vehicle.text(0.5, 0.5, 'Vehicle image\nnot available', ha='center', va='center', 
                           transform=ax_vehicle.transAxes, fontsize=16, color='gray')
            ax_vehicle.set_title('Vehicle Camera View')
            ax_vehicle.axis('off')
            if not vehicle_image_path:
                print("No vehicle image path found")
            else:
                print(f"Vehicle image file does not exist: {vehicle_image_path}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert matplotlib figure to image array for video creation BEFORE saving/closing
        fig.canvas.draw()
        combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        
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
        
        # Close figure if not shown
        if not show_plot:
            plt.close(fig)
        
        return combined_image

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

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Bird\'s Eye View of LiDAR data with 3D boxes and vehicle position')
    parser.add_argument('--dataroot', type=str, default='data/v2x-seq-nuscenes/infrastructure-side',
                       help='Path to the NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--scene_idx', type=int, default=0,
                       help='Index of the scene to visualize')
    parser.add_argument('--sample_idx', type=int, default=5,
                       help='Index of the sample to visualize (if not animating)')
    parser.add_argument('--range', type=float, nargs=4, default=[-40, -70, 100, 70],
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
    
    # New cooperative visualization parameters
    parser.add_argument('--cooperative_data_root', type=str, default='data/v2x-seq-nuscenes/cooperative',
                       help='Path to the cooperative dataset root (containing i2v_pair.json and vehicle-side data)')
    parser.add_argument('--show_vehicle_position', action='store_true', default=True,
                       help='Show vehicle position in BEV visualization using cooperative data')
    parser.add_argument('--no_vehicle_position', action='store_true',
                       help='Disable vehicle position visualization')
    
    # Point cloud color parameters
    parser.add_argument('--infra_point_color', type=str, default='cyan',
                       help='Color for infrastructure point cloud (default: cyan)')
    parser.add_argument('--vehicle_point_color', type=str, default='orange',
                       help='Color for vehicle point cloud (default: orange)')
    
    args = parser.parse_args()
    
    # Handle the no_vehicle_position flag
    if args.no_vehicle_position:
        args.show_vehicle_position = False
    
    print(f"Using point cloud range: {args.range}")
    print(f"Using transparency settings - Point cloud: {args.point_cloud_alpha}, Boxes: {args.box_alpha}, Arrows: {args.arrow_alpha}")
    print(f"Point cloud colors - Infrastructure: {args.infra_point_color}, Vehicle: {args.vehicle_point_color}")
    print(f"Cooperative data root: {args.cooperative_data_root}")
    print(f"Show vehicle position: {args.show_vehicle_position}")
    
    # Initialize NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Initialize visualizer with cooperative parameters
    visualizer = BEVVisualizer(nusc, scene_idx=args.scene_idx, 
                              point_cloud_range=args.range,
                              point_cloud_alpha=args.point_cloud_alpha,
                              box_alpha=args.box_alpha,
                              arrow_alpha=args.arrow_alpha,
                              cooperative_data_root=args.cooperative_data_root,
                              show_vehicle_position=args.show_vehicle_position,
                              infra_point_color=args.infra_point_color,
                              vehicle_point_color=args.vehicle_point_color)
    
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
