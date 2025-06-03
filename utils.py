import os
import cv2
import re
import math
import json
import yaml
import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.prediction.metrics import mean_distances, final_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def quaternion_to_yaw(quaternion):
    """
    Convert a quaternion (w, x, y, z) to yaw (rotation about the z-axis) in radians.

    Parameters:
    quaternion (tuple): A tuple (w, x, y, z) representing the quaternion.

    Returns:
    float: Yaw angle in radians.
    """
    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return yaw


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).
    Handles non-orthogonal matrices by finding the closest orthogonal matrix using SVD.
    
    Parameters:
    rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
    
    Returns:
    tuple: A tuple (w, x, y, z) representing the quaternion.
    """
    # Ensure input is a numpy array
    R = np.array(rotation_matrix, dtype=np.float64)
    
    # Check if matrix is 3x3
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")
    
    # Check if matrix is close to orthogonal
    should_be_identity = R @ R.T
    identity = np.eye(3)
    orthogonality_error = np.linalg.norm(should_be_identity - identity)
    
    if orthogonality_error > 1e-6:
        print(f"Warning: Matrix is not orthogonal (error: {orthogonality_error:.2e}). "
              f"Finding closest orthogonal matrix using SVD.")
        
        # Use SVD to find the closest orthogonal matrix
        U, s, Vt = np.linalg.svd(R)
        
        # The closest orthogonal matrix is U @ Vt
        R_orthogonal = U @ Vt
        
        # Ensure proper rotation matrix (determinant = 1, not -1)
        if np.linalg.det(R_orthogonal) < 0:
            # Flip the last column of U to ensure det = 1
            U[:, -1] *= -1
            R_orthogonal = U @ Vt
        
        R = R_orthogonal
        print(f"Corrected matrix orthogonality error: {np.linalg.norm((R @ R.T) - identity):.2e}")
    
    try:
        # Use pyquaternion to convert rotation matrix to quaternion
        q = Quaternion(matrix=R)
        return (q.w, q.x, q.y, q.z)
    except Exception as e:
        print(f"Error converting matrix to quaternion with pyquaternion: {e}")
        
        # Fallback: manual conversion using Shepperd's method
        return _manual_rotation_matrix_to_quaternion(R)


def _manual_rotation_matrix_to_quaternion(R):
    """
    Manual conversion from rotation matrix to quaternion using Shepperd's method.
    This is a fallback when pyquaternion fails.
    
    Parameters:
    R (numpy.ndarray): A 3x3 orthogonal rotation matrix.
    
    Returns:
    tuple: A tuple (w, x, y, z) representing the quaternion.
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # Normalize the quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 0:
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    return (qw, qx, qy, qz)


def global_to_ego_frame(cur_pos, cur_heading, positions):
    """
    Transforms a list of global positions to the ego vehicle's local frame.

    Args:
        cur_pos (tuple): The current position of the ego vehicle (x, y).
        cur_heading (float): The heading of the ego vehicle in radians.
        positions (list of tuples): A list of global positions [(x1, y1), (x2, y2), ...].

    Returns:
        list of tuples: Transformed positions in the ego frame.
    """
    x0, y0 = cur_pos
    cos_h, sin_h = np.cos(-cur_heading), np.sin(
        -cur_heading
    )  # Negate heading for inverse rotation

    transformed_positions = []
    for x, y in positions:
        dx, dy = x - x0, y - y0  # Translate
        x_ego = cos_h * dx - sin_h * dy  # Rotate
        y_ego = sin_h * dx + cos_h * dy  # Rotate
        transformed_positions.append((x_ego, y_ego))

    return transformed_positions


def compute_speed(points, past_timestamps):
    """
    Computes the speed at each point using Euclidean distance between adjacent points.

    Parameters:
        points (list of tuples): List of (x, y) coordinates.
        time_interval (float): Time interval between points (default is 0.5s).

    Returns:
        list of floats: Speed values at each segment.
    """
    speeds = []
    for i in range(1, len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        t1 = past_timestamps[i]
        t2 = past_timestamps[i + 1]

        # Compute Euclidean distance
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        time_interval = (t2 - t1) / 1e6  # Convert to seconds

        # Compute speed (distance / time)
        speed = distance / time_interval

        if speed < 0.2:
            speed = 0.0

        speeds.append(round(speed, 3))

    return speeds


def compute_curvature(points):
    """
    Computes the curvature at each point using its neighboring points.
    The curvature is computed using the osculating circle formula.

    Parameters:
        points (list of tuples): List of (x, y) coordinates.

    Returns:
        list of floats: Curvature values for each valid point.
    """
    curvatures = []
    for i in range(1, len(points) - 1):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        x3, y3 = points[i + 1]

        # Compute distances
        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        # Compute signed area using determinant formula
        area = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

        # Curvature formula: 4 * area / (a * b * c)
        if a * b * c != 0:  # Avoid division by zero
            curvature = 4 * area / (a * b * c)
            if curvature > 0.2:
                curvature = 0.0
            elif curvature < -0.2:
                curvature = -0.0
        else:
            curvature = 0  # Assign zero curvature for degenerate cases

        curvatures.append(round(curvature, 3))

    return curvatures


def integrate_driving_commands(commands, dt=0.5):
    """
    Computes the trajectory given an initial position (0,0), orientation 0, and a list of driving commands.

    Parameters:
    commands (list of tuples): List of (speed, curvature) pairs.
    dt (float): Time step duration in seconds (default is 0.5s).

    Returns:
    list of tuples: List of (x, y, theta) positions over time.
    """
    x, y, heading = 0.0, 0.0, 0.0
    trajectory = [(x, y)]

    for v, kappa in commands:
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        heading += kappa * v * dt  # Update heading
        trajectory.append((x, y))

    return trajectory[1:]  # Exclude the initial position

def OverlayTrajectory(
    img_path: str,
    wp_world1: list,  
    wp_world2: list,
    cam_to_ego,
    ego_pos: tuple,
    ego_heading: float,
    save_path: str,
):
    # Load the image from file
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Failed to load image from path: {img_path}")

    img = original_img.copy()  # Work on a copy to keep the original unchanged

    # --- Form Transformation Matrices ---
    T_ego_global = np.eye(4)
    T_ego_global[:3, :3] = np.array(
        [
            [np.cos(ego_heading), -np.sin(ego_heading), 0],
            [np.sin(ego_heading), np.cos(ego_heading), 0],
            [0, 0, 1],
        ]
    )
    T_ego_global[:3, 3] = np.array([ego_pos[0], ego_pos[1], 0])

    T_cam_ego = np.eye(4)
    T_cam_ego[:3, :3] = Quaternion(cam_to_ego["rotation"]).rotation_matrix
    T_cam_ego[:3, 3] = cam_to_ego["translation"]

    T_cam_global = T_ego_global @ T_cam_ego
    T_global_cam = np.linalg.inv(T_cam_global)

    # Only using ground truth and prediction colors
    colors = {
        "wp_world1": (0.0, 0.23, 0.34, 1.0),    # Dark blue for ground truth
        "wp_world2": (0.56, 0.79, 0.90, 1.0),   # Light blue for prediction   
    }
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    # Define common styling properties
    linewidth = 3
    linestyle = "solid"
    arrowstyle = "->"
    mutation_scale = 20
    arrow_extension = 10

    # Process only ground truth and prediction trajectories
    for wp_world, key in zip(
        [wp_world1, wp_world2], ["wp_world1", "wp_world2"]
    ):
        color = colors[key]
        
        # Transform 3D World Points to Camera Coordinates
        points3d_world = [np.array([p[0], p[1], 0]) for p in wp_world]
        points3d_cam = np.array(
            [(T_global_cam @ np.append(p, 1))[:3] for p in points3d_world]
        )
        
        valid = points3d_cam[:, 2] > 0
        if not valid.any():
            continue
            
        # Project valid points onto the image plane
        points_valid = points3d_cam[valid]
        proj = (cam_to_ego["camera_intrinsic"] @ points_valid.T).T
        points3d_img = proj[:, :2] / proj[:, 2][:, np.newaxis]
        
        if len(points3d_img) <= 1:
            continue
            
        # Draw trajectory line
        x, y = zip(*points3d_img)
        ax.plot(x, y, linestyle=linestyle, color=color, linewidth=linewidth)
        
        # Draw arrows on each segment
        for i in range(1, len(points3d_img)):
            start = np.array(points3d_img[i - 1])
            end = np.array(points3d_img[i])
            vector = end - start
            norm = np.linalg.norm(vector)
            if norm != 0:
                end = end + arrow_extension * (vector / norm)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color,
                    lw=linewidth,
                    mutation_scale=mutation_scale,
                ),
            )

    # Convert Matplotlib figure to OpenCV image
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    buf = buf[:-5, :, :]  # Remove extra padding
    img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    # Close the matplotlib figure to prevent memory leaks
    plt.close(fig)

    # Save the Updated Image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"Image saved at {save_path}")

def compute_metrics(predicted_trajectory, ground_truth_trajectory):
    """
    Calculate comprehensive metrics between predicted and ground truth trajectories:
    L2_1s, L2_2s, L2_3s, ADE, FDE, and missRate_2
    
    Args:
        predicted_trajectory: List of (x, y) points of the predicted trajectory
        ground_truth_trajectory: List of (x, y) points of the ground truth trajectory
        
    Returns:
        dict: Dictionary containing all metrics
    """
    from nuscenes.eval.prediction.metrics import mean_distances, final_distances
    
    # Convert to numpy arrays if they aren't already
    pred_traj = np.array(predicted_trajectory)
    gt_traj = np.array(ground_truth_trajectory)
    
    # Ensure both trajectories have the same length (use the shorter one)
    min_length = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_length]
    gt_traj = gt_traj[:min_length]
    
    # Initialize metrics dictionary
    metrics = {
        "L2_1s": None,
        "L2_2s": None,
        "L2_3s": None,
        "ADE": None,
        "FDE": None,
        "missRate_2": None
    }
    
    # For L2_1s (1 second horizon - first 2 points)
    if len(pred_traj) >= 2 and len(gt_traj) >= 2:
        gt_pos_1s = gt_traj[:2]
        pred_traj_1s = pred_traj[:2]
        l2_per_timestep_1s = np.linalg.norm(gt_pos_1s - pred_traj_1s, axis=1)
        metrics["L2_1s"] = np.mean(l2_per_timestep_1s)
    
    # For L2_2s (2 second horizon - middle 2 points)
    if len(pred_traj) >= 4 and len(gt_traj) >= 4:
        gt_pos_2s = gt_traj[2:4]
        pred_traj_2s = pred_traj[2:4]
        l2_per_timestep_2s = np.linalg.norm(gt_pos_2s - pred_traj_2s, axis=1)
        metrics["L2_2s"] = np.mean(l2_per_timestep_2s)
    
    # For L2_3s (3 second horizon - last 2 points)
    if len(pred_traj) >= 6 and len(gt_traj) >= 6:
        gt_pos_3s = gt_traj[4:6]
        pred_traj_3s = pred_traj[4:6]
        l2_per_timestep_3s = np.linalg.norm(gt_pos_3s - pred_traj_3s, axis=1)
        metrics["L2_3s"] = np.mean(l2_per_timestep_3s)
    
    # Prepare data for nuScenes metrics calculation
    # Reshape to match nuScenes expected format: [num_modes, n_timesteps, 2]
    pred_stacked = np.expand_dims(pred_traj, axis=0)  # Shape: [1, n_timesteps, 2]
    gt_stacked = np.expand_dims(gt_traj, axis=0)  # Shape: [1, n_timesteps, 2]
    
    # Calculate metrics using nuScenes functions
    ade_errors = mean_distances(pred_stacked, gt_stacked)
    fde_errors = final_distances(pred_stacked, gt_stacked)
    
    # ADE: Average Displacement Error
    metrics["ADE"] = float(np.min(ade_errors))
    
    # FDE: Final Displacement Error
    metrics["FDE"] = float(np.min(fde_errors))
    
    # MissRate_2: determine if prediction has max distance > 2m
    # Calculate L2 distance for each timestep
    distances = np.sqrt(np.sum((pred_traj - gt_traj)**2, axis=1))
    max_distance = np.max(distances)
    
    # Count as miss if max distance > 2m
    metrics["missRate_2"] = 1.0 if max_distance > 2.0 else 0.0
    
    return metrics

def save_dict_to_json(data, file_path):
    """
    Saves a dictionary to a JSON file.
    
    Args:
        data (dict): Dictionary to save
        file_path (str): Path where the JSON file should be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the dictionary to a JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json_file(file_path):
    """
    Loads a JSON file into a dictionary.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        dict: The loaded dictionary or empty dict if the file doesn't exist
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_config(config_path):
    """
    Loads a YAML file into a dictionary.
    
    Args:
        file_path (str): Path to the config file to load
        
    Returns:
        dict: The loaded dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_scene_number(scene_name):
    """
    Extract numeric part from scene name (e.g., "scene-0495" -> 495)

    Args:
        scene_name (str): Scene name

    Returns:
        int: Extracted scene number or infinity if not found
    """
    match = re.search(r"(\d+)", scene_name)
    if match:
        return int(match.group(1))
    return float("inf")

def update_scene_json(scene_file_path, frame_data, metadata=None):
    """
    Updates a scene JSON file with new frame data in the hierarchical structure.
    If the file doesn't exist, creates it with the proper structure.
    
    Args:
        scene_file_path (str): Path to the scene JSON file
        frame_data (dict): Data for a single frame to add to the scene
        metadata (dict, optional): Scene metadata to update/add
    """
    # Load existing data or create new structure
    scene_data = load_json_file(scene_file_path)
    
    # Initialize structure if needed
    if not scene_data:
        scene_data = {
            "scene_info": {
                "name": frame_data.get("scene_name", ""),
                "description": frame_data.get("description", ""),
                "first_sample_token": frame_data.get("first_sample_token", ""),
                "last_sample_token": frame_data.get("last_sample_token", "")
            },
            "frames": [],
            "metadata": metadata or {}
        }
    
    # Add the new frame
    scene_data["frames"].append(frame_data)
    
    # Update metadata if provided
    if metadata:
        scene_data["metadata"].update(metadata)
        
    # Update total frames count
    scene_data["metadata"]["total_frames"] = len(scene_data["frames"])
    
    # Save updated data
    save_dict_to_json(scene_data, scene_file_path)

def format_long_text(text, max_line_length=80):
    """
    Break long text into structured paragraphs for better readability in JSON.
    Instead of trying to add literal newlines (which get escaped in JSON),
    split the text into separate paragraph elements in a list.
    
    Args:
        text (str): The text to format
        max_line_length (int): Maximum length per paragraph
        
    Returns:
        list: List of paragraph strings
    """
    if not text:
        return []
    
    # Split text into paragraphs first (if there are any)
    paragraphs = text.split('\n')
    
    # Further split long paragraphs
    result = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) == 0:
            continue
            
        # If paragraph is short enough, add it as is
        if len(paragraph) <= max_line_length:
            result.append(paragraph.strip())
        else:
            # Split long paragraph into chunks
            words = paragraph.split()
            current_line = []
            current_length = 0
            
            for word in words:
                # Check if adding this word would exceed the line length
                if current_length + len(word) + (1 if current_length > 0 else 0) > max_line_length:
                    # Add the current line to result and start a new line
                    if current_line:
                        result.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Add the word to the current line
                    current_line.append(word)
                    current_length += len(word) + (1 if current_length > 0 else 0)
            
            # Add the last line if there's anything left
            if current_line:
                result.append(' '.join(current_line))
    
    return result

