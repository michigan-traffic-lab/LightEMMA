import os
import numpy as np
import argparse
import glob
import re
from collections import OrderedDict

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: Single VLM Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing the VLM prediction results (default: from config)",
    )
    parser.add_argument(
        "--no_vis", action="store_true", help="Disable visualization generation"
    )
    parser.add_argument(
        "--local_samples_path",
        type=str,
        default=None,
        help="Local path to the nuscenes samples directory. If provided, will extract the relative path after '/samples/' and use this path instead.",
    )
    return parser.parse_args()


def convert_image_path(image_path, data_dir):
    """
    Convert image path to full path if needed.

    Args:
        image_path: Original image path from the results file
        data_dir: Base directory containing the nuScenes dataset

    Returns:
        str: Updated image path for local file system
    """
    # Try to find absolute path
    if os.path.exists(image_path):
        return image_path

    # Try relative to data_dir
    full_path = os.path.join(data_dir, image_path)
    if os.path.exists(full_path):
        return full_path

    # Extract the relative path (after 'samples/') if '/samples/' in path
    if "/samples/" in image_path:
        rel_path = image_path.split("/samples/")[-1]
        return os.path.join(data_dir, "samples", rel_path)

    # If 'samples' is not in the path, try to extract the CAM_FRONT portion
    if "/CAM_FRONT/" in image_path:
        rel_path = image_path.split("/CAM_FRONT/")[-1]
        return os.path.join(data_dir, "samples", "CAM_FRONT", rel_path)

    # If we can't do a clean replacement, just return the original path
    print(f"Warning: Could not find image at {image_path}")
    return image_path


def sort_scene_frames(scene_data):
    """
    Sort frames within a scene by frame index.

    Args:
        scene_data: Dictionary containing scene data with frames

    Returns:
        Dictionary with sorted frames
    """
    # Sort frames by frame_index
    if "frames" in scene_data:
        scene_data["frames"] = sorted(
            scene_data["frames"], key=lambda x: x.get("frame_index", float("inf"))
        )

    # Sort frame_results by frame_index if present
    if "frame_results" in scene_data:
        scene_data["frame_results"] = sorted(
            scene_data["frame_results"],
            key=lambda x: x.get("frame_index", float("inf")),
        )

    return scene_data


def extract_error_frames(evaluation_results, error_file_path):
    """
    Extract all error frames from evaluation results and save to error.txt.

    Args:
        evaluation_results: Dictionary containing evaluation results
        error_file_path: Path to save the error.txt file
    """
    # Collect all error frames
    error_frames = []

    # Extract error frames from each scene
    for scene_name, scene_data in evaluation_results["per_scene"].items():
        for frame_result in scene_data.get("frame_results", []):
            # Check if this is an error frame
            if frame_result.get("error", False):
                # Get frame index
                frame_idx = frame_result.get("frame_index", -1)

                # Get prediction string if available, otherwise use "N/A"
                pred_str = frame_result.get("pred_actions_str", "N/A")

                # Clean up prediction string - remove newlines and limit length
                if pred_str != "N/A":
                    # Replace newlines with spaces and limit length
                    pred_str = pred_str.replace("\n", " ").replace("\r", " ")
                    # Limit string length to avoid overly long lines
                    if len(pred_str) > 100:
                        pred_str = pred_str[:97] + "..."

                # Add to error frames list
                error_frames.append((scene_name, frame_idx, pred_str))

    # Sort error frames by scene name and frame index
    # Extract scene number and use it for sorting
    def get_scene_number(scene_name):
        # Extract numeric part from scene name (e.g., "scene-0495" -> 495)
        match = re.search(r"(\d+)", scene_name)
        if match:
            return int(match.group(1))
        return float("inf")  # For any scenes without numbers

    error_frames.sort(key=lambda x: (get_scene_number(x[0]), x[1]))

    # Write sorted error frames to file
    with open(error_file_path, "w") as error_file:
        # Write header
        error_file.write("scene\tframe\tpred_actions_str\n")

        # Write sorted error frames
        for scene_name, frame_idx, pred_str in error_frames:
            error_file.write(f"{scene_name}\t{frame_idx}\t{pred_str}\n")

    print(f"Error frames extracted and saved to {error_file_path}")


def evaluate_predictions(results_dir, data_dir, generate_vis=True):
    """
    Evaluate all predictions in the results directory.

    Args:
        results_dir: Directory containing the VLM results
        data_dir: Directory containing the nuScenes dataset
        generate_vis: Whether to generate visualizations

    Returns:
        dict: Dictionary containing evaluation results
    """
    print(f"Evaluating predictions in {results_dir}")

    args = parse_args()
    # Create or verify output directory structure
    output_dir = os.path.join(results_dir, "output")
    frame_dir = os.path.join(results_dir, "frame")
    analysis_dir = os.path.join(results_dir, "analysis")

    os.makedirs(analysis_dir, exist_ok=True)
    if generate_vis:
        os.makedirs(frame_dir, exist_ok=True)

    # Find all scene JSON files
    scene_files = glob.glob(os.path.join(output_dir, "*.json"))

    if not scene_files:
        print(f"No JSON files found in {output_dir}")
        return None

    # Initialize metrics
    total_frames = 0
    successful_frames = 0
    parse_error_frames = 0

    # Metrics per frame
    all_l2_1s = []
    all_l2_2s = []
    all_l2_3s = []
    all_ade = []
    all_fde = []
    all_miss_rates = []

    # Token and time usage metrics
    total_token_usage = {
        "scene_prompt": {"input": 0, "output": 0},
        "intent_prompt": {"input": 0, "output": 0},
        "waypoint_prompt": {"input": 0, "output": 0},
        "total": {"input": 0, "output": 0},
    }

    total_time_usage = {
        "scene_prompt": 0,
        "intent_prompt": 0,
        "waypoint_prompt": 0,
        "total": 0,
    }

    # Metrics per scene
    scene_metrics = {}

    for scene_file in scene_files:
        scene_name = os.path.basename(scene_file).split(".")[0]
        print(f"Processing scene {scene_name}")

        scene_data = load_json_file(scene_file)

        if not scene_data or "frames" not in scene_data or not scene_data["frames"]:
            print(f"No valid data in {scene_file}")
            continue

        # Sort frames within the scene data
        scene_data = sort_scene_frames(scene_data)

        # Initialize metrics for this scene
        scene_l2_1s = []
        scene_l2_2s = []
        scene_l2_3s = []
        scene_ade = []
        scene_fde = []
        scene_miss_rates = []
        scene_frames = 0
        scene_successful = 0
        scene_parse_errors = 0

        # Initialize token and time usage for this scene
        scene_token_usage = {
            "scene_prompt": {"input": 0, "output": 0},
            "intent_prompt": {"input": 0, "output": 0},
            "waypoint_prompt": {"input": 0, "output": 0},
            "total": {"input": 0, "output": 0},
        }

        scene_time_usage = {
            "scene_prompt": 0,
            "intent_prompt": 0,
            "waypoint_prompt": 0,
            "total": 0,
        }

        # Initialize frame results for this scene
        frame_results = []

        # Process each frame in the scene
        for frame_idx, frame in enumerate(scene_data["frames"]):
            total_frames += 1
            scene_frames += 1

            frame_result = {
                "frame_index": frame["frame_index"],
                "sample_token": frame["sample_token"],
                "image_path": frame["image_path"],
            }

            try:
                # Get ground truth positions
                gt_positions = frame["ego_info"]["gt_positions"]

                # Check if predictions exist and are valid
                if "predictions" in frame and "trajectory" in frame["predictions"]:
                    successful_frames += 1
                    scene_successful += 1

                    # Get predicted trajectory
                    pred_trajectory = frame["predictions"]["trajectory"]

                    # Calculate metrics
                    metrics = compute_metrics(pred_trajectory, gt_positions)

                    # Store metrics
                    if metrics["L2_1s"] is not None:
                        all_l2_1s.append(metrics["L2_1s"])
                        scene_l2_1s.append(metrics["L2_1s"])

                    if metrics["L2_2s"] is not None:
                        all_l2_2s.append(metrics["L2_2s"])
                        scene_l2_2s.append(metrics["L2_2s"])

                    if metrics["L2_3s"] is not None:
                        all_l2_3s.append(metrics["L2_3s"])
                        scene_l2_3s.append(metrics["L2_3s"])

                    if metrics["ADE"] is not None:
                        all_ade.append(metrics["ADE"])
                        scene_ade.append(metrics["ADE"])

                    if metrics["FDE"] is not None:
                        all_fde.append(metrics["FDE"])
                        scene_fde.append(metrics["FDE"])

                    if metrics["missRate_2"] is not None:
                        all_miss_rates.append(metrics["missRate_2"])
                        scene_miss_rates.append(metrics["missRate_2"])

                    # Add metrics to frame result
                    frame_result["metrics"] = metrics

                    # Add token usage to frame result and accumulate for scene
                    if "token_usage" in frame:
                        frame_result["token_usage"] = frame["token_usage"]

                        # Accumulate token usage for this scene
                        for prompt_type, counts in frame["token_usage"].items():
                            scene_token_usage[prompt_type]["input"] += counts["input"]
                            scene_token_usage[prompt_type]["output"] += counts["output"]
                            scene_token_usage["total"]["input"] += counts["input"]
                            scene_token_usage["total"]["output"] += counts["output"]

                    # Add time usage to frame result and accumulate for scene
                    if "time_usage" in frame:
                        frame_result["time_usage"] = frame["time_usage"]

                        # Accumulate time usage for this scene
                        for prompt_type, time_val in frame["time_usage"].items():
                            scene_time_usage[prompt_type] += time_val
                            scene_time_usage["total"] += time_val

                    # Create visualization if requested
                    if generate_vis:
                        print(f"Generating visualization for frame {frame_idx} in scene {scene_name}")
                        # Convert image path to accessible path
                        image_path = convert_image_path(
                            frame["image_path"],
                            (
                                data_dir
                                if args.local_samples_path is None
                                else args.local_samples_path
                            ),
                        )

                        # Check if the image file exists
                        if os.path.exists(image_path):
                            camera_params = frame["camera_params"]

                            # Convert rotation matrix to quaternion if it's in matrix format
                            if "rotation" in camera_params:
                                rotation = camera_params["rotation"]
                                # Check if rotation is a matrix (list of lists or numpy array)
                                if isinstance(rotation, (list, np.ndarray)):
                                    # Check for 3x3 matrix (list of lists or 2D array)
                                    if (isinstance(rotation, list) and len(rotation) == 3 and 
                                        all(isinstance(row, (list, np.ndarray)) and len(row) == 3 for row in rotation)):
                                        # 3x3 rotation matrix as list of lists
                                        rotation_matrix = np.array(rotation)
                                        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
                                        camera_params["rotation"] = quaternion
                                        print(f"Converted 3x3 rotation matrix (list of lists) to quaternion: {quaternion}")
                                    elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
                                        # 3x3 rotation matrix as numpy array
                                        quaternion = rotation_matrix_to_quaternion(rotation)
                                        camera_params["rotation"] = quaternion
                                        print(f"Converted 3x3 rotation matrix (numpy array) to quaternion: {quaternion}")
                                    elif len(rotation) == 4:
                                        # Already in quaternion format (w, x, y, z)
                                        print(f"Rotation already in quaternion format: {rotation}")
                                    elif len(rotation) == 9:
                                        # Flattened 3x3 rotation matrix
                                        rotation_matrix = np.array(rotation).reshape(3, 3)
                                        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
                                        camera_params["rotation"] = quaternion
                                        print(f"Converted flattened rotation matrix to quaternion: {quaternion}")
                                    else:
                                        print(f"Unknown rotation format: {type(rotation)}, shape/length: {rotation.shape if hasattr(rotation, 'shape') else len(rotation) if hasattr(rotation, '__len__') else 'N/A'}")
                                else:
                                    print(f"Rotation is not a list or numpy array: {type(rotation)}")
                            
                            print(camera_params)
                            ego_pos = frame["ego_info"]["position"]
                            ego_heading = frame["ego_info"]["heading"]

                            # Get reconstructed trajectory from ground truth actions
                            gt_actions = frame["ego_info"]["gt_actions"]

                            # Create visualization filename
                            viz_filename = f"{scene_name}_frame{frame_idx}.png"
                            viz_path = os.path.join(frame_dir, viz_filename)

                            # Create visualization
                            OverlayTrajectory(
                                img_path=image_path,
                                wp_world1=gt_positions,  # Ground truth
                                wp_world2=pred_trajectory,  # VLM prediction
                                cam_to_ego=camera_params,
                                ego_pos=(0, 0),
                                ego_heading=0.0,
                                save_path=viz_path,
                            )
                else:
                    # Count frames with parse errors
                    parse_error_frames += 1
                    scene_parse_errors += 1

                    # For frames with errors, just add the error information
                    frame_result["error"] = True
                    if (
                        "predictions" in frame
                        and "pred_actions_str" in frame["predictions"]
                    ):
                        frame_result["pred_actions_str"] = frame["predictions"][
                            "pred_actions_str"
                        ]

            except Exception as e:
                print(f"Error processing frame {frame_idx} in scene {scene_name}: {e}")
                frame_result["error"] = True
                frame_result["error_message"] = str(e)
                continue

            # Add frame result to the list for this scene
            frame_results.append(frame_result)

        # Add token usage to overall totals
        for prompt_type in total_token_usage.keys():
            total_token_usage[prompt_type]["input"] += scene_token_usage[prompt_type][
                "input"
            ]
            total_token_usage[prompt_type]["output"] += scene_token_usage[prompt_type][
                "output"
            ]

        # Add time usage to overall totals
        for prompt_type in total_time_usage.keys():
            total_time_usage[prompt_type] += scene_time_usage[prompt_type]

        # Calculate scene-level metrics
        if scene_successful > 0:
            scene_metrics[scene_name] = {
                "frames_total": scene_frames,
                "frames_successful": scene_successful,
                "frames_parse_errors": scene_parse_errors,
                "success_rate": scene_successful / scene_frames,
                "metrics": {
                    "L2_1s": np.mean(scene_l2_1s).item() if scene_l2_1s else None,
                    "L2_2s": np.mean(scene_l2_2s).item() if scene_l2_2s else None,
                    "L2_3s": np.mean(scene_l2_3s).item() if scene_l2_3s else None,
                    "ADE": np.mean(scene_ade).item() if scene_ade else None,
                    "FDE": np.mean(scene_fde).item() if scene_fde else None,
                    "missRate_2": (
                        np.mean(scene_miss_rates).item() if scene_miss_rates else None
                    ),
                },
                "token_usage": scene_token_usage,
                "time_usage": scene_time_usage,
                "frame_results": sorted(
                    frame_results, key=lambda x: x.get("frame_index", float("inf"))
                ),
            }

    # Sort scenes by scene number
    def get_scene_number(scene_name):
        match = re.search(r"(\d+)", scene_name)
        if match:
            return int(match.group(1))
        return float("inf")

    # Create sorted OrderedDict for scene_metrics
    sorted_scene_metrics = OrderedDict()
    for scene_name in sorted(scene_metrics.keys(), key=get_scene_number):
        sorted_scene_metrics[scene_name] = scene_metrics[scene_name]

    # Calculate overall metrics
    overall_metrics = {
        "frames_total": total_frames,
        "frames_successful": successful_frames,
        "frames_parse_errors": parse_error_frames,
        "success_rate": successful_frames / total_frames if total_frames > 0 else 0,
        "metrics": {
            "L2_1s": np.mean(all_l2_1s).item() if all_l2_1s else None,
            "L2_2s": np.mean(all_l2_2s).item() if all_l2_2s else None,
            "L2_3s": np.mean(all_l2_3s).item() if all_l2_3s else None,
            "ADE": np.mean(all_ade).item() if all_ade else None,
            "FDE": np.mean(all_fde).item() if all_fde else None,
            "missRate_2": np.mean(all_miss_rates).item() if all_miss_rates else None,
        },
        "token_usage": total_token_usage,
        "time_usage": total_time_usage,
    }

    # Create evaluation results dictionary
    evaluation_results = {"overall": overall_metrics, "per_scene": sorted_scene_metrics}

    return evaluation_results


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get paths from config or arguments
    data_dir = config["data"]["root"]

    # Use provided results_dir or find the latest model directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Assuming we're evaluating the most recent model run
        model_dirs = glob.glob(os.path.join(config["data"]["results"], "*"))
        if not model_dirs:
            print("Error: No model directories found in results path")
            return
        # Sort by modification time to get the latest
        results_dir = max(model_dirs, key=os.path.getmtime)

    print(f"Using results directory: {results_dir}")

    # Evaluate predictions
    evaluation_results = evaluate_predictions(
        results_dir,
        data_dir if args.local_samples_path is None else args.local_samples_path,
        generate_vis=not args.no_vis,
    )

    if evaluation_results:
        # Save evaluation results to analysis directory
        analysis_dir = os.path.join(results_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        evaluation_file = os.path.join(analysis_dir, "evaluation.json")
        save_dict_to_json(evaluation_results, evaluation_file)
        print(f"Evaluation results saved to {evaluation_file}")

        # Extract error frames and save to error.txt
        error_file_path = os.path.join(analysis_dir, "errors.txt")
        extract_error_frames(evaluation_results, error_file_path)

        # Print overall metrics summary
        overall = evaluation_results["overall"]["metrics"]
        print("\nOverall Metrics:")
        print(f"L2 1s: {overall['L2_1s']:.4f}")
        print(f"L2 2s: {overall['L2_2s']:.4f}")
        print(f"L2 3s: {overall['L2_3s']:.4f}")
        print(f"ADE: {overall['ADE']:.4f}")
        print(f"FDE: {overall['FDE']:.4f}")
        print(f"missRate_2: {overall['missRate_2']:.4f}")

        # Print token usage summary
        token_usage = evaluation_results["overall"]["token_usage"]["total"]
        print(f"\nTotal tokens: {token_usage['input'] + token_usage['output']}")
        print(f"Input tokens: {token_usage['input']}")
        print(f"Output tokens: {token_usage['output']}")

        # Print time usage summary
        time_usage = evaluation_results["overall"]["time_usage"]["total"]
        print(f"\nTotal inference time: {time_usage:.2f} seconds")

        # Print error summary
        error_frames = evaluation_results["overall"]["frames_parse_errors"]
        print(
            f"\nFrames with errors: {error_frames}/{evaluation_results['overall']['frames_total']} "
            f"({error_frames/evaluation_results['overall']['frames_total']*100:.2f}%)"
        )
    else:
        print("No valid predictions to evaluate")


if __name__ == "__main__":
    main()
