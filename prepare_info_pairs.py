#!/usr/bin/env python3
"""
Script to convert data_infos.json to v2i_pair.json and i2v_pair.json formats.
Transforms array format to nested dictionaries organized by vehicle/infrastructure sequences and frames.
"""

import json
import argparse
import sys
from pathlib import Path


def load_data_infos(input_file):
    """Load data_infos.json file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {input_file}: {e}")
        sys.exit(1)


def convert_to_v2i_pairs(data_infos):
    """
    Convert data_infos array format to v2i_pair nested dictionary format.
    
    Input format: [{"vehicle_frame": "002244", "infrastructure_frame": "002393", 
                   "vehicle_sequence": "0010", "infrastructure_sequence": "0010", 
                   "system_error_offset": {"delta_x": -1.1, "delta_y": -0.94}}]
    
    Output format: {"vehicle_sequence": {"vehicle_frame": {
                     "infrastructure_sequence": "0010", 
                     "infrastructure_frame": "002393", 
                     "system_error_offset": {"delta_x": -1.1, "delta_y": -0.94}}}}
    """
    v2i_pairs = {}
    
    for item in data_infos:
        vehicle_sequence = item["vehicle_sequence"]
        vehicle_frame = item["vehicle_frame"]
        
        # Initialize nested structure if not exists
        if vehicle_sequence not in v2i_pairs:
            v2i_pairs[vehicle_sequence] = {}
        
        # Create the nested entry
        v2i_pairs[vehicle_sequence][vehicle_frame] = {
            "infrastructure_sequence": item["infrastructure_sequence"],
            "infrastructure_frame": item["infrastructure_frame"],
            "system_error_offset": item["system_error_offset"]
        }
    
    return v2i_pairs


def convert_to_i2v_pairs(data_infos):
    """
    Convert data_infos array format to i2v_pair nested dictionary format.
    
    Input format: [{"vehicle_frame": "002244", "infrastructure_frame": "002393", 
                   "vehicle_sequence": "0010", "infrastructure_sequence": "0010", 
                   "system_error_offset": {"delta_x": -1.1, "delta_y": -0.94}}]
    
    Output format: {"infrastructure_sequence": {"infrastructure_frame": {
                     "vehicle_sequence": "0010", 
                     "vehicle_frame": "002244", 
                     "system_error_offset": {"delta_x": -1.1, "delta_y": -0.94}}}}
    """
    i2v_pairs = {}
    
    for item in data_infos:
        infrastructure_sequence = item["infrastructure_sequence"]
        infrastructure_frame = item["infrastructure_frame"]
        
        # Initialize nested structure if not exists
        if infrastructure_sequence not in i2v_pairs:
            i2v_pairs[infrastructure_sequence] = {}
        
        # Create the nested entry
        i2v_pairs[infrastructure_sequence][infrastructure_frame] = {
            "vehicle_sequence": item["vehicle_sequence"],
            "vehicle_frame": item["vehicle_frame"],
            "system_error_offset": item["system_error_offset"]
        }
    
    return i2v_pairs


def save_pairs(pairs_data, output_file, pair_type):
    """Save pairs data to output file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {pair_type} to {output_file}")
    except Exception as e:
        print(f"Error: Failed to save output file {output_file}: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and orchestrate the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert data_infos.json to v2i_pair.json and i2v_pair.json formats"
    )
    parser.add_argument(
        '--input',
        default='data/v2x-seq-nuscenes/cooperative/data_info.json',
        help='Input data_infos.json file path (default: data_infos.json)'
    )
    parser.add_argument(
        '--output',
        default='data/v2x-seq-nuscenes/cooperative/v2i_pair.json',
        help='Output v2i_pair.json file path (default: v2i_pair.json)'
    )
    parser.add_argument(
        '--i2v-output',
        default='data/v2x-seq-nuscenes/cooperative/i2v_pair.json',
        help='Output i2v_pair.json file path (default: i2v_pair.json)'
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}...")
    data_infos = load_data_infos(args.input)
    print(f"Loaded {len(data_infos)} entries from data_infos.json")
    
    # Convert to v2i_pairs format
    print("Converting to v2i_pair format...")
    v2i_pairs = convert_to_v2i_pairs(data_infos)
    
    # Convert to i2v_pairs format
    print("Converting to i2v_pair format...")
    i2v_pairs = convert_to_i2v_pairs(data_infos)
    
    # Count total entries for verification
    v2i_total_entries = sum(len(frames) for frames in v2i_pairs.values())
    i2v_total_entries = sum(len(frames) for frames in i2v_pairs.values())
    
    print(f"V2I: Converted to {len(v2i_pairs)} vehicle sequences with {v2i_total_entries} total entries")
    print(f"I2V: Converted to {len(i2v_pairs)} infrastructure sequences with {i2v_total_entries} total entries")
    
    # Save outputs
    print(f"Saving v2i_pair.json to {args.output}...")
    save_pairs(v2i_pairs, args.output, "v2i_pair.json")
    
    print(f"Saving i2v_pair.json to {args.i2v_output}...")
    save_pairs(i2v_pairs, args.i2v_output, "i2v_pair.json")


if __name__ == "__main__":
    main()
