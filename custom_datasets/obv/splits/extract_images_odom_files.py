#!/usr/bin/env python3

import os
import rosbag
import numpy as np
import cv2
import csv
from tqdm import tqdm
import glob
from collections import defaultdict
import bisect

def find_closest_timestamp(target_time, timestamp_list):
    """
    Find the closest timestamp in a sorted list to the target time
    Returns the index of the closest timestamp
    """
    if not timestamp_list:
        return None
    
    # Use binary search to find insertion point
    idx = bisect.bisect_left(timestamp_list, target_time)
    
    # Check which timestamp is closer
    if idx == 0:
        return 0
    elif idx == len(timestamp_list):
        return len(timestamp_list) - 1
    else:
        # Compare distances
        if abs(timestamp_list[idx] - target_time) < abs(timestamp_list[idx-1] - target_time):
            return idx
        else:
            return idx - 1

def extract_synchronized_data(sensor_bag_path, gt_bag_path, output_dir, interval_seconds=1.0):
    """
    Extract synchronized RGB, thermal, and odometry data at regular intervals
    
    Args:
        sensor_bag_path: Path to the bag with RGB and thermal data
        gt_bag_path: Path to the bag with ground truth odometry
        output_dir: Directory where extracted data will be saved
        interval_seconds: Time interval between extracted frames (default: 1 second)
    """
    
    # Create output directories
    rgb_dir = os.path.join(output_dir, 'rgb')
    thermal_dir = os.path.join(output_dir, 'thermal')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    
    bag_name = os.path.splitext(os.path.basename(sensor_bag_path))[0]
    print(f"\nProcessing: {bag_name}")
    print(f"Sensor bag: {sensor_bag_path}")
    print(f"GT bag: {gt_bag_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Read all odometry data from GT bag
        print("Reading odometry data from GT bag...")
        gt_bag = rosbag.Bag(gt_bag_path, 'r')
        
        gt_info = gt_bag.get_type_and_topic_info()
        gt_topics = list(gt_info.topics.keys())
        print(f"GT bag topics: {gt_topics}")
        
        # Find odometry topic
        odom_topics = ['/aft_mapped_to_init', '/odom', '/odometry']
        found_odom_topic = None
        for topic in odom_topics:
            if topic in gt_topics:
                found_odom_topic = topic
                break
        
        if not found_odom_topic:
            print("No odometry topic found in GT bag!")
            gt_bag.close()
            return
        
        print(f"Found odometry topic: {found_odom_topic}")
        
        # Read all odometry messages
        odom_data = {}  # timestamp -> odometry message
        odom_timestamps = []
        
        for topic, msg, t in gt_bag.read_messages(topics=[found_odom_topic]):
            timestamp = t.to_nsec()
            odom_data[timestamp] = msg
            odom_timestamps.append(timestamp)
        
        gt_bag.close()
        odom_timestamps.sort()
        print(f"Read {len(odom_data)} odometry messages")
        
        # Step 2: Read all image data from sensor bag
        print("Reading image data from sensor bag...")
        sensor_bag = rosbag.Bag(sensor_bag_path, 'r')
        
        sensor_info = sensor_bag.get_type_and_topic_info()
        sensor_topics = list(sensor_info.topics.keys())
        print(f"Sensor bag topics: {sensor_topics}")
        
        # Find image topics
        rgb_topics = ['/camera/color/image_raw', '/camera/rgb/image_raw', '/rgb/image_raw']
        thermal_topics = ['/flir_boson/image_raw', '/thermal/image_raw', '/flir/image_raw']
        
        found_rgb_topic = None
        found_thermal_topic = None
        
        for topic in rgb_topics:
            if topic in sensor_topics:
                found_rgb_topic = topic
                break
                
        for topic in thermal_topics:
            if topic in sensor_topics:
                found_thermal_topic = topic
                break
        
        print(f"Found RGB topic: {found_rgb_topic}")
        print(f"Found thermal topic: {found_thermal_topic}")
        
        # Read all image messages
        rgb_data = {}  # timestamp -> image message
        thermal_data = {}  # timestamp -> image message
        rgb_timestamps = []
        thermal_timestamps = []
        
        image_topics = []
        if found_rgb_topic:
            image_topics.append(found_rgb_topic)
        if found_thermal_topic:
            image_topics.append(found_thermal_topic)
        
        if not image_topics:
            print("No image topics found!")
            sensor_bag.close()
            return
        
        for topic, msg, t in tqdm(sensor_bag.read_messages(topics=image_topics)):
            timestamp = t.to_nsec()
            
            if topic == found_rgb_topic:
                rgb_data[timestamp] = msg
                rgb_timestamps.append(timestamp)
            elif topic == found_thermal_topic:
                thermal_data[timestamp] = msg
                thermal_timestamps.append(timestamp)
        
        sensor_bag.close()
        rgb_timestamps.sort()
        thermal_timestamps.sort()
        
        print(f"Read {len(rgb_data)} RGB images")
        print(f"Read {len(thermal_data)} thermal images")
        
        # Step 3: Determine synchronization time range
        if not odom_timestamps:
            print("No odometry data available!")
            return
        
        # Use odometry time range as reference
        start_time = odom_timestamps[0]
        end_time = odom_timestamps[-1]
        
        print(f"Time range: {(end_time - start_time) / 1e9:.2f} seconds")
        
        # Step 4: Extract synchronized data at regular intervals
        print(f"Extracting synchronized data every {interval_seconds} seconds...")
        
        interval_ns = int(interval_seconds * 1e9)  # Convert to nanoseconds
        current_time = start_time
        
        triplets_csv = os.path.join(output_dir, 'synchronized_data.csv')
        
        with open(triplets_csv, 'w', newline='') as csvfile:
            fieldnames = ['sequence_id', 'timestamp_ns', 'rgb_filename', 'thermal_filename',
                         'pos_x', 'pos_y', 'pos_z', 
                         'orient_x', 'orient_y', 'orient_z', 'orient_w',
                         'linear_vel_x', 'linear_vel_y', 'linear_vel_z',
                         'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            sequence_id = 0
            
            while current_time <= end_time:
                sequence_id += 1
                
                # Find closest odometry
                odom_idx = find_closest_timestamp(current_time, odom_timestamps)
                if odom_idx is None:
                    current_time += interval_ns
                    continue
                
                closest_odom_time = odom_timestamps[odom_idx]
                odom_msg = odom_data[closest_odom_time]
                
                # Find closest RGB image
                rgb_filename = None
                if rgb_timestamps:
                    rgb_idx = find_closest_timestamp(current_time, rgb_timestamps)
                    if rgb_idx is not None:
                        closest_rgb_time = rgb_timestamps[rgb_idx]
                        rgb_msg = rgb_data[closest_rgb_time]
                        
                        # Extract and save RGB image
                        rgb_filename = f"seq_{sequence_id:04d}_{closest_rgb_time}.png"
                        try:
                            np_arr = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
                            
                            if rgb_msg.encoding == 'rgb8':
                                cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
                            elif rgb_msg.encoding == 'bgr8':
                                cv_image = np_arr
                            else:
                                cv_image = np_arr
                            
                            cv2.imwrite(os.path.join(rgb_dir, rgb_filename), cv_image)
                        except Exception as e:
                            print(f"Error saving RGB image {rgb_filename}: {e}")
                            rgb_filename = None
                
                # Find closest thermal image
                thermal_filename = None
                if thermal_timestamps:
                    thermal_idx = find_closest_timestamp(current_time, thermal_timestamps)
                    if thermal_idx is not None:
                        closest_thermal_time = thermal_timestamps[thermal_idx]
                        thermal_msg = thermal_data[closest_thermal_time]
                        
                        # Extract and save thermal image
                        thermal_filename = f"seq_{sequence_id:04d}_{closest_thermal_time}.png"
                        try:
                            if thermal_msg.encoding == 'mono8':
                                np_arr = np.frombuffer(thermal_msg.data, dtype=np.uint8).reshape(thermal_msg.height, thermal_msg.width)
                            elif thermal_msg.encoding == 'mono16':
                                np_arr = np.frombuffer(thermal_msg.data, dtype=np.uint16).reshape(thermal_msg.height, thermal_msg.width)
                            else:
                                np_arr = np.frombuffer(thermal_msg.data, dtype=np.uint8).reshape(thermal_msg.height, thermal_msg.width)
                            
                            cv2.imwrite(os.path.join(thermal_dir, thermal_filename), np_arr)
                        except Exception as e:
                            print(f"Error saving thermal image {thermal_filename}: {e}")
                            thermal_filename = None
                
                # Save synchronized data row
                try:
                    row = {
                        'sequence_id': sequence_id,
                        'timestamp_ns': current_time,
                        'rgb_filename': rgb_filename or '',
                        'thermal_filename': thermal_filename or '',
                        'pos_x': odom_msg.pose.pose.position.x,
                        'pos_y': odom_msg.pose.pose.position.y,
                        'pos_z': odom_msg.pose.pose.position.z,
                        'orient_x': odom_msg.pose.pose.orientation.x,
                        'orient_y': odom_msg.pose.pose.orientation.y,
                        'orient_z': odom_msg.pose.pose.orientation.z,
                        'orient_w': odom_msg.pose.pose.orientation.w,
                        'linear_vel_x': odom_msg.twist.twist.linear.x,
                        'linear_vel_y': odom_msg.twist.twist.linear.y,
                        'linear_vel_z': odom_msg.twist.twist.linear.z,
                        'angular_vel_x': odom_msg.twist.twist.angular.x,
                        'angular_vel_y': odom_msg.twist.twist.angular.y,
                        'angular_vel_z': odom_msg.twist.twist.angular.z
                    }
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error writing odometry data at sequence {sequence_id}: {e}")
                
                current_time += interval_ns
        
        print(f"Extracted {sequence_id} synchronized triplets")

    except Exception as e:
        print(f"Error processing bags: {e}")
        return

def main():
    parser = argparse.ArgumentParser(
        description="Extract synchronized RGB, thermal, and odometry data from ROS bags"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g. OdomBeyondVision)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Base paths derived from root_dir
    root_dir = os.path.abspath(args.root_dir)

    rosbags_dir = os.path.join(root_dir, "Rosbags", "Handheld")
    gt_dir = os.path.join(rosbags_dir, "gt")
    output_base_dir = os.path.join(root_dir, "ExtractedData")

    print(f"Root dir          : {root_dir}")
    print(f"Rosbags dir       : {rosbags_dir}")
    print(f"GT dir            : {gt_dir}")
    print(f"Output base dir   : {output_base_dir}")
    print(f"Interval (sec)    : {args.interval}")

    # Sequence mapping (unchanged)
    sequence_mapping = {
        1: "2020-01-28-11-10-12",
        2: "2020-01-28-11-15-11",
        3: "2020-01-28-11-20-29",
        4: "2020-01-28-11-24-40",
        5: "2020-01-28-11-28-50",
        6: "2020-01-28-11-34-05",
        7: "2020-01-28-11-36-12",
        8: "2020-01-28-11-38-18",
        9: "2020-01-28-11-39-07",
        10: "2020-01-28-11-43-26",
        11: "2020-01-28-11-45-44",
        12: "2020-01-28-11-47-09",
        13: "2020-01-28-11-48-44",
        14: "2020-01-28-11-50-54",
        15: "2020-01-28-11-53-26",
        16: "2020-01-28-11-55-54",
        17: "2020-01-28-11-57-59",
        18: "2020-01-28-11-59-42",
        19: "2020-01-28-12-01-24",
        20: "2020-01-29-17-49-34",
        21: "2020-02-06-15-03-24",
        22: "2020-02-06-15-07-25",
        23: "2020-02-06-15-15-32",
        24: "2020-02-06-15-24-24",
        25: "2020-02-06-15-29-15",
        26: "2020-02-06-15-35-57",
        27: "2020-02-06-15-48-35",
        28: "2020-02-06-15-58-41",
        29: "2020-02-06-16-04-15",
        30: "2020-02-06-16-09-40",
        31: "2020-02-06-16-15-45",
        32: "2020-02-06-16-21-04",
        33: "2020-02-07-15-57-43",
        34: "2020-02-07-16-09-47",
        35: "2020-02-07-16-25-22",
        36: "2020-02-07-16-31-44",
        37: "2020-02-07-16-36-31",
        38: "2020-02-07-16-41-08",
        39: "2020-02-07-16-46-50",
        40: "2020-02-07-17-00-15",
        41: "2020-02-07-17-01-24",
        42: "2020-02-07-17-04-08",
        43: "2020-02-07-17-08-21",
        44: "2020-02-07-17-11-56"
    }

    os.makedirs(output_base_dir, exist_ok=True)

    gt_files = glob.glob(os.path.join(gt_dir, "gt_*.bag"))
    print(f"Found {len(gt_files)} GT bag files")

    for gt_file in gt_files:
        try:
            gt_filename = os.path.basename(gt_file)
            seq_num = int(gt_filename.replace("gt_", "").replace(".bag", ""))

            if seq_num not in sequence_mapping:
                print(f"Sequence {seq_num} not in mapping, skipping")
                continue

            sensor_bag_path = os.path.join(
                rosbags_dir, sequence_mapping[seq_num] + ".bag"
            )

            if not os.path.exists(sensor_bag_path):
                print(f"Missing sensor bag: {sensor_bag_path}")
                continue

            output_dir = os.path.join(
                output_base_dir,
                f"sequence_{seq_num:02d}_{sequence_mapping[seq_num]}"
            )

            extract_synchronized_data(
                sensor_bag_path,
                gt_file,
                output_dir,
                interval_seconds=args.interval
            )

        except Exception as e:
            print(f"Error processing {gt_file}: {e}")

    print("\nAll sequences processed!")


if __name__ == "__main__":
    main()