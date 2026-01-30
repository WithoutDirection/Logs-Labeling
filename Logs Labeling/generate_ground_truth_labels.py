"""
This script processes CSV log files and generates ground truth labels based on a provided JSON file.
Optimized for performance and accuracy by matching events to logs (instead of logs to events)
and using hierarchical strict filtering.
Updated to strict 1-to-1 matching respecting chronological order.
"""

import os
import json
import pandas as pd
import glob
from tqdm import tqdm
import warnings

# Configuration
GROUND_TRUTH_PATH = "/tmp2/b11902050/Logs-Labeling/data/groundtruth/ability2events.json"
INPUT_LOGS_DIR = "/tmp2/b11902050/Logs-Labeling/data/input_logs"
OUTPUT_DIR = "/tmp2/b11902050/Logs-Labeling/data/input_logs_labeled"

# Operation Mapping (JSON Action -> CSV Operation)
OP_MAP = {
    "Process Create": ["Process Start", "Process Create"],
    "TCP Connect": ["TCP Connect"],
    "TCP Send": ["TCP Send"],
    "TCP Receive": ["TCP Receive"],
    "TCP Disconnect": ["TCP Disconnect"],
    "CreateFile": ["CreateFile"],
    "WriteFile": ["WriteFile"],
    "CloseFile": ["CloseFile"],
    "ReadFile": ["ReadFile"],
    "RegSetValue": ["RegSetValue"],
    "RegCreateKey": ["RegCreateKey"],
    "RegDeleteValue": ["RegDeleteValue"],
    "RegOpenKey": ["RegOpenKey"],
    "RegQueryKey": ["RegQueryKey"],
    "RegQueryValue": ["RegQueryValue"],
    "RegCloseKey": ["RegCloseKey"],
    "QueryDirectory": ["QueryDirectory"],
    "QueryAllInformationFile": ["QueryAllInformationFile"],
    "QueryAttributeTagFile": ["QueryAttributeTagFile"],
    "QueryBasicInformationFile": ["QueryBasicInformationFile"],
    "QueryNetworkOpenInformationFile": ["QueryNetworkOpenInformationFile"],
    "SetBasicInformationFile": ["SetBasicInformationFile"],
    "SetDispositionInformationEx": ["SetDispositionInformationEx"],
    "SetDispositionInformationFile": ["SetDispositionInformationFile"],
    "QueryStandardInformationFile": ["QueryStandardInformationFile"], 
    "UDP Send": ["UDP Send"],
    "UDP Receive": ["UDP Receive"]
}


def load_ground_truth():
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    with open(GROUND_TRUTH_PATH, 'r') as f:
        data = json.load(f)
    print(f"Loaded ground truth for {len(data)} datasets.")
    return data

def normalize_path(path):
    if not isinstance(path, str): return ""
    return path.lower().replace("\\", "/").strip().rstrip('/')

def normalize_cmd(cmd):
    if not isinstance(cmd, str): return ""
    return " ".join(cmd.split()).lower().strip()

def process_datasets():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ground_truth = load_ground_truth()
    
    csv_files = glob.glob(os.path.join(INPUT_LOGS_DIR, "*_raw_events.csv"))
    print(f"Found {len(csv_files)} CSV files in {INPUT_LOGS_DIR}")
    
    for csv_path in tqdm(csv_files, desc="Labeling Datasets"):
        basename = os.path.basename(csv_path)
        dataset_id = basename.replace("_raw_events.csv", "")
        
        if dataset_id not in ground_truth:
            continue
            
        events = ground_truth[dataset_id]
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
            
        # Parse Dates and Sort
        try:
            # Format: '4/14/2022 1:21:12 PM' -> '%m/%d/%Y %I:%M:%S %p'
            df['parsed_time'] = pd.to_datetime(df['Date & Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            # If format fails (mixed formats?), fallback with warning suppression or handle
            if df['parsed_time'].isnull().all():
                 # Fallback if specific format fails totally
                 df['parsed_time'] = pd.to_datetime(df['Date & Time'], errors='coerce')
            
            df = df.sort_values('parsed_time').reset_index(drop=True)
        except Exception as e:
            # print(f"Warning: Could not parse dates for {dataset_id}: {e}")
            pass

        # Initialize Label
        df["Label"] = "benign"
        
        # Create helper columns for faster vectorized filtering
        df['norm_op'] = df['Operation'].fillna("")
        df['norm_pid'] = pd.to_numeric(df['PID'], errors='coerce')
        df['norm_parent_pid'] = pd.to_numeric(df['Parent PID'], errors='coerce')
        
        # Keep track of last matched index for sequential search
        last_idx = 0
        malicious_indices = set()
        
        for event in events:
            # Event structure: [src_id, src_obj, tgt_id, tgt_obj, action, timestamp]
            if len(event) < 5: continue
            
            src_obj = event[1]
            tgt_obj = event[3]
            action = event[4]
            
            # --- Search Space (Sequential) ---
            # Only search after the previous match
            subset = df.iloc[last_idx:]
            if subset.empty:
                print(f"Error: dataset {dataset_id[:8]} - End of log reached before event {action}.")
                break # Stop processing events for this dataset if we ran out of logs
            
            # --- Filter 1: Action (Operation) ---
            possible_ops = OP_MAP.get(action, [action])
            candidates = subset[subset['norm_op'].isin(possible_ops)]
            
            if candidates.empty:
                # If Strict Sequential fails (maybe out of order?), you could try searching entire DF
                # But prioritizing false positive reduction: skip this event.
                # print(f"Warning: dataset {dataset_id[:8]} - Event {action} matched 0 records (Action).")
                continue
                
            # --- Filter 2: PIDs (Context Sensitive) ---
            if action == "Process Create":
                if "Pid" in src_obj and src_obj["Pid"]:
                    matches = candidates[candidates['norm_parent_pid'] == src_obj["Pid"]]
                    if not matches.empty: candidates = matches
                if "Pid" in tgt_obj and tgt_obj["Pid"]:
                    matches = candidates[candidates['norm_pid'] == tgt_obj["Pid"]]
                    if not matches.empty: candidates = matches
            else:
                if "Pid" in src_obj and src_obj["Pid"]:
                    matches = candidates[candidates['norm_pid'] == src_obj["Pid"]]
                    if not matches.empty: candidates = matches
            
            # --- Filter 3: Command Line ---
            if not candidates.empty:
                target_cmd = None
                if action == "Process Create":
                    if "Cmdline" in tgt_obj and tgt_obj["Cmdline"]:
                        target_cmd = normalize_cmd(tgt_obj["Cmdline"])
                else:
                    if "Cmdline" in src_obj and src_obj["Cmdline"]:
                        target_cmd = normalize_cmd(src_obj["Cmdline"])
                
                if target_cmd:
                    mask = candidates['Command Line'].apply(
                        lambda x: target_cmd in normalize_cmd(str(x)) or normalize_cmd(str(x)) in target_cmd
                    )
                    matches = candidates[mask]
                    if not matches.empty: candidates = matches

            # --- Filter 4: Target Specific Attributes (Path/Name) ---
            if not candidates.empty:
                if action == "TCP Connect":
                    dst_addr = tgt_obj.get("Dstaddress")
                    dst_port = tgt_obj.get("Port")
                    
                    if dst_addr:
                        matches = candidates[candidates['Path'].astype(str).str.contains(dst_addr, case=False, na=False)]
                        if not matches.empty: candidates = matches
                    if dst_port:
                        matches = candidates[candidates['Path'].astype(str).str.contains(str(dst_port), na=False)]
                        if not matches.empty: candidates = matches
                
                elif action != "Process Create":
                    # For File/Reg/etc, check Path
                    if "Name" in tgt_obj and tgt_obj["Name"]:
                        tgt_name = normalize_path(tgt_obj["Name"])
                        if len(tgt_name) > 3: 
                            # Try Exact match first for path components (stricter)
                            mask_exact = candidates['Path'].apply(
                                lambda x: normalize_path(str(x)) == tgt_name
                            )
                            matches_exact = candidates[mask_exact]
                            
                            if not matches_exact.empty:
                                candidates = matches_exact
                            else:
                                # Fallback to substring only if exact match fails
                                mask = candidates['Path'].apply(
                                    lambda x: tgt_name in normalize_path(str(x))
                                )
                                matches = candidates[mask]
                                if not matches.empty: candidates = matches
            
            if candidates.empty:
                # print(f"Warning: dataset {dataset_id[:8]} - Event {action} matched 0 records (Final).")
                continue
            
            # --- Selection ---
            # Prioritize:
            # 1. Exact "identical" unlabeled log (Already filtered as best as we can)
            # 2. Pick the FIRST candidate (Chronological order) to map 1 match per GT event
            
            best_match = candidates.iloc[0]
            best_idx = best_match.name
            
            # If multiple candidates, check if they are identical
            # (Just for warning/debugging)
            # check_cols = [c for c in candidates.columns if c not in ['Label', 'norm_op', 'norm_pid', 'norm_parent_pid', 'parsed_time']]
            # unique_rows = candidates.drop_duplicates(subset=check_cols)
            # if len(unique_rows) > 1:
            #     # If we are picking index 0 out of non-identical rows, we are making a choice based on time order.
            #     pass
            
            # Mark ONLY the best match
            malicious_indices.add(best_idx)
            
            # Update last_idx to start AFTER this match
            last_idx = best_idx + 1

        # Apply labels
        if malicious_indices:
            df.loc[list(malicious_indices), 'Label'] = "malicious"

        # Drop temporary columns
        df.drop(columns=['norm_op', 'norm_pid', 'norm_parent_pid', 'parsed_time'], errors='ignore', inplace=True)
         
        # Save
        out_path = os.path.join(OUTPUT_DIR, basename)
        df.to_csv(out_path, index=False)
        # print(f"Processed {basename}, marked {len(malicious_indices)} events.")

if __name__ == "__main__":
    process_datasets()
