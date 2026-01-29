"""
This script processes CSV log files and generates ground truth labels based on a provided JSON file

It is still faulty and needs to be fixed.
"""



import os
import json
import pandas as pd
import glob
from tqdm import tqdm

# Configuration
GROUND_TRUTH_PATH = "/tmp2/b11902050/Logs-Labeling/data/groundtruth/ability2events.json"
INPUT_LOGS_DIR = "/tmp2/b11902050/Logs-Labeling/data/input_logs"
OUTPUT_DIR = "/tmp2/b11902050/Logs-Labeling/data/input_logs_labeled"

# Operation Mapping (JSON Action -> CSV Operation)
OP_MAP = {
    "Process Create": ["Process Start"],
    "TCP Connect": ["TCP Connect"],
    "File Create": ["File Create"],
    "File Write": ["File Write", "File Modify"],
    "Image Load": ["Image Load"],
    "Process Terminate": ["Process Terminate", "Process End"],
    "RegSetValue": ["RegSetValue"],
    "QueryDirectory": ["QueryDirectory", "File Enumeration"] # Need to check CSV for exact string
}

def load_ground_truth():
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    with open(GROUND_TRUTH_PATH, 'r') as f:
        data = json.load(f)
    print(f"Loaded ground truth for {len(data)} datasets.")
    return data

def normalize_path(path):
    if not isinstance(path, str): return ""
    return path.lower().replace("\\", "/")

def match_row(row, event_infos):
    """
    Check if a CSV row matches any of the malicious events.
    event_infos: list of malicious events for this dataset.
    Each event is [src_uuid, src_obj, tgt_uuid, tgt_obj, action, timestamp]
    """
    row_pid = row.get("PID")
    row_ppid = row.get("Parent PID")
    row_op = row.get("Operation")
    row_cmd = str(row.get("Command Line", "")).lower()
    row_path = str(row.get("Path", "")).lower()
    row_img = str(row.get("Image Path", "")).lower()
    
    # Pre-process row PID/PPID to int if possible
    try: row_pid = int(row_pid)
    except: pass
    try: row_ppid = int(row_ppid)
    except: pass

    for event in event_infos:
        # Unpack event
        # Format: [src_id, src_obj, tgt_id, tgt_obj, action, timestamp]
        src_obj = event[1]
        tgt_obj = event[3]
        action = event[4]
        
        # 1. Match Action
        csv_ops = OP_MAP.get(action, [action]) # Default to exact match if not in map
        if row_op not in csv_ops:
            # Fallback: fuzzy match or skip?
            # Let's be strict first.
            # Some CSV ops might be "Process Create" actually? 
            # We observed "Process Start" in CSV.
            continue
            
        # 2. Match Attributes based on Action Type
        is_match = False
        
        # Helper for command line matching
        def check_cmd(obj, row_c):
            # If object has Cmdline, it MUST match the row's command line
            j_cmd = str(obj.get("Cmdline", "")).lower()
            if j_cmd:
                # Basic normalization: remove extra whitespace
                j_cmd = " ".join(j_cmd.split())
                r_cmd = " ".join(row_c.split())
                # Check for substring match in either direction (to handle truncation or prefixing)
                if j_cmd not in r_cmd and r_cmd not in j_cmd:
                    return False
            
            # If object describes Image/Name, it usually must match too (unless we matched specific cmdline)
            # But relying on Image name alone for things like "cmd.exe" is dangerous.
            # If we matched Cmdline, that's strong. If no Cmdline in JSON, check Image.
            # If Cmdline in JSON but failed match above -> we returned False.
            
            if not j_cmd:
                # Only check Image if no Cmdline available or if we want to be redundant?
                # Let's enforce Image check always if present.
                j_img = str(obj.get("Image", "")).lower()
                # If Image path is not full, matching might be loose.
                if j_img and normalize_path(j_img) not in normalize_path(row_img):
                     # Try Name?
                     j_name = str(obj.get("Name", "")).lower()
                     if j_name and j_name not in normalize_path(row_img):
                         return False
            return True

        if action == "Process Create":
            # JSON: Source=Parent, Target=Child
            json_child_pid = tgt_obj.get("Pid")
            json_parent_pid = src_obj.get("Pid")
            
            # Check PIDs
            if json_child_pid and row_pid != json_child_pid: continue
            if json_parent_pid and row_ppid != json_parent_pid: continue
            
            # Check Target Command Line (The process being created)
            if not check_cmd(tgt_obj, row_cmd):
                continue
            
            is_match = True

        elif action == "TCP Connect":
            # JSON: Source=Process, Target=Network
            json_pid = src_obj.get("Pid")
            if json_pid and row_pid != json_pid: continue
            
            # Check Source Command Line (The process making connection)
            if not check_cmd(src_obj, row_cmd):
                continue
            
            # Check Destination
            # CSV Path: "SourceIP:Port -> DestIP:Port"
            dst_addr = tgt_obj.get("Dstaddress")
            dst_port = tgt_obj.get("Port")
            
            if dst_addr and dst_addr.lower() not in row_path: continue
            if dst_port and str(dst_port) not in row_path: continue
            
            is_match = True
            
        else:
            # Generic matching for other events (File, Reg, etc.)
            # Usually Source is the Process
            json_pid = src_obj.get("Pid")
            if json_pid and row_pid != json_pid: continue
            
            # Check Source Command Line
            if not check_cmd(src_obj, row_cmd):
                continue
            
            # Try to match target object name in Path
            tgt_name = tgt_obj.get("Name", "")
            if tgt_name and normalize_path(tgt_name) not in normalize_path(row_path):
                 continue
                 
            is_match = True
            
        if is_match:
            return 1
            
    return 0

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
            # print(f"Skipping {dataset_id} (No ground truth)")
            continue
            
        events = ground_truth[dataset_id]
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
            
        # Add Label Column
        # Doing row-by-row for now (slow but precise logic implementation)
        # Optimization: Filter by Action first?
        # For validation, let's just do apply.
        
        tqdm.pandas(desc=f"Labeling {dataset_id[:8]}", leave=False)
        df["Label"] = df.apply(lambda row: match_row(row, events), axis=1)
        
        # Stats
        malicious_count = df["Label"].sum()
        # print(f"  {dataset_id}: {malicious_count} malicious events found.")
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, basename)
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    process_datasets()
