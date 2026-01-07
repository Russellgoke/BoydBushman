import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
import numpy as np
import mplcursors  # Used for hover-based information
import re

# --- Manual Configuration ---
PLOT_X = False
PLOT_Y = True
PLOT_ACCEL = False

def get_csv_files(inputs):
    csv_files = []
    for item in inputs:
        if os.path.isfile(item) and item.endswith('.csv'):
            csv_files.append(item)
        elif os.path.isdir(item):
            folder_files = glob.glob(os.path.join(item, "*.csv"))
            csv_files.extend(folder_files)
    return sorted(list(set(csv_files)))

def parse_ignore_ids(file_path):
    """Read first line and extract IDs to ignore if it's a comment."""
    ignore_ids = set()
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            # Remove the # and parse comma/space separated IDs
            content = first_line[1:].strip()
            # Split by comma, semicolon, or whitespace
            parts = re.split(r'[,;\s]+', content)
            for part in parts:
                part = part.strip()
                if part:
                    try:
                        ignore_ids.add(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid ID format in {file_path}: {part}")
    return ignore_ids

def plot_trajectories(file_list):
    if not file_list:
        print("No CSV files found.")
        return

    data_to_plot = {'x': [], 'y': [], 'accel': []}

    for file in file_list:
        try:
            # Check for ignore list in first line comment
            ignore_ids = parse_ignore_ids(file)
            if ignore_ids:
                print(f"Ignoring trajectory IDs {ignore_ids} from {os.path.basename(file)}")
            
            df = pd.read_csv(file, comment='#')
            fname = os.path.basename(file)
            for traj_id, group in df.groupby('trajectory_id'):
                if traj_id in ignore_ids:
                    continue  # Skip ignored trajectories
                frame_nums = group['frame_num'].values
                start_frame = frame_nums[0]
                time = frame_nums - start_frame  # Relative time from first frame
                x_pos = group['x'].values
                y_pos = group['y'].values
                
                label = f"File: {fname}\nID: {traj_id}\nStart Frame: {start_frame}"
                
                if PLOT_X:
                    data_to_plot['x'].append((time, x_pos, label))
                if PLOT_Y:
                    data_to_plot['y'].append((time, y_pos, label))
                if PLOT_ACCEL:
                    accel_y = np.diff(y_pos, n=2)
                    time_accel = time[2:]
                    data_to_plot['accel'].append((time_accel, accel_y, label))
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Helper function to set up interactive hovering
    def setup_interaction(title, data_list, ylabel, invert_y=False):
        fig, ax = plt.subplots(num=title, figsize=(10, 6))
        for t, val, lbl in data_list:
            ax.plot(t, val, alpha=0.4, label=lbl) # Plot lines with some transparency
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frames Since Start")
        ax.grid(True, alpha=0.3)
        if invert_y:
            ax.invert_yaxis()
        
        # Use mplcursors to show label on hover
        cursor = mplcursors.cursor(ax.get_lines(), hover=True)
        
        @cursor.connect("add")
        def _(sel):
            # Show the label we defined earlier
            sel.annotation.set_text(sel.artist.get_label())
            # Briefly highlight the line
            sel.artist.set_alpha(1.0)
            sel.artist.set_linewidth(2.5)

        @cursor.connect("remove")
        def _(sel):
            sel.artist.set_alpha(0.4)
            sel.artist.set_linewidth(1.5)

        plt.tight_layout()

    # Trigger independent windows
    if PLOT_X and data_to_plot['x']:
        setup_interaction("X Position", data_to_plot['x'], "X Position (px)")

    if PLOT_Y and data_to_plot['y']:
        setup_interaction("Y Position", data_to_plot['y'], "Y Position (px)", invert_y=True)

    if PLOT_ACCEL and data_to_plot['accel']:
        setup_interaction("Y Acceleration", data_to_plot['accel'], "Acceleration (px/fr²)")

    if any([PLOT_X, PLOT_Y, PLOT_ACCEL]):
        print("Hover over lines to see file, ID, and start frame.")
        plt.show()
    else:
        print("No plots selected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()
    input_paths = args.paths if args.paths else ["."]
    files = get_csv_files(input_paths)
    plot_trajectories(files)