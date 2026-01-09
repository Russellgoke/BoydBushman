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
PLOT_Y = False
PLOT_ACCEL = False
PLOT_ACCEL_HIST = True  # Histogram of accelerations from parabola fits
PRINT_ACCEL_LIST = False  # Print sorted list of accelerations with ID and start frame

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
    """Read comment lines and extract IDs to ignore from #OUTLIERS: lines."""
    ignore_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                break  # Stop at first non-comment line (header)
            if line.upper().startswith('#OUTLIERS:'):
                # Extract content after #OUTLIERS:
                content = line[len('#OUTLIERS:'):].strip()
                if not content:
                    continue
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

def collect_trajectory_data(file_list):
    """Load CSV files and extract trajectory data for plotting."""
    data = {'x': [], 'y': [], 'accel': []}
    parabola_accels = []  # List of (accel, file, traj_id, start_frame) tuples

    for file in file_list:
        try:
            ignore_ids = parse_ignore_ids(file)
            if ignore_ids:
                print(f"Ignoring trajectory IDs {ignore_ids} from {os.path.basename(file)}")
            
            df = pd.read_csv(file, comment='#')
            fname = os.path.basename(file)
            
            for traj_id, group in df.groupby('trajectory_id'):
                if traj_id in ignore_ids:
                    continue
                
                frame_nums = group['frame_num'].values
                start_frame = frame_nums[0]
                time = frame_nums - start_frame
                x_pos = group['x'].values
                y_pos = group['y'].values
                hover_label = f"File: {fname}\nID: {traj_id}\nStart Frame: {start_frame}"
                
                if PLOT_X:
                    data['x'].append((time, x_pos, fname, hover_label))
                if PLOT_Y:
                    data['y'].append((time, y_pos, fname, hover_label))
                if PLOT_ACCEL:
                    if len(time) <= 3:
                        print(f"Not enough data to calculate acceleration for {fname} ID {traj_id}")
                        continue
                    dt = np.diff(time)
                    velocity = np.diff(y_pos) / dt
                    time_vel = time[:-1] + dt / 2
                    dt_vel = np.diff(time_vel)
                    accel_y = np.diff(velocity) / dt_vel
                    
                    time_accel = time_vel[:-1] + dt_vel / 2
                    data['accel'].append((time_accel, accel_y, fname, hover_label))
                
                if (PLOT_ACCEL_HIST or PRINT_ACCEL_LIST) and len(time) >= 3:
                    coeffs = np.polyfit(time, y_pos, 2)
                    accel_val = 2 * coeffs[0]
                    parabola_accels.append((accel_val, fname, traj_id, start_frame))
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return data, parabola_accels

def setup_interaction(title, data_list, ylabel, invert_y=False):
    """Create an interactive plot window with hover labels, colored by file."""
    fig, ax = plt.subplots(num=title, figsize=(10, 6))
    
    # Get unique files and assign colors
    unique_files = list(dict.fromkeys(item[2] for item in data_list))  # Preserve order
    colors = plt.cm.tab10.colors[:len(unique_files)]
    file_to_color = {f: colors[i % len(colors)] for i, f in enumerate(unique_files)}
    
    # Track which files we've added to legend
    legend_added = set()
    
    for t, val, fname, hover_label in data_list:
        color = file_to_color[fname]
        # Only add legend label for first trace of each file
        if fname not in legend_added:
            ax.plot(t, val, alpha=0.4, color=color, label=fname)
            legend_added.add(fname)
        else:
            ax.plot(t, val, alpha=0.4, color=color)
        # Store hover label on the line for mplcursors
        ax.get_lines()[-1].set_gid(hover_label)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Frames Since Start")
    ax.grid(True, alpha=0.3)
    if invert_y:
        ax.invert_yaxis()
    
    # Add legend for files
    if len(unique_files) > 1:
        ax.legend(loc='best', fontsize=8)
    
    cursor = mplcursors.cursor(ax.get_lines(), hover=True)
    
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(sel.artist.get_gid())
        sel.artist.set_alpha(1.0)
        sel.artist.set_linewidth(2.5)

    @cursor.connect("remove")
    def _(sel):
        sel.artist.set_alpha(0.4)
        sel.artist.set_linewidth(1.5)

    plt.tight_layout()

def plot_trajectories(file_list):
    if not file_list:
        print("No CSV files found.")
        return

    data, parabola_accels = collect_trajectory_data(file_list)

    if PLOT_X and data['x']:
        setup_interaction("X Position", data['x'], "X Position (px)")
    if PLOT_Y and data['y']:
        setup_interaction("Y Position", data['y'], "Y Position (px)", invert_y=True)
    if PLOT_ACCEL and data['accel']:
        setup_interaction("Y Acceleration", data['accel'], "Acceleration (px/fr²)")
    
    if PLOT_ACCEL_HIST and parabola_accels:
        fig, ax = plt.subplots(num="Acceleration Histogram (Parabola Fit)", figsize=(10, 6))
        
        # Group accelerations by file
        file_data = {}
        for accel_val, fname, traj_id, start_frame in parabola_accels:
            if fname not in file_data:
                file_data[fname] = []
            file_data[fname].append(accel_val)
        
        # Get unique files and assign colors (matching the line plots)
        unique_files = sorted(file_data.keys())
        colors = plt.cm.tab10.colors[:len(unique_files)]
        file_to_color = {f: colors[i % len(colors)] for i, f in enumerate(unique_files)}
        
        # Determine bins from all data
        all_accel_values = [item[0] for item in parabola_accels]
        bins = np.histogram_bin_edges(all_accel_values, bins='auto')
        
        # Plot overlapping histograms for each file
        for fname in unique_files:
            accel_vals = file_data[fname]
            ax.hist(accel_vals, bins=bins, edgecolor='black', 
                   alpha=0.6, color=file_to_color[fname], label=fname)
        
        ax.set_xlabel("Acceleration (px/fr²)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        
        # Calculate and plot mean for each file
        for fname in unique_files:
            accel_vals = file_data[fname]
            mean_accel = np.mean(accel_vals)
            std_accel = np.std(accel_vals)
            color = file_to_color[fname]
            ax.axvline(mean_accel, color=color, linestyle='--', linewidth=2, 
                      alpha=0.8, label=f'{fname} Mean: {mean_accel:.4f}')
        
        # Calculate overall statistics for title
        overall_mean = np.mean(all_accel_values)
        overall_std = np.std(all_accel_values)
        
        # Add legend
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f"Acceleration Histogram (n={len(all_accel_values)}, Overall μ={overall_mean:.4f}, σ={overall_std:.4f})")
        plt.tight_layout()
    
    if PRINT_ACCEL_LIST and parabola_accels:
        sorted_accels = sorted(parabola_accels, key=lambda x: x[0])
        print("\n--- Accelerations (sorted by value) ---")
        print(f"{'Accel':>12}  {'File':<30}  {'ID':>4}  {'Start Frame':>11}")
        print("-" * 65)
        for accel_val, fname, traj_id, start_frame in sorted_accels:
            print(f"{accel_val:>12.4f}  {fname:<30}  {traj_id:>4}  {start_frame:>11}")


    if any([PLOT_X, PLOT_Y, PLOT_ACCEL, PLOT_ACCEL_HIST]):
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