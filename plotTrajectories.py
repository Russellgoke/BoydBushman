import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob

def get_csv_files(inputs):
    """Collects all CSV files from a list of files and folders."""
    csv_files = []
    for item in inputs:
        if os.path.isfile(item) and item.endswith('.csv'):
            csv_files.append(item)
        elif os.path.isdir(item):
            # Get only first-level CSV files in the folder
            folder_files = glob.glob(os.path.join(item, "*.csv"))
            csv_files.extend(folder_files)
    return sorted(list(set(csv_files))) # Remove duplicates and sort

def plot_trajectories(file_list):
    if not file_list:
        print("No CSV files found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    total_trajectories = 0

    for file in file_list:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file)
            
            # Group by trajectory_id to plot individual drops
            for traj_id, group in df.groupby('trajectory_id'):
                # 'point_index' represents time steps (frames relative to start)
                time = group['point_index']
                
                # Plot X vs Time
                ax1.plot(time, group['x'], alpha=0.6, label=f"{filename} ID:{traj_id}")
                
                # Plot Y vs Time
                ax2.plot(time, group['y'], alpha=0.6)
                
                total_trajectories += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Formatting X plot
    ax1.set_ylabel('X Position (px)')
    ax1.set_title('Horizontal Drift (X) vs Time')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Formatting Y plot
    ax2.set_ylabel('Y Position (px)')
    ax2.set_xlabel('Relative Frame Index (Time)')
    ax2.set_title('Vertical Fall (Y) vs Time')
    ax2.invert_yaxis()  # Invert so falling down looks like it's going down
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    print(f"Plotted {total_trajectories} trajectories from {len(file_list)} files.")
    plt.show()

if __name__ == "__main__":
    # You can pass paths via CLI or hardcode them in the list below
    # Example: python plot_drops.py my_data_folder/ run1.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='List of files or folders')
    args = parser.parse_args()

    # If no CLI args provided, it looks in the current directory
    input_paths = args.paths if args.paths else ["."]
    
    files_to_plot = get_csv_files(input_paths)
    plot_trajectories(files_to_plot)