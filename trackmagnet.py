import cv2
import csv
import numpy as np
import os
from datetime import datetime  # Added for timestamping

#TODO sometimes splits into two objects idk why, if one is still wide enough fail
START_FRAME = 44040

# --- Configuration ---
# VIDEO_PATH = r'Videos\MVI_1635.mp4' # 4800, attracting
# LANE_ROI = (659, 0, 130, 749)
# CROP_VIEW = None #(659, 0, 130, 749)  # Set to None to select interactively, or (x, y, w, h) tuple for default
VIDEO_PATH = r'Videos\MVI_1636.MP4' # repelling
LANE_ROI = (609, 0, 143, 749)
CROP_VIEW = None # (639, 0, 130, 749)  # Set to None to select interactively, or (x, y, w, h) tuple for default
MIN_FRAMES_TO_VALIDATE = 20  # Persistence threshold
TOP_ZONE_PERCENT = 0.1      # Must start in top 10% of ROI
MAX_X_DRIFT = 20            # Max horizontal pixel shift between frames
MAX_VELOCITY = 60             # Max distance to look for next match, accounting for missed frames
MAX_MISSED_FRAMES = 3       # Grace period for flickering
MIN_WIDTH = 53   # Added minimum width
MIN_HEIGHT = 30  # Added minimum height

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f'trajectories_{TIMESTAMP}.csv'

BACKGROUND_WINDOW_WIDTH = 5

# --- Colors (BGR) ---
COLOR_TOO_SKINNY = (255, 0, 0)      # Blue
COLOR_TOO_SHORT = (147, 20, 255)     # Pink
COLOR_NEW_CANDIDATE = (144, 238, 144) # Light Green
COLOR_DELETED = (0, 0, 255)        # Red
COLOR_SUCCESS = (0, 255, 0)        # Solid Green
COLOR_TRACKING = (0, 255, 255)     # Yellow
COLOR_ROI = (255, 0, 0)            # Blue for ROI box

class FallingCandidate:
    def __init__(self, initial_pos, initial_frame):
        self.positions = [(initial_frame, initial_pos[0], initial_pos[1])] # List of (frame_num, x, y)
        self.missed_frames = 0
        self.is_valid = False
        self.is_dead = False

    def update(self, new_pos, frame_num):
        _, prev_x, prev_y = self.positions[-1]
        new_x, new_y = new_pos
        
        # Kinematic Check: Must move DOWN (new_y > prev_y) and within X drift limits
        if new_y > prev_y and abs(new_x - prev_x) < MAX_X_DRIFT:
            self.positions.append((frame_num, new_x, new_y))
            self.missed_frames = 0
            if len(self.positions) >= MIN_FRAMES_TO_VALIDATE:
                self.is_valid = True
            return True
        return False

    def increment_missed(self):
        self.missed_frames += 1
        if self.missed_frames > MAX_MISSED_FRAMES:
            self.is_dead = True

def initialize_roi(cap, default_roi):
    """Handles ROI selection or uses the hardcoded default."""
    ret, first_frame = cap.read()
    if not ret: return None
    if default_roi is None:
        print("Select the vertical lane, then press SPACE.")
        roi = cv2.selectROI("Select Drop Lane", first_frame, fromCenter=False)
        cv2.destroyWindow("Select Drop Lane")
        print(f"Selected ROI: ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")
        return roi
    return default_roi

def initialize_crop_view(cap, default_crop):
    """Handles crop view selection or uses the hardcoded default."""
    ret, first_frame = cap.read()
    if not ret: return None
    if default_crop is None:
        print("Select the crop view region to display, then press SPACE.")
        crop = cv2.selectROI("Select Crop View", first_frame, fromCenter=False)
        cv2.destroyWindow("Select Crop View")
        print(f"Selected crop view: ({crop[0]}, {crop[1]}, {crop[2]}, {crop[3]})")
        return crop
    return default_crop

def process_frame(roi_frame, backSub):
    """Cleans up noise using morphological operations."""
    fg_mask = backSub.apply(roi_frame)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def filter_trailing_skip(positions):
    """Remove trailing single frame if it skipped frames before it.
    
    If the last frame skipped (frame_num gap > 1 from second-to-last) and
    there's only one frame after the skip, remove it as it's likely noise.
    """
    if len(positions) < 2:
        return positions
    
    last_frame = positions[-1][0]
    second_last_frame = positions[-2][0]
    
    if last_frame - second_last_frame > 1:
        # Last frame skipped - remove it
        return positions[:-1]
    
    return positions

def save_to_csv(measurements):
    """Saves all trajectories to a CSV file with frame number per point."""
    if not measurements:
        print("No valid trajectories to save.")
        return

    # Filter out trailing single-frame skips from each trajectory
    filtered_measurements = [filter_trailing_skip(pos) for pos in measurements]
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        # Write metadata comment
        f.write(f"#META: VIDEO={VIDEO_PATH}, ROI={LANE_ROI}, TIMESTAMP={TIMESTAMP}\n")
        # Placeholder for outliers (user can edit this line)
        f.write("#OUTLIERS:\n")
        writer = csv.writer(f)
        writer.writerow(['trajectory_id', 'point_index', 'frame_num', 'x', 'y'])
        for t_id, positions in enumerate(filtered_measurements):
            for p_idx, (frame_num, px, py) in enumerate(positions):
                writer.writerow([t_id, p_idx, frame_num, px, py])
    
    print(f"Successfully saved {len(filtered_measurements)} trajectories to {OUTPUT_CSV}")

def main():
    # Check if video file exists, try alternate case if needed
    video_path = VIDEO_PATH
    if not os.path.exists(video_path):
        # Try uppercase extension (Windows is case-insensitive but OpenCV might care)
        alt_path = video_path.replace('.mp4', '.MP4')
        if os.path.exists(alt_path):
            print(f"Found video with uppercase extension: {alt_path}")
            video_path = alt_path
        else:
            print(f"Error: Video file not found at {VIDEO_PATH}")
            print(f"Also tried: {alt_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Please check the video path and try again.")
            return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    # Verify video has frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: Video file appears to be empty or invalid")
        cap.release()
        return
    
    print(f"Video loaded: {total_frames} frames")
    if START_FRAME >= total_frames:
        print(f"Warning: START_FRAME ({START_FRAME}) is beyond video length ({total_frames})")
    
    lane_roi = initialize_roi(cap, LANE_ROI)
    if lane_roi is None:
        print("Error: Could not initialize ROI")
        cap.release()
        return

    crop_view = initialize_crop_view(cap, CROP_VIEW)
    # Check if crop view was cancelled or invalid (width/height = 0)
    if crop_view is None or (crop_view[2] == 0 or crop_view[3] == 0):
        # If no crop view selected, use full frame (set to None to disable cropping)
        crop_view = None

    backSub = cv2.createBackgroundSubtractorMOG2(history=BACKGROUND_WINDOW_WIDTH, varThreshold=40, detectShadows=False)
    
    paused = True
    x, y, w, h = [int(v) for v in lane_roi]
    
    candidates = []
    final_measurements = []
    print("Controls: 'space': Pause/Play, 'a': Step Backwards 20, 's': Step Backward, 'd': Step Forward, 'f': Step Forward 20 frames, 'r': Reset, 'q': Quit")
    print(f"Starting at frame {START_FRAME}. Press SPACE to start playback.")

    # Create resizable window
    cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)
    
    # Set initial window size to match display frame dimensions (maintains aspect ratio)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop_view is not None:
        display_width = crop_view[2]
        display_height = crop_view[3]
    else:
        display_width = frame_width
        display_height = frame_height
    cv2.resizeWindow('Tracker', display_width, display_height)

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 1. Pre-process
        roi_frame = frame[y:y+h, x:x+w]
        fg_mask = process_frame(roi_frame, backSub)

        # 2. Detect and Filter by Size
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_detections = []
        for cnt in contours:
            mx, my, mw, mh = cv2.boundingRect(cnt)
            abs_x, abs_y = mx + x, my + y   
            if mw < MIN_WIDTH:
                # Visualization: Blue for too skinny
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), COLOR_TOO_SKINNY, 1)
                continue
            if mh < MIN_HEIGHT:
                # Visualization: Pink for too short
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), COLOR_TOO_SHORT, 1)
                continue
            # Skip if bounding box touches top or bottom of ROI (object clipping through edge)
            if my <= 0 or my + mh >= h:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # Calculate Centroid (CX, CY)
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                
                # current_detections now uses the centroid, not the box center
                current_detections.append((cx, cy))
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), COLOR_TRACKING, 1)
            cv2.circle(frame, (cx, cy), 3, COLOR_TRACKING, -1)            # Calculate absolute coordinates on the main frame

        matched_detections_indices = set()
        
        # Update candidates if they continue or are gone
        for cand in candidates:
            best_dist = float('inf')
            best_match_idx = None

            # Find nearest detection within MAX_VELOCITY radius
            _, last_x, last_y = cand.positions[-1]
            for i, det in enumerate(current_detections):
                dist = np.linalg.norm(np.array(det) - np.array((last_x, last_y)))
                if dist < MAX_VELOCITY and dist < best_dist: # Distance gate
                    best_dist = dist
                    best_match_idx = i # TODO two objects merge into one
            
            if best_match_idx is not None:
                # Try to update. If update fails (e.g. moved up), treat as missed frame.
                if cand.update(current_detections[best_match_idx], current_frame_num):
                    matched_detections_indices.add(best_match_idx)
                else:
                        cand.increment_missed()
            else:
                cand.increment_missed()
                
        # Check for new candidates
        for i, det in enumerate(current_detections):
            if i not in matched_detections_indices:
                if (det[1] - y) < (h * TOP_ZONE_PERCENT):
                    candidates.append(FallingCandidate(det, current_frame_num))
                    # Visualization: Light Green hollow circle for new birth
                    cv2.rectangle(frame, (det[0]-15, det[1]-15), (det[0]+15, det[1]+15), COLOR_NEW_CANDIDATE, 2)
        # Cleanup: Move valid ones to final, remove dead ones could be done while looping above but whatever
        for i in range(len(candidates) - 1, -1, -1):
            cand = candidates[i]
            _, last_x, last_y = cand.positions[-1]
            if cand.is_dead:
                if cand.is_valid:
                    # Visualization: SUCCESS (Solid Green Circle)
                    final_measurements.append(cand.positions)
                    print(f"Frame {current_frame_num}: Saved measurement ({len(cand.positions)} pts)")
                    cv2.rectangle(frame, (last_x-20, last_y-20), (last_x+20, last_y+20), COLOR_SUCCESS, 3)      
                else:
                    # Visualization: DELETED/FAILED (Red X)
                    cv2.rectangle(frame, (last_x-15, last_y-15), (last_x+15, last_y+15), COLOR_DELETED, 2)
                candidates.pop(i)

        # 4. Visualize
        for cand in candidates:
            # Draw trail line
            if len(cand.positions) > 1:
                for j in range(1, len(cand.positions)):
                    _, x1, y1 = cand.positions[j-1]
                    _, x2, y2 = cand.positions[j]
                    cv2.line(frame, (x1, y1), (x2, y2), COLOR_TRACKING, 2)
        # Draw ROI box
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ROI, 2)
        
        # Apply crop view if specified
        display_frame = frame
        if crop_view is not None:
            cx, cy, cw, ch = [int(v) for v in crop_view]
            display_frame = frame[cy:cy+ch, cx:cx+cw]
        
        # Info Overlay - drawn on cropped display frame at top left
        cv2.putText(display_frame, f"{current_frame_num}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_DELETED, 2)
        cv2.putText(display_frame, f"{len(final_measurements)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_DELETED, 2)
        
        cv2.imshow('Tracker', display_frame)

        # Handle keyboard input
        while True:
            # When paused, wait indefinitely for a key. When playing, wait 30ms.
            wait_time = 0 if paused else 30
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord(' '):
                paused = not paused
                break
            elif key == ord('a'): # Step Backward 20
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_num - 20))
                break
            elif key == ord('s'): # Step Backward
                # Set to current frame - 2 because the next loop iteration will cap.read() + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_num - 2))
                break
            elif key == ord('d'): # Step Forward
                break 
            elif key == ord('f'): #step 20
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num + 20)
                break
            elif key == ord('r'):
                backSub = cv2.createBackgroundSubtractorMOG2(history=BACKGROUND_WINDOW_WIDTH, varThreshold=40)
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                save_to_csv(final_measurements)
                return
            
            # If not paused, exit the inner loop to process the next frame automatically
            if not paused:
                break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv(final_measurements)

if __name__ == "__main__":
    main()