import cv2
import numpy as np

# --- Configuration ---
VIDEO_PATH = r'Videos\35cropped.mov'
LANE_ROI = (659, 0, 130, 749)
MIN_FRAMES_TO_VALIDATE = 20  # Persistence threshold TODO calibrate
TOP_ZONE_PERCENT = 0.1      # Must start in top 10% of ROI
MAX_X_DRIFT = 20            # Max horizontal pixel shift between frames
MAX_VELOCITY = 60             # Max distance to look for next match, accounting for missed frames
MAX_MISSED_FRAMES = 3       # Grace period for flickering
MIN_SIZE = 1200 # TODO Calibrate
MAX_SIZE = 2800

BACKGROUND_WINDOW_WIDTH = 5

# --- Colors (BGR) ---
COLOR_TOO_SMALL = (255, 0, 0)      # Blue
COLOR_TOO_BIG = (147, 20, 255)     # Pink
COLOR_NEW_CANDIDATE = (144, 238, 144) # Light Green
COLOR_DELETED = (0, 0, 255)        # Red
COLOR_SUCCESS = (0, 255, 0)        # Solid Green
COLOR_TRACKING = (0, 255, 255)     # Yellow
COLOR_ROI = (255, 0, 0)            # Blue for ROI box

class FallingCandidate:
    def __init__(self, initial_pos, initial_frame):
        self.positions = [initial_pos] # List of (x, y)
        self.start_frame = initial_frame
        self.missed_frames = 0
        self.is_valid = False
        self.is_dead = False

    def update(self, new_pos):
        prev_x, prev_y = self.positions[-1]
        new_x, new_y = new_pos
        
        # Kinematic Check: Must move DOWN (new_y > prev_y) and within X drift limits
        if new_y > prev_y and abs(new_x - prev_x) < MAX_X_DRIFT:
            self.positions.append(new_pos)
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
        return roi
    return default_roi

def process_frame(roi_frame, backSub):
    """Cleans up noise using morphological operations."""
    fg_mask = backSub.apply(roi_frame)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    lane_roi = initialize_roi(cap, LANE_ROI)
    if lane_roi is None: return

    backSub = cv2.createBackgroundSubtractorMOG2(history=BACKGROUND_WINDOW_WIDTH, varThreshold=40, detectShadows=False)
    
    paused = True
    x, y, w, h = [int(v) for v in lane_roi]
    
    candidates = []
    final_measurements = []
    print("Controls: 'space': Pause/Play, 'd': Step Forward, 'a': Step Backward, 'f': Step Forward 20 frames, 'r': Reset, 'q': Quit")

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
            area = cv2.contourArea(cnt)
            mx, my, mw, mh = cv2.boundingRect(cnt)
            abs_x, abs_y = mx + x, my + y   
            if area < MIN_SIZE:
                # Visualization: Blue for too small
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), COLOR_TOO_SMALL, 1)
                continue
            if area > MAX_SIZE:
                # Visualization: Pink for too big
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), COLOR_TOO_BIG, 1)
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
            for i, det in enumerate(current_detections):
                dist = np.linalg.norm(np.array(det) - np.array(cand.positions[-1]))
                if dist < MAX_VELOCITY and dist < best_dist: # Distance gate
                    best_dist = dist
                    best_match_idx = i # TODO two objects merge into one
            
            if best_match_idx is not None:
                # Try to update. If update fails (e.g. moved up), treat as missed frame.
                if cand.update(current_detections[best_match_idx]):
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
            last_pos = cand.positions[-1]                
            if cand.is_dead:
                if cand.is_valid:
                    # Visualization: SUCCESS (Solid Green Circle)
                    final_measurements.append(cand.positions)
                    print(f"Frame {current_frame_num}: Saved measurement ({len(cand.positions)} pts)")
                    cv2.rectangle(frame, (last_pos[0]-20, last_pos[1]-20), (last_pos[0]+20, last_pos[1]+20), COLOR_SUCCESS, 3)      
                else:
                    # Visualization: DELETED/FAILED (Red X)
                    cv2.rectangle(frame, (last_pos[0]-15, last_pos[1]-15), (last_pos[0]+15, last_pos[1]+15), COLOR_DELETED, 2)
                candidates.pop(i)

        # 4. Visualize
        for cand in candidates:
            # Draw trail line
            if len(cand.positions) > 1:
                for j in range(1, len(cand.positions)):
                    cv2.line(frame, cand.positions[j-1], cand.positions[j], COLOR_TRACKING, 2)
        # Draw ROI box
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ROI, 2)
        
        # Info Overlay
        cv2.putText(frame, f"Frame: {current_frame_num} Active: {len(candidates)} Saved: {len(final_measurements)}", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRACKING, 2)
        cv2.imshow('Tracker', frame)

        # Handle keyboard input
        while True:
            # When paused, wait indefinitely for a key. When playing, wait 30ms.
            wait_time = 0 if paused else 30
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord(' '):
                paused = not paused
                break
            elif key == ord('d'): # Step Forward
                break 
            elif key == ord('f'): #step 20
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num + 20)
                break
            elif key == ord('a'): # Step Backward
                # Set to current frame - 2 because the next loop iteration will cap.read() + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_num - 2))
                break
            elif key == ord('q'): # Step Backward 20
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame_num - 20))
                break
            elif key == ord('r'):
                backSub = cv2.createBackgroundSubtractorMOG2(history=BACKGROUND_WINDOW_WIDTH, varThreshold=40)
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            
            # If not paused, exit the inner loop to process the next frame automatically
            if not paused:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()