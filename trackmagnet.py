import cv2
import numpy as np

# --- Configuration ---
VIDEO_PATH = r'Videos\35cropped.mov'
LANE_ROI = (659, 0, 120, 779)
MIN_FRAMES_TO_VALIDATE = 20  # Persistence threshold TODO calibrate
TOP_ZONE_PERCENT = 0.1      # Must start in top 10% of ROI
MAX_X_DRIFT = 20            # Max horizontal pixel shift between frames
MAX_VELOCITY = 50          # Max pixels per frame, tolerant for missed frames
MAX_MISSED_FRAMES = 3       # Grace period for flickering
MIN_SIZE = 5 # TODO Calibrate
MAX_SIZE = 50000 # TODO Calibrate

class FallingCandidate:
    def __init__(self, initial_pos, initial_frame, roi_h):
        self.positions = [initial_pos] # List of (x, y)
        self.start_frame = initial_frame
        self.missed_frames = 0
        self.is_valid = False
        self.is_dead = False
        self.roi_h = roi_h

    def update(self, new_pos):
        prev_x, prev_y = self.positions[-1]
        new_x, new_y = new_pos
        
        # Check downward motion and horizontal drift
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
    if not ret:
        return None
    
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

def find_best_candidate(contours, x_offset, y_offset):
    """Filters contours and returns the most likely falling object."""
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 500000:
            mx, my, mw, mh = cv2.boundingRect(cnt)
            cx = mx + x_offset + mw // 2
            cy = my + y_offset + mh // 2
            candidates.append({'pos': (cx, cy), 'area': area, 'bbox': (mx + x_offset, my + y_offset, mw, mh)})
    
    # Simple heuristic: return the lowest object (largest y) if multiple exist, 
    # or implement velocity tracking here.
    if not candidates:
        return None
    return max(candidates, key=lambda c: c['pos'][1])

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    lane_roi = initialize_roi(cap, LANE_ROI)
    if lane_roi is None: return

    backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    
    paused = False
    x, y, w, h = [int(v) for v in lane_roi]
    
    candidates = []
    final_measurements = []

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 1. Pre-process
            roi_frame = frame[y:y+h, x:x+w]
            fg_mask = process_frame(roi_frame, backSub)

            # 2. Detect objects that are not background
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_detections = []
            matched_detections = set()
            # Filter detections by size
            for cnt in contours:
                if MIN_SIZE < cv2.contourArea(cnt) < MAX_SIZE:
                    mx, my, mw, mh = cv2.boundingRect(cnt)
                    current_detections.append((mx + x + mw//2, my + y + mh//2))
                    
            
            # Update candidates if they continue or are gone
            for cand in candidates:
                best_dist = float('inf')
                best_match = None
                
                for i, det in enumerate(current_detections):
                    dist = np.linalg.norm(np.array(det) - np.array(cand.positions[-1]))
                    if dist < MAX_VELOCITY and dist < best_dist: # Distance gate
                        best_dist = dist
                        best_match = i # TODO two objects merge into one
                
                if best_match is not None:
                    if cand.update(current_detections[best_match]):
                        matched_detections.add(best_match)
                else:
                    cand.increment_missed()
            
            # Cleanup: Move valid ones to final, remove dead ones could be done while looping above but whatever
            for cand in candidates[:]:
                if cand.is_dead:
                    if cand.is_valid:
                        final_measurements.append(cand.positions)
                        print(f"Recorded measurement with {len(cand.positions)} points.")
                candidates.remove(cand)
            
            # Check for new candidates
            for i, det in enumerate(current_detections):
                if i not in matched_detections:
                    if (det[1] - y) < (h * TOP_ZONE_PERCENT):
                        candidates.append(FallingCandidate(det, current_frame, h))


            # 4. Visualize
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Tracker', frame)
            cv2.imshow('Mask', fg_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('r'):
            backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()