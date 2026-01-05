import cv2
import numpy as np

# --- Configuration ---
video_path = r'Videos\35cropped.mov'
# Set this to True once you have your lane coordinates to skip the UI
LANE_ROI = (659, 0, 120, 779) # Placeholder for your coordinates

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 1. Select the "Falling Lane" (ROI)
# This ignores messy background noise outside the drop zone
ret, first_frame = cap.read()
if LANE_ROI is None:
    print("Select the vertical lane where the object falls, then press SPACE.")
    lane_roi = cv2.selectROI("Select Drop Lane", first_frame, fromCenter=False)
    cv2.destroyWindow("Select Drop Lane")
else:
    lane_roi = LANE_ROI

# 2. Initialize Background Subtractor
# varThreshold: Increase if you get too much "ghosting" from the background
# detectShadows: Set to False to reduce processing and ignore faint shadows
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

print("Controls: 'q': Quit, 'space': Pause/Play, 'r': Reset Background")

paused = False
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret: break

    # Crop frame to the falling lane
    x, y, w, h = [int(v) for v in lane_roi]
    roi_frame = frame[y:y+h, x:x+w]

    # 3. Apply Background Subtraction
    fg_mask = backSub.apply(roi_frame)

    # 4. Clean up the mask (Morphological Filtering)
    # This removes tiny 'salt and pepper' noise from the messy background
    kernel = np.ones((3,3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) 
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # 5. Find the Object
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by expected size of your dropped object (in pixels)
        if 50 < area < 5000:
            mx, my, mw, mh = cv2.boundingRect(cnt)
            
            # Draw on the original frame (adjusting for ROI offset)
            cv2.rectangle(frame, (mx+x, my+y), (mx+x+mw, my+y+mh), (0, 255, 255), 2)
            
            # Calculate Centroid
            cx = mx + x + mw//2
            cy = my + y + mh//2
            
            print(f"Frame {current_frame_num}: Object at ({cx}, {cy})")

    # Visuals
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Show the lane
    cv2.imshow('Dropped Object Tracker', frame)
    cv2.imshow('Mask (What the AI sees)', fg_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): paused = not paused
    elif key == ord('r'): # Reset background model if it gets too messy
        backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

cap.release()
cv2.destroyAllWindows()