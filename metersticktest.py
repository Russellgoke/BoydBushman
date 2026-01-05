import cv2
import numpy as np
import os
import math

# --- Constants ---
MANUALLY_DEFINED_PTS = [(636, 25), (680, 26), (728, 729), (685, 736)] 
video_path = r'Videos\MVI_1635.MP4'

if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}")
    exit()

roi_str = "_".join([f"{p[0]}-{p[1]}" for p in MANUALLY_DEFINED_PTS])
log_filename = f"track_log_{roi_str}.csv"

if not os.path.exists(log_filename):
    with open(log_filename, 'w') as f:
        # Updated header to reflect 8 corner values
        f.write("frame,x0,y0,x1,y1,x2,y2,x3,y3\n")

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()

src_pts = np.float32(MANUALLY_DEFINED_PTS)
width = int(max(np.linalg.norm(src_pts[0]-src_pts[1]), np.linalg.norm(src_pts[2]-src_pts[3])))
height = int(max(np.linalg.norm(src_pts[0]-src_pts[3]), np.linalg.norm(src_pts[1]-src_pts[2])))
dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
M_warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
template = cv2.warpPerspective(first_frame, M_warp, (width, height))

sift = cv2.SIFT_create()
kp_temp, des_temp = sift.detectAndCompute(template, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

print(f"Logging to: {log_filename}")
print("Controls: 'q': Quit, 'f': Jump to frame, 'space': Pause/Play")

paused = False
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret: break
    
    curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # --- CHANGE: Process and Log only every 100 frames ---
    if curr_frame % 100 == 0:
        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        
        if des_frame is not None and len(kp_frame) > 10:
            matches = flann.knnMatch(des_temp, des_frame, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good_matches) > 10:
                src_pts_m = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts_m = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M_homog, _ = cv2.findHomography(src_pts_m, dst_pts_m, cv2.RANSAC, 5.0)
                
                if M_homog is not None:
                    rect_pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
                    dst_frame_pts = cv2.perspectiveTransform(rect_pts, M_homog)
                    
                    bc_x = np.mean(dst_frame_pts[1:3], axis=0)[0][0]
                    bc_y = np.mean(dst_frame_pts[1:3], axis=0)[0][1]
                    
                    corners_flat = dst_frame_pts.flatten()
                    corners_str = ",".join([f"{val:.2f}" for val in corners_flat])

                    with open(log_filename, 'a') as f:
                        f.write(f"{curr_frame},{corners_str}\n")
                    
                    cv2.polylines(frame, [np.int32(dst_frame_pts)], True, (0, 255, 0), 2)
                    cv2.circle(frame, (int(bc_x), int(bc_y)), 7, (0, 0, 255), -1)

    cv2.putText(frame, f"Frame: {curr_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Tracking', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): paused = not paused
    elif key == ord('f'):
        target = input("Enter frame: ")
        if target.isdigit():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()