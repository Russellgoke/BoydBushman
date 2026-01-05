import cv2
import os

# --- Configuration ---
video_path = r'Videos\MVI_1635.MP4'

if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def display_frame(frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_num}")
        return None
    
    # Overlay info
    total_seconds = frame_num / fps
    ts = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
    cv2.putText(frame, f"Frame: {frame_num} / {total_frames} ({ts})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("ROI Selector Tool", frame)
    return frame

print(f"Video Loaded: {total_frames} frames at {fps:.2f} FPS")
print("-" * 30)
print("COMMANDS:")
print("  [Number] : Jump to frame number")
print("  mm:ss    : Jump to timestamp")
print("  's'      : Select 4-Point ROI (TL, TR, BR, BL)")
print("  'b'      : Select Standard Bounding Box ROI")
print("  'q'      : Quit")
print("-" * 30)

current_frame_idx = 0
current_img = display_frame(current_frame_idx)

while True:
    cmd = input("Enter Frame, Time, or Command: ").strip().lower()

    if cmd == 'q':
        break
    
    elif cmd == 's':
        # 4-Point Manual Selection
        pts = []
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                cv2.circle(temp_img, (x, y), 4, (0, 0, 255), -1)
                if len(pts) > 1:
                    cv2.line(temp_img, pts[-2], pts[-1], (0, 255, 0), 1)
                cv2.imshow("Select 4 Points", temp_img)

        temp_img = current_img.copy()
        cv2.imshow("Select 4 Points", temp_img)
        cv2.setMouseCallback("Select 4 Points", click_event)
        print("Click 4 corners: TL, TR, BR, BL. Press any key to finish.")
        cv2.waitKey(0)
        cv2.destroyWindow("Select 4 Points")
        print(f"\n4-Point ROI: {pts}\n")

    elif cmd == 'b':
        # Standard OpenCV Bounding Box
        roi = cv2.selectROI("Select Bounding Box", current_img, fromCenter=False)
        cv2.destroyWindow("Select Bounding Box")
        if roi != (0, 0, 0, 0):
            print(f"\nBounding Box (x, y, w, h): {roi}\n")

    else:
        # Attempt to parse as frame number or timestamp
        try:
            if ':' in cmd:
                m, s = map(int, cmd.split(':'))
                target = int((m * 60 + s) * fps)
            else:
                target = int(cmd)
            
            if 0 <= target < total_frames:
                current_frame_idx = target
                current_img = display_frame(current_frame_idx)
            else:
                print(f"Frame {target} out of range.")
        except ValueError:
            print("Invalid input. Use frame number, mm:ss, 's', 'b', or 'q'.")

cap.release()
cv2.destroyAllWindows()