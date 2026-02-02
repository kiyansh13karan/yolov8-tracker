
import cv2
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/video/test.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_v8.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model to use')
    parser.add_argument('--show', action='store_true', help='Show video output')
    args = parser.parse_args()

    # Load the YOLOv8 model
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    # Open the video file
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {args.output}")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output file
            out.write(annotated_frame)

            # Display the annotated frame (optional)
            if args.show:
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and release the video write object
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == '__main__':
    main()
