import cv2
import numpy as np
import easyocr
import concurrent.futures
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. Load the video
video_path = "input.mp4"  # Replace with your video file
output_path = "output.mp4"  # Output file

cap = cv2.VideoCapture(video_path)

# 2. Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4

# 3. Initialize video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 4. Initialize EasyOCR Reader (Use GPU if available)
reader = easyocr.Reader(["ch_sim", "en"], gpu=True)

# 5. Function to process a frame
def process_frame(frame):
    try:
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect text
        results = reader.readtext(gray)

        # Create a blank mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for bbox, text, prob in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))

            # Fill mask where text is detected
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=cv2.FILLED)

        # Inpaint the frame to remove text
        inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return inpainted_frame

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return frame  # Return original frame if error occurs

# 6. Process video frames in parallel
start_time = time.time()
frame_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Submit frame processing to the thread pool
        future = executor.submit(process_frame, frame)
        future_frames.append(future)

        frame_count += 1
        if frame_count % 10 == 0:
            logging.info(f"Processing frame {frame_count}/{total_frames}...")

    # Retrieve and write processed frames
    for future in concurrent.futures.as_completed(future_frames):
        processed_frame = future.result()
        out.write(processed_frame)

# 7. Release resources
cap.release()
out.release()
end_time = time.time()
logging.info(f"✅ Video processing complete! Saved as: {output_path}")
logging.info(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")

