import os
from datetime import datetime
import time
import numpy as np
import cv2
from cv2 import VideoWriter_fourcc

IMSHOW = False
IMWRITE = True

REFRESH_MASK_SECONDS_INTERVAL = 10
NO_MOVEMENT_SPAWN_SECONDS_INTERVAL = 30
MAX_RECORD_SECONDS_DURATION = 60 * 15
SIMULTANEOUS_RECORDING_SECONDS_OVERLAP = 15

FPS = 10.0

FOURCC = VideoWriter_fourcc(*"mp4v")


def main():
    # Initialize video capture object (0 for webcam)
    cap = cv2.VideoCapture(0)

    mask = None

    refresh_mask_time = time.time()
    time_last_recording_frame = time.time()
    recording_started_time = 0
    aux_recording_started_time = 0

    output_path = "./bin/"
    recording_writer = None
    aux_recording_writer = None

    is_recording = False

    while True:
        aux_tmp = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # If no frame is captured, break
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and detail
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Set the mask frame as the reference (background frame)
        if (
            mask is None
            or time.time() - refresh_mask_time > REFRESH_MASK_SECONDS_INTERVAL
        ):
            mask = gray
            refresh_mask_time = time.time()

        # Compute the absolute difference between the current frame and first frame
        frame_delta = cv2.absdiff(mask, gray)

        # Apply a threshold to eliminate small changes and create a binary image
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate the threshold image to fill in holes
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)

        # Find contours (edges or boundaries) in the thresholded image
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if IMSHOW:
            boxes_frame = frame.copy()
            # Loop over the contours
            for contour in contours:
                # If the contour is too small, ignore it
                if cv2.contourArea(contour) < 500:
                    continue

                # Get the bounding box for each motion area
                (x, y, w, h) = cv2.boundingRect(contour)
                # Draw a rectangle around the motion area
                cv2.rectangle(boxes_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(contours) > 0:
            is_recording = True
            time_last_recording_frame = time.time()
        elif (
            is_recording
            and time.time() - time_last_recording_frame
            > NO_MOVEMENT_SPAWN_SECONDS_INTERVAL
        ):
            is_recording = False

        if is_recording:
            # Display the resulting frames
            if IMSHOW:
                cv2.imshow("Frame with boxes", boxes_frame)
                # cv2.imshow("Threshold", thresh)
                # cv2.imshow("Mask", mask)
            if IMWRITE:
                if recording_writer is None:
                    file_name = os.path.join(
                        output_path,
                        f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.mp4",
                    )
                    recording_writer = cv2.VideoWriter(
                        file_name, FOURCC, FPS, frame.shape[1::-1]
                    )
                    recording_started_time = time.time()
                recording_writer.write(frame)

                if (time.time() - recording_started_time) > (
                    MAX_RECORD_SECONDS_DURATION - SIMULTANEOUS_RECORDING_SECONDS_OVERLAP
                ):
                    if aux_recording_writer is None:
                        file_name = os.path.join(
                            output_path,
                            f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.mp4",
                        )
                        aux_recording_writer = cv2.VideoWriter(
                            file_name, FOURCC, FPS, frame.shape[1::-1]
                        )
                        aux_recording_started_time = time.time()

                if aux_recording_writer is not None:
                    aux_recording_writer.write(frame)

                if (time.time() - recording_started_time) > MAX_RECORD_SECONDS_DURATION:
                    recording_writer.release()
                    recording_writer = aux_recording_writer
                    aux_recording_writer = None
                    recording_started_time = aux_recording_started_time

        else:
            if IMSHOW:
                cv2.destroyAllWindows()
            if IMWRITE:
                if recording_writer is not None:
                    recording_writer.release()
                    recording_writer = None
                if aux_recording_writer is not None:
                    aux_recording_writer.release()
                    aux_recording_writer = None

        # Break the loop if 'q' is pressed
        diff_frame = int((1e3 / FPS) - ((time.time() - aux_tmp) * 1e3))
        if cv2.waitKey(max(1, diff_frame)) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
