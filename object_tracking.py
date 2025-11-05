"""
Object Tracking using OpenCV
This script demonstrates multiple tracking algorithms available in OpenCV
and allows you to track objects using your MacBook's built-in camera.
"""

import cv2
import sys

# Dictionary of available tracking algorithms in OpenCV
TRACKER_TYPES = {
    "1": "BOOSTING",
    "2": "MIL",
    "3": "KCF",
    "4": "TLD",
    "5": "MEDIANFLOW",
    "6": "MOSSE",
    "7": "CSRT",
}


def create_tracker(tracker_type):
    """
    Create a tracker object based on the specified type.

    Args:
        tracker_type (str): Name of the tracker algorithm

    Returns:
        tracker: OpenCV tracker object
    """
    if tracker_type == "BOOSTING":
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == "TLD":
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        print(f"Unknown tracker type: {tracker_type}")
        sys.exit(1)

    return tracker


def main():
    """
    Main function to run the object tracking application.
    """
    # Display available tracker types
    print("=" * 50)
    print("Object Tracking with OpenCV")
    print("=" * 50)
    print("\nAvailable Tracker Algorithms:")
    for key, value in TRACKER_TYPES.items():
        print(f"{key}. {value}")

    # Get user input for tracker selection
    tracker_choice = input("\nSelect tracker type (1-7): ")

    if tracker_choice not in TRACKER_TYPES:
        print("Invalid choice. Using default tracker: CSRT")
        tracker_type = "CSRT"
    else:
        tracker_type = TRACKER_TYPES[tracker_choice]

    print(f"\nUsing {tracker_type} tracker")

    # Initialize video capture from MacBook camera (index 0)
    print("\nInitializing MacBook camera...")
    video = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Read the first frame
    ok, frame = video.read()
    if not ok:
        print("Error: Cannot read video stream")
        sys.exit(1)

    print("\nCamera initialized successfully!")
    print("\nInstructions:")
    print("1. Select the object you want to track by drawing a bounding box")
    print("2. Press ENTER or SPACE to confirm the selection")
    print("3. Press ESC to cancel and exit")

    # Select ROI (Region of Interest) - the object to track
    bbox = cv2.selectROI("Select Object to Track", frame, False)
    cv2.destroyWindow("Select Object to Track")

    # Check if a valid selection was made
    if bbox[2] == 0 or bbox[3] == 0:
        print("No valid selection made. Exiting...")
        video.release()
        sys.exit(1)

    # Create tracker and initialize with first frame and bounding box
    tracker = create_tracker(tracker_type)
    ok = tracker.init(frame, bbox)

    if not ok:
        print("Error: Failed to initialize tracker")
        video.release()
        sys.exit(1)

    print("\nTracking started! Press 'q' to quit.")

    # Variables for FPS calculation
    fps_timer = cv2.getTickCount()

    # Main tracking loop
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print("End of video stream")
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)
        fps_timer = cv2.getTickCount()

        # Draw bounding box and information
        if ok:
            # Tracking success - draw green box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

            # Display tracking status
            cv2.putText(
                frame,
                "Tracking",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            # Tracking failure - display message
            cv2.putText(
                frame,
                "Tracking Failed",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Display tracker type and FPS
        cv2.putText(
            frame,
            f"Tracker: {tracker_type}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Display instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show the frame
        cv2.imshow("Object Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nExiting...")
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    print("Tracking completed successfully!")


if __name__ == "__main__":
    main()
