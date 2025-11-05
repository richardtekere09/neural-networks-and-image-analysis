#!/usr/bin/env python3
"""
Enhanced Multi-Object Tracking with Advanced Features
- Pause/Resume tracking
- Delete lost objects
- Add objects during tracking
- Better visual feedback
"""

import sys
import cv2
from random import randint
import time

# Available tracker types
TRACKER_TYPES = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "CSRT"]


class MultiObjectTracker:
    """Class to manage multiple object tracking"""

    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []
        self.active = []  # Track which trackers are active
        self.paused = False

    def create_tracker(self):
        """Create a new tracker instance"""
        if self.tracker_type == "BOOSTING":
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == "MIL":
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "TLD":
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == "MEDIANFLOW":
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        else:
            return None

    def add_object(self, frame, bbox):
        """Add a new object to track"""
        tracker = self.create_tracker()
        if tracker is None:
            return False

        success = tracker.init(frame, bbox)
        if success:
            self.trackers.append(tracker)
            self.colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            self.active.append(True)
            return True
        return False

    def update(self, frame):
        """Update all active trackers"""
        boxes = []
        for i, tracker in enumerate(self.trackers):
            if self.active[i]:
                success, box = tracker.update(frame)
                if success:
                    boxes.append((i, box))
                else:
                    self.active[i] = False
                    boxes.append((i, None))
            else:
                boxes.append((i, None))
        return boxes

    def get_active_count(self):
        """Get number of active trackers"""
        return sum(self.active)

    def remove_tracker(self, index):
        """Mark a tracker as inactive"""
        if 0 <= index < len(self.active):
            self.active[index] = False


def draw_help_menu(frame):
    """Draw help menu overlay"""
    overlay = frame.copy()
    height, width = frame.shape[:2]

    # Semi-transparent background
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Help text
    help_text = [
        "CONTROLS:",
        "ESC / Q - Quit",
        "SPACE - Pause/Resume",
        "A - Add new object",
        "R - Remove lost trackers",
        "H - Toggle help",
    ]

    y_offset = 35
    for i, text in enumerate(help_text):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (20, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def main():
    """Main function for enhanced multi-object tracking"""
    print("=" * 70)
    print("ENHANCED MULTI-OBJECT TRACKING")
    print("=" * 70)

    # Select tracker type
    print("\nAvailable tracking algorithms:")
    for i, t in enumerate(TRACKER_TYPES, 1):
        print(f"{i}. {t}")

    tracker_choice = input("\nSelect tracker (1-7) or press Enter for CSRT: ").strip()

    if tracker_choice and tracker_choice.isdigit() and 1 <= int(tracker_choice) <= 7:
        tracker_type = TRACKER_TYPES[int(tracker_choice) - 1]
    else:
        tracker_type = "CSRT"

    print(f"\nUsing tracker: {tracker_type}")

    # Select video source
    print("\n" + "=" * 70)
    print("Video Source Options:")
    print("1. MacBook Camera (default)")
    print("2. Video File")
    source_choice = input("\nSelect source (1-2) or press Enter for camera: ").strip()

    if source_choice == "2":
        videoPath = input("Enter video file path: ").strip()
        cap = cv2.VideoCapture(videoPath)
    else:
        print("\nInitializing MacBook camera...")
        cap = cv2.VideoCapture(0)
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Failed to open video source")
        sys.exit(1)

    # Read first frame
    success, frame = cap.read()
    if not success:
        print("Error: Failed to read video")
        sys.exit(1)

    print("\nVideo source opened successfully!")

    # Initialize tracker manager
    tracker_manager = MultiObjectTracker(tracker_type)

    # Initial object selection
    print("\n" + "=" * 70)
    print("SELECT OBJECTS TO TRACK:")
    print("=" * 70)
    print("1. Draw bounding box around object")
    print("2. Press SPACE/ENTER to confirm")
    print("3. Press 'q' when done selecting")
    print("=" * 70)

    while True:
        bbox = cv2.selectROI(
            "Select Objects", frame, fromCenter=False, showCrosshair=True
        )

        if bbox[2] > 0 and bbox[3] > 0:
            if tracker_manager.add_object(frame, bbox):
                print(f"Object {len(tracker_manager.trackers)} added successfully")
            else:
                print("Failed to add object")

        k = cv2.waitKey(0) & 0xFF
        if k == ord("q") or k == ord("Q"):
            break

    cv2.destroyWindow("Select Objects")

    if len(tracker_manager.trackers) == 0:
        print("\nNo objects selected. Exiting...")
        cap.release()
        sys.exit(0)

    print(f"\n{len(tracker_manager.trackers)} objects selected")
    print("\nStarting tracking...")

    # Tracking loop variables
    fps_timer = cv2.getTickCount()
    frame_count = 0
    show_help = True
    adding_object = False

    print("\n" + "=" * 70)
    print("TRACKING CONTROLS:")
    print("=" * 70)
    print("ESC or Q - Quit tracking")
    print("SPACE   - Pause/Resume")
    print("A       - Add new object during tracking")
    print("R       - Remove all lost trackers")
    print("H       - Toggle help overlay")
    print("=" * 70)

    while cap.isOpened():
        if not tracker_manager.paused:
            success, frame = cap.read()
            if not success:
                print("\nEnd of video")
                break

            frame_count += 1

            # Update all trackers
            boxes = tracker_manager.update(frame)

            # Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)
            fps_timer = cv2.getTickCount()

            # Draw tracked objects
            for idx, box in boxes:
                if box is not None:
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    cv2.rectangle(frame, p1, p2, tracker_manager.colors[idx], 2, 1)

                    # Draw label with background
                    label = f"#{idx + 1}"
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    cv2.rectangle(
                        frame,
                        (p1[0], p1[1] - text_size[1] - 10),
                        (p1[0] + text_size[0], p1[1]),
                        tracker_manager.colors[idx],
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (p1[0], p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Display info bar
            active_count = tracker_manager.get_active_count()
            total_count = len(tracker_manager.trackers)

            info_bg = frame.copy()
            cv2.rectangle(info_bg, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(info_bg, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame,
                f"Tracker: {tracker_type}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Objects: {active_count}/{total_count}",
                (200, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Show help menu if enabled
            if show_help:
                draw_help_menu(frame)

        else:
            # Show pause indicator
            cv2.putText(
                frame,
                "PAUSED - Press SPACE to resume",
                (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Multi-Object Tracker", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q") or key == ord("Q"):  # ESC or Q
            print("\nExiting...")
            break

        elif key == ord(" "):  # SPACE - Pause/Resume
            tracker_manager.paused = not tracker_manager.paused
            if tracker_manager.paused:
                print("PAUSED")
            else:
                print("RESUMED")

        elif key == ord("a") or key == ord("A"):  # Add object
            tracker_manager.paused = True
            print("\nSelect new object to track...")
            bbox = cv2.selectROI(
                "Add New Object", frame, fromCenter=False, showCrosshair=True
            )
            cv2.destroyWindow("Add New Object")

            if bbox[2] > 0 and bbox[3] > 0:
                if tracker_manager.add_object(frame, bbox):
                    print(f"Object {len(tracker_manager.trackers)} added successfully")
                else:
                    print("Failed to add object")

            tracker_manager.paused = False

        elif key == ord("r") or key == ord("R"):  # Remove lost trackers
            removed = 0
            for i in range(len(tracker_manager.active)):
                if not tracker_manager.active[i]:
                    removed += 1

            # Reset lists to only keep active trackers
            new_trackers = []
            new_colors = []
            new_active = []

            for i, active in enumerate(tracker_manager.active):
                if active:
                    new_trackers.append(tracker_manager.trackers[i])
                    new_colors.append(tracker_manager.colors[i])
                    new_active.append(True)

            tracker_manager.trackers = new_trackers
            tracker_manager.colors = new_colors
            tracker_manager.active = new_active

            print(f"Removed {removed} lost tracker(s)")

        elif key == ord("h") or key == ord("H"):  # Toggle help
            show_help = not show_help

    # Cleanup
    print(f"\nTotal frames processed: {frame_count}")
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking completed successfully!")


if __name__ == "__main__":
    main()
