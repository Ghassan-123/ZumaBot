import pygetwindow as gw
import numpy as np
import pyautogui
import keyboard
import time
import mss
import cv2
from WindowCapturer import WindowCapturer


class GamePlayer:
    def __init__(self):
        self.start_pos = None
        self.finish_pos = []
        self.frog_pos = None
        self.WindowCapturer = WindowCapturer()

    def GetFinishPos(self, hsv):
        if not self.finish_pos :
            finish_mask = cv2.inRange(
                hsv, np.array([25, 100, 50]), np.array([30, 255, 195])
            )

            finish_mask = cv2.dilate(finish_mask, (5, 5), iterations=10)

            finish_contours, _ = cv2.findContours(
                finish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            finish_contour = None
            if finish_contours:
                # Get the largest contour by area
                finish_contour = max(finish_contours, key=cv2.contourArea)
                (finish_cx, finish_cy), finish_radius = cv2.minEnclosingCircle(
                    finish_contour
                )

                self.finish_pos.append((finish_cx, finish_cy))
                return (finish_cx, finish_cy)
        else:
            return self.finish_pos

    def GetFrogPos(self, hsv):
        if self.finish_pos is None:
            frog_mask = cv2.inRange(
                hsv, np.array([14, 26, 130]), np.array([19, 78, 240])
            )

            frog_mask = cv2.dilate(frog_mask, (5, 5), iterations=10)

            frog_contours, _ = cv2.findContours(
                frog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            frog_contour = None
            if frog_contour:
                # Get the largest contour by area
                frog_contour = max(frog_contours, key=cv2.contourArea)
                (frog_cx, frog_cy), frog_radius = cv2.minEnclosingCircle(frog_contour)

                self.frog_pos = (frog_cx, frog_cy)
                return (frog_cx, frog_cy)
        else:
            return self.frog_pos

    def RunLoop(self):

        #  Start timer
        prev_time = time.time()

        # Get game window
        window = self.WindowCapturer.GetWindow("Zuma Deluxe 1.1.0.0")

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print((y, x))
                print(hsv[y, x])

        while True:
            # Get current frame
            frame = self.WindowCapturer.CaptureWindow(window)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Draw FPS
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            finish_mask = cv2.inRange(
                hsv, np.array([25, 100, 50]), np.array([30, 255, 195])
            )
            frog_mask = cv2.inRange(
                hsv, np.array([14, 26, 130]), np.array([19, 72, 240])
            )

            # finish_mask = cv2.morphologyEx(finish_mask, cv2.MORPH_OPEN, (3, 3))
            finish_mask = cv2.dilate(finish_mask, (5, 5), iterations=10)

            frog_mask = cv2.dilate(frog_mask, (5, 5), iterations=10)

            finish_contours, _ = cv2.findContours(
                finish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            frog_contours, _ = cv2.findContours(
                frog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            finish_contour = None
            frog_contour = None
            if finish_contours:
                # Get the largest contour by area
                finish_contour = max(finish_contours, key=cv2.contourArea)
                (fininsh_cx, fininsh_cy), fininsh_radius = cv2.minEnclosingCircle(
                    finish_contour
                )
                cv2.circle(
                    frame,
                    (int(fininsh_cx), int(fininsh_cy)),
                    int(fininsh_radius),
                    (255, 255, 255),
                    2,
                )  # Center dot

            # for frog
            if frog_contours:
                # Get the largest contour by area
                frog_contour = max(frog_contours, key=cv2.contourArea)
                (frog_cx, frog_cy), frog_radius = cv2.minEnclosingCircle(frog_contour)
                cv2.circle(
                    frame,
                    (int(frog_cx), int(frog_cy)),
                    int(frog_radius),
                    (255, 255, 255),
                    2,
                )  # Center dot

            # cv2.drawContours(frame, frog_contour, -1, (255, 255, 255), 10)

            # Show Frame
            cv2.imshow("Test", frame)
            cv2.imshow("Screen Capture", hsv)
            cv2.imshow("Finish Mask", finish_mask)
            cv2.imshow("Frog Mask", frog_mask)

            if keyboard.is_pressed("c"):
                win_x = 30
                win_y = 60

                # Convert window-relative to screen coordinates
                screen_x = window.left + win_x
                screen_y = window.top + win_y

                # Move mouse
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                # Click
                pyautogui.click()

                # Small delay
                time.sleep(0.1)

            cv2.setMouseCallback("Screen Capture", on_mouse)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


gameplayer = GamePlayer()
gameplayer.RunLoop()
