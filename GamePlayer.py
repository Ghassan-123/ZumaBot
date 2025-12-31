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

    def GetFinishPos(self, hsv, area_threshold=4800):
        if not self.finish_pos:
            finish_mask = cv2.inRange(
                hsv, np.array([25, 100, 50]), np.array([30, 255, 195])
            )

            finish_mask = cv2.dilate(finish_mask, (5, 5), iterations=10)
            # cv2.imshow("finish", finish_mask)

            finish_contours, _ = cv2.findContours(
                finish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if finish_contours:
                # Sort contours by area biggest → smallest
                finish_contours = sorted(
                    finish_contours, key=cv2.contourArea, reverse=True
                )

                # Take the largest two (if available)
                top_two = finish_contours[:2]

                for cnt in top_two:
                    area = cv2.contourArea(cnt)

                    # Only append if larger than threshold
                    if (area_threshold - 200) < area < (area_threshold + 200):
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        self.finish_pos.append(((cx, cy), radius))

                if self.finish_pos:
                    return self.finish_pos
        else:
            return self.finish_pos

    def GetFrogPos(self, hsv, area_threshold=4000):
        # if self.frog_pos is None:
        frog_mask = cv2.inRange(hsv, np.array([14, 26, 130]), np.array([19, 78, 240]))

        frog_mask = cv2.dilate(frog_mask, (5, 5), iterations=10)
        # cv2.imshow("frog", frog_mask)

        frog_contours, _ = cv2.findContours(
            frog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if frog_contours:
            # Get the largest contour by area
            frog_contour = max(frog_contours, key=cv2.contourArea)
            area = cv2.contourArea(frog_contour)

            if (area_threshold - 400) < area < (area_threshold + 400):
                (frog_cx, frog_cy), frog_radius = cv2.minEnclosingCircle(frog_contour)
                self.frog_pos = ((frog_cx, frog_cy), frog_radius)

            if self.frog_pos:
                return self.frog_pos

    # else:
    #     return self.frog_pos

    def DetectFrogSift(self, frame):
        # sift
        # 1. Setup
        template = cv2.imread("frog_template.png")
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect and Compute
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(frame_gray, None)

        # 3. Match using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # 4. Sort and Filter
        # Sort matches by distance (lower distance means a better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep only the best 20 matches, or filter by a distance value
        good = [m for m in matches if m.distance < 50]

        # 5. Draw Matches
        res = cv2.drawMatches(
            template_gray,
            kp1,
            frame_gray,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("Final Detection", res)

    def DetectFrogOrb(self, frame):
        # 1. Setup
        template = cv2.imread("frog_template.png")
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Initialize ORB detector
        # nfeatures=500 is the default; increase it if the frog is hard to find
        orb = cv2.ORB_create(nfeatures=1000)

        # 3. Detect and Compute
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)

        # Check if descriptors were found to avoid crashing the matcher
        if des1 is None or des2 is None:
            return

        # 4. Match using BFMatcher
        # IMPORTANT: Use NORM_HAMMING for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # 5. Sort and Filter
        matches = sorted(matches, key=lambda x: x.distance)

        # ORB distances are integers (0 to 256).
        # A distance < 30 is usually a very strong match.
        good = [m for m in matches if m.distance < 40]

        # 6. Draw
        res = cv2.drawMatches(
            template_gray,
            kp1,
            frame_gray,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("ORB Detection", res)

    # need payment and rebuilding open cv
    def DetectFrogSurf(self, frame):
        # 1. Setup
        template = cv2.imread("frog_template.png")
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect and Compute using SURF
        # 400 is the Hessian Threshold (higher = fewer, but stronger features)
        surf = cv2.xfeatures2d.SURF_create(400)

        kp1, des1 = surf.detectAndCompute(template_gray, None)
        kp2, des2 = surf.detectAndCompute(frame_gray, None)

        # 3. Match using BFMatcher
        # SURF descriptors are floating point, so use NORM_L2
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # 4. Sort and Filter
        matches = sorted(matches, key=lambda x: x.distance)
        good = [m for m in matches if m.distance < 0.1]  # SURF distances vary from SIFT

        # 5. Draw
        res = cv2.drawMatches(
            template_gray,
            kp1,
            frame_gray,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("SURF Detection", res)

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

            finishPos = self.GetFinishPos(hsv)
            frogPos = self.GetFrogPos(hsv)

            if finishPos is not None:
                for pos in finishPos:
                    (cx, cy), radius = pos
                    cv2.circle(
                        frame,
                        (int(cx), int(cy)),
                        int(radius),
                        (255, 255, 255),
                        2,
                    )  # Center dot

            if frogPos is not None:
                (cx, cy), radius = frogPos
                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    int(radius),
                    (255, 255, 255),
                    2,
                )  # Center dot

            # cv2.drawContours(frame, frog_contour, -1, (255, 255, 255), 10)

            # Show Frame
            # cv2.imshow("Test", frame)
            cv2.imshow("Screen Capture", hsv)
            # self.DetectFrogSift(frame)
            self.DetectFrogOrb(frame)

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
