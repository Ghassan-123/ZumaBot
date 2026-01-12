from WindowCapturer import WindowCapturer
from collections import defaultdict
from Ball import Ball

import numpy as np
import pyautogui
import keyboard
import time
import sys
import cv2

import threading


class GamePlayerChains:
    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio
        self.window = None

        self.start_pos = None
        self.finish_pos = []
        self.frog_pos = None
        self.WindowCapturer = WindowCapturer()
        self.frog_template = cv2.imread("frog_template.png")

        self.masks_row = {}
        self.masks_cleaned = {}
        self.current_ball = None
        self.second_color = None

        self.all_balls_mask = None
        self.balls_centers = []

        self.blobs = {}

        self.start = False
        self.can_shoot = False
        self.can_shoot_time = 0
        self.can_shoot_duration = 2  # seconds

        self.kernels = {
            1: np.ones((3, 3), np.uint8),
            2: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            3: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            4: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        }

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
                        self.finish_pos.append(((cx, cy), radius + 2))

                if self.finish_pos:
                    return self.finish_pos
        else:
            return self.finish_pos

    def GetFrogPos(self, hsv, area_threshold=1200):
        if self.frog_pos is None:
            frog_mask = cv2.inRange(
                hsv, np.array([15, 15, 200]), np.array([20, 45, 255])
            )

            frog_mask = cv2.dilate(frog_mask, (7, 7), iterations=10)
            # cv2.imshow("frog", frog_mask)

            frog_contours, _ = cv2.findContours(
                frog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if frog_contours:
                # Get the largest contour by area
                frog_contour = max(frog_contours, key=cv2.contourArea)
                area = cv2.contourArea(frog_contour)

                if (area_threshold - 400) < area < (area_threshold + 400):
                    (frog_cx, frog_cy), frog_radius = cv2.minEnclosingCircle(
                        frog_contour
                    )
                    self.frog_pos = ((frog_cx, frog_cy), frog_radius + 20)

                if self.frog_pos:
                    return self.frog_pos
        else:
            return self.frog_pos

    def GetRed(self, hsv):
        # Red part 1 (low hue)
        lower_red1 = np.array([0, 110, 210])
        upper_red1 = np.array([12, 255, 255])

        # Red part 2 (high hue)
        lower_red2 = np.array([170, 110, 210])
        upper_red2 = np.array([179, 255, 255])

        # Red (two ranges)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        red_mask = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, self.kernels[1])

        self.masks_row["red"] = red_mask

        temp = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE, self.kernels[2], iterations=1
        )

        temp = cv2.dilate(temp, self.kernels[3], iterations=1)

        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, self.kernels[4], iterations=1)

        self.masks_cleaned["red"] = temp

    def GetGreen(self, hsv):
        lower_green = (50, 160, 165)
        upper_green = (70, 230, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        green_mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, self.kernels[1])

        self.masks_row["green"] = green_mask

        temp = cv2.morphologyEx(
            green_mask, cv2.MORPH_CLOSE, self.kernels[2], iterations=1
        )

        temp = cv2.dilate(temp, self.kernels[3], iterations=1)

        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, self.kernels[4], iterations=1)

        self.masks_cleaned["green"] = temp

    def GetBlue(self, hsv):
        lower_blue = (95, 180, 190)
        upper_blue = (130, 230, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        blue_mask = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.kernels[1])

        self.masks_row["blue"] = blue_mask  # mask_blue

        temp = cv2.morphologyEx(
            blue_mask, cv2.MORPH_CLOSE, self.kernels[2], iterations=1
        )

        temp = cv2.dilate(temp, self.kernels[3], iterations=1)

        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, self.kernels[4], iterations=1)

        self.masks_cleaned["blue"] = temp

    def GetYellow(self, hsv):
        lower_yellow = (20, 160, 200)
        upper_yellow = (30, 230, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        yellow_mask = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernels[1])

        self.masks_row["yellow"] = yellow_mask

        temp = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_CLOSE, self.kernels[2], iterations=1
        )

        if self.finish_pos:
            for fpos in self.finish_pos:
                (ffx, ffy), ffr = fpos
                cv2.circle(
                    temp,
                    center=(int(ffx), int(ffy)),
                    radius=int(ffr),
                    color=0,
                    thickness=-1,  # filled circle
                )

        temp = cv2.dilate(temp, self.kernels[3], iterations=1)

        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, self.kernels[4], iterations=1)

        self.masks_cleaned["yellow"] = temp

    def process_colors(self, hsv):
        threads = [
            threading.Thread(target=self.GetRed, args=(hsv,)),
            threading.Thread(target=self.GetGreen, args=(hsv,)),
            threading.Thread(target=self.GetBlue, args=(hsv,)),
            threading.Thread(target=self.GetYellow, args=(hsv,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def RunLoop(self):
        #  Start timer
        prev_time = time.time()

        # Get game window
        self.window = self.WindowCapturer.WaitForWindow("Zuma Deluxe 1.1.0.0", interval_seconds=0.1)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print((y, x))
                print(hsv[y, x])

        while True:
            # Get current frame
            frame = self.WindowCapturer.CaptureWindow(self.window)

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
            if self.aspect_ratio == 1.6:
                # cut top
                hsv[:64, :] = (0, 0, 0)
                hsv[:80, :112] = (0, 0, 0)
                hsv[:80, -112:] = (0, 0, 0)

                # edges (left, right, bottom)
                hsv[:, :10] = (0, 0, 0)
                hsv[:, -10:] = (0, 0, 0)
                hsv[-10:, :] = (0, 0, 0)
            else:
                # cut top
                hsv[:50, :] = (0, 0, 0)
                hsv[:60, :100] = (0, 0, 0)
                hsv[:60, 100:] = (0, 0, 0)

                # edges (left, right, bottom)
                hsv[:, :10] = (0, 0, 0)
                hsv[:, -10:] = (0, 0, 0)
                hsv[-10:, :] = (0, 0, 0)

            if not self.start and keyboard.is_pressed("s"):
                self.start = True

            if self.start:
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
                self.process_colors(hsv)

                self.GetAllBalls(frame)
                self.process_blobs()

            # Show Frame
            cv2.imshow("Test", frame)
            cv2.imshow("Screen Capture", hsv)

            if keyboard.is_pressed("r"):
                self.start_pos = None
                self.finish_pos = []
                self.frog_pos = None

            if keyboard.is_pressed("e"):
                sys.exit(0)

            cv2.setMouseCallback("Screen Capture", on_mouse)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def GetAllBalls(self, frame):
        if self.masks_cleaned:
            first_test = cv2.bitwise_or(
                self.masks_cleaned["red"], self.masks_cleaned["green"]
            )
            second_test = cv2.bitwise_or(
                self.masks_cleaned["blue"], self.masks_cleaned["yellow"]
            )
            all_balls = cv2.bitwise_or(first_test, second_test)

            if self.frog_pos:
                (frx, fry), frr = self.frog_pos
                cv2.circle(
                    all_balls,
                    center=(int(frx), int(fry)),
                    radius=int(frr),
                    color=0,
                    thickness=-1,  # filled circle
                )
            if self.finish_pos:
                for fpos in self.finish_pos:
                    (ffx, ffy), ffr = fpos
                    cv2.circle(
                        all_balls,
                        center=(int(ffx), int(ffy)),
                        radius=int(ffr),
                        color=0,
                        thickness=-1,  # filled circle
                    )

            # Convert to float32 for distance transform
            dist = cv2.distanceTransform(all_balls, cv2.DIST_L2, 5)

            ball_centers = self.local_maxima(dist, min_distance=30)
            for center in ball_centers:
                cv2.circle(
                    frame,
                    center=(int(center[0]), int(center[1])),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,  # filled circle
                )

            self.all_balls_mask = all_balls
            self.balls_centers = ball_centers

            cv2.imshow("all balls", all_balls)
            return ball_centers

    def RedBlobs(self):
        red_mask = cv2.bitwise_and(self.all_balls_mask, self.masks_cleaned["red"])
        red_centers = []
        for c in self.balls_centers:
            x, y = int(c[0]), int(c[1])
            if red_mask[y, x] > 0:
                red_centers.append(c)
        self.blobs["red"] = self.blob_center_and_count(red_mask, red_centers)

    def GreenBlobs(self):
        green_mask = cv2.bitwise_and(self.all_balls_mask, self.masks_cleaned["green"])
        green_centers = []
        for c in self.balls_centers:
            x, y = int(c[0]), int(c[1])
            if green_mask[y, x] > 0:
                green_centers.append(c)
        self.blobs["green"] = self.blob_center_and_count(green_mask, green_centers)

    def BlueBlobs(self):
        blue_mask = cv2.bitwise_and(self.all_balls_mask, self.masks_cleaned["blue"])
        blue_centers = []
        for c in self.balls_centers:
            x, y = int(c[0]), int(c[1])
            if blue_mask[y, x] > 0:
                blue_centers.append(c)
        self.blobs["blue"] = self.blob_center_and_count(blue_mask, blue_centers)

    def YellowBlobs(self):
        yellow_mask = cv2.bitwise_and(self.all_balls_mask, self.masks_cleaned["yellow"])
        yellow_centers = []
        for c in self.balls_centers:
            x, y = int(c[0]), int(c[1])
            if yellow_mask[y, x] > 0:
                yellow_centers.append(c)
        self.blobs["yellow"] = self.blob_center_and_count(yellow_mask, yellow_centers)

    def process_blobs(self):
        threads = [
            threading.Thread(target=self.RedBlobs),
            threading.Thread(target=self.GreenBlobs),
            threading.Thread(target=self.BlueBlobs),
            threading.Thread(target=self.YellowBlobs),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()
