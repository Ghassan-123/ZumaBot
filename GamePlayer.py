from WindowCapturer import WindowCapturer
import numpy as np
import pyautogui
import keyboard
import time
import sys
import cv2


class GamePlayer:
    def __init__(self):
        self.start_pos = None
        self.finish_pos = []
        self.frog_pos = None
        self.WindowCapturer = WindowCapturer()
        self.frog_template = cv2.imread("frog_template.png")

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

    def GetFrogPos(self, hsv, area_threshold=1200):
        if self.frog_pos is None:
            frog_mask = cv2.inRange(
                hsv, np.array([15, 15, 200]), np.array([20, 45, 255])
            )

            frog_mask = cv2.dilate(frog_mask, (7, 7), iterations=10)
            cv2.imshow("frog", frog_mask)

            frog_contours, _ = cv2.findContours(
                frog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if frog_contours:
                # Get the largest contour by area
                frog_contour = max(frog_contours, key=cv2.contourArea)
                area = cv2.contourArea(frog_contour)
                print(area)

                if (area_threshold - 400) < area < (area_threshold + 400):
                    (frog_cx, frog_cy), frog_radius = cv2.minEnclosingCircle(
                        frog_contour
                    )
                    self.frog_pos = ((frog_cx, frog_cy), frog_radius + 10)

                if self.frog_pos:
                    return self.frog_pos
        else:
            return self.frog_pos

    def DetectFrogSift(self, frame):
        # 1. Setup
        template_gray = cv2.cvtColor(self.frog_template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect and Compute
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(frame_gray, None)

        # 3. Match using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # 4. Filter good matches
        # Lowe’s ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

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

    def DetectFrogOrb(self, frame):
        # 1. Setup
        template_gray = cv2.cvtColor(self.frog_template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Initialize ORB detector
        orb = cv2.ORB_create()

        # 3. Detect and Compute
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)

        # Check if descriptors were found to avoid crashing the matcher
        if des1 is None or des2 is None:
            return

        # 4. Match using BFMatcher
        # IMPORTANT: Use NORM_HAMMING for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # 5. Filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

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

        # 7. Localize the Frog (Homography)
        # We need at least 4 matches to find a homography matrix
        if len(good) > 4:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Find the transformation matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Get dimensions of the template
                h, w = template_gray.shape
                # Define the corners of the template
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)

                # Project corners into the frame to find the "contour"
                dst = cv2.perspectiveTransform(pts, M)

                # 8. Calculate Minimum Enclosing Circle
                # dst contains the 4 corners of the detected object in the frame
                (x, y), radius = cv2.minEnclosingCircle(dst)
                center = (int(x), int(y))
                radius = int(radius)

                # 9. Draw Results
                # Draw the contour (the bounding box)
                frame = cv2.polylines(
                    frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA
                )
                # Draw the circle
                cv2.circle(frame, center, radius, (255, 0, 255), 2)

                print(f"Frog found at {center} with radius {radius}")

        cv2.imshow("Detection with Circle", frame)

    def RunLoop(self):
        #  Start timer
        prev_time = time.time()

        # Get game window
        window = self.WindowCapturer.GetWindow("Zuma Deluxe 1.1.0.0")

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print((y, x))
                print(hsv[y, x])

        start = False

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

            if keyboard.is_pressed("s"):
                start = True

            if start:
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

            # Show Frame
            cv2.imshow("Test", frame)
            cv2.imshow("Screen Capture", hsv)
            # self.DetectFrogSift(frame)
            # self.DetectFrogSurf(frame)
            # self.DetectFrogOrb(frame)

            if keyboard.is_pressed("r"):
                self.start_pos = None
                self.finish_pos = []
                self.frog_pos = None

            if keyboard.is_pressed("e"):
                sys.exit(0)

            # if keyboard.is_pressed("p"):
            #     win_x = 30
            #     win_y = 60

            #     # Convert window-relative to screen coordinates
            #     screen_x = window.left + win_x
            #     screen_y = window.top + win_y

            #     # Move mouse
            #     pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            #     # Click
            #     pyautogui.click()

            #     # Small delay
            #     time.sleep(0.1)

            cv2.setMouseCallback("Screen Capture", on_mouse)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


gameplayer = GamePlayer()
gameplayer.RunLoop()
