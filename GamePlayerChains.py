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
        self.test_mask = None

        self.balls = {}  # id -> Ball
        self.next_ball_id = 0

        self.MATCH_DIST = 30  # px (tune)
        self.MAX_MISSING_TIME = 0.4  # seconds

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
                    self.frog_pos = ((frog_cx, frog_cy), frog_radius + 10)

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

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        self.masks_row["red"] = red_mask

        # temp = np.zeros_like(red_mask)
        # contours, _ = cv2.findContours(
        #     red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # if contours:
        #     for contour in contours:
        #         (cx, cy), _ = cv2.minEnclosingCircle(contour)
        #         area = cv2.contourArea(contour)
        #         if 50 < area < 450:
        #             cv2.circle(temp, (int(cx), int(cy)), 10, (255, 255, 255), -1)
        # self.masks_cleaned["red"] = temp
        # cv2.imshow("red", temp)

    def GetGreen(self, hsv):
        lower_green = (50, 160, 165)
        upper_green = (70, 230, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        self.masks_row["green"] = green_mask

        # temp = np.zeros_like(green_mask)
        # contours, _ = cv2.findContours(
        #     green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # if contours:
        #     for contour in contours:
        #         (cx, cy), _ = cv2.minEnclosingCircle(contour)
        #         area = cv2.contourArea(contour)
        #         if 100 < area < 500:
        #             cv2.circle(temp, (int(cx), int(cy)), 10, (255, 255, 255), -1)
        # self.masks_cleaned["green"] = temp
        # cv2.imshow("green", temp)

    def GetBlue(self, hsv):
        lower_blue = (95, 180, 190)
        upper_blue = (130, 230, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        self.masks_row["blue"] = blue_mask  # mask_blue

        # temp = np.zeros_like(blue_mask)
        # contours, _ = cv2.findContours(
        #     blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # if contours:
        #     for contour in contours:
        #         (cx, cy), radius = cv2.minEnclosingCircle(contour)
        #         area = cv2.contourArea(contour)
        #         if 40 < area < 600:
        #             cv2.circle(temp, (int(cx), int(cy)), 10, (255, 255, 255), -1)
        # self.masks_cleaned["blue"] = temp
        # cv2.imshow("blue", temp)

    def GetYellow(self, hsv):
        lower_yellow = (20, 160, 200)
        upper_yellow = (30, 230, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        self.masks_row["yellow"] = yellow_mask

        # temp = np.zeros_like(yellow_mask)
        # contours, _ = cv2.findContours(
        #     yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # if contours:
        #     for contour in contours:
        #         (cx, cy), _ = cv2.minEnclosingCircle(contour)
        #         area = cv2.contourArea(contour)
        #         if 50 < area < 500:
        #             cv2.circle(temp, (int(cx), int(cy)), 10, (255, 255, 255), -1)
        # self.masks_cleaned["yellow"] = temp
        # cv2.imshow("yellow", temp)

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
            if self.aspect_ratio == 1.6:
                hsv[:64, :] = (0, 0, 0)
            else:
                hsv[:50, :] = (0, 0, 0)

            if not start and keyboard.is_pressed("s"):
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
                self.process_colors(hsv)

                self.GetAllBalls(frame)
                self.GetCurrentPlayBall()

                detections = self.DetectBalls()
                self.UpdateBallTracks(detections)
                chains = self.GetSortedBallsPerChain()
                # frame = frame[100:400, 100:500]

                for chain_id, chain in enumerate(chains):
                    for order, ball in enumerate(chain):
                        x, y = ball.center.astype(int)
                        cv2.putText(
                            frame,
                            f"C{chain_id}:{order}",
                            (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            2,
                        )
                cv2.imshow("Test", frame)

            # Show Frame
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

    def GetAllBalls(self, frame):
        if self.masks_row:
            # if self.masks_cleaned:
            # first_half = cv2.bitwise_or(
            #     self.masks_cleaned["red"], self.masks_cleaned["green"]
            # )
            # second_half = cv2.bitwise_or(
            #     self.masks_cleaned["blue"], self.masks_cleaned["yellow"]
            # )
            # all_balls = cv2.bitwise_or(first_half, second_half)

            first_test = cv2.bitwise_or(self.masks_row["red"], self.masks_row["green"])
            second_test = cv2.bitwise_or(
                self.masks_row["blue"], self.masks_row["yellow"]
            )
            all_balls = cv2.bitwise_or(first_test, second_test)

            radius = 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            all_balls = cv2.morphologyEx(
                all_balls, cv2.MORPH_CLOSE, kernel, iterations=1
            )

            radius = 4
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )  # adjust size for your balls
            dilated_mask = cv2.dilate(all_balls, kernel, iterations=1)

            radius = 3
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            dilated_mask = cv2.morphologyEx(
                dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=1
            )

            contours, _ = cv2.findContours(
                dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Convert to float32 for distance transform
            dist = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            ball_centers = self.local_maxima(dist, min_distance=30)
            for center in ball_centers:
                cv2.circle(
                    frame,
                    center=(int(center[0]), int(center[1])),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,  # filled circle
                )
            cv2.imshow("testing",frame)

            if self.frog_pos:
                (frx, fry), frr = self.frog_pos
                cv2.circle(
                    all_balls,
                    center=(int(frx), int(fry)),
                    radius=int(frr) + 1,
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
            self.all_balls_mask = all_balls
            # self.test_mask = test_balls

            cv2.imshow("all balls", dilated_mask)
            # cv2.imshow("test balls", test_balls)

    def DetectBalls(self):
        if self.all_balls_mask is None:
            return []

        detections = []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.all_balls_mask, connectivity=8
        )

        indices = sorted(
            range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True
        )

        min_dist_sq = 25 * 25

        for i in indices:
            area = stats[i, cv2.CC_STAT_AREA]
            if 50 < area < 500:
                cx, cy = centroids[i]
                candidate = np.array([cx, cy])

                if all(np.sum((candidate - d) ** 2) >= min_dist_sq for d in detections):
                    detections.append(candidate)

        return detections

    def GetCurrentPlayBall(self):
        if self.masks_row and self.frog_pos:
            for key, mask in self.masks_row.items():
                (cx, cy), radius = self.frog_pos

                roi_mask = mask[
                    int(cy - radius) : int(cy + radius),
                    int(cx - radius) : int(cx + radius),
                ]

                contours, _ = cv2.findContours(
                    roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    for contour in contours:
                        area = cv2.contourArea(contour)

                        if area > 200:
                            self.current_ball = key
                        elif 5 < area < 50:
                            self.second_color = key
                        else:
                            pass

    def UpdateBallTracks(self, detections):
        assigned = set()

        for ball in self.balls.values():
            best_dist = float("inf")
            best_idx = None

            for i, center in enumerate(detections):
                if i in assigned:
                    continue

                dist = np.linalg.norm(ball.center - center)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist < self.MATCH_DIST:
                ball.center = detections[best_idx]
                ball.last_seen = time.time()
                assigned.add(best_idx)

        for i, center in enumerate(detections):
            if i not in assigned:
                self.balls[self.next_ball_id] = Ball(
                    self.next_ball_id, center, color=None
                )
                self.next_ball_id += 1

        now = time.time()
        self.balls = {
            k: v
            for k, v in self.balls.items()
            if now - v.last_seen < self.MAX_MISSING_TIME
        }

    def BuildAdjacency(self):
        adj = defaultdict(list)
        balls = list(self.balls.values())

        # Precompute distances
        for ball in balls:
            neighbors = []

            for other in balls:
                if ball.id == other.id:
                    continue

                d = np.linalg.norm(ball.center - other.center)

                if d < self.MATCH_DIST * 1.5:
                    neighbors.append((d, other.id))

            # Sort neighbors by distance (closest first)
            neighbors.sort(key=lambda x: x[0])

            # Keep ONLY the closest 2
            for _, nid in neighbors[:2]:
                adj[ball.id].append(nid)

        # Ensure symmetry (undirected graph)
        clean_adj = defaultdict(list)
        for a in adj:
            for b in adj[a]:
                if a not in clean_adj[b]:
                    clean_adj[b].append(a)
                if b not in clean_adj[a]:
                    clean_adj[a].append(b)

        return clean_adj

    def GetChains(self, adj):
        visited = set()
        chains = []

        for ball_id in adj:
            if ball_id in visited:
                continue

            stack = [ball_id]
            component = set()

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.add(cur)
                for n in adj[cur]:
                    if n not in visited:
                        stack.append(n)

            chains.append(component)

        return chains

    def FindLeadingBall(self, chain, adj):
        ends = []
        # x_range = (100, 500)
        # y_range = (100, 400)
        x_range = None
        y_range = None

        candidates = []

        for bid in chain:
            # only loose ends
            if len(adj[bid]) != 1:
                continue

            ball = self.balls.get(bid)
            if ball is None:
                continue

            cx, cy = ball.center

            # check position ranges
            if x_range and not (x_range[0] <= cx <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= cy <= y_range[1]):
                continue

            candidates.append(bid)

        if not candidates:
            return None  # no valid loose end

        # if reference point given, pick closest to it
        if self.frog_pos:
            (rx, ry), _ = self.frog_pos
            closest = min(
                candidates,
                key=lambda bid: np.linalg.norm(
                    self.balls[bid].center - np.array([rx, ry])
                ),
            )
            return closest

        # otherwise, just return first candidate
        return candidates[0]

    def OrderChain(self, start_id, adj):
        ordered = []
        visited = set()

        current = start_id
        prev = None

        while True:
            ordered.append(current)
            visited.add(current)

            # choose only unvisited neighbors
            next_nodes = [n for n in adj[current] if n != prev and n not in visited]

            if not next_nodes:
                break

            prev = current
            current = next_nodes[0]

        return ordered

    def GetSortedBallsPerChain(self):
        adj = self.BuildAdjacency()
        chains = self.GetChains(adj)

        result = []  # list of ordered chains

        for chain in chains:
            lead = self.FindLeadingBall(chain, adj)
            if lead is None:
                continue

            ordered_ids = self.OrderChain(lead, adj)
            ordered_balls = [self.balls[i] for i in ordered_ids]
            result.append(ordered_balls)

        return result

    def local_maxima(self, dist, min_distance=20):
        centers = []

        # Pad the distance map to avoid boundary issues
        padded = np.pad(
            dist, pad_width=min_distance, mode="constant", constant_values=0
        )

        while True:
            # Find the maximum
            _, max_val, _, max_loc = cv2.minMaxLoc(padded)
            if max_val <= 0:
                break  # no more peaks

            # Convert to original coordinates
            y, x = max_loc[1] - min_distance, max_loc[0] - min_distance
            centers.append((x, y))

            # Zero out a circular region around this peak to enforce min_distance
            cv2.circle(padded, max_loc, min_distance, 0, -1)

        return centers


# def DetectFrogSift(self, frame):
#     # 1. Setup
#     template_gray = cv2.cvtColor(self.frog_template, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 2. Detect and Compute
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(template_gray, None)
#     kp2, des2 = sift.detectAndCompute(frame_gray, None)

#     # 3. Match using BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
#     matches = bf.knnMatch(des1, des2, k=2)

#     # 4. Filter good matches
#     # Lowe’s ratio test
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append(m)

#     # 5. Draw Matches
#     res = cv2.drawMatches(
#         template_gray,
#         kp1,
#         frame_gray,
#         kp2,
#         good,
#         None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
#     )

#     cv2.imshow("Final Detection", res)

# # need payment and rebuilding open cv
# def DetectFrogSurf(self, frame):
#     # 1. Setup
#     template = cv2.imread("frog_template.png")
#     template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 2. Detect and Compute using SURF
#     # 400 is the Hessian Threshold (higher = fewer, but stronger features)
#     surf = cv2.xfeatures2d.SURF_create(400)

#     kp1, des1 = surf.detectAndCompute(template_gray, None)
#     kp2, des2 = surf.detectAndCompute(frame_gray, None)

#     # 3. Match using BFMatcher
#     # SURF descriptors are floating point, so use NORM_L2
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(des1, des2)

#     # 4. Sort and Filter
#     matches = sorted(matches, key=lambda x: x.distance)
#     good = [m for m in matches if m.distance < 0.1]  # SURF distances vary from SIFT

#     # 5. Draw
#     res = cv2.drawMatches(
#         template_gray,
#         kp1,
#         frame_gray,
#         kp2,
#         good,
#         None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
#     )

#     cv2.imshow("SURF Detection", res)

# def DetectFrogOrb(self, frame):
#     # 1. Setup
#     template_gray = cv2.cvtColor(self.frog_template, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 2. Initialize ORB detector
#     orb = cv2.ORB_create()

#     # 3. Detect and Compute
#     kp1, des1 = orb.detectAndCompute(template_gray, None)
#     kp2, des2 = orb.detectAndCompute(frame_gray, None)

#     # Check if descriptors were found to avoid crashing the matcher
#     if des1 is None or des2 is None:
#         return

#     # 4. Match using BFMatcher
#     # IMPORTANT: Use NORM_HAMMING for ORB
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#     matches = bf.knnMatch(des1, des2, k=2)

#     # 5. Filter good matches
#     good = []
#     for m, n in matches:
#         if m.distance < 0.5 * n.distance:
#             good.append(m)

#     # 6. Draw
#     res = cv2.drawMatches(
#         template_gray,
#         kp1,
#         frame_gray,
#         kp2,
#         good,
#         None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
#     )

#     cv2.imshow("ORB Detection", res)

#     # 7. Localize the Frog (Homography)
#     # We need at least 4 matches to find a homography matrix
#     if len(good) > 4:
#         # Extract location of good matches
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#         # Find the transformation matrix
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#         if M is not None:
#             # Get dimensions of the template
#             h, w = template_gray.shape
#             # Define the corners of the template
#             pts = np.float32(
#                 [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
#             ).reshape(-1, 1, 2)

#             # Project corners into the frame to find the "contour"
#             dst = cv2.perspectiveTransform(pts, M)

#             # 8. Calculate Minimum Enclosing Circle
#             # dst contains the 4 corners of the detected object in the frame
#             (x, y), radius = cv2.minEnclosingCircle(dst)
#             center = (int(x), int(y))
#             radius = int(radius)

#             # 9. Draw Results
#             # Draw the contour (the bounding box)
#             frame = cv2.polylines(
#                 frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA
#             )
#             # Draw the circle
#             cv2.circle(frame, center, radius, (255, 0, 255), 2)

#             # print(f"Frog found at {center} with radius {radius}")

#     cv2.imshow("Detection with Circle", frame)
