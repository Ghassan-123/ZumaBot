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
        self.can_shoot_duration = 1.8  # seconds

        self.kernels = {
            1: np.ones((3, 3), np.uint8),
            2: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            3: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            4: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        }

        self.frame_id = 0

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
        self.GetRed(hsv)
        self.GetGreen(hsv)
        self.GetBlue(hsv)
        self.GetYellow(hsv)

    # def process_colors(self, hsv):
    #     threads = [
    #         threading.Thread(target=self.GetRed, args=(hsv,)),
    #         threading.Thread(target=self.GetGreen, args=(hsv,)),
    #         threading.Thread(target=self.GetBlue, args=(hsv,)),
    #         threading.Thread(target=self.GetYellow, args=(hsv,)),
    #     ]

    #     for t in threads:
    #         t.start()
    #     for t in threads:
    #         t.join()

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
            self.frame_id += 1

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

                if self.frame_id % 3 == 0:
                    self.frame_id %= 3
                    detections = self.GetAllBalls(frame)
                    self.process_blobs()

                    all_chains = self.order_balls_by_finishes(detections)
                    self.all_chains = all_chains
                    # self.danger_centers = self.get_danger_centers(all_chains, limit=7)
                    # self.danger_keys = {self.center_key(c) for c in self.danger_centers}
                    for idx, chain in enumerate(all_chains):
                        for order, center in enumerate(chain):
                            cv2.putText(
                                frame,
                                f"C{idx}:{order}",
                                (int(center[0]) + 6, int(center[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.25,
                                (0, 255, 255),
                                2,
                            )

                    for key, val in self.blobs.items():
                        if key == "red":
                            color = (0, 0, 255)
                        elif key == "green":
                            color = (0, 255, 0)
                        elif key == "blue":
                            color = (255, 0, 0)
                        elif key == "yellow":
                            color = (0, 255, 255)
                        else:
                            color = (255, 255, 255)

                        for center, count, _ in val:
                            cv2.putText(
                                frame,
                                f"{count}",
                                (int(center[0]) + 12, int(center[1]) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                    self.GetCurrentPlayBall()

                    self.choose_ball_play(detections)
                    self.reset_shooting()

                # print(f"current_ball: {self.current_ball}")
                # print(f"second_ball: {self.second_color}")

                # cv2.imshow("Red", self.masks_cleaned["red"])
                # cv2.imshow("Green", self.masks_cleaned["green"])
                # cv2.imshow("Blue", self.masks_cleaned["blue"])
                # cv2.imshow("Yellow", self.masks_cleaned["yellow"])

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
                    radius=int(frr) + 2,
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
                    color=(255, 255, 255),
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
        self.RedBlobs()
        self.GreenBlobs()
        self.BlueBlobs()
        self.YellowBlobs()

    # def process_blobs(self):
    #     threads = [
    #         threading.Thread(target=self.RedBlobs),
    #         threading.Thread(target=self.GreenBlobs),
    #         threading.Thread(target=self.BlueBlobs),
    #         threading.Thread(target=self.YellowBlobs),
    #     ]

    #     for t in threads:
    #         t.start()
    #     for t in threads:
    #         t.join()

    def GetCurrentPlayBall(self):
        if self.masks_cleaned and self.frog_pos:
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
                        # print("current ball area:", area)

                        if 480 > area > 120:
                            self.current_ball = key
                        elif area > 20:
                            self.second_color = key
                        else:
                            pass

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
            centers.append(np.array([x, y]))

            # Zero out a circular region around this peak to enforce min_distance
            cv2.circle(padded, max_loc, min_distance, 0, -1)

        return centers

    def order_balls_by_finishes(self, detections):
        """
        detections: list of ball centers (np.array)
        returns: list of chains (each chain is a list of centers)
        """
        if not self.finish_pos:
            return []

        chains = []
        remaining = detections.copy()  # balls that haven't been assigned yet

        for finish in self.finish_pos:
            (fx, fy), _ = finish
            start_point = np.array([fx, fy])

            chain = []
            if not remaining:
                chains.append(chain)
                continue

            # Start with the ball closest to this finish
            dists = [np.linalg.norm(c - start_point) for c in remaining]
            idx = np.argmin(dists)
            current = remaining.pop(idx)
            chain.append(current)

            # Build chain by nearest neighbor
            while remaining:
                dists = [np.linalg.norm(c - chain[-1]) for c in remaining]
                min_idx = np.argmin(dists)
                # Optional: stop chain if distance too big (avoid jumping chains)
                if dists[min_idx] > 80:  # adjust threshold based on your scale
                    break
                current = remaining.pop(min_idx)
                chain.append(current)

            chains.append(chain)

        # sort chains by length descending
        chains.sort(key=lambda c: len(c), reverse=True)
        return chains

    def blob_center_and_count(self, mask, centers):
        """
        mask: binary mask of all balls (single-channel, 0 or 255)
        centers: list of np.array([x, y]) detected from distance transform
        returns: list of tuples [(center, count), ...] per blob
                center: np.array([x, y]) representing the mean of centers in blob
                count: number of centers in that blob
        """
        num_labels, labels = cv2.connectedComponents(mask)
        blob_data = []

        for label in range(1, num_labels):  # skip background
            # Find all centers in this blob
            blob_centers = [c for c in centers if labels[int(c[1]), int(c[0])] == label]

            if blob_centers:
                blob_centers_np = np.array(blob_centers)
                mean_center = np.mean(blob_centers_np, axis=0)  # average x and y
                count = len(blob_centers)
                blob_data.append((mean_center, count, blob_centers))
            else:
                pass
                # No detected centers inside this blob
                # Optionally, you can use the blob's centroid from connectedComponents stats
                # blob_data.append((None, 0))

        return blob_data

    def choose_ball_play(self, balls_centers):
        valid_blobs = self.blobs.get(self.current_ball, [])
        if not valid_blobs:
            return

        scored_blobs = []

        for blob in valid_blobs:
            target_center, count, in_centers = blob

            # --- BASE SCORE ---
            score = count

            # --- BRIDGE-MERGE ---
            left_color, right_color = self.get_blob_side_colors(
                in_centers, self.all_chains
            )

            if count >= 2 and left_color is not None and left_color == right_color:
                score += 10  # strong priority boost

            # --- PATH CLEAR CHECK ---
            filtered_centers = [
                c
                for c in balls_centers
                if not any(np.allclose(c, ic, atol=5.0) for ic in in_centers)
            ]

            if not self.is_path_clear(target_center, filtered_centers):
                continue

            scored_blobs.append((score, blob))

        if not scored_blobs:
            return

        # --- PICK BEST SCORING SHOT ---
        scored_blobs.sort(key=lambda x: x[0], reverse=True)
        best_score, best_blob = scored_blobs[0]

        target_center, _, _ = best_blob

        if not self.can_shoot:
            return

        print(f"🔥 Clear shot! {self.current_ball} | score={best_score}")

        win_x, win_y = target_center

        screen_x = self.window.left + win_x
        screen_y = self.window.top + win_y

        pyautogui.moveTo(screen_x, screen_y)
        pyautogui.click()

        # Rotate balls
        self.current_ball = self.second_color
        self.second_color = None

        # Reset shooting cooldown
        self.can_shoot = False
        self.can_shoot_time = time.time()

    def is_path_clear(self, target_center, all_centers, radius=20):
        """
        Checks if the path from the frog to target is blocked by any other balls.

        target_center: np.array([x, y])
        all_blob_centers: list of np.array([x, y])
        radius: allowed sideways tolerance (how far a ball can be from the line)
        """
        if self.frog_pos:
            (fx, fy), _ = self.frog_pos
            frog = np.array([fx, fy], dtype=np.float32)
            target = np.array(target_center, dtype=np.float32)

            shot_vec = target - frog
            shot_len = np.linalg.norm(shot_vec)
            if shot_len < 1e-5:
                return False

            shot_dir = shot_vec / shot_len

            # cv2.line(
            #     frame,
            #     (int(frog[0]), int(frog[1])),
            #     (int(target_center[0]), int(target_center[1])),
            #     (0, 255, 0),
            #     1,
            # )
            for c in all_centers:
                c = np.array(c, dtype=np.float32)

                # Skip the target itself
                if np.allclose(c, target, atol=5.0):
                    continue

                to_blob = c - frog
                proj = np.dot(to_blob, shot_dir)

                # Only balls strictly between frog and target
                if proj <= 0 or proj >= shot_len:
                    continue

                # Distance perpendicular to shot line
                perp_dist = np.linalg.norm(to_blob - proj * shot_dir)

                # If ball is close enough to line, it blocks the shot
                if perp_dist <= radius:
                    return False  # blocked

            return True  # clear
        return False

    def reset_shooting(self):
        if not self.can_shoot:
            if time.time() - self.can_shoot_time >= self.can_shoot_duration:
                self.can_shoot = True

    def find_blob_range_in_chain(self, blob_centers, chain, max_dist=30):
        if not chain or not blob_centers:
            return None
        indices = []

        for bc in blob_centers:
            dists = [np.linalg.norm(c - bc) for c in chain]
            if not dists:
                continue

            idx = np.argmin(dists)
            if dists[idx] <= max_dist:
                indices.append(idx)

        if not indices:
            return None

        return min(indices), max(indices)

    def get_blob_side_colors(self, blob_centers, chains):
        for chain in chains:
            rng = self.find_blob_range_in_chain(blob_centers, chain)
            if rng is None:
                continue

            start_idx, end_idx = rng

            left_color = None
            right_color = None

            if start_idx > 0:
                left_color = self.get_ball_color_at(chain[start_idx - 1])
            if end_idx < len(chain) - 1:
                right_color = self.get_ball_color_at(chain[end_idx + 1])

            return left_color, right_color

        return None, None

    def get_ball_color_at(self, center):
        """
        Returns the color name ('red', 'green', 'blue', 'yellow')
        of the ball at the given center position.
        """
        x, y = int(center[0]), int(center[1])

        for color, mask in self.masks_cleaned.items():
            # Defensive bounds check
            if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
                continue

            if mask[y, x] > 0:
                return color

        return None

    # def get_danger_centers(self, chains, limit=7):
    #     """
    #     Returns a set of ball centers that are within the first `limit`
    #     balls of each chain (closest to finish).
    #     """
    #     danger = []

    #     for chain in chains:
    #         if not chain:
    #             continue
    #         danger.extend(chain[:limit])

    #     return danger

    # def center_key(self, c, q=5):
    #     return (int(c[0] // q), int(c[1] // q))

    # def blob_in_danger_zone(self, blob_centers):
    #     for bc in blob_centers:
    #         if self.center_key(bc) in self.danger_keys:
    #             return True
    #     return False

    # def order_balls_nearest(self, centers, start_point):
    #     centers = centers.copy()
    #     ordered = []

    #     # Find the ball closest to start_point
    #     dists = [np.linalg.norm(c - start_point) for c in centers]
    #     idx = np.argmin(dists)
    #     current = centers.pop(idx)
    #     ordered.append(current)

    #     while centers:
    #         # Find the ball closest to the last in ordered
    #         dists = [np.linalg.norm(c - ordered[-1]) for c in centers]
    #         idx = np.argmin(dists)
    #         current = centers.pop(idx)
    #         ordered.append(current)

    #     return ordered


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
