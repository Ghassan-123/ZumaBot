                    # self.danger_centers = self.get_danger_centers(all_chains, limit=7)
                    # self.danger_keys = {self.center_key(c) for c in self.danger_centers}
    
    
    
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
