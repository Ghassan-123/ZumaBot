# import cv2
# import numpy as np

# # Read image
# img = cv2.imread("test5.png", cv2.IMREAD_GRAYSCALE)

# # Threshold to binary
# _, binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)

# # Invert if needed (foreground must be white)
# binary = cv2.bitwise_not(binary)

# # Skeletonization
# skeleton = np.zeros(binary.shape, np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# while True:
#     eroded = cv2.erode(binary, kernel)
#     temp = cv2.dilate(eroded, kernel)
#     temp = cv2.subtract(binary, temp)
#     skeleton = cv2.bitwise_or(skeleton, temp)
#     binary = eroded.copy()

#     if cv2.countNonZero(binary) == 0:
#         break

# cv2.imshow("Binary", binary)
# cv2.imshow("Skeleton", skeleton)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image in grayscale
# img = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)

# # Threshold to binary
# _, binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY_INV)

# # Create an empty skeleton
# skeleton = np.zeros_like(binary)

# # Structuring element
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# cv2.imshow("kernel", kernel)

# temp = np.zeros_like(binary)
# done = False

# # while not done:
# for i in range(10):
#     eroded = cv2.erode(binary, kernel)
#     opened = cv2.dilate(eroded, kernel)
#     diff = cv2.subtract(binary, opened)
#     skeleton = cv2.bitwise_or(skeleton, diff)
#     stop = eroded.copy()

#     if cv2.countNonZero(stop) == 0:
#         done = True

# # Show result
# cv2.imshow("diff", diff)
# cv2.imshow("binary", binary)
# cv2.imshow("Skeleton", skeleton)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# # Load stage image
# img = cv2.imread("test5.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Blur to reduce noise
# blur = cv2.GaussianBlur(gray, (5, 5), 0)

# # Edge detection
# edges = cv2.Canny(blur, 50, 150)

# # Create skeleton using morphological thinning
# skeleton = np.zeros_like(edges)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# img_bin = edges.copy()
# done = False

# while not done:
#     eroded = cv2.erode(img_bin, kernel)
#     temp = cv2.dilate(eroded, kernel)
#     temp = cv2.subtract(img_bin, temp)
#     skeleton = cv2.bitwise_or(skeleton, temp)
#     img_bin = eroded.copy()

#     if cv2.countNonZero(img_bin) == 0:
#         done = True

# points = np.column_stack(np.where(skeleton > 0))  # [y, x]
# from scipy.spatial import KDTree

# # Build tree
# tree = KDTree(points)
# ordered_path = [points[0]]  # start at first point

# for _ in range(len(points)-1):
#     last = ordered_path[-1]
#     dist, idx = tree.query(last, k=2)  # nearest neighbors
#     next_point = points[idx[1]]  # nearest unvisited
#     ordered_path.append(next_point)


# cv2.imshow("Edges", edges)
# cv2.imshow("Skeleton", skeleton)
# cv2.waitKey(0)


# import cv2
# import numpy as np

# # -----------------------------
# # Count 8-connected neighbors
# # -----------------------------
# def count_neighbors(img, y, x):
#     h, w = img.shape
#     count = 0
#     for dy in (-1, 0, 1):
#         for dx in (-1, 0, 1):
#             if dy == 0 and dx == 0:
#                 continue
#             ny, nx = y + dy, x + dx
#             if 0 <= ny < h and 0 <= nx < w:
#                 if img[ny, nx] > 0:
#                     count += 1
#     return count


# # -----------------------------
# # Find skeleton endpoints
# # -----------------------------
# def find_endpoints(skel):
#     endpoints = []
#     ys, xs = np.where(skel > 0)

#     for y, x in zip(ys, xs):
#         if count_neighbors(skel, y, x) == 1:
#             endpoints.append((y, x))

#     return endpoints


# # -----------------------------
# # Remove closed loops + short junk
# # -----------------------------
# def clean_skeleton(skeleton, min_length=300):
#     """
#     Keeps only open skeleton components (2 endpoints)
#     and removes short junk components.
#     """

#     skel = (skeleton > 0).astype(np.uint8) * 255
#     cleaned = np.zeros_like(skel)

#     num_labels, labels = cv2.connectedComponents(skel)

#     for label in range(1, num_labels):
#         component = (labels == label).astype(np.uint8) * 255

#         pixel_count = np.sum(component > 0)
#         if pixel_count < min_length:
#             continue

#         endpoints = find_endpoints(component)

#         # KEEP only open curves (real path)
#         if len(endpoints) >= 1:
#             cleaned[component > 0] = 255

#     return cleaned


# # -----------------------------
# # Example usage
# # -----------------------------
# if __name__ == "__main__":
#     # skeleton must be a binary image: 0 / 255
#     # skeleton = cv2.imread("test1.png", cv2.IMREAD_GRAYSCALE)

#     # Load stage image
#     img = cv2.imread("test1.png")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Blur to reduce noise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Edge detection
#     edges = cv2.Canny(blur, 50, 150)

#     # Create skelton using morphological thinning
#     skeleton = np.zeros_like(edges)
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

#     img_bin = edges.copy()
#     done = False

#     while not done:
#         eroded = cv2.erode(img_bin, kernel)
#         temp = cv2.dilate(eroded, kernel)
#         temp = cv2.subtract(img_bin, temp)
#         skeleton = cv2.bitwise_or(skeleton, temp)
#         img_bin = eroded.copy()

#         if cv2.countNonZero(img_bin) == 0:
#             done = True


#     # _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)

#     # # Invert if needed (foreground must be white)
#     # binary = cv2.bitwise_not(binary)

#     # # Skeletonization
#     # skeleton = np.zeros(binary.shape, np.uint8)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

#     # while True:
#     #     eroded = cv2.erode(binary, kernel)
#     #     temp = cv2.dilate(eroded, kernel)
#     #     temp = cv2.subtract(binary, temp)
#     #     skeleton = cv2.bitwise_or(skeleton, temp)
#     #     binary = eroded.copy()

#     #     if cv2.countNonZero(binary) == 0:
#     #         break


#     cleaned = clean_skeleton(skeleton, min_length=400)

#     cv2.imshow("Original Skeleton", skeleton)
#     cv2.imshow("Cleaned Skeleton", cleaned)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



import cv2
import numpy as np

# -----------------------------
# Count 8-connected neighbors
# -----------------------------
def count_neighbors(img, y, x):
    h, w = img.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if img[ny, nx] > 0:
                    count += 1
    return count

# -----------------------------
# Find skeleton endpoints
# -----------------------------
def find_endpoints(skel):
    endpoints = []
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        if count_neighbors(skel, y, x) == 1:
            endpoints.append((y, x))
    return endpoints

# -----------------------------
# Keep only skeleton connected to end
# -----------------------------
def keep_connected_to_end(skel, end_point):
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=np.uint8)
    queue = [end_point]
    while queue:
        y, x = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = 255
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skel[ny, nx] > 0 and visited[ny, nx] == 0:
                        queue.append((ny, nx))
    return visited

# -----------------------------
# Remove closed loops and short junk
# -----------------------------
def clean_skeleton(skeleton, min_length=300):
    skel = (skeleton > 0).astype(np.uint8) * 255
    cleaned = np.zeros_like(skel)

    num_labels, labels = cv2.connectedComponents(skel)

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        pixel_count = np.sum(component > 0)
        if pixel_count < min_length:
            continue
        endpoints = find_endpoints(component)
        if len(endpoints) >= 1:  # keep only open curves
            cleaned[component > 0] = 255
    return cleaned

# -----------------------------
# Create cost map from skeleton
# -----------------------------
def create_cost_map(skel):
    # invert skeleton for distance transform
    inv = 255 - skel
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    cost_map = 1 + dist  # linear cost; use np.exp(dist) for sharper penalties
    return cost_map

# -----------------------------
# Skeletonize using morphological thinning
# -----------------------------
def skeletonize(img_bin):
    skel = np.zeros_like(img_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img_bin, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_bin = eroded.copy()
        if cv2.countNonZero(img_bin) == 0:
            done = True
    return skel


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print((y, x))

# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    # Load stage image
    img = cv2.imread("test1.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Skeletonize edges
    skeleton = skeletonize(edges)

    # TODO: replace with actual skull/end point detection
    # For now, manually pick (y, x) as the end
    end_point = (152, 393)  # example coordinates; adjust for your image

    # Keep only skeleton connected to end
    skeleton_connected = keep_connected_to_end(skeleton, end_point)

    # Clean skeleton: remove loops and short junk
    cleaned = clean_skeleton(skeleton_connected, min_length=400)

    # Optional: dilate + re-skeletonize to bridge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    skeleton_final = skeletonize(dilated)

    # Create cost map for path-finding
    cost_map = create_cost_map(skeleton_final)

    # Visualization
    cv2.imshow("Original Skeleton", skeleton)
    cv2.imshow("Connected Skeleton", skeleton_connected)
    cv2.imshow("Cleaned Skeleton", cleaned)
    cv2.imshow("Final Skeleton", skeleton_final)
    cv2.imshow("Cost Map", cost_map / cost_map.max())  # normalize for display
    cv2.setMouseCallback("Original Skeleton", on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        

