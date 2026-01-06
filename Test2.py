# import cv2
# import numpy as np
# from scipy.spatial import cKDTree

# # ------------------------------------------------
# # Skeletonization (robust)
# # ------------------------------------------------
# def skeletonize(img_bin):
#     skel = np.zeros_like(img_bin)
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
#     while True:
#         eroded = cv2.erode(img_bin, kernel)
#         temp = cv2.dilate(eroded, kernel)
#         temp = cv2.subtract(img_bin, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img_bin = eroded.copy()
#         if cv2.countNonZero(img_bin) == 0:
#             break
#     return skel

# # ------------------------------------------------
# # Count neighbors
# # ------------------------------------------------
# def neighbors(img, y, x):
#     h, w = img.shape
#     out = []
#     for dy in (-1,0,1):
#         for dx in (-1,0,1):
#             if dy == 0 and dx == 0:
#                 continue
#             ny, nx = y+dy, x+dx
#             if 0 <= ny < h and 0 <= nx < w:
#                 if img[ny, nx] > 0:
#                     out.append((ny, nx))
#     return out

# # ------------------------------------------------
# # Find endpoints
# # ------------------------------------------------
# def find_endpoints(skel):
#     ys, xs = np.where(skel > 0)
#     endpoints = []
#     for y, x in zip(ys, xs):
#         if len(neighbors(skel, y, x)) == 1:
#             endpoints.append((y, x))
#     return endpoints

# # ------------------------------------------------
# # AUTO PATH FOLLOW + GAP BRIDGING
# # ------------------------------------------------
# def extract_zuma_path(skel):
#     points = np.column_stack(np.where(skel > 0))
#     if len(points) == 0:
#         return []

#     tree = cKDTree(points)
#     visited = set()

#     # pick start automatically
#     endpoints = find_endpoints(skel)
#     start = endpoints[0] if endpoints else tuple(points[0])

#     path = [start]
#     visited.add(start)
#     current = start

#     while True:
#         # try pixel neighbors first
#         nbrs = neighbors(skel, current[0], current[1])
#         next_pixel = None
#         for p in nbrs:
#             if p not in visited:
#                 next_pixel = p
#                 break

#         if next_pixel:
#             path.append(next_pixel)
#             visited.add(next_pixel)
#             current = next_pixel
#             continue

#         # --- GAP BRIDGING ---
#         dists, idxs = tree.query(current, k=len(points))
#         bridged = False
#         for i in idxs:
#             p = tuple(points[i])
#             if p not in visited:
#                 path.append(p)
#                 visited.add(p)
#                 current = p
#                 bridged = True
#                 break

#         if not bridged:
#             break

#     return path

# # ------------------------------------------------
# # DRAW PATH
# # ------------------------------------------------
# def draw_path(img_shape, path):
#     canvas = np.zeros(img_shape, dtype=np.uint8)
#     for i in range(len(path)-1):
#         cv2.line(
#             canvas,
#             (path[i][1], path[i][0]),
#             (path[i+1][1], path[i+1][0]),
#             255,
#             1
#         )
#     return canvas

# # ------------------------------------------------
# # MAIN
# # ------------------------------------------------
# if __name__ == "__main__":
#     img = cv2.imread("test1.png")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)

#     edges = cv2.Canny(blur, 50, 150)
#     skeleton = skeletonize(edges)

#     path = extract_zuma_path(skeleton)
#     path_img = draw_path(skeleton.shape, path)

#     cv2.imshow("Edges", edges)
#     cv2.imshow("Skeleton", skeleton)
#     cv2.imshow("Zuma Path (FINAL)", path_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# ---------------------------------------------
# Skeletonize
# ---------------------------------------------
def skeletonize(img):
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while True:
        eroded = cv2.erode(img,kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        if cv2.countNonZero(img)==0:
            break
    return skel

# ---------------------------------------------
# Build graph from skeleton + gap bridging
# ---------------------------------------------
def build_graph(skel, max_bridge=20):
    pts = np.column_stack(np.where(skel>0))
    tree = cKDTree(pts)
    G = nx.Graph()

    for i,(y,x) in enumerate(pts):
        G.add_node(i,pos=(y,x))
        dists, idxs = tree.query((y,x), k=8)
        for d,j in zip(dists,idxs):
            if i!=j and d <= max_bridge:
                G.add_edge(i,j,weight=d)
    return G, pts

# ---------------------------------------------
# Extract longest path (TRUE Zuma path)
# ---------------------------------------------
def longest_path(G):
    # tree = minimum spanning tree removes loops
    T = nx.minimum_spanning_tree(G)
    nodes = list(T.nodes)

    # double BFS to find diameter
    a = nodes[0]
    b = max(nx.single_source_dijkstra_path_length(T,a), key=lambda x: nx.single_source_dijkstra_path_length(T,a)[x])
    c = max(nx.single_source_dijkstra_path_length(T,b), key=lambda x: nx.single_source_dijkstra_path_length(T,b)[x])

    return nx.shortest_path(T,b,c,weight='weight')

# ---------------------------------------------
# Draw path
# ---------------------------------------------
def draw_path(shape, pts, path):
    img = np.zeros(shape,dtype=np.uint8)
    for i in range(len(path)-1):
        y1,x1 = pts[path[i]]
        y2,x2 = pts[path[i+1]]
        cv2.line(img,(x1,y1),(x2,y2),255,1)
    return img

# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    img = cv2.imread("test1.png",0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    skel = skeletonize(edges)

    G, pts = build_graph(skel, max_bridge=25)
    path = longest_path(G)
    path_img = draw_path(skel.shape, pts, path)

    cv2.imshow("Skeleton", skel)
    cv2.imshow("FINAL ZUMA PATH", path_img)
    cv2.waitKey(0)




# while True:
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('s'):
#         start = True

#     if key == ord('q'):
#         break
