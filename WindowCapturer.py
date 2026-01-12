import numpy as np
import time
import mss
import cv2


class RegionWindow:
    def __init__(self, left: int, top: int, width: int, height: int):
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)


class WindowCapturer:
    def __init__(self):
        self.sct = mss.mss()
        self._cached_window = None

    def _grab_bgr(self, monitor):
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def CaptureScreen(self):
        monitor = self.sct.monitors[0]
        return self._grab_bgr(monitor)

    def _circle_count_score(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return 0
        h, w = roi_bgr.shape[:2]
        if w < 260 or h < 220:
            return 0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        max_r = int(min(w, h) * 0.48)
        min_r = int(min(w, h) * 0.10)
        if max_r <= min_r:
            return 0

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(25, min(w, h) // 10),
            param1=140,
            param2=32,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return 0
        return int(circles.shape[1])

    def _border_density_score(self, edges_u8, rect, thickness=4):
        x, y, w, h = rect
        ih, iw = edges_u8.shape[:2]
        if w <= 0 or h <= 0:
            return 0.0

        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(iw, int(x + w))
        y1 = min(ih, int(y + h))
        if x1 - x0 < 2 * thickness + 10 or y1 - y0 < 2 * thickness + 10:
            return 0.0

        t = int(max(1, thickness))
        top = edges_u8[y0 : min(ih, y0 + t), x0:x1]
        bottom = edges_u8[max(0, y1 - t) : y1, x0:x1]
        left = edges_u8[y0:y1, x0 : min(iw, x0 + t)]
        right = edges_u8[y0:y1, max(0, x1 - t) : x1]

        def density(band):
            if band.size == 0:
                return 0.0
            return float(cv2.countNonZero(band)) / float(band.size)

        return (density(top) + density(bottom) + density(left) + density(right)) / 4.0

    def _detect_zuma_window_rect(self, screen_bgr):
        result = self._detect_zuma_window_result(screen_bgr)
        if result is None:
            return None
        return result["rect"]

    def _detect_zuma_window_result(self, screen_bgr):
        h, w = screen_bgr.shape[:2]
        gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30_000:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 320 or ch < 260:
                continue
            if cw > int(w * 0.90) or ch > int(h * 0.90):
                continue

            ar = cw / float(ch)
            if ar < 1.10 or ar > 1.85:
                continue

            extent = float(area) / float(cw * ch + 1e-6)
            if extent < 0.20:
                continue

            candidates.append((x, y, cw, ch, float(area)))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[4], reverse=True)
        candidates = candidates[:35]

        best = None
        for x, y, cw, ch, area in candidates:
            ar = cw / float(ch)
            ar_center = 1.22
            ar_sigma = 0.22
            ar_score = float(np.exp(-((ar - ar_center) / ar_sigma) ** 2))

            title_h = min(70, max(0, ch // 6))
            pad = 10
            inner = screen_bgr[
                y + title_h : max(y + title_h, y + ch - pad),
                x + pad : max(x + pad, x + cw - pad),
            ]

            circle_score = self._circle_count_score(inner)

            inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if inner.size else None
            texture = float(cv2.Laplacian(inner_gray, cv2.CV_64F).var()) if inner_gray is not None else 0.0

            border_density = self._border_density_score(edges, (x, y, cw, ch), thickness=4)

            score = (
                border_density * 140.0
                + ar_score * 30.0
                + circle_score * 18.0
                + min(texture, 2500.0) * 0.01
                + min(area, 700_000.0) * 0.000001
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "rect": (int(x), int(y), int(cw), int(ch)),
                    "border_density": float(border_density),
                    "ar": float(ar),
                    "ar_score": float(ar_score),
                    "circle_score": int(circle_score),
                }

        if best is None:
            return None

        if best["score"] < 35.0:
            return None

        best["candidates"] = int(len(candidates))
        return best


    def GetWindow(self, title):
        if self._cached_window is not None:
            return self._cached_window

        screen = self.CaptureScreen()
        rect = self._detect_zuma_window_rect(screen)
        if rect is None:
            raise RuntimeError("Zuma window not found")

        x, y, w, h = rect
        self._cached_window = RegionWindow(x, y, w, h)
        return self._cached_window

    def WaitForWindow(self, title, interval_seconds=1.0):
        attempts = 0
        start_time = time.time()
        last_log_time = 0.0
        last_best = None
        while True:
            try:
                return self.GetWindow(title)
            except Exception:
                attempts += 1
                try:
                    screen = self.CaptureScreen()
                    last_best = self._detect_zuma_window_result(screen)
                except Exception:
                    last_best = None

                now = time.time()
                if now - last_log_time >= 1.0:
                    elapsed = now - start_time
                    if last_best is None:
                        print(
                            f"[WindowCapturer] waiting... attempts={attempts} elapsed={elapsed:.1f}s best=None"
                        )
                    else:
                        rect = last_best["rect"]
                        score = float(last_best["score"])
                        candidates = int(last_best.get("candidates", 0))
                        border_density = float(last_best.get("border_density", 0.0))
                        ar = float(last_best.get("ar", 0.0))
                        circle_score = int(last_best.get("circle_score", 0))
                        print(
                            f"[WindowCapturer] waiting... attempts={attempts} elapsed={elapsed:.1f}s "
                            f"candidates={candidates} best_score={score:.1f} "
                            f"border={border_density:.3f} ar={ar:.3f} circles={circle_score} best_rect={rect}"
                        )
                    last_log_time = now
                time.sleep(float(interval_seconds))

    def CaptureWindow(self, window):
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
        }

        # Skip invalid states (minimized, zero size)
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            time.sleep(0.05)

        return self._grab_bgr(monitor)
