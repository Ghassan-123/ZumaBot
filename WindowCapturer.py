import pygetwindow as gw
import numpy as np
import pyautogui
import keyboard
import time
import mss
import cv2


class WindowCapturer:
    def __init__(self):
        self.sct = mss.mss()

    def GetWindow(self, title):
        windows = gw.getAllTitles()
        print("windows", windows)

        windows = gw.getWindowsWithTitle(title)
        if not windows:
            raise RuntimeError(f"Window '{title}' not found")
        return windows[0]

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

        # Capture screen
        screenshot = self.sct.grab(monitor)

        # Convert to NumPy array
        frame = np.array(screenshot)

        # Convert BGRA → BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # return the captured screen
        return frame