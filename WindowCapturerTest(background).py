# import win32gui
# import win32ui
# import win32con
# import ctypes
# import numpy as np
# import cv2

# class BackgroundCapturer:
#     def __init__(self, window_title):
#         self.hwnd = win32gui.FindWindow(None, window_title)
#         if not self.hwnd:
#             raise RuntimeError(f"Window '{window_title}' not found")

#     def capture(self):
#         # Get window dimensions
#         left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
#         w = right - left
#         h = bot - top

#         # Create Device Contexts
#         hwndDC = win32gui.GetWindowDC(self.hwnd)
#         mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
#         saveDC = mfcDC.CreateCompatibleDC()

#         # Create Bitmap
#         saveBitMap = win32ui.CreateBitmap()
#         saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
#         saveDC.SelectObject(saveBitMap)

#         # FIX: Use ctypes to call PrintWindow directly from user32.dll
#         # 2 = PW_RENDERFULLCONTENT (captures even if obscured)
#         result = ctypes.windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)

#         # Convert to numpy array
#         signedIntsArray = saveBitMap.GetBitmapBits(True)
#         img = np.frombuffer(signedIntsArray, dtype='uint8')
#         img.shape = (h, w, 4)

#         # Cleanup
#         win32gui.DeleteObject(saveBitMap.GetHandle())
#         saveDC.DeleteDC()
#         mfcDC.DeleteDC()
#         win32gui.ReleaseDC(self.hwnd, hwndDC)

#         if result != 1:
#             return None # Or handle error: capture failed

#         return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# # Usage
# capturer = BackgroundCapturer("Zuma Deluxe 1.1.0.0")
# frame = capturer.capture()
# cv2.imshow("frame", frame)
# cv2.waitKey(0)



import win32gui
import win32ui
from ctypes import windll
from PIL import Image

hwnd = win32gui.FindWindow(None, "Zuma Deluxe 1.1.0.0")

# Uncomment the following line if you use a high DPI display or >100% scaling size
windll.user32.SetProcessDPIAware()

# Change the line below depending on whether you want the whole window
# or just the client area. 
left, top, right, bot = win32gui.GetClientRect(hwnd)
# left, top, right, bot = win32gui.GetWindowRect(hwnd)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()

saveBitMap = win32ui.CreateBitmap()
saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

saveDC.SelectObject(saveBitMap)

# Change the line below depending on whether you want the whole window
# or just the client area.
result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
# result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
print(result)

bmpinfo = saveBitMap.GetInfo()
bmpstr = saveBitMap.GetBitmapBits(True)

im = Image.frombuffer(
    "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1
)

win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(hwnd, hwndDC)

if result == 1:
    # PrintWindow Succeeded
    im.save("test.png")
