from GamePlayer import GamePlayer
import tkinter as tk

class Main:
    def __init__(self):
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        self.aspect_ratio = screen_width / screen_height
        print(f"Screen resolution: {screen_width}x{screen_height}")
        print(f"Aspect ratio: {self.aspect_ratio:.2f} ({screen_width}:{screen_height})")

    def start(self):
        gameplayer = GamePlayer(self.aspect_ratio)
        gameplayer.RunLoop()


if __name__ == "__main__":
    Main().start()
