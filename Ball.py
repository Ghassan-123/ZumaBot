import numpy as np
import time


class Ball:
    def __init__(self, ball_id, center, color):
        self.id = ball_id
        self.center = np.array(center, dtype=np.float32)
        self.color = color
        self.last_seen = time.time()
        self.finish_id = None
        self.cost = None
