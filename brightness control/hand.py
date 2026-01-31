import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def create_hands():
    return mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)
