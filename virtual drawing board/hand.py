import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def create_hands():
    return mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(lm[tips[0]].x < lm[tips[0] - 1].x)
    for i in range(1, 5):
        fingers.append(lm[tips[i]].y < lm[tips[i] - 2].y)
    return fingers
