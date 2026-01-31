import cv2
import numpy as np
from hand import create_hands, mp_hands, mp_draw, fingers_up

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

hands = create_hands()

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
DRAW_COLOR = (255, 0, 255)
DRAW_THICKNESS = 8
ERASER_SIZE = 40

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        lm = hand_lm.landmark
        fingers = fingers_up(lm)

        h, w, _ = frame.shape
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        if fingers == [False, True, False, False, False]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), DRAW_COLOR, DRAW_THICKNESS)
            prev_x, prev_y = x, y

        elif fingers == [False, True, True, True, False]:
            cv2.circle(canvas, (x, y), ERASER_SIZE, (0, 0, 0), -1)
            prev_x, prev_y = 0, 0

        elif fingers == [False, True, True, False, False]:
            prev_x, prev_y = 0, 0

        elif fingers == [True, True, True, True, True]:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            prev_x, prev_y = 0, 0

        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Virtual Drawing Board", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()