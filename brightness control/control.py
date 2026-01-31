import cv2
import numpy as np
import screen_brightness_control as sbc
from hand import create_hands, mp_hands, mp_draw, distance

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

hands = create_hands()

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

        d = distance(lm[4], lm[8])
        brightness = int(np.interp(d, [0.03, 0.25], [0, 100]))
        brightness = max(0, min(100, brightness))
        sbc.set_brightness(brightness)

        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        cv2.putText(
            frame,
            f"Brightness: {brightness}%",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Hand Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
