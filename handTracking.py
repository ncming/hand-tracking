import cv2
import mediapipe as mp
import numpy as np
import math

CAMERA_INDEX = 1 # camera laptop = 0 else 1
wCam, hCam = 640, 480 
WINDOW_NAME = ":3"

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

tips_ids = [4, 8, 12, 16, 20] #thumb index mid ring pinky

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img,1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #openCV-AI MediaPipe 
    results = hands.process(img_rgb)
    servo_angles = [0, 0, 0, 0, 0] 

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                x_wrist, y_wrist = lm_list[0][1], lm_list[0][2]
                for i in range(5):
                    x_tip, y_tip = lm_list[tips_ids[i]][1], lm_list[tips_ids[i]][2]
                    length = math.hypot(x_tip - x_wrist, y_tip - y_wrist)
                    if i == 0: #thumb
                        angle = np.interp(length, [20, 150], [0, 180])
                    else:
                        angle = np.interp(length, [30, 200], [0, 180])
                    servo_angles[i] = int(angle)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    print(servo_angles)
    cv2.imshow(WINDOW_NAME, img)
    key = cv2.waitKey(1)
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

cap.release()
cv2.destroyAllWindows()