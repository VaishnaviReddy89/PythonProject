import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
    
# Disable PyAutoGUI fail-safe   
pyautogui.FAILSAFE = False  
    
# Initialize MediaPipe      
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)       
cap.set(3, 1280)
cap.set(4, 720)

# Cooldown timers (prevents spam)
last_click_time = 0 
last_tab_time = 0
last_close_time = 0
cooldown = 0.6  # seconds   

while True:
    success, frame = cap.read()
    if not success: 
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger tips
            index = hand_landmarks.landmark[8] 
            thumb = hand_landmarks.landmark[4]
            middle = hand_landmarks.landmark[12]
            ring = hand_landmarks.landmark[16]

            # Convert to pixels
            ix, iy = int(index.x * w), int(index.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)
            rx, ry = int(ring.x * w), int(ring.y * h)

            # Map camera to full screen
            screen_x = np.interp(ix, [0, w], [0, screen_width])
            screen_y = np.interp(iy, [0, h], [0, screen_height])

            pyautogui.moveTo(screen_x, screen_y)

            # Draw points
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)   # Index  
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), cv2.FILLED) # Thumb
            cv2.circle(frame, (mx, my), 10, (255, 0, 0), cv2.FILLED) # Middle
            cv2.circle(frame, (rx, ry), 10, (0, 255, 255), cv2.FILLED) # Ring

            # Distances
            click_distance = math.hypot(tx - ix, ty - iy)
            tab_distance = math.hypot(tx - mx, ty - my)
            close_distance = math.hypot(tx - rx, ty - ry)

            current_time = time.time()

            #  LEFT CLICK (Thumb + Index)
            if click_distance < 40 and current_time - last_click_time > cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "CLICK", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            #  PRESS TAB (Thumb + Middle)
            elif tab_distance < 40 and current_time - last_tab_time > cooldown:
                pyautogui.press('tab')
                last_tab_time = current_time
                cv2.putText(frame, "TAB", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            #  CLOSE BROWSER TAB (Thumb + Ring)
            elif close_distance < 40 and current_time - last_close_time > cooldown:
                pyautogui.hotkey('ctrl', 'w')
                last_close_time = current_time
                cv2.putText(frame, "TAB CLOSED", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("AI Virtual Mouse", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()     