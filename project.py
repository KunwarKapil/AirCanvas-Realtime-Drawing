import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pytesseract
from gtts import gTTS
import os  # To play the generated audio file

# Set the path to your Tesseract-OCR installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to convert text to speech using Google TTS
def speak_text(text):
    tts = gTTS(text=text, lang='en')  
    tts.save("output.mp3")  # Save the speech as an mp3 file
    os.system("start output.mp3")  

# Giving different arrays to handle color points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Drawing canvas
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Black canvas

# Indexes to mark the points in specific color arrays
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Color palette(BGRY)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create paint window UI
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize Mediapipe Hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
#this is used for detecting the landmarks on the detected hand
mpDraw = mp.solutions.drawing_utils

# Apply a moving average filter over the last N frames for smoothing
class SmoothFinger: 
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def apply(self, point):
        self.history.append(point)
        # Calculate average of the last N points
        avg_point = np.mean(self.history, axis=0)
        return int(avg_point[0]), int(avg_point[1])

# Initialize finger smoothener
finger_smoothener = SmoothFinger(window_size=10)  # Use a larger window for smoother movement

# initilize the webcam
cap = cv2.VideoCapture(0)
ret = True
prev_fore_finger = None  # Initialize previous finger position

while ret:
    ret, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    #hand lamdmarks predection
    result = hands.process(framergb)

    #post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            #drawing lamdmarks
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

        # Apply smoothing to the finger's position
        fore_finger = finger_smoothener.apply(fore_finger)

        if thumb[1] - fore_finger[1] < 30:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif fore_finger[1] <= 65:
            if 40 <= fore_finger[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                blue_index = green_index = red_index = yellow_index = 0
                drawing_canvas[:] = 0
                paintWindow[67:, :, :] = 255
            elif 160 <= fore_finger[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= fore_finger[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= fore_finger[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= fore_finger[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(fore_finger)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(fore_finger)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(fore_finger)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(fore_finger)

    # Draw the lines on both frames
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(drawing_canvas, points[i][j][k - 1], points[i][j][k], (255, 255, 255), 2)

    # Image preprocessing before OCR
    gray_canvas = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
    _, thresh_canvas = cv2.threshold(gray_canvas, 150, 255, cv2.THRESH_BINARY)
    dilated_canvas = cv2.dilate(thresh_canvas, kernel, iterations=1)

    key = cv2.waitKey(1)
    if key == ord('r'):
        if np.any(drawing_canvas > 0):
            text = pytesseract.image_to_string(dilated_canvas, config='--psm 6')
            print("Recognized Text:", text.strip())
            
            # Speak the recognized text using Google TTS
            speak_text(text.strip())
            
            drawing_canvas[:] = 0

    if key == ord('q'):
        break

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

cap.release()
cv2.destroyAllWindows()
