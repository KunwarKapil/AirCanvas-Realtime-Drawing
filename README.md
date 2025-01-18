Air Canvas: Gesture-Based Drawing Tool with OCR and TTS
Description
Air Canvas is an innovative tool that allows users to draw in real-time using hand gestures, powered by Mediapipe for hand tracking, OpenCV for video processing, and PyTesseract for text recognition (OCR). Additionally, the recognized text can be converted to speech using Google Text-to-Speech (gTTS).

Features
Hand Gesture Recognition: Draw on a virtual canvas by tracking hand and finger movements using Mediapipe.
Dynamic Color Selection: Choose from multiple colors for drawing.
Text Recognition (OCR): Extract text from the drawing using Tesseract OCR.
Text-to-Speech (TTS): Convert the recognized text into speech for auditory feedback.
Interactive UI: Intuitive interface with a "CLEAR" button and color palette for ease of use.
Tech Stack
Programming Language: Python
Libraries:
OpenCV: Video processing and drawing functionalities.
Mediapipe: Hand tracking and gesture recognition.
PyTesseract: Optical Character Recognition for text extraction.
gTTS: Text-to-Speech conversion.
NumPy: Image data manipulation.
Collections (deque): Efficient point tracking for smooth drawing.
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/AirCanvas-RealTime-Drawing.git
cd AirCanvas-RealTime-Drawing
Install Dependencies:
Ensure Python 3.7+ is installed. Then, install the required libraries:

bash
Copy
Edit
pip install opencv-python mediapipe pytesseract gTTS numpy
Install Tesseract OCR:
Download and install Tesseract from the official site.
Update the Tesseract path in the code:

python
Copy
Edit
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
Usage
Run the Script:
Execute the Python file to start the application:

bash
Copy
Edit
python air_canvas.py
How to Use:

Show your hand to the webcam to activate tracking.
Use your index finger to draw on the virtual canvas.
Select colors by pointing to the corresponding buttons at the top of the screen.
Press 'r' to recognize and speak the drawn text using OCR and TTS.
Press 'q' to quit the application.
Screenshots

(Replace this placeholder with actual screenshots of your application)

Real-World Applications
Education: A tool for interactive learning and presentations.
Assistive Technology: Helps individuals with disabilities write or draw.
Prototyping: Quickly sketch ideas without requiring traditional drawing tools.
Future Improvements
AI Integration: Improve gesture recognition accuracy using machine learning.
Eye Tracking: Add features for drawing using eye movements.
Custom Gestures: Allow users to define custom hand gestures for different actions.
Mobile App: Extend the project to mobile platforms.
Contributing
Contributions are welcome! If you’d like to improve this project, feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgements
Mediapipe by Google
Tesseract OCR
OpenCV Community
