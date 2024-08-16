from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import numpy as np
import sqlite3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

with open('model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights('model_weights.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

id = 0
global count
count=0
names = ['None', 'bibin','dijo']

# Initialize the Flask app

with sqlite3.connect('users.db') as db:
    c = db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password TEXT NOT NULL);')
db.commit()
db.close()


app = Flask(__name__)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def SendMail(ImgFileName):
    with open(ImgFileName, 'rb') as f:
        img_data = f.read()

    msg = MIMEMultipart()
    msg['Subject'] = 'Face Emotion recogntion'
    msg['From'] = 'sender@gmail.com'
    msg['To'] = 'receiver@gmail.com'

    text = MIMEText("Result")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('sender@gmail.com', 'pass')
    s.sendmail('sender@gmail.com', 'receiver@gmail.com', msg.as_string())
    s.quit()


# Define a function to capture video frames and perform emotion recognition
def gen_frames():  
    cap = cv2.VideoCapture(0)
    global count
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face = gray[y:y+h, x:x+w]

            id, confidence = recognizer.predict(face)

        # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))



            
            # Resize the face image to match the input size of the model
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=-0)
            prediction = emotion_model.predict(face)
            print(prediction)
            maxindex = int(np.argmax(prediction,axis=1))
            
            # Draw bounding box and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Name:" +str(id), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, "Emotion:" +emotion_dict[maxindex], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if maxindex==1 or maxindex==5:
                count=count+1
            else:
                count=count-1
            if count<0:
                count=0
            if count==5:
                cv2.imwrite('img.jpg',frame)   
                SendMail('img.jpg')
                count=0


            
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define a route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the main route
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users WHERE username=?', (username,))
        user = c.fetchone()
        
        if not user:
            return render_template('login.html', error='Invalid username or password')

        if user[2]!= password:
            return render_template('login.html', error='Invalid username or password')

        return render_template('index.html')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()


        c.execute('SELECT * FROM users WHERE username=?', (username,))
        if c.fetchone():
            return render_template('signup.html', error='Username already exists')

        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()

        return render_template('login.html')
    
    return render_template('signup.html')


if __name__ == '__main__':
    app.run(debug=True)
