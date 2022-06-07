import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pyttsx3
from tensorflow import keras
from keras.models import model_from_json, Model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

model = model_from_json(open("data.json", "r").read())
model.load_weights("fer_61.h5")
jokes = pd.read_csv('new_shortjokes.csv')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

predicted_emotion = ''
def telljoke(emotion):
    if emotion == "sad":
            joke = random.choice(jokes["Joke"])
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
            engine.say(joke)
            engine.runAndWait()

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        
        faces= faceCascade.detectMultiScale(gray, 1.32, 5)
        for (x,y,w,h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                facess = faceCascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh)in facess:
                    face_roi = roi_color[ey: ey+eh, ex:ex +ew] 
                
                final_image =cv2.resize(face_roi, (48,48))
                gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                gray=gray.reshape((1,48,48))

                predictions = model.predict(gray)  
        
                max_index = np.argmax(predictions[0])
        
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
                predicted_emotion = emotions[max_index] 

                cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                telljoke(predicted_emotion)      
        return img

def main():
    st.title("Real Time Face Emotion Recognition App")
    mode = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Choose The App Mode", mode)

    if choice =="Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px>
                                        <h4 style="color:white;text-align:center;">
                                        Face Emotion Recognition</h4>
                                </div>
                                </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 This application has three functionalities.
                 
                 1. Real time face detection using webcam feed.
                 
                 2. Real time face emotion recognition.
                 
                 3. Tells a joke when sad emotion is detected.
                 
                 """)
    elif choice =="Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice =="About":
        st.subheader("About this app")
        html_temp_home1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    This application is developed by Damola Babayato using Streamlit Framework, OpenCV, Custom Trained CNN model, Pyttsx3 for speech initiatialization and Streamlit.
                                    The app tells a joke when it detects a sad emotion.</h4>
                            </div>
                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

        html_temp4= """
                        
                            <div style="background-color:#6D7B8D;padding:10px">
                            <h4 style="color:white;text-align:center;">If you have any suggestions or comments please contact me on:</h4>
                            <a href="mailto:damolababayato@gmail.com">damolababayato@yahoo.com</a>
                            </div>
                            <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    
    else:
        pass

if __name__ == "__main__":
    main()                           