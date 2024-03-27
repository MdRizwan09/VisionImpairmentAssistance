from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import requests
import imutils
import pyttsx3
from mutagen.mp3 import MP3
from gtts import gTTS
import time
import vlc

# from readchar import readkey, key

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("Models/Keras/Model_6/keras_model.h5", compile=False)

# Load the labels
class_names = open("Models/Keras/Model_6/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# camera = cv2.VideoCapture(2) # For external webcam

# try: 
#     camera = cv2.VideoCapture(2) # For external webcam
# except cv2.error:
#     camera = cv2.VideoCapture(0)


def splay(mytext):

    # Language in which you want to convert
    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)

    fn = "Voices/CurrenceyDetection"
    fileName = fn+".mp3"
    myobj.save(str(fileName))

    audio = MP3(fileName)

    audio_info = audio.info
    length = int(audio_info.length)

    play = vlc.MediaPlayer(fileName)
    play.play()
    time.sleep(length)
    play.stop()


def splayOffline(txt):
    tts = pyttsx3.init()

    """ RATE"""
    rate = tts.getProperty('rate')   # getting details of current speaking rate
    # print (rate)                        #printing current voice rate
    tts.setProperty('rate', 145)     # setting up new voice rate

    tts.say(txt)
    tts.runAndWait()


url = "http://192.168.141.221:8080/shot.jpg"


while True:
    try:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=480, height=400)
        image = img
    except:        
        ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]


    det = class_name[2:-1]
    conf = np.round(confidence_score * 100)

    if conf > 65:
        if det != "Not a Currency":
            print("Index of object: ", index)
            print("Class:"+det)
            print("Confidence Score:", str(conf)[:-2], "%\n")
            detected = det+" detected."
            try:
                splay(detected)
            except:
                splayOffline(detected)
            # break
    

    # 27 is the ASCII for the esc key on your keyboard.
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()


# -----------------------------------------------------
