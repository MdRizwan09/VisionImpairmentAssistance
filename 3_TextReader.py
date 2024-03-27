# import the opencv library
import cv2
import requests
import numpy as np
import imutils
from PIL import Image
import pyttsx3
import pytesseract
from mutagen.mp3 import MP3
from gtts import gTTS
import time
import vlc


# define a video capture object 
vid = cv2.VideoCapture(0)

def splay(mytext):
    # mytext = items

    # Language in which you want to convert
    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)

    fn = "Voices/ObjectDetection"
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

def docProcessing():
    img = cv2.imread('TextReader.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)

    ret,thresh1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)

    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    # _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def biggestRectangle(contours):
        biggest = None
        max_area = 0
        indexReturn = -1
        for index in range(len(contours)):
                i = contours[index]
                area = cv2.contourArea(i)
                if area > 100:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.1*peri,True)
                    if area > max_area: #and len(approx)==4:
                            biggest = approx
                            max_area = area
                            indexReturn = index
        return indexReturn

    indexReturn = biggestRectangle(contours)
    hull = cv2.convexHull(contours[indexReturn])
    cv2.imwrite('ProcessedDoc.jpg',cv2.drawContours(img, [hull], 0, (0,255,0),3))
    
    # cv2.imwrite('ProcessedDoc.jpg',thresh1)


def textReader():
    docProcessing()
    txt = pytesseract.image_to_string(Image.open('ProcessedDoc.jpg'))
    print("Text: \n", txt)
    return txt

url = "http://192.168.141.221:8080/shot.jpg"

while(True):
    try:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=480, height=400)
        image = img
    except:        
        ret, image = vid.read()

    # sky = image[0:80, 0:80]
    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite("TextReader.jpg", image)
        t = textReader()
        cv2.imshow('Output', image)
        cv2.waitKey(0)
        try:
            splay(t)
        except:
            splayOffline(t)
        cv2.destroyWindow('Output')
        

    # elif cv2.waitKey(1) & 0xFF == ord('q'):
    elif cv2.waitKey(1) == 27:
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
