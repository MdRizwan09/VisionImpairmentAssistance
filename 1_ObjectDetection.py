import cv2
import vlc
import numpy as np
import imutils
import requests
import pyttsx3
from mutagen.mp3 import MP3
from gtts import gTTS
import time


def splay(mytext):
    # mytext = items

    # Language in which you want to convert
    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)

    fn = "Voices/ObjectDetection"
    fileName = fn+".mp3"
    print(fileName)
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


lst = []
s = ""

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def objDet(image):
    s = ""

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # net = cv2.dnn.readNet("Vision Impairment Assistance/yolov3.weights", "Vision Impairment Assistance/yolov3.cfg")

    net = cv2.dnn.readNetFromDarknet('Models/yolov3.cfg', 'Models/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # ln = net.getLayerNames()
    # print(len(ln), ln)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    # print("net : ", net)
    # print("outs: ", outs)
    

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
                file = open("Models/Yolo/yolov3.txt")
                all_lines = file.readlines()
                
                val = all_lines[class_id]
                if val not in lst:
                    lst.append(val)
                
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))    
  
  
    for i in lst:
        s += i
    lst.clear()


    def replaceNewLine(value):
        return ' and '.join(value.splitlines())
    ss = replaceNewLine(s)

    if ss != "":
        
        items = ss+" is ahead."
        print(items)

        try:
            splay(items)
        except:
            splayOffline(items)
        # break
        
    s = ""
    

url = "http://192.168.141.22q1:8080/shot.jpg"
# image = cv2.imread("/home/mx/Desktop/Project/Vision Impairment Assistance/object-detection-opencv/dog.jpg")
vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture(2)

while(True):
    try:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=480, height=400)
        image = img
    except:        
        ret, image = vid.read()

    classes = None

    with open("Models/Yolo/yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    

    cv2.imshow('-Object Detection-', image)


    if cv2.waitKey(1) & 0xFF == ord(' '):
        objDet(image)
    
    # elif cv2.waitKey(1) & 0xFF == ord('q'):
    elif cv2.waitKey(1) == 27:
        break
  

vid.release()
cv2.destroyAllWindows()

