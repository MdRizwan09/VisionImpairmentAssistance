## Vision Impairment Assistance

**We need to insatall all these libraries first:**
- ```pip install opencv-python ```
- ```pip install python-vlc```
- ```pip install imutils```
- ```pip install pyttsx3```
- ```pip install mutagen```
- ```pip install gtts```
- ```pip install vlc```



## YOLO (You Only Look Once)

Download the pre-trained YOLO v3 weights file from https://pjreddie.com/media/files/yolov3.weights this link and place it in ```VisionImpairmentAssistance\Models\Yolo``` directory, or you can directly download it from terminal using these commands:

**For Windows or Powershell:**
```bash
cd Models/Yolo ; wget https://pjreddie.com/media/files/yolov3.weights ; cd ../..
```

**For Linux:**
```bash
cd Models/Yolo && wget https://pjreddie.com/media/files/yolov3.weights && cd ../..
```
