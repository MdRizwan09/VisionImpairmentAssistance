from tkinter import *
import subprocess
from PIL import Image, ImageTk

# Create the main window
root = Tk()
root.geometry("700x600")
root.title("VISION IMPAIRMENT ASSISTANCE")
root['background']='purple'


# Add a label widget
label = Label(root, text="VISION IMPAIRMENT ASSISTANCE", font=("Arial", 25, "bold"))
label.pack(pady=20, ipadx=10)


# Add a button widget
button = Button(root, text="Object Detection", bg="light green", command=lambda: subprocess.run(["python3", "/home/mx/Desktop/Project/8th_Sem/ModulesDone/1_ObjectDetection.py"]))
button.pack(pady=10, ipadx= 15)
button = Button(root, text="Currency Detection", bg="light green", command=lambda: subprocess.run(["python3", "/home/mx/Desktop/Project/8th_Sem/ModulesDone/2_CurrencyDetection02.py"]))
button.pack(pady=10, ipadx= 7)
button = Button(root, text="Text Reader", bg="light green", command=lambda: subprocess.run(["python3", "/home/mx/Desktop/Project/8th_Sem/ModulesDone/3_TextReader.py"]))
button.pack(pady=10, ipadx= 31)

button = Button(root, text="EXIT", command=lambda: root.destroy(), bg="#E74C3C")
button.pack(pady=10)

# Start the main event loop
root.mainloop()
