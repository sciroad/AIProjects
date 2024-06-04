import os
import tkinter as tk
from tkinter import simpledialog

import camera
import cv2 as cv
import PIL.Image
import PIL.ImageTk
import model


class App:

    def __init__(self, window=tk.TK(), window_title="Camera Classifier") -> None:
        self.window = window
        self.window.title(window_title)

        self.counters=[1,1]

        self.model = model.Model()

        self.auto_predict = False

        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15

        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()
    
    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Predict", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Class Name", "Enter the name of the first class",
                                                    parant=self.window)
        self.classname_two = simpledialog.askstring("Class Name", "Enter the name of the second class",
                                                    parant=self.window)
        

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_data(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_data(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train", width=50, 
                                   command=lambda: self.model.train(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS", font=("Helvetica", 16))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

        # self.btn_snapshot = tk.Button(self.window, text="Snapshot", width=50, command=self.snapshot)
        # self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        # self.btn_auto_predict = tk.Button(self.window, text="Auto Predict: OFF", width=50, command=self.auto_predict_toggle)
        # self.btn_auto_predict.pack(anchor=tk.CENTER, expand=True)
    
    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_data(self, class_no):
        ret, frame = self.camera.get_frame()
        if not os.path.exists(f"dataset/{class_no}"):
            os.mkdir(f"dataset/{class_no}")
        
        cv.imwrite(f"dataset/{class_no}/{self.counters[class_no-1]}.jpg", frame)
        img = PIL.Image.open(f"dataset/{class_no}/{self.counters[class_no-1]}.jpg")
        img.thumbnail((128,128), PIL.Image.ANTIALIAS)
        img.save(f"dataset/{class_no}/{self.counters[class_no-1]}.jpg")

        self.counters[class_no-1] += 1

    def reset(self):
        for i in range(1,3):
            for file in os.listdir(f"dataset/{i}"):
                if os.path.isfile(f"dataset/{i}/{file}"):
                    os.remove(f"dataset/{i}/{file}")

        self.counters = [1,1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            self.predict()
        
        ret, frame = self.camera.get_frame()    
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)
        

    def predict(self, class_no):
        ret, frame = self.camera.get_frame()
        class_no = self.model.predict(frame)
        if class_no == 1:
            self.class_label.config(text=self.classname_one)
            return self.classname_one
        elif class_no == 2:
            self.class_label.config(text=self.classname_two)
            return self.classname_two
        else:
            self.class_label.config(text="CLASS")
            return "CLASS"


        


    

        