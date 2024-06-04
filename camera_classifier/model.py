from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train(self, counters):
        images = np.array([])
        labels = np.array([])

        for i in range(len(counters)):
            for j in range(1, counters[i]):
                img = cv.imread(f"dataset/{i+1}/{j}.jpg", cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (128, 128))
                img = img.flatten()

                images = np.append(images, img)
                labels = np.append(labels, i+1)
        
        images = images.reshape(-1, 128*128)
        self.model.fit(images, labels)

        print("Training complete")
    
    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("temp.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = cv.imread("temp.jpg", cv.IMREAD_GRAYSCALE) 
        img = cv.resize(img, (128, 128))
        img = img.flatten()

        img = img.reshape(1, -1)

        return self.model.predict([img])[0]