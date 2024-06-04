import cv2 as cv

class Camera:

    def __init__(self):
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        
        self.width = int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()
    
    def get_frame(self):
        if not self.camera.isOpened():
            raise Exception("Camera is not opened")
        
        ret, frame = self.camera.read()
        
        if not ret:
            raise Exception("Could not read frame")
        
        return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    