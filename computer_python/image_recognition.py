import cv2
import numpy as np
from typing import Union,List,Tuple

# Start kameraet
class Camera:
    def __init__(self):
        self.capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    
    def displayFrame(self,frame:np.array):
        cv2.imshow("Bold Detektion", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return
        pass

    def displayWithCircles(self,frame:np.array,circles:List[Tuple[int,str]]) -> None:
        if circles is not None:
            names = [a[1] for a in circles]
            circles = [a[0] for a in circles]
            circles = np.round(circles).astype("int")
            
            for i in range(len(names)):
                (x, y, r) = circles[i]
                # Tegn en cirkel og et mærke på midten
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(frame, names[i], (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print("no circles")
        self.displayFrame(frame)

    
    def getFrame(self) -> np.array:
        ret, frame = self.capture.read()
        if not ret:
            print("Kunne ikke hente billede fra kamera")
            return None
        return frame
    
    def findCircle(self,frame:np.array):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange farveområde (kan justeres)
        lower_orange = np.array([0, 30, 100])
        upper_orange = np.array([30, 255, 255])
        
        # Hvid farveområde (HSV)
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([180, 35, 255])
        
        # Skab maske kun med orange
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Kombinerer masker
        mask = cv2.bitwise_or(mask_orange, mask_white)
        
        # Brug masken til at finde relevante områder
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        #self.displayFrame(masked)


        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Find cirkler med Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=7
        )
        if(circles is not None):
            names = ["ball"]*len(circles[0])
            if(len(circles[0]) == 1):
                circles = [*zip(circles[0],names)]
            else:
                circles = [*zip(circles[0],names)]
        return circles
    
    def findEgg(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange farveområde (kan justeres)
        lower_orange = np.array([0, 30, 100])
        upper_orange = np.array([30, 255, 255])
        
        # Hvid farveområde (HSV)
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([180, 35, 255])
        
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Brug masken til at finde relevante områder
        masked = cv2.bitwise_and(frame, frame, mask=mask_white)
        #self.displayFrame(masked)


        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Find cirkler med Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=12,
            maxRadius=15
        )
        if(circles is not None):
            names = ["eggs"]*len(circles[0])
            if(len(circles[0]) == 1):
                circles = [*zip(circles[0],names)]
            else:
                circles = [*zip(circles[0],names)]
        return circles

    def findWall(self, frame):
        
        pass

    def findCar(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hueWidth = 15
        huemiddle = 227/2
        generalWidth = 65
        # Hvid farveområde (HSV)
        lower_blue = np.array([huemiddle - hueWidth, 0.82 * 255 - generalWidth, 0.43 * 255 - generalWidth])
        upper_blue = np.array([huemiddle + hueWidth, 0.82 * 255 + generalWidth, 0.43 * 255 + generalWidth])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Brug masken til at finde relevante områder
        masked = cv2.bitwise_and(frame, frame, mask=mask_blue)
        #self.displayFrame(masked)


        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Find cirkler med Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=7
        )
        if(circles is not None):
            names = ["car"]*len(circles[0])
            if(len(circles[0]) == 1):
                circles = [*zip(circles[0],names)]
            else:
                circles = [*zip(circles[0],names)]
        self.displayWithCircles(masked,circles)
        return circles

    def close(self):
        self.capture.release()
    
    def Test(self):
        frame:np.array = self.getFrame()
        eggs = self.findEgg(np.copy(frame))
        circles = self.findCircle(np.copy(frame))
        cars = self.findCar(np.copy(frame))
        return
        if(circles is None):
            circles = []
        if(eggs is None):
            eggs = []
        if(cars is None):
            cars = []
        self.displayWithCircles(frame, circles + eggs + cars)


