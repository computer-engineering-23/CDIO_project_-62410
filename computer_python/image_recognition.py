import cv2
import numpy as np
from typing import Union

# Start kameraet
class Camera:
    def __init__(self):
        self.capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    
    def displayFrame(self,frame:np.array):
        cv2.imshow("Bold Detektion", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return
        pass

    def displayWithCircles(self,frame:np.array,circles) -> None:
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Tegn en cirkel og et mærke på midten
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Bold", (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print("no circles")
        self.displayFrame(frame)

    
    def getFrame(self) -> Union[None, np.array]:
        ret, frame = self.capture.read()
        if not ret:
            print("Kunne ikke hente billede fra kamera")
            return None
        return frame
    
    def findCircle(self,frame:np.array):
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange farveområde (kan justeres)
        lower_orange = np.array([10, 150, 150])
        upper_orange = np.array([25, 255, 255])
        
        # Hvid farveområde (HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 80, 255])
        
        # Skab maske kun med orange
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Kombinerer masker
        mask = cv2.bitwise_or(mask_orange, mask_white)
        
        # Brug masken til at finde relevante områder
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)

        
        # Find cirkler med Hough Circle Transform
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
        # circles = cv2.HoughCircles(
        #     gray,
        #     cv2.HOUGH_GRADIENT,
        #     dp=1.2,
        #     minDist=50,
        #     param1=50,
        #     param2=30,
        #     minRadius=10,
        #     maxRadius=60
        # )

        return circles
    
    def close(self):
        cap.release()
    
    def Test(self):
        frame = self.getFrame()
        circles = self.findCircle(frame)
        self.displayWithCircles(frame,circles)
