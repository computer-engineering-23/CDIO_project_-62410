import cv2
import numpy as np
from typing import List,Tuple,Union
import math

# Start kameraet
class Camera:
    def __init__(self, APIid:int = cv2.CAP_DSHOW, debug:bool = False):
        self.debug = debug
        self.capture = cv2.VideoCapture(1, APIid)
        self.walls:List[List[List[int | float]]] = []
        if not self.capture.isOpened():
            print("Kunne ikke åbne kamera")
            exit(1)
        initial_frame:Union[np.ndarray,None] = self.getFrame()
        if initial_frame is None:
            print("Kunne ikke hente billede fra kamera")
            exit(1)
        self.shape:Tuple[int,...] = np.shape(initial_frame)
    
    def displayFrame(self,frame:np.ndarray,name:str = "detection window", debug:bool = False):
        if(debug == True and not self.debug):
            return
        cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return
        pass

    def displayWithDetails(self,frame:np.ndarray,circles:Union[List[Tuple[List[Union[int,float]],str]],None] = None, lines:Union[List[list[tuple[int,int,int,int]]],None] = None, goals:Union[List[Tuple[int,int]],None] = None, name:Union[str,None] = None, debug:bool = False) -> None:
        if(debug == True and not self.debug):
            return
        data = np.zeros(np.shape(frame), dtype=np.uint8)
        frame = self.drawToFrame(frame, circles, lines, goals)
        data = self.drawToFrame(data, circles, lines, goals)
        if(name is not None):
            self.displayFrame(frame, name, debug)
        else:
            self.displayFrame(frame, "with camera", debug)
            self.displayFrame(data, "without camera", True)

    def drawToFrame(self, frame:np.ndarray, circles:Union[List[Tuple[List[Union[int,float]],str]],None] = None, lines:Union[List[list[tuple[int,int,int,int]]],None] = None, goals:Union[List[Tuple[int,int]],None] = None) -> np.ndarray:
        if circles is not None:
            names:List[str] = [a[1] for a in circles]
            _circles:List[List[Union[int,float]]] = [a[0] for a in circles]
            _circles_:np.ndarray = np.round(_circles).astype("int")
            
            for i in range(len(names)):
                (x, y, r) = _circles_[i]
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.putText(frame, names[i], (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if lines is not None:
            for i in range (0,len(lines)):
                (x1, y1, x2,y2) = lines[i][0]
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3, cv2.LINE_AA)
        
        if goals is not None:
            for goal in goals:
                cv2.circle(frame, goal, 5, (255, 0, 0), 0)
                cv2.putText(frame, "goal", (goal[0] - 10, goal[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def getFrame(self) -> Union[np.ndarray,None]:
        ret, frame = self.capture.read()
        if not ret:
            print("Kunne ikke hente billede fra kamera")
            return None
        self.shape = np.shape(frame)
        return frame
    
    def findCircle(self,frame:np.ndarray) -> Union[List[Tuple[List[Union[int,float]],str]],None]:
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
        self.displayFrame(masked, "masked circle", debug=True)

        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Find cirkler med Hough Circle Transform
        circles:Union[List[List[List[Union[int,float]]]],None] = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=7
        ).tolist()
        if(circles is not None):
            names = ["ball"]*len(circles[0])
            if(len(circles[0]) == 1):
                return [*zip(circles[0],names)]
            else:
                return [*zip(circles[0],names)]
        return circles
    
    def findEgg(self, frame:np.ndarray) -> Union[List[Tuple[List[int | float], str]],None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Hvid farveområde (HSV)
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([180, 35, 255])
        
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Brug masken til at finde relevante områder
        masked = cv2.bitwise_and(frame, frame, mask=mask_white)
        self.displayFrame(masked, "masked egg", debug=True)


        # Konverter til gråskala og blur igen
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Find cirkler med Hough Circle Transform
        circles: Union[List[List[List[Union[int,float]]]],None] = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=12,
            maxRadius=15
        ).tolist()
        if(circles is not None):
            names = ["eggs"]*len(circles[0])
            if(len(circles[0]) == 1):
                return [*zip(circles[0],names)]
            else:
                return [*zip(circles[0],names)]
        return circles

    def findWall(self, frame:np.ndarray, noMask:bool = False) -> List[List[List[Union[int,float]]]]:
        if(not noMask):
            hsv:np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            hueMid = 170
            hueWidth = 25
            sat = 30
            bri = 60
            hue0 = hueMid - hueWidth
            hue1 = (hueMid + hueWidth) % 180
            
            # Rød farveområde (HSV)
            lower_red0 = np.array([hue0, sat, bri])
            upper_red0 = np.array([180 , 200, 200])

            lower_red1 = np.array([0   , sat, bri])
            upper_red1 = np.array([hue1, 200, 200])

            # Skab maske
            mask0:np.ndarray = cv2.inRange(hsv, lower_red0, upper_red0)
            mask1:np.ndarray = cv2.inRange(hsv, lower_red1, upper_red1)
            
            mask:np.ndarray = cv2.bitwise_or(mask0,mask1)


            # Brug masken til at finde relevante områder
            masked:np.ndarray = cv2.bitwise_and(frame, frame, mask=mask)
            masked:np.ndarray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            edges:np.ndarray = cv2.Canny(masked, 50, 200, None, 3)
            self.displayFrame(masked, "walls masked", debug = True)
            self.displayFrame(edges, "walls edges", debug = True)
        
        else:
            edges:np.ndarray = frame
            edges:np.ndarray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        

        linesP:List[List[List[Union[int,float]]]] = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, None, 5, 5).tolist()
        
        self.walls = linesP

        return linesP

    def generateWall(self, frameNumber) ->List[List[List[int | float]]]:
        rawWalls = []
        for i in range (frameNumber):
            current = self.getFrame()
            if(current is None):
                print("Kunne ikke hente billede fra kamera")
                return []
            rawWalls.append(self.findWall(current))
        buffer = np.zeros(self.shape, dtype=np.uint8)
        for walls in rawWalls:
            if walls is not None:
                for i in range (0,len(walls)):
                    (x1, y1, x2,y2) = walls[i][0]
                    cv2.line(buffer,(x1,y1),(x2,y2),(255,255,255),1, cv2.LINE_AA)
        self.displayFrame(buffer, "walls", True)
        self.walls = self.findWall(buffer,True)
        return self.walls

    def findCar(self, frame:np.ndarray) -> Tuple[List[Tuple[List[int | float],str]],Tuple[List[int | float],str]] | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hueWidth = 22
        huemiddle = 166//2
        generalWidth = 67
        # grøn farveområde (HSV)
        lower_green = np.array([huemiddle - hueWidth, 0.87 * 255 - generalWidth, 0.59 * 255 - generalWidth])
        upper_green = np.array([huemiddle + hueWidth, 0.87 * 255 + generalWidth, 0.59 * 255 + generalWidth])
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Konverter til gråskala og blur igen
        gray = cv2.GaussianBlur(mask_green, (15, 15), 0)

        # Find cirkler med Hough Circle Transform
        circles: Union[List[List[List[Union[int,float]]]],None] = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,
            param1=40,
            param2=10,
            minRadius=3,
            maxRadius=20
        ).tolist()
        self.displayFrame(mask_green,"car mask", debug=True)
        self.displayFrame(gray, "car blur", debug=True)
        closest:Union[Tuple[int,int], None] = None
        distance = 1000000
        front: Union[List[Union[int, float]],None] = None
        if(circles is not None):
            for i in range (0, len(circles[0])):
                for j in range (0, len(circles[0])):
                    if(i == j):
                        continue
                    (y0, x0, r) = circles[0][i]
                    (y1, x1, r) = circles[0][j]
                    currentDistance = math.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
                    if(currentDistance < distance):
                        distance = currentDistance
                        closest = (i,j)
            if(closest is not None):
                front = [(circles[0][closest[0]][0] + circles[0][closest[1]][0]) // 2, (circles[0][closest[0]][1] + circles[0][closest[1]][1]) // 2, 5]
                circles[0].remove(circles[0][closest[1]])
                circles[0].remove(circles[0][closest[0]])
                circles[0].append(front)
            if(len(circles[0]) > 1):
                lines:List[List[Tuple[int,int,int,int]]] = []
                for i in range (0,len(circles[0])):
                    (y0, x0, r) = circles[0][i]
                    (y1, x1, r) = circles[0][(i+1) % len(circles[0])]
                    lines.append([(int(y0), int(x0), int(y1), int(x1))])
                self.displayWithDetails(frame, lines= lines, name="car", debug=True)
                
            names = ["car"]*len(circles[0])
            circles_ = [*zip(circles[0],names)]
            if(front is not None):
                return (circles_, (front, "front"))
        else:
            return None

    def close(self):
        self.capture.release()
    
    def midpointWalls(self, width, lines:List[List[List[Union[int,float]]]]) -> List[Tuple[int,int]]:
        
        if(lines is None or len(lines) == 0):
            return [(0,0),(0,0)]

        veritcalLines = []
        for i in range (0,len(lines)):
            (y1, x1, y2,x2) = lines[i][0]
            a = abs(x1 - x2) / abs(y1 - y2)
            if(a < -7 or a > 7):
                veritcalLines.append(lines[i])
        
        rightLines = []
        leftLines = []
        for i in range (0,len(veritcalLines)):
            (x1, y1, x2, y2) = veritcalLines[i][0]
            if(x1 > ((width // 3) * 2) and x2 > ((width // 3) * 2)):
                rightLines.append(veritcalLines[i])
            elif(x1 < (width // 3) and x2 < (width // 3)):
                leftLines.append(veritcalLines[i])
        
        rightTop = -1
        rightBottom = -1    
        rightIner = -1
        rightOut = -1
        for i in range (0,len(rightLines)):
            (y1, x1, y2, x2) = rightLines[i][0]
            if(y1 > rightBottom or rightBottom == -1):
                rightBottom = y1
            
            if(y2 > rightBottom):
                rightBottom = y2
            
            if(y1 < rightTop or rightTop == -1):
                rightTop = y1
            
            if(y2 < rightTop):
                rightTop = y2
            
            if(x1 > rightIner or rightIner == -1):
                rightIner = x1
            
            if(x2 > rightIner):
                rightIner = x2
            
            if(x1 < rightOut or rightOut == -1):
                rightOut = x1
            
            if(x2 < rightOut):
                rightOut = x2
        
        rightMidY = (rightTop + rightBottom) // 2
        rightMidX = (rightIner + rightOut) // 2

        leftTop = -1
        leftBottom = -1    
        leftIner = -1
        leftOut = -1
        for i in range (0,len(leftLines)):
            (y1, x1, y2, x2) = leftLines[i][0]
            if(y1 > leftBottom or leftBottom == -1):
                leftBottom = y1
            
            if(y2 > leftBottom):
                leftBottom = y2
            
            if(y1 < leftTop or leftTop == -1):
                leftTop = y1
            
            if(y2 < leftTop):
                leftTop = y2
            
            if(x1 > leftIner or leftIner == -1):
                leftIner = x1
            
            if(x2 > leftIner):
                leftIner = x2
            
            if(x1 < leftOut or leftOut == -1):
                leftOut = x1
            
            if(x2 < leftOut):
                leftOut = x2
        
        leftMidY = (leftTop + leftBottom) // 2
        leftMidX = (leftIner + leftOut) // 2


        goals = [(rightMidY,rightMidX), (leftMidY,leftMidX)]

        self.displayFrame(self.drawToFrame(np.zeros(self.shape, dtype=np.uint8),lines=rightLines, goals=[goals[0]]), "right lines", debug=True)
        self.displayFrame(self.drawToFrame(np.zeros(self.shape, dtype=np.uint8),lines=leftLines, goals=[goals[1]]), "left lines", debug=True)
        
        return goals

    def Test(self, useOldWall = False):
        if(useOldWall):
            walls = self.walls
        else:
            walls = self.generateWall(40)
        frame:Union[np.ndarray,None] = self.getFrame()
        if(frame is None): return
        eggs = self.findEgg(np.copy(frame))
        circles = self.findCircle(np.copy(frame))
        car, front = self.findCar(np.copy(frame))# type: ignore
        goals = self.midpointWalls(self.shape[1], walls)# type: ignore
        if(circles is None):
            circles = []
        if(eggs is None):
            eggs = []
        if(car is None):
            car = []
        if(walls is None):
            walls = []
        if(goals is None):
            goals = []
        if(front is None):
            front = []
        else:
            front = [front]
        self.displayWithDetails(frame, circles + eggs + car + front, walls, goals)# type: ignore

    def setDebug(self, debug:bool):
        self.debug = debug