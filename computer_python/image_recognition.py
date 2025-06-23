import cv2
import numpy as np
from typing import List, Tuple, Union
import math
from classes import Point, Wall, Line, polygonArea
from Log import printLog

# Start kameraet
class Camera:
    def __init__(self, APIid:int = cv2.CAP_DSHOW, debug:bool = False):
        self.debug = debug
        self.capture = cv2.VideoCapture(1, APIid)
        self.walls:List[List[tuple[int | float,int | float,int | float,int | float]]] = []
        if not self.capture.isOpened():
            printLog("error","Kunne ikke åbne kamera", producer="Camera")
            exit(1)
        initial_frame:Union[np.ndarray,None] = self.getFrame()
        if initial_frame is None:
            printLog("error","Kunne ikke hente billede fra kamera", producer="Camera")
            exit(1)
        self.shape:Tuple[int,...] = np.shape(initial_frame)
        self.corners:Tuple[Point|None,Point|None,Point|None,Point|None] = (None,None,None,None)
    
    def displayFrame(self,frame:np.ndarray,name:str = "detection window", debug:bool = False):
        if(debug == True and not self.debug):
            return
        cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return
        pass

    def displayWithDetails(self,frame:np.ndarray,circles:Union[List[Tuple[List[Union[int,float]],str]],None] = None, lines:list[Line]|None = None, goals:tuple[Point, Point] | None = None, name:Union[str,None] = None, debug:bool = False) -> None:
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

    def drawToFrame(self, frame:np.ndarray, circles:Union[List[Tuple[List[Union[int,float]],str]],None] = None, lines:list[Line]|None = None, goals:tuple[Point, Point] | None = None) -> np.ndarray:
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
                (x1, y1, x2, y2) = (lines[i].start.x, lines[i].start.y, lines[i].end.x, lines[i].end.y)
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3, cv2.LINE_AA)
        
        if goals is not None:
            for goal in goals:
                cv2.circle(frame, (int(goal.x),int(goal.y)), 5, (255, 0, 0), 0)
                cv2.putText(frame, "goal", (int(goal.x) - 10, int(goal.y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i,corner in enumerate(self.corners):
            if(corner is not None):
                cv2.circle(frame, (int(corner.x), int(corner.y)), 5, (0, 0, 255), 0)
                cv2.putText(frame, "corner" + str(i), (int(corner.x) - 10, int(corner.y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def getFrame(self) -> Union[np.ndarray,None]:
        ret, frame = self.capture.read()
        if not ret:
            printLog("error","Kunne ikke hente billede fra kamera", producer="Camera")
            return None
        self.shape = np.shape(frame)
        return frame
    
    def findCircle(self,frame:np.ndarray) -> Union[List[Tuple[List[Union[int,float]],str]],None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Hvid farveområde (HSV)
        lower_white = np.array([0, 0, 165])
        upper_white = np.array([180, 35, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        masked = cv2.bitwise_and(frame, frame, mask=mask_white)
        self.displayFrame(masked, "masked circle", debug=True)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(mask_white, (15, 15), 0)
        self.displayFrame(gray, "brur circle", debug=False)
        __circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=14,
            minRadius=5,
            maxRadius=8
        )
        if(__circles is not None):
            circles:List[List[List[Union[int,float]]]] = __circles.tolist()
            names = ["ball"]*len(circles[0])
            return [*zip(circles[0],names)]
        return __circles
    
    def findEgg(self, frame:np.ndarray) -> Union[List[Tuple[List[int | float], str]],None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([180, 35, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        masked = cv2.bitwise_and(frame, frame, mask=mask_white)
        self.displayFrame(masked, "masked egg", debug=True)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        __circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=12,
            maxRadius=15
        )
        if(__circles is not None):
            circles = __circles.tolist()
            names = ["eggs"]*len(circles[0])
            return [*zip(circles[0],names)]
        return __circles

    def findWall(self, frame:np.ndarray, noMask:bool = False) -> List[List[tuple[Union[int,float],int | float,int | float,int | float]]]:
        """a single iteration to find walls"""
        if(not noMask):
            hsv:np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hueMid = 0
            hueWidth = 15
            minSaturation = 0
            maxSaturation = 255
            minBrightness = 0
            maxBrightness = 255
            hue0 = hueMid - hueWidth if hueMid - hueWidth >= 0 else (hueMid - hueWidth + 180)
            hue1 = (hueMid + hueWidth) % 180
            lower_red0 = np.array([hue0, minSaturation, minBrightness])
            upper_red0 = np.array([180 , maxSaturation, maxBrightness])
            lower_red1 = np.array([0   , minSaturation, minBrightness])
            upper_red1 = np.array([hue1, maxSaturation, maxBrightness])
            mask0:np.ndarray = cv2.inRange(hsv, lower_red0, upper_red0)
            mask1:np.ndarray = cv2.inRange(hsv, lower_red1, upper_red1)
            mask:np.ndarray = cv2.bitwise_or(mask0,mask1)
            masked:np.ndarray = cv2.bitwise_and(frame, frame, mask=mask)
            masked:np.ndarray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            edges = masked
            self.displayFrame(masked, "walls masked", debug = True)
            self.displayFrame(edges, "walls edges", debug = True)
        else:
            edges:np.ndarray = frame
            edges:np.ndarray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        linesP:List[List[tuple[Union[int,float],int | float,int | float,int | float]]] = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, None, 5, 5).tolist()
        self.walls = linesP
        return linesP

    def generateWall(self, frameNumber) ->List[List[tuple[int | float,int | float,int | float,int | float]]]:
        """many iterations to find walls more acurately"""
        
        rawWalls = []
        for i in range (frameNumber):
            current = self.getFrame()
            if(current is None):
                printLog("error","Kunne ikke hente billede fra kamera", producer="generateWall")
                return []
            rawWalls.append(self.findWall(current))
        buffer = np.zeros(self.shape, dtype=np.uint8)
        for walls in rawWalls:
            if walls is not None:
                for i in range (0,len(walls)):
                    (x1, y1, x2, y2) = walls[i][0]
                    cv2.line(buffer,(x1,y1),(x2,y2),(255,255,255),1, cv2.LINE_AA)
        self.displayFrame(buffer, "walls", True)
        self.walls = self.findWall(buffer,True)
        self.corners = self.findCorners(self.walls)
        return self.walls

    def findCorners(self,walls:List[List[tuple[Union[int,float],int | float,int | float,int | float]]]) -> tuple[Point|None,Point|None,Point|None,Point|None]:
        old = self.corners
        if(not any (corner is None for corner in self.corners)):
            oldArea = polygonArea([c for c in old if c is not None])
        else:
            oldArea = -1
        
        wallClass:list[Wall] = []
        for wall in walls:
            wallClass.append(Wall(wall))
        intersects:List[Point] = []
        for i in range (0,len(wallClass)):
            for j in range (i+1, len(wallClass)):
                if wallClass[i]._asLine()._intersects(wallClass[j],30):
                    intersection = wallClass[i].intersect(wallClass[j])
                    if(type(intersection) != bool):
                        intersects.append(intersection)
        split:tuple[list[Point],list[Point],list[Point],list[Point]] = ([],[],[],[])
        for intersect in intersects:
            if(intersect.x < self.shape[1] / 2 and intersect.y < self.shape[0] / 2): # top-left
                split[0].append(intersect)
            elif(intersect.x >= self.shape[1] / 2 and intersect.y < self.shape[0] / 2): # top-right
                split[1].append(intersect)
            elif(intersect.x < self.shape[1] / 2 and intersect.y >= self.shape[0] / 2): # bottom-left
                split[2].append(intersect)
            elif(intersect.x >= self.shape[1] / 2 and intersect.y >= self.shape[0] / 2): # bottom-right
                split[3].append(intersect)
        if(any(len(split[i]) == 0 for i in range(0,4))):
            printLog("error","Kunne ikke finde alle hjørner", producer="findCorners")
            return (None,None,None,None)
        x0 = float(np.median(np.array([p.x for p in split[0]])))
        y0 = float(np.median(np.array([p.y for p in split[0]])))
        x1 = float(np.median(np.array([p.x for p in split[1]])))
        y1 = float(np.median(np.array([p.y for p in split[1]])))
        x2 = float(np.median(np.array([p.x for p in split[2]])))
        y2 = float(np.median(np.array([p.y for p in split[2]])))
        x3 = float(np.median(np.array([p.x for p in split[3]])))
        y3 = float(np.median(np.array([p.y for p in split[3]])))
        corners:Tuple[Point|None,Point|None,Point|None,Point|None] = (
            Point(x0, y0), Point(x2, y2), Point(x3, y3),Point(x1, y1)
        )
        
        margin = 0.05
        
        for i in range(0,4):
            if(corners[i] is None):
                printLog("error","Kunne ikke finde hjørner, et eller flere hjørner er None", producer="findCorners")
                return old
        
        newArea = polygonArea([c for c in corners if c is not None])
        if(oldArea != -1 and (newArea < oldArea * (1 - margin))):
            printLog("error","Kunne ikke finde hjørner, for lille område", producer="findCorners")
            return old
        
        for i in range(0,4):
            j = (i + 1) % 4
            old1 = old[i]
            old2 = old[j]
            if(old1 is None or old2 is None):
                break
            if(Line(corners[i], corners[(i + 1) % 4]).length() < Line(old1, old2).length() * (1 - margin)):
                printLog("error","Kunne ikke finde hjørner, for lille afstand mellem hjørner", producer="findCorners")
                return old
        
        self.corners = corners
        return corners

    def findCar(self, frame:np.ndarray) -> Tuple[List[Tuple[List[int | float],str]],Tuple[List[int | float],str]] | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        huemiddle = 150//2
        satmiddle = 60
        brightmiddle = 60
        hueWidth = 20
        satWidth = 60
        brightWidth = 15
        lower_green = np.array([max(huemiddle - hueWidth,0), 20, 20])
        upper_green = np.array([min(huemiddle + hueWidth,360), 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        gray = cv2.GaussianBlur(mask_green, (15, 15), 0)
        gray = cv2.inRange(gray, np.array([20]), np.array([255]))
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        __circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,
            param1=40,
            param2=10,
            minRadius=2,
            maxRadius=10
        )
        self.displayFrame(mask_green,"car mask", debug=True)
        self.displayFrame(gray, "car blur", debug=True)
        closest:Union[Tuple[int,int], None] = None
        distance = 1000000
        front: Union[List[Union[int, float]],None] = None
        if(__circles is not None):
            circles:List[List[List[Union[int,float]]]] = __circles.tolist()
            while(len(circles[0]) > 4):
                furtherstDist = 0
                furthestID = -1
                for i in range (0, len(circles[0])):
                    dist = 0
                    for j in range (0, len(circles[0])):
                        if(i == j):
                            continue
                        dist += math.sqrt((circles[0][i][0] - circles[0][j][0]) ** 2 + (circles[0][i][1] - circles[0][j][1]) ** 2)
                    if(dist > furtherstDist):
                        furtherstDist = dist
                        furthestID = i
                circles[0].remove(circles[0][furthestID])
            for i in range (0, len(circles[0])):
                for j in range (0, len(circles[0])):
                    if(i == j):
                        continue
                    (x0, y0, r) = circles[0][i]
                    (x1, y1, r) = circles[0][j]
                    currentDistance = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
                    if(currentDistance < distance):
                        distance = currentDistance
                        closest = (i,j)
            if(closest is not None):
                front = [
                    (circles[0][closest[0]][0] + circles[0][closest[1]][0]) // 2,
                    (circles[0][closest[0]][1] + circles[0][closest[1]][1]) // 2,
                    5
                ]
                circles[0].remove(circles[0][closest[1]])
                circles[0].remove(circles[0][closest[0]])
                circles[0].append(front)
            if(len(circles[0]) > 1):
                lines:List[Line] = []
                for i in range (0,len(circles[0])):
                    (x0, y0, r) = circles[0][i]
                    (x1, y1, r) = circles[0][(i+1) % len(circles[0])]
                    lines.append(Line(Point(x0, y0), Point(x1, y1)))
                self.displayWithDetails(frame, lines= lines, name="car", debug=True)
            names = ["car"]*len(circles[0])
            circles_ = [*zip(circles[0],names)]
            if(front is not None):
                return (circles_, (front, "front"))
        else:
            return None

    def close(self):
        self.capture.release()
    
    def makeWalls(self, corners:tuple[Point|None,Point|None,Point|None,Point|None]) -> List[Wall]|None:
        if corners is None or len(corners) < 4:
            printLog("error","Kunne ikke lave vægge, hjørner mangler", producer="makeWalls")
            return None
        
        output:List[Wall] = []
        for i in range(0,4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]
            if corner1 is None or corner2 is None:
                printLog("error","Kunne ikke lave væg, et eller flere hjørner er None", producer="makeWalls")
                return None
            output.append(Wall([(corner1.x, corner1.y, corner2.x, corner2.y)]))
        return output
    
    def makeGoals(self, corners:tuple[Point | None,Point  | None,Point  | None,Point  | None]) -> tuple[Point,Point] | None:
        if corners is None or len(corners) < 4:
            printLog("error","Kunne ikke lave mål, hjørner mangler", producer="makeGoals")
            return None
        if(corners[0] is None or corners[1] is None or corners[2] is None or corners[3] is None):
            printLog("error","Kunne ikke lave mål, et eller flere hjørner er None", producer="makeGoals")
            return None
        return ( \
            Point( \
                (corners[0].x + corners[1].x) / 2, \
                (corners[0].y + corners[1].y) / 2  \
            ), \
            Point( \
                (corners[2].x + corners[3].x) / 2, \
                (corners[2].y + corners[3].y) / 2  \
            ) \
        )
    
    def midpointWalls(self, width, lines:List[List[tuple[int | float,int | float,int | float,int | float]]]) -> List[Tuple[int,int]]:
        if(lines is None or len(lines) == 0):
            return [(0,0),(0,0)]
        veritcalLines = []
        for i in range (0,len(lines)):
            (x1, y1, x2, y2) = lines[i][0]
            if(x1 == x2): # For at undgå division med 0
                veritcalLines.append(lines[i])
                break
            a = abs(y1 - y2) / abs(x1 - x2)
            if(a < -4 or a > 4):
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
            (x1, y1, x2, y2) = rightLines[i][0]
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
            (x1, y1, x2, y2) = leftLines[i][0]
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
        goals = [(rightMidX,rightMidY), (leftMidX,leftMidY)]
        return goals

    def Test(self, generateWalls = False):
        if(generateWalls):
            self.walls = self.generateWall(40)
        else:
            if(self.walls is None or len(self.walls) == 0):
                self.walls = self.generateWall(40)
        self.corners = self.findCorners(self.walls)
        goals:tuple[Point, Point] | None = self.makeGoals(self.corners)
        frame:Union[np.ndarray,None] = self.getFrame()
        if(frame is None): return
        walls:list[Wall]|None = self.makeWalls(self.corners)
        if walls is None or len(walls) == 0: walls = []
        lines:list[Line] = [wall._asLine() for wall in walls]
        self.displayWithDetails(frame, lines=lines, goals=goals)

    def setDebug(self, debug:bool):
        self.debug = debug