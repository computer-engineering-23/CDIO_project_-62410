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

        self.hsv_thresholds = {
        'Ball_Orange': {
            'low': np.array([11, 111, 186]),
            'high': np.array([30, 255, 255])
        },
        'Ball_White': {
            'low': np.array([0, 0, 208]),
            'high': np.array([180, 38, 255])
        },
        'Egg': {
            'low': np.array([0, 0, 208]),
            'high': np.array([180, 38, 255])
        },
        'Wall': {
            'low': np.array([13, 139, 158]),  # red range
            'high': np.array([173, 243, 255])
        },
        'Car': {
            'low': np.array([62, 115, 94]),   # green range
            'high': np.array([100, 228, 143])
        }
}
    
    def displayFrame(self,frame:np.ndarray,name:str = "detection window", debug:bool = False):
        if(debug == True and not self.debug):
            return
        cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return
    
    def adjustWithSliders(self):
        object_names = ['Ball_Orange', 'Ball_White', 'Egg', 'Wall', 'Car']

        for name in object_names:
            cv2.namedWindow(name)

            # Initialize trackbars with defaults from self.hsv_thresholds
            low = self.hsv_thresholds[name]['low']
            high = self.hsv_thresholds[name]['high']
            cv2.createTrackbar('H low', name, low[0], 179, lambda x: None)
            cv2.createTrackbar('H high', name, high[0], 179, lambda x: None)
            cv2.createTrackbar('S low', name, low[1], 255, lambda x: None)
            cv2.createTrackbar('S high', name, high[1], 255, lambda x: None)
            cv2.createTrackbar('V low', name, low[2], 255, lambda x: None)
            cv2.createTrackbar('V high', name, high[2], 255, lambda x: None)

        while True:
            frame = self.getFrame()
            if frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for name in object_names:
                h_low = cv2.getTrackbarPos('H low', name)
                h_high = cv2.getTrackbarPos('H high', name)
                s_low = cv2.getTrackbarPos('S low', name)
                s_high = cv2.getTrackbarPos('S high', name)
                v_low = cv2.getTrackbarPos('V low', name)
                v_high = cv2.getTrackbarPos('V high', name)

                
                # Update the live threshold values
                self.hsv_thresholds[name]['low'] = np.array([h_low, s_low, v_low])
                self.hsv_thresholds[name]['high'] = np.array([h_high, s_high, v_high])

                if name is "Wall":
                    lower = self.hsv_thresholds[name]['low']
                    upper = self.hsv_thresholds[name]['high']
                    low_hue = lower[0]
                    high_hue = upper[0]
                    
                    low_std = lower[1:3]
                    high_std = upper[1:3]
                    
                    low_ = cv2.inRange(hsv, np.array([0,*low_std]), np.array([low_hue,*high_std]))
                    high_= cv2.inRange(hsv, np.array([high_hue,*low_std]), np.array([180,*high_std]))
                    mask = cv2.bitwise_or(low_, high_)
                else:
                    # Generate and show the mask
                    lower = self.hsv_thresholds[name]['low']
                    upper = self.hsv_thresholds[name]['high']
                    mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow(name, result)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


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
    
    def findCircle(self, frame: np.ndarray) -> Union[List[Tuple[List[Union[int, float]], str]], None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current HSV thresholds for orange and white balls
        low_orange = self.hsv_thresholds['Ball_Orange']['low']
        high_orange = self.hsv_thresholds['Ball_Orange']['high']
        low_white = self.hsv_thresholds['Ball_White']['low']
        high_white = self.hsv_thresholds['Ball_White']['high']

        # Create masks
        mask_orange = cv2.inRange(hsv, low_orange, high_orange)
        mask_white = cv2.inRange(hsv, low_white, high_white)

        # Combine both masks
        mask = cv2.bitwise_or(mask_orange, mask_white)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        self.displayFrame(masked, "masked circle", debug=True)

        # Convert to grayscale and blur
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Detect circles
        __circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=7
        )

        if __circles is not None and len(__circles) > 0:
            circle_data = __circles[0]
            if isinstance(circle_data, np.ndarray):
                circle_list = circle_data.tolist()
            else:
                circle_list = circle_data

            if isinstance(circle_list[0], list) and len(circle_list[0]) == 3:
                names = ["ball"] * len(circle_list)
                return list(zip(circle_list, names))

        return None


    
    def findEgg(self, frame: np.ndarray) -> Union[List[Tuple[List[int | float], str]], None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get white range from sliders
        low = self.hsv_thresholds['Egg']['low']
        high = self.hsv_thresholds['Egg']['high']

        mask_white = cv2.inRange(hsv, low, high)
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

        if __circles is not None and len(__circles) > 0:
            circle_data = __circles[0]
            if isinstance(circle_data, np.ndarray):
                circle_list = circle_data.tolist()
            else:
                circle_list = circle_data

            if isinstance(circle_list[0], list) and len(circle_list[0]) == 3:
                names = ["eggs"] * len(circle_list)
                return list(zip(circle_list, names))

        return None



    def findWall(self, frame: np.ndarray, noMask: bool = False) -> List[List[tuple[int | float, int | float, int | float, int | float]]]:
        if not noMask:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low = self.hsv_thresholds['Wall']['low']
            high = self.hsv_thresholds['Wall']['high']
            
            low_hue = low[0]
            high_hue = high[0]
            
            low_std = low[1:3]
            high_std = high[1:3]
            
            low_ = cv2.inRange(hsv, np.array([0,*low_std]), np.array([low_hue,*high_std]))
            high_= cv2.inRange(hsv, np.array([high_hue,*low_std]), np.array([180,*high_std]))
            
            # low_hue = cv2.inRange(hsv, low, np.array([180,*high[2:4]]))
            # high_hue = cv2.inRange(hsv, np.array([0,*low[2:4]]), high)
            
            mask = cv2.bitwise_or(low_, high_)
            # mask = cv2.inRange(hsv, low, high)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            self.displayFrame(masked, "walls masked", debug=True)
            self.displayFrame(masked_gray, "walls edges", debug=True)

            edges = masked_gray
        else:
            edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        linesP: Union[np.ndarray, None] = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, None, 5, 5)

        # Pylance-safe assignment
        wall_lines: List[List[tuple[int | float, int | float, int | float, int | float]]] = linesP.tolist() if linesP is not None else []

        self.walls = wall_lines
        return wall_lines




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

    def findCar(self, frame: np.ndarray) -> Tuple[List[Tuple[List[int | float], str]], Tuple[List[int | float], str]] | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = self.hsv_thresholds['Car']['low']
        high = self.hsv_thresholds['Car']['high']
        mask_green = cv2.inRange(hsv, low, high)

        gray = cv2.GaussianBlur(mask_green, (15, 15), 0)
        gray = cv2.inRange(gray, np.array([20]), np.array([255]))
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        self.displayFrame(mask_green, "car mask", debug=True)
        self.displayFrame(gray, "car blur", debug=True)

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

        closest = None
        front = None
        if __circles is not None and len(__circles) > 0:
            circle_data = __circles[0]
            if isinstance(circle_data, np.ndarray):
                circle_list = circle_data.tolist()
            else:
                circle_list = circle_data

            if isinstance(circle_list, list) and isinstance(circle_list[0], list):
                while len(circle_list) > 4:
                    i_to_remove = max(range(len(circle_list)), key=lambda i: sum(
                        math.dist(circle_list[i], circle_list[j]) for j in range(len(circle_list)) if i != j
                    ))
                    circle_list.pop(i_to_remove)

                for i in range(len(circle_list)):
                    for j in range(len(circle_list)):
                        if i == j:
                            continue
                        d = math.dist(circle_list[i][:2], circle_list[j][:2])
                        if closest is None or d < math.dist(circle_list[closest[0]][:2], circle_list[closest[1]][:2]):
                            closest = (i, j)

                if closest:
                    x = (circle_list[closest[0]][0] + circle_list[closest[1]][0]) // 2
                    y = (circle_list[closest[0]][1] + circle_list[closest[1]][1]) // 2
                    front = [x, y, 5]
                    for idx in sorted(closest, reverse=True):
                        circle_list.pop(idx)
                    circle_list.append(front)

                names = ["car"] * len(circle_list)
                results = list(zip(circle_list, names))
                if front:
                    return results, (front, "front")

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
        self.debug = True
        if(generateWalls):
            self.walls = self.generateWall(40)
        else:
            pass
        frame:Union[np.ndarray,None] = self.getFrame()
        if(frame is None): return

        balls = self.findCircle(frame)
        eggs = self.findEgg(frame)
        car = self.findCar(frame)
        if balls is None:
            balls = []
        if eggs is None:
            eggs = []
        if car is None:
            car = ([], None)
        self.corners = self.findCorners(self.walls)
        goals:tuple[Point, Point] | None = self.makeGoals(self.corners)
        walls:list[Wall]|None = self.makeWalls(self.corners)
        if walls is None or len(walls) == 0: walls = []
        lines:list[Line] = [wall._asLine() for wall in walls]
        self.displayWithDetails(frame,circles=balls+eggs, lines=lines, goals=goals, name="detection", debug=True)

    def setDebug(self, debug:bool):
        self.debug = debug