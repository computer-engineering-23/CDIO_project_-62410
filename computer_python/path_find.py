import cv2
import numpy as np
from typing import List, Tuple, Union
from image_recognition import Camera
from classes import Point, Wall, Car, Pickup, Movement, Rotation
import math

def deltaRotation(newAngle:float, currentAngle:float) -> float:
    """Generates the rotation needed to turn the car to the new angle"""
    if(newAngle < 0):
        newAngle += 2 * math.pi
    if(currentAngle < 0):
        currentAngle += 2 * math.pi
    rotation = newAngle - currentAngle
    if(rotation > math.pi):
        rotation -= 2 * math.pi
    elif(rotation < -math.pi):
        rotation += 2 * math.pi
    return rotation

class track:
    def __init__(self,cam:Camera ,
            walls:Union[List[List[List[Union[int,float]]]],None] = None, 
            goals:Union[List[Tuple[int,int]],None] = None, 
            targets:Union[List[Tuple[List[int | float],str]],None] = None, 
            obsticles: Union[List[Tuple[List[int | float],str]],None] = None, 
            car:Union[List[Tuple[List[int |float], str]],None] = None, 
            front:Union[Tuple[List[int | float],str],None] = None
        ):

        self.walls:List[Wall] = self.formatWalls(walls)
        self.goals:List[Point] = self.formatGoals(goals)
        self.targets:List[Point] = self.formatTargets(targets)
        self.cam:Camera = cam
        if(cam is None):
            self.cam = Camera()
        self.obsticles = self.formatObsticles(obsticles)
        self.car:Car = self.formatCar(car, front if front is not None else None)
    
    def update(self, walls:bool = False, goals:bool = False, targets:bool = False, obsticles:bool = False, car:bool = False):
        frame:np.ndarray | None = self.cam.getFrame()
        
        if(frame is None):
            print("No frame received from camera.")
            return
        
        if(walls):
            self.walls = self.formatWalls(self.cam.generateWall(40))
        
        if(goals):
            self.goals = self.formatGoals(self.cam.midpointWalls(self.cam.shape[1], self.cam.walls))
        
        if(targets):
            self.targets = self.formatTargets(self.cam.findCircle(np.copy(frame)))
        
        if(obsticles is not None):
            self.obsticles = self.formatObsticles(self.cam.findEgg(np.copy(frame)))
        if(car):
            tempCar = self.cam.findCar(np.copy(frame))
            if(tempCar is not None and len(tempCar) == 2): 
                car_,front_ = tempCar
                self.car = self.formatCar(car_, front_)
            else:
                self.car = self.formatCar(None, None)
    
    def formatWalls(self, walls:Union[List[List[List[int | float]]],None]) -> List[Wall]:
        realWalls = []
        if(walls is None or len(walls) == 0):
            return []
        for wall in walls:
            realWalls.append(Wall(wall))
        return realWalls
    def formatGoals(self, goals:Union[List[Tuple[int,int]],None]) -> List[Point]:
        realGoals = []
        if(goals is None or len(goals) == 0):
            return []
        for goal in goals:
            realGoals.append(Point(goal[1], goal[0]))
        return realGoals
    def formatTargets(self, targets:Union[List[Tuple[List[int | float],str]],None]) -> List[Point]:
        realTargets = []
        if(targets is None or len(targets) == 0):
            return []
        for target in targets:
            realTargets.append(Point(target[0][1], target[0][0]))
        return realTargets
    def formatObsticles(self, obsticles:Union[List[Tuple[List[int | float],str]],None]) -> List[Point]:
        realObsticles = []
        if(obsticles is None or len(obsticles) == 0):
            return []
        for obsticle in obsticles:
            realObsticles.append(Point(obsticle[0][1], obsticle[0][0]))
        return realObsticles
    def formatCar(self, car:Union[List[Tuple[List[int | float],str]],None], front:Union[Tuple[List[int | float],str],None]) -> Car:
        triangle:List[Point] = []
        if(car is not None and len(car) != 0):
            for point in car:
                triangle.append(Point(point[0][1], point[0][0]))
            if(len(triangle) < 3):
                return self.car
            if(len(triangle) > 3):
                triangle = triangle[0:3]
            front_point:Point = Point(0, 0)
            if(front is not None):
                front_point = Point(front[0][1], front[0][0])
            return Car(triangle,front_point)
        return Car([Point(0,0),Point(0,1),Point(1,0)], Point(0, 0))  # Default car if no car is provided

    def generatepath(self) -> List[Pickup | Movement | Rotation]:
        """Generates a path from the car to the first goal"""
        if(self.goals is None or self.car.front is None):
            return []
        path:List[Pickup | Movement | Rotation] = []
        front = self.car.front
        self.targets.sort(key=lambda target: math.sqrt((target.x - front.x) ** 2 + (target.y - front.y) ** 2))
        target = self.targets[0]
        direction = self.car.getRotation()
        car = self.car
        #generate the starting position

        #Make vector from fron  of th 
        dx = target.x - front.x
        dy = target.y - front.y
        angle = math.atan2(dx, dy)
        angle_diff = angle - direction
        print(f"Target: x={angle_diff} y={target.y}")
        
        path.append(Rotation(deltaRotation(angle, direction)))  # Rotate to face the target 
        car = car.applySelf(path[-1])
        
        path.append(Movement(front.distanceTo(target)))
        
        return path

    def drawCar(self, frame:np.ndarray,car:Car) -> np.ndarray:
        """Draws the car on the frame"""
        for i in range(len(car.triangle)):
            p0 = car.triangle[i]
            p1 = car.triangle[(i + 1) % 3]
            cv2.line(frame, (int(p0.y), int(p0.x)), (int(p1.y), int(p1.x)), (255, 255, 0), 2)
        return frame
    def Draw(self, frame:np.ndarray):
        """Draws the track on the frame"""
        for wall in self.walls:
            cv2.line(frame, (int(wall.start.y), int(wall.start.x)), (int(wall.end.y), int(wall.end.x)), (0, 0, 255), 1)
        
        for goal in self.goals:
            cv2.circle(frame, (int(goal.y), int(goal.x)), 5, (255, 0, 0), -1)
        
        for target in self.targets:
            cv2.circle(frame, (int(target.y), int(target.x)), 5, (0, 255, 0), -1)
        
        for obsticle in self.obsticles:
            cv2.circle(frame, (int(obsticle.y), int(obsticle.x)), 5, (0, 255, 255), -1)
        
        frame =self.drawCar(frame,self.car)
        
        car:Car = self.car.copy()
        for i in self.generatepath():
            front = car.front
            car.applySelf(i)
            if isinstance(i, Movement):
                cv2.arrowedLine(frame, (int(front.y),int(front.x)),(int(car.front.y), int(car.front.x)), (0, 255, 0), 1)
            if isinstance(i, Rotation):
                cv2.arrowedLine(frame, (int(front.y),int(front.x)),(int(car.front.y), int(car.front.x)), (255, 255, 255), 1)
                cv2.putText(frame, f"Rotate: {i.angle:.2f} rad", (int(car.front.y),int(car.front.x)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frame = self.drawCar(frame,car)

        cv2.putText(frame, "walls: red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "goals: blue", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "targets: green", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "obsticles: yellow", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "car: cyan", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Press 'q' to exit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def test(self):
        """Test function to show the track"""
        while True:
            self.update(walls=False, goals=False, targets=True, obsticles=True, car=True)
            frame:np.ndarray | None = self.cam.getFrame()
            if(frame is None):
                print("No frame received from camera.")
                break
            self.Draw(frame)
            self.cam.displayFrame(frame,"Track")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        self.cam.close()
        cv2.destroyAllWindows()
    
    def testPath(self):
        """Test function to show the path"""
        return [Movement(10), Rotation(math.pi), Movement(5), Pickup()]
