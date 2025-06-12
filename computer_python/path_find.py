import cv2
import numpy as np
from typing import List, Tuple, Union
from image_recognition import Camera
from classes import Point, Movement, Rotation, Wall, Car, Pickup,RobotInfo
import math

class track:
    def __init__(self,cam:cv2.VideoCapture,walls:List[np.array] = None, goals:List[Tuple[int,int]] = None, targets:List[Tuple[int,int]] = None, obsticles: List[Tuple[int,int]] = None, car:List[Tuple[int,int]] = None, front:Tuple[int,int] = None):
        self.walls:List[Wall] = self.formatWalls(walls)
        self.goals:List[Point] = self.formatGoals(goals)
        self.targets:List[Point] = self.formatTargets(targets)
        self.cam:cv2.VideoCapture = cam
        if(cam is None):
            self.cam = Camera()
        self.obsticles = self.formatObsticles(obsticles)
        self.car:Car = self.formatCar(car, [front])
    
    def update(self, walls:bool = False, goals:bool = False, targets:bool = False, obsticles:bool = False, car:bool = False):
        frame:np.array = self.cam.getFrame()
        
        if(walls):
            self.walls = self.formatWalls(self.cam.generateWall(40))
        
        if(goals):
            self.goals = self.formatGoals(self.cam.midpointWalls(self.cam.shape[1], self.cam.walls))
        
        if(targets):
            self.targets = self.formatTargets(self.cam.findCircle(np.copy(frame)))
        
        if(obsticles is not None):
            self.obsticles = self.formatObsticles(self.cam.findEgg(np.copy(frame)))
        
        if(car):
            self.car = self.formatCar(*self.cam.findCar(np.copy(frame)))
    
    def formatWalls(self, walls:List[np.array]) -> List[Wall]:
        realWalls = []
        if(walls is None or len(walls) == 0):
            return []
        for wall in walls:
            realWalls.append(Wall(wall))
        return realWalls
    def formatGoals(self, goals:List[Tuple[int,int]]) -> List[Point]:
        realGoals = []
        if(goals is None or len(goals) == 0):
            return []
        for goal in goals:
            realGoals.append(Point(goal[1], goal[0]))
        return realGoals
    def formatTargets(self, targets:List[Tuple[int,int]]) -> List[Point]:
        realTargets = []
        if(targets is None or len(targets) == 0):
            return []
        for target in targets:
            realTargets.append(Point(target[0][1], target[0][0]))
        return realTargets
    def formatObsticles(self, obsticles:List[Tuple[int,int]]) -> List[Point]:
        realObsticles = []
        if(obsticles is None or len(obsticles) == 0):
            return []
        for obsticle in obsticles:
            realObsticles.append(Point(obsticle[0][1], obsticle[0][0]))
        return realObsticles
    def formatCar(self, car:List[Tuple[int,int]], front:Tuple[int,int]) -> Car:
        if(car is None or len(car) == 0):
            return Car([[(0, 0)],[(0, 1)],[(0, 2)]], [(0, 0)])
        return Car(car, front)
    
    def VectorPath(self) -> Tuple[int,int]:
        """generates Vector from front to first goal"""
        target:Point = self.goals[0]
        vector =  Point(self.car.front.y - target.y, self.car.front.x - target.x)
        if(abs(vector.x) < 10):
            vector.x = 0
        if(abs(vector.y) < 10):
            vector.y = 0
        return vector
    
    def generatepath(self) -> List[Union[Rotation,Movement]]:
        """Generates a path from the car to the first goal"""
        if(self.goals is None or self.car.front is None):
            return []
        path = []
        
        vector = self.VectorPath()
        if(vector.x == 0 and vector.y == 0):
            return [Pickup()]
        
        if(vector.y > 0):
            path.append(Rotation(math.pi / 2 - self.car.getRotation()))
            path.append(Movement(vector.y,math.pi / 2))
        elif(vector.y < 0):
            path.append(Rotation(-math.pi / 2 - self.car.getRotation()))
            path.append(Movement(vector.y,-math.pi / 2))
        
        if(vector.x > 0):
            path.append(Rotation(0 - path[-1].direction))
            path.append(Movement(vector.x, 0))
        elif(vector.x < 0):
            path.append(Rotation(math.pi - path[-1].direction))
            path.append(Movement(vector.x, math.pi))

        return path + [Pickup()]
    
    def Draw(self, frame:np.array):
        """Draws the track on the frame"""
        for wall in self.walls:
            cv2.line(frame, (int(wall.start.y), int(wall.start.x)), (int(wall.end.y), int(wall.end.x)), (0, 0, 255), 1)
        
        for goal in self.goals:
            cv2.circle(frame, (int(goal.y), int(goal.x)), 5, (255, 0, 0), -1)
        
        for target in self.targets:
            cv2.circle(frame, (int(target.y), int(target.x)), 5, (0, 255, 0), -1)
        
        for obsticle in self.obsticles:
            cv2.circle(frame, (int(obsticle.y), int(obsticle.x)), 5, (0, 255, 255), -1)
        
        for i in range(len(self.car.triangle)):
            p0 = self.car.triangle[i]
            p1 = self.car.triangle[(i + 1) % 3]
            cv2.line(frame, (int(p0.x), int(p0.y)), (int(p1.x), int(p1.y)), (255, 255, 0), 2)
        
        for i in self.generatepath():
            if isinstance(i, Movement):
                cv2.arrowedLine(frame, (int(self.car.front.y), int(self.car.front.x)), (int(self.car.front.y + i.distance * math.cos(i.direction)), int(self.car.front.x + i.distance * math.sin(i.direction))), (0, 255, 0), 1)
            elif isinstance(i, Rotation):
                cv2.putText(frame, f"Rotate: {i.angle:.2f} rad", (i.location.x,i.location.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
            frame = self.cam.getFrame()
            self.Draw(frame)
            self.cam.displayFrame(frame,"Track")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        self.cam.close()
        cv2.destroyAllWindows()
    
    def testPath(self):
        """Test function to show the path"""
        return [Movement(10, 0), Rotation(math.pi, Point(0, 0)), Movement(5, math.pi), Pickup()]
