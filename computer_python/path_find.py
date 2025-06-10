import cv2
import numpy as np
from typing import List, Tuple, Union
from computer_python.image_recognition import Camera
from computer_python.classes import Point, Movement, Rotation, Wall


class track:
    def __init__(self,cam:cv2.VideoCapture,walls:List[np.array] = None, goals:List[Tuple[int,int]] = None, targets:List[Tuple[int,int]] = None, obsticles: List[Tuple[int,int]] = None, car:List[Tuple[int,int]] = None, front:Tuple[int,int] = None):
        self.walls:List[np.array] = walls
        self.goals:List[Tuple[int,int]] = goals
        self.targets:List[Tuple[int,int]] = targets
        self.cam:cv2.VideoCapture = cam
        if(cam is None):
            self.cam = Camera()
        self.obsticles = obsticles
        self.car:List[Tuple[int,int]] = car
        self.front:Tuple[int,int] = front
    
    def update(self, walls:bool = False, goals:bool = False, targets:bool = False, obsticles:bool = False, car:bool = False):
        frame:np.array = self.cam.getFrame()
        
        if(walls):
            self.walls = self.cam.generateWall(40)
        
        if(goals):
            self.goals = self.cam.midpointWalls(self.cam.shape[1], self.walls)
        
        if(targets):
            self.targets = self.cam.findCircle(np.copy(frame))
        
        if(obsticles is not None):
            self.obsticles = self.cam.findEgg(np.copy(frame))
        
        if(car):
            self.car, self.front = self.cam.findCar(np.copy(frame))
    
    def formatWalls(self, walls:List[np.array]) -> List[Wall]:
        realWalls = []
        if(walls is None or len(walls) == 0):
            return []
        for wall in walls:
            realWalls.append(Wall(wall))
        return realWalls
    
    def VectorPath(self) -> Tuple[int,int]:
        """generates Vector from front to first goal"""
        target:Tuple[int,int] = self.goals[0]
        return (self.front[0] - target[0], self.front[1] - target[1])
    
    def generatepath(self) -> List[Union[Rotation,Movement]]:
        """Generates a path from the car to the first goal"""
        if(self.goals is None or self.front is None):
            return []
        path = []
        vector = self.VectorPath()
        
        
        return path
