import math
import numpy as np
from typing import List

class Point:
    pass

class Point:
    """Point class to represent a point in 2D space"""
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y
    def __eq__(self, other: Point):
        return self.x == other.x and self.y == other.y

class Movement:
    """Movement class to represent a movement in 2D space"""
    def __init__(self, distance:float, direction:float):
        self.distance = distance
        self.direction = direction

class Rotation:
    """
        Rotation class to represent an angle change in radians
        possitive angle is counter-clockwise, negative angle is clockwise
    """
    def __init__(self, angle:float):
        self.angle = angle
    def __init__(self, PointBase0:Point,PointBase1:Point, PointTarget:Point):
        self.angle = math.atan2(PointTarget[1] - PointBase0[1], PointTarget[0] - PointBase0[0]) - math.atan2(PointBase1[1] - PointBase0[1], PointBase1[0] - PointBase0[0])

class Wall:
    def __init__(self, wall:np.array):
        self.wall = wall
        self.start = Point(wall[0][1], wall[0][0])
        self.end = Point(wall[0][3], wall[0][2])

class Car:
    def __init__(self, car:List[np.array], front:List[np.array] = None):
        self.triangle = []
        for point in car:
            self.triangle.append(Point(point[0][1], point[0][0]))
        self.front = Point(front[0][1], front[0][0]) if front is not None and len(front) == 2  else Point(0, 0)
        
        while(len(self.triangle) < 3):
            self.triangle.append(Point(0, 0))
        if(len(self.triangle) > 3):
            self.triangle = self.triangle[:3]
        
        while(not self.triangle[0] == self.front):
            self.triangle.append(self.triangle.pop(0))
        self.base = self.triangle[1:3]

class Pickup:
    """Pickup class to represent a pickup action"""
    def __init__(self):
        pass