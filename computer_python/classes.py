import math
import numpy as np
from typing import List,Union,Tuple
from Log import printLog

def __sgn(y:float) -> int:
    """Returns the sign of a number"""
    if y >= 0:
        return 1
    else:
        return -1

def pointAverage(points:List['Point']) -> 'Point':
    """Calculates the average of a list of points"""
    if not points or len(points) == 0:
        return Point(0, 0)
    y_sum = sum(point.y for point in points)
    x_sum = sum(point.x for point in points)
    return Point(y_sum // len(points), x_sum // len(points))

class Point:
    """Point class to represent a point in 2D space"""
    def __init__(self, y:Union[int,float], x:Union[int,float]):
        self.y:Union[int,float] = y
        self.x:Union[int,float] = x
    def __eq__(self, other: object) -> bool:
        """Checks if two points are equal"""
        if(isinstance(other,Point)):
            return self.y == other.y and self.x == other.x
        return False
    def move(self, point: 'Point') -> 'Point':
        """Moves the point by a given point"""
        return Point(self.y + point.y, self.x + point.x)
    def negate(self) -> 'Point':
        """Negates the point by swapping the y and y coordinates"""
        return Point(-self.y, -self.x)
    def angleTo(self, point: 'Point') -> float:
        return math.atan2(point.y - self.y, point.x - self.x)
    def rotateAround(self, center: 'Point', angle: float) -> 'Point':
        """Rotates the point around a center point by a given angle in radians"""
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        y_new = cos_angle * (self.y - center.y) - sin_angle * (self.x - center.x) + center.y
        x_new = sin_angle * (self.y - center.y) + cos_angle * (self.x - center.x) + center.x
        return Point(y_new, x_new)
    def distanceTo(self, point: 'Point') -> float:
        """Calculates the distance to another point"""
        return math.sqrt((point.y - self.y) ** 2 + (point.x - self.x) ** 2)
    def copy(self) -> 'Point':
        """Creates a deep copy of the point"""
        return Point(self.y, self.x)

class Movement:
    """
        Movement class to represent a movement in 2D space for the robot
        robot will only read distance
    """
    def __init__(self, distance:float):
        self.distance:float = distance

class Rotation:
    """
        Rotation class to represent an angle change in radians for the robot
        possitive angle is counter-clockwise, negative angle is clockwise
        robot will only read angle
    """
    def __init__(self, angle:float):
        self.angle:float = angle


class deliver(Movement):
    """
        deliver class to represent a delivery action for the robot
    """
    def __init__(self, distance:float):
        self.distance:float = distance

class Wall:
    """
        Wall class to represent a wall in the environment
        wall is represented by two points: start and end
        can generate the angle of the wall
    """
    def __init__(self, wall:List[List[Union[int,float]]]):
        self.start:Point = Point(wall[0][1], wall[0][0])
        self.end:Point = Point(wall[0][3], wall[0][2])
    def _asLine(self) -> 'Line':
        """Converts the wall to a Line object"""
        return Line(self.start, self.end)

class Line:
    """
        Line class to represent a line in the environment
        line is represented by two points: start and end
        can generate the angle of the line
    """
    def __init__(self, start:Point, end:Point):
        self.start:Point = start
        self.end:Point = end
    def angle(self) -> float:
        """Calculates the angle of the line in radians"""
        return self.start.angleTo(self.end)
    def length(self) -> float:
        """Calculates the length of the line"""
        return self.start.distanceTo(self.end)
    def intersects(self, walls:List[Wall]) -> bool:
        """Checks if the line intersects with any of the walls"""
        for wall in walls:
            if self._intersects(wall):
                return True
        return False
    def _intersects(self, wall:Wall) -> bool:
        col:Point = Point(0, 0)
        func1 = self._asFunction()
        func2 = wall._asLine()._asFunction()
        col.y = (func1[1] * func2[2] - func2[1] * func1[2]) / (func1[0] * func2[1] - func2[0] * func1[1])
        col.x = (func1[2] * func2[0] - func2[2] * func1[0]) / (func1[0] * func2[1] - func2[0] * func1[1])
        if col.y < min(self.start.y, self.end.y) or col.y > max(self.start.y, self.end.y):
            return False
        if col.x < min(self.start.x, self.end.x) or col.x > max(self.start.x, self.end.x):
            return False
        if col.y < min(wall.start.y, wall.end.y) or col.y > max(wall.start.y, wall.end.y):
            return False
        if col.x < min(wall.start.x, wall.end.x) or col.x > max(wall.start.x, wall.end.x):
            return False
        return True  # returns True if the line intersects with the wall
    
    def _asFunction(self) ->List[float]:
        """Converts the line to a function of y = mx + b"""
        out = []
        if self.start.y < self.end.y:
            self.start, self.end = self.end, self.start
        if(self.start.y == self.end.y):
            return [0,0,0]
        
        out.append(self.end.x - self.start.x)  # a
        out.append(self.start.y - self.end.y)  # b
        out.append(self.start.x * (self.end.y - self.start.y) - (self.end.x - self.start.x) * self.start.y)  # c
        
        return out  # returns [a, b, c] for the line equation ay + bx + c = 0
    
    def Shift(self, offset:int, angle:Union[float,None] = None) -> List['Line']:
        """Shifts the line by a given offset in both directions"""
        if angle is None:
            angle = self.angle()
        
        if offset == 0:
            return [self,self]  # returns the original line if offset is zero or negative
        
        offset = abs(offset)
        
        out = []
        
        offset_y = offset * math.cos(angle)
        offset_x = offset * math.sin(angle)
        
        new_start = Point(self.start.y + offset_y, self.start.x + offset_x)
        new_end = Point(self.end.y + offset_y, self.end.x + offset_x)
        out.append(Line(new_start, new_end))
        new_start = Point(self.start.y - offset_y, self.start.x - offset_x)
        new_end = Point(self.end.y - offset_y, self.end.x - offset_x)
        out.append(Line(new_start, new_end))
        return out  # returns a list of two lines shifted by the offset in both directions
    
    def move(self, point:Point) -> 'Line':
        """Moves the line by a given point"""
        return Line(self.start.move(point), self.end.move(point))
    def negate(self) -> 'Line':
        """Negates the line by swapping the start and end points"""
        return Line(self.start.negate(), self.end.negate())

class Car:
    """
        Car class to represent the car's position and orientation in the environment
        car is represented by a triangle formed by three points
        front is the point in front of the car, used to determine the direction
    """
    def __init__(self, triangle:List[Point], front:Point):
        self.triangle:List[Point] = triangle
        self.front:Point = front
        
        self.valid()
    def copy(self) -> 'Car':
        """Creates a deep copy of the car with independent points"""
        new_triangle = [Point(p.y, p.x) for p in self.triangle]
        new_front = Point(self.front.y, self.front.x)
        return Car(new_triangle, new_front)
    def area(self) -> float:
        """Calculates the area of the triangle formed by the car's points"""
        y1, x1 = self.triangle[0].y, self.triangle[0].x
        y2, x2 = self.triangle[1].y, self.triangle[1].x
        y3, x3 = self.triangle[2].y, self.triangle[2].x
        
        return abs((y1*(x2 - x3) + y2*(x3 - x1) + y3*(x1 - x2)) / 2.0)
    
    def valid(self) -> bool:
        """Checks if the front point is valid (not at the same position as the triangle points)"""
        i:int = 0
        while(not self.triangle[0] == self.front and i < len(self.triangle)):
            self.triangle.append(self.triangle.pop(0))
            i += 1
        
        if(i >= len(self.triangle)):
            return False
        
        if(self.triangle[0] != self.front):
            return False
        
        if self.area() < 20:
            return False
        
        return True
    
    def getRotation(self) -> float:
        """Calculates the rotation of the car based on its triangle points"""
        dx = self.front.x - self.getRotationCenter().x
        dy = self.front.y - self.getRotationCenter().y
        return math.atan2(dy, dx)
    
    def getWidth(self) -> float:
        """Calculates the width of the car based on the distance between the base points"""
        if not self.valid():
            return 0.0
        return Line(*self.triangle[1:3]).length()
    
    def getRotationCenter(self) -> Point:
        """Calculates the center of the car's rotation based on its triangle points"""
        if not self.valid():
            return Point(0, 0)
        return pointAverage(self.triangle[1:3])
    
    def getRotationDiameter(self) -> float:
        """Calculates the diameter of the car based on the distance between the front point and the base points"""
        if not self.valid():
            return 0.0
        return Line(self.front, self.getRotationCenter()).length()
    
    def apply(self, robotInfo:Movement | Rotation) -> 'Car':
        """Applies the robot info to the car and returns a new RobotInfo object"""
        CarCopy = Car(self.triangle.copy(), self.front)
        CarCopy.applySelf(robotInfo)
        return CarCopy
    
    def applySelf(self, robotInfo:Movement | Rotation) -> None:
        """Applies the robot info to the car"""
        if isinstance(robotInfo, Rotation):
            self.rotate(robotInfo.angle)
        if isinstance(robotInfo, Movement):
            self.move(robotInfo.distance)
    
    def rotate(self, angle:float) -> None:
        """Rotates the car around its rotation center by a given angle in radians"""
        center = self.getRotationCenter()
        if(self.triangle[0] != self.front):
            printLog("ERROR","invalid triangle points, default action will be taken")
        for i in range(len(self.triangle)):
            self.triangle[i] = self.triangle[i].rotateAround(center, angle)
        self.front = self.triangle[0]
    
    def move(self, distance:float) -> None:
        # Move in the direction from rotation center to front
        center = self.getRotationCenter()
        dy = self.front.y - center.y
        dx = self.front.x - center.x
        norm = math.sqrt(dx**2 + dy**2)
        if norm == 0:
            vector = Point(0, 0)
        else:
            vector = Point(distance * dy / norm, distance * dx / norm)
        for i in range(len(self.triangle)):
            self.triangle[i] = self.triangle[i].move(vector)
        self.front = self.front.move(vector)
    
    def validTarget(self, target:Point) -> bool:
        """Checks if the target point is valid (not at the same position as the triangle points)"""
        return self.getRotationCenter().distanceTo(target) > self.front.distanceTo(self.getRotationCenter())

class Arc:
    """
        Arc class to represent an arc in the environment
        arc is represented by a center point, radius, and angle
        is actualluy a circle segment, but for simplicity we call it an arc
    """
    def __init__(self, center:Point, startAngle:float, endAngle:float, radius:float):
        """Initializes the Arc with a center point, start point, and end point"""
        self.center:Point = center
        self.start:float = startAngle
        self.end:float = endAngle
        self.radius:float = radius
    def Intersects(self, walls:List[Wall]) -> bool:
        """Checks if the arc intersects with any of the walls"""
        for wall in walls:
            if self._intersects(wall):
                return True
        return False
    
    def points(self) -> List[Point]:
        """Generates start and end points of the arc"""
        return[
            Point(
                self.center.y + self.radius * math.cos(self.start), #y0
                self.center.x + self.radius * math.sin(self.start)  #x0
            ),Point(
                self.center.y + self.radius * math.cos(self.end),   #y1
                self.center.x + self.radius * math.sin(self.end)    #x1
            )
        ]
    
    def _intersects(self, wall:Wall) -> bool:
        """Checks if the arc intersects with a given wall"""
        # Calculate the distance from the center of the arc to the line segment defined by the wall
        line = wall._asLine()
        
        polarradius = math.sqrt((line.start.y - self.center.y) ** 2 + (line.start.x - self.center.x) ** 2)
        if (line.angle() >= self.start and line.angle() <= self.end and polarradius < self.radius):
            return True  # returns True if the start is within the arc
        
        polarradius = math.sqrt((line.end.y - self.center.y) ** 2 + (line.end.x - self.center.x) ** 2)
        if (line.angle() >= self.start and line.angle() <= self.end and polarradius < self.radius):
            return True  # returns True if the end is within the arc
        #chack if the arc's radius intersects with the wall
        arcPoints = self.points()
        arcLines = [Line(arcPoints[0], self.center), Line(arcPoints[1], self.center)]
        for arcLine in arcLines:
            if arcLine._intersects(wall):
                return True
        
        # Check if the line segment intersects with the arc
        line = line.move(self.center.negate())  # move the line to the origin
        
        deltay = line.end.y - line.start.y
        deltax = line.end.x - line.start.x
        deltaR = math.sqrt(deltay ** 2 + deltax ** 2)
        delta = line.start.y * line.end.x - line.end.y * line.start.x
        
        # Check if the line segment intersects with the arc
        discriminant = self.radius ** 2 * deltaR ** 2 - delta ** 2
        
        if discriminant < 0:
            return False
        
        # Calculate the intersection points
        y0 = (delta * deltax + __sgn(deltax) * deltay * math.sqrt(discriminant))/(deltaR ** 2)
        x0 = (-delta * deltay + abs(deltax) * math.sqrt(discriminant))/(deltaR ** 2)
        y1 = (delta * deltax - __sgn(deltax) * deltay * math.sqrt(discriminant))/(deltaR ** 2)
        x1 = (-delta * deltay - abs(deltax) * math.sqrt(discriminant))/(deltaR ** 2)
        
        intersections = [Point(y0, x0).move(self.center), Point(y1, x1).move(self.center)]
        for intersection in intersections:
            # Check if the intersection point is within the arc's angle range
            angle = math.atan2(intersection.x - self.center.x, intersection.y - self.center.y)
            if self.start <= angle <= self.end:
                return True
        
        return False