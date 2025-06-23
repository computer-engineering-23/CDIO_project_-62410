import math
import numpy as np
from typing import List,Union,Tuple
from Log import printLog

def polygonArea(corners:list['Point']) -> float:
    area = 0.0
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        area += corners[i].x * corners[j].y
        area -= corners[j].x * corners[i].y
    area = abs(area) / 2.0
    return area

def sgn(x:float) -> int:
    """Returns the sign of a number"""
    if x >= 0:
        return 1
    else:
        return -1

def pointAverage(points:List['Point']) -> 'Point':
    """Calculates the average of a list of points"""
    if not points or len(points) == 0:
        return Point(0, 0)
    x_sum = sum(point.x for point in points)
    y_sum = sum(point.y for point in points)
    return Point(x_sum / len(points), y_sum / len(points))

class Point:
    """Point class to represent a point in 2D space"""
    def __init__(self, x:Union[int,float], y:Union[int,float]):
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def move(self, point: 'Point') -> 'Point':
        return Point(self.x + point.x, self.y + point.y)

    def negate(self) -> 'Point':
        return Point(-self.x, -self.y)

    def angleTo(self, point: 'Point') -> float:
        return math.atan2(point.y - self.y, point.x - self.x)

    def rotateAround(self, center: 'Point', angle: float) -> 'Point':
        sin, cos = math.sin(angle), math.cos(angle)
        x_shifted = self.x - center.x
        y_shifted = self.y - center.y
        x_new = x_shifted * cos - y_shifted * sin + center.x
        y_new = x_shifted * sin + y_shifted * cos + center.y
        return Point(x_new, y_new)

    def distanceTo(self, point: 'Point') -> float:
        return math.hypot(self.x - point.x, self.y - point.y)

    def copy(self) -> 'Point':
        return Point(self.x, self.y)
    
    def __str__(self) -> str:
        return f"Point[{self.x},{self.y}]"

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
    def __init__(self, wall:List[tuple[Union[int,float],int | float,int | float,int | float]]):
        self.start = Point(wall[0][0], wall[0][1])
        self.end = Point(wall[0][2], wall[0][3])
    def _asLine(self) -> 'Line':
        """Converts the wall to a Line object"""
        return Line(self.start, self.end)
    def intersect(self, other:'Wall') -> Point | bool:
        """Checks if the wall intersects with any of the walls"""
        a1,b1,c1 = self._asLine()._asFunction()
        a2,b2,c2 = other._asLine()._asFunction()
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            return c1 == c2  # lines are parallel or coincident
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return Point(y, x)  # returns the intersection point if it exists

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
    def _intersects(self, wall:Wall, extend:int = 0) -> bool:
        col:Point = Point(0, 0)
        func1 = self._asFunction()
        func2 = wall._asLine()._asFunction()
        if(func1[0] * func2[1] - func2[0] * func1[1]) == 0:
            return func1[2] == func2[2]  # lines are parallel or coincident
        col.y = (func1[1] * func2[2] - func2[1] * func1[2]) / (func1[0] * func2[1] - func2[0] * func1[1])
        col.x = (func1[2] * func2[0] - func2[2] * func1[0]) / (func1[0] * func2[1] - func2[0] * func1[1])
        if col.y < min(self.start.y, self.end.y) - extend or col.y > max(self.start.y, self.end.y) + extend:
            return False
        if col.x < min(self.start.x, self.end.x) - extend or col.x > max(self.start.x, self.end.x) + extend:
            return False
        if col.y < min(wall.start.y, wall.end.y) - extend or col.y > max(wall.start.y, wall.end.y) + extend:
            return False
        if col.x < min(wall.start.x, wall.end.x) - extend or col.x > max(wall.start.x, wall.end.x) + extend:
            return False
        return True  # returns True if the line intersects with the wall
    
    def _asFunction(self) ->tuple[float,float,float]:
        """Converts the line to a function of ax + by + c = 0"""
        if self.start.y < self.end.y:
            self.start, self.end = self.end, self.start
        if(self.start.y == self.end.y):
            return (0,0,0)
        
        a = (self.end.x - self.start.x)  # a
        b = (self.start.y - self.end.y)  # b
        c = (self.start.x * (self.end.y - self.start.y) - (self.end.x - self.start.x) * self.start.y)  # c
        
        return (a,b,c)  # returns [a, b, c] for the line equation ay + bx + c = 0
    
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
        
        if(not self.valid()):
            printLog("debug", f"generating invalid car, {self.triangle}; {self.front}", "Class car constructor")
    def copy(self) -> 'Car':
        """Creates a deep copy of the car with independent points"""
        new_triangle = [p.copy() for p in self.triangle]
        new_front = self.front.copy()
        new_car = Car(new_triangle, new_front)
        return new_car
    def area(self) -> float:
        """Calculates the area of the triangle formed by the car's points"""
        return polygonArea(self.triangle)
    
    def valid(self) -> bool:
        """Checks if the front point is valid (not at the same position as the triangle points)"""
        i:int = 0
        while(not self.triangle[0] == self.front and i < len(self.triangle)):
            self.triangle.append(self.triangle.pop(0))
            i += 1
        
        if(i >= len(self.triangle)):
            printLog("ERROR","invalid triangle points, front not found in car",producer="Class Car Valid")
            return False
        
        if(self.triangle[0] != self.front):
            printLog("ERROR","invalid triangle points, front not stored correctly",producer="Class Car Valid")
            return False
        
        if self.area() < 2000:
            printLog("ERROR","invalid triangle points, area too small:", self.area() ,producer="Class Car Valid")
            return False
        
        printLog("INFO","valid triangle points",producer="Class Car Valid")
        printLog("INFO", "area:", self.area(),producer="Class Car Valid")
        return True
    
    def getRotation(self) -> float:
        """Calculates the rotation of the car based on its triangle points"""
        direction = Line(self.getRotationCenter(),self.front)
        return direction.angle()
    
    def getWidth(self) -> float:
        """Calculates the width of the car based on the distance between the base points"""
        if not self.valid():
            if(len(self.triangle) >= 3):
                printLog("debug","invalid car, attempting calculation","Class car width")
                return Line(*self.triangle[1:3]).length()
            else:
                printLog("ERRor","invalid car, default width used","Class car width")
                return 0.0
        return Line(*self.triangle[1:3]).length()
    
    def getRotationCenter(self) -> Point:
        """Calculates the center of the car's rotation based on its triangle points"""
        if not self.valid():
            if(len(self.triangle) >= 3):
                printLog("debug","invalid car, attempting calculation","Class car rotation center")
                return pointAverage(self.triangle[1:3])
            else:
                printLog("ERRor","invalid car, default width used","Class car rotation center")
                return Point(0, 0)
        return pointAverage(self.triangle[1:3])
    
    def getRotationDiameter(self) -> float:
        """Calculates the diameter of the car based on the distance between the front point and the base points"""
        if not self.valid():
            if(len(self.triangle) >= 3):
                printLog("debug","invalid car, attempting calculation","Class car rotation diameter")
                return Line(self.front, self.getRotationCenter()).length()
            else:
                printLog("ERRor","invalid car, default width used","Class car rotation diameter")
                return 0.0
        return Line(self.front, self.getRotationCenter()).length()
    
    def apply(self, robotInfo:Movement | Rotation) -> 'Car':
        """Applies the robot info to the car and returns a new RobotInfo object"""
        CarCopy = self.copy()  # Create a copy of the car to avoid modifying the original
        CarCopy.applySelf(robotInfo)
        return CarCopy
    
    def applySelf(self, robotInfo:Movement | Rotation) -> None:
        """Applies the robot info to the car"""
        if isinstance(robotInfo, Rotation):
            self.rotate(robotInfo.angle)
        if isinstance(robotInfo, Movement):
            self.move(robotInfo.distance)
        if not self.valid():
            printLog("debug","apply failed, invalid car unpredictable outcome from this point on","Class car apply")
    
    def rotate(self, angle:float) -> None:
        """Rotates the car around its rotation center by a given angle in radians"""
        center = self.getRotationCenter()
        if(not self.valid()):
            printLog("ERROR","invalid Car, default action will be taken",producer="Class Car rotate")
        for i in range(len(self.triangle)):
            self.triangle[i] = self.triangle[i].rotateAround(center, angle)
        self.front = self.triangle[0]
    
    def move(self, distance:float) -> None:
        """Moves the car in the direction from its rotation center to its front point by a given distance"""
        if not self.valid():
            printLog("ERROR","invalid Car, default action will be taken",producer="Class Car move")

        # Move in the direction from rotation center to front
        center = self.getRotationCenter()
        dx = self.front.x - center.x
        dy = self.front.y - center.y
        norm = math.sqrt(dx**2 + dy**2)
        if norm == 0:
            vector = Point(0, 0)
        else:
            vector = Point(distance * dx / norm, distance * dy / norm)
        for i in range(len(self.triangle)):
            self.triangle[i] = self.triangle[i].move(vector)
        self.front = self.front.move(vector)
    
    def validTarget(self, target:Point) -> bool:
        """Checks if the target point is valid (not at the same position as the triangle points)"""
        return self.getRotationCenter().distanceTo(target) > self.front.distanceTo(self.getRotationCenter())
    @property
    def radius(self) -> float:
        """Returns the turning radius of the car (from rotation center to front)"""
        return Line(self.front, self.getRotationCenter()).length()
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
        self.angle = abs(endAngle - startAngle)
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
    def point_at(self, t: float) -> Point:
        """Returns a point on the arc at angle offset `t` from the start angle"""
        angle = self.start + t  # assumes counter-clockwise
        return Point(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle)
        )
    
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
        y0 = (delta * deltax + sgn(deltax) * deltay * math.sqrt(discriminant))/(deltaR ** 2)
        x0 = (-delta * deltay + abs(deltax) * math.sqrt(discriminant))/(deltaR ** 2)
        y1 = (delta * deltax - sgn(deltax) * deltay * math.sqrt(discriminant))/(deltaR ** 2)
        x1 = (-delta * deltay - abs(deltax) * math.sqrt(discriminant))/(deltaR ** 2)
        
        intersections = [Point(y0, x0).move(self.center), Point(y1, x1).move(self.center)]
        for intersection in intersections:
            # Check if the intersection point is within the arc's angle range
            angle = math.atan2(intersection.x - self.center.x, intersection.y - self.center.y)
            if self.start <= angle <= self.end:
                return True
        
        return False