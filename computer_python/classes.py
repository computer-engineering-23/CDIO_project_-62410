import math
import numpy as np
from typing import List

def __sgn(x:float) -> int:
    """Returns the sign of a number"""
    if x >= 0:
        return 1
    elif x < 0:
        return -1

class Point:
    """Point class to represent a point in 2D space"""
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y
    def average(self, points:List['Point']) -> 'Point':
        """Calculates the average of a list of points"""
        if not points or len(points) == 0:
            return Point(0, 0)
        x_sum = sum(point.x for point in points)
        y_sum = sum(point.y for point in points)
        return Point(x_sum // len(points), y_sum // len(points))
    def __eq__(self, other: 'Point') -> bool:
        """Checks if two points are equal"""
        return self.x == other.x and self.y == other.y
    def move(self, point: 'Point') -> 'Point':
        """Moves the point by a given point"""
        return Point(self.x + point.x, self.y + point.y)
    def negate(self) -> 'Point':
        """Negates the point by swapping the x and y coordinates"""
        return Point(-self.x, -self.y)

class RobotInfo:
    """
        RobotInfo class to represent the robot's information
        contains the robot's position, direction, and other relevant information
    """
    def __init__(self, location:Point, direction:float, action:str = None):
        self.location:Point = location
        self.direction:float = direction  # in radians
        self.action:str = action  # action to be performed by the robot, e.g. "move", "rotate", "pickup"

class Movement(RobotInfo):
    """
        Movement class to represent a movement in 2D space for the robot
        robot will only read distance
    """
    def __init__(self, distance:float, location:Point, direction:float):
        super().__init__(location, direction, "move")
        self.distance:float = distance

class Rotation(RobotInfo):
    """
        Rotation class to represent an angle change in radians for the robot
        possitive angle is counter-clockwise, negative angle is clockwise
        robot will only read angle
    """
    def __init__(self, angle:float, location:Point, direction:float):
        super(location, direction,"rotate")
        self.angle:float = angle

class Pickup(RobotInfo):
    """
        Pickup class to represent a pickup action for the robot
    """
    def __init__(self, location:Point, direction:float):
        super(location, direction, "pickup")

class Wall:
    """
        Wall class to represent a wall in the environment
        wall is represented by two points: start and end
        can generate the angle of the wall
    """
    def __init__(self, wall:np.array):
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
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
    def length(self) -> float:
        """Calculates the length of the line"""
        return math.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)
    def intersects(self, walls:List[Wall]) -> bool:
        """Checks if the line intersects with any of the walls"""
        for wall in walls:
            if self._intersects(wall):
                return True
        return False
    def _intersects(self, wall:Wall) -> bool:
        col = Point(0, 0)
        func1 = self._asFunction()
        func2 = wall._asLine()._asFunction()
        col.x = (func1[1] * func2[2] - func2[1] * func1[2]) / (func1[0] * func2[1] - func2[0] * func1[1])
        col.y = (func1[2] * func2[0] - func2[2] * func1[0]) / (func1[0] * func2[1] - func2[0] * func1[1])
        if col.x < min(self.start.x, self.end.x) or col.x > max(self.start.x, self.end.x):
            return False
        if col.y < min(self.start.y, self.end.y) or col.y > max(self.start.y, self.end.y):
            return False
        if col.x < min(wall.start.x, wall.end.x) or col.x > max(wall.start.x, wall.end.x):
            return False
        if col.y < min(wall.start.y, wall.end.y) or col.y > max(wall.start.y, wall.end.y):
            return False
        return True  # returns True if the line intersects with the wall
    
    def _asFunction(self) -> List[float]:
        """Converts the line to a function of y = mx + b"""
        out = []
        if self.start.x < self.end.x:
            self.start, self.end = self.end, self.start
        if(self.start.x == self.end.x):
            return None
        
        out.append(self.end.y - self.start.y)  # a
        out.append(self.start.x - self.end.x)  # b
        out.append(self.start.y * (self.end.x - self.start.x) - (self.end.y - self.start.y) * self.start.x)  # c
        
        return out  # returns [a, b, c] for the line equation ax + by + c = 0
    
    def Shift(self, offset:int, angle:float = None) -> List['Line']:
        """Shifts the line by a given offset in both directions"""
        if angle is None:
            angle = self.angle()
        
        if offset == 0:
            return [self,self]  # returns the original line if offset is zero or negative
        
        offset = abs(offset)
        
        out = []
        
        offset_x = offset * math.cos(angle)
        offset_y = offset * math.sin(angle)
        
        new_start = Point(self.start.x + offset_x, self.start.y + offset_y)
        new_end = Point(self.end.x + offset_x, self.end.y + offset_y)
        out.append(Line(new_start, new_end))
        new_start = Point(self.start.x - offset_x, self.start.y - offset_y)
        new_end = Point(self.end.x - offset_x, self.end.y - offset_y)
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
    def __init__(self, car:List[np.array], front:List[np.array] = None):
        self.triangle:List[Point] = []
        for point in car:
            self.triangle.append(Point(point[0][1], point[0][0]))
        self.front:Point = Point(front[0][1], front[0][0]) if front is not None and len(front) == 2 else Point(0, 0)
        
        while(len(self.triangle) < 3):
            self.triangle.append(Point(0, 0))
        if(len(self.triangle) > 3):
            self.triangle = self.triangle[0:3]
        
        self.valid()

        self.offset:int = 10
    
    def area(self) -> float:
        """Calculates the area of the triangle formed by the car's points"""
        x1, y1 = self.triangle[0].x, self.triangle[0].y
        x2, y2 = self.triangle[1].x, self.triangle[1].y
        x3, y3 = self.triangle[2].x, self.triangle[2].y
        
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)

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
        if not self.valid():
            return 0.0
        return Line(self.getRotationCenter(), self.triangle[0]).angle()
    
    def getWidth(self) -> float:
        """Calculates the width of the car based on the distance between the base points"""
        if not self.valid():
            return 0.0
        return Line(self.triangle[1:3]).length()
    
    def getRotationCenter(self) -> Point:
        """Calculates the center of the car's rotation based on its triangle points"""
        if not self.valid():
            return Point(0, 0)
        return Point.average(self.triangle[1:3])
    
    def getRotationDiameter(self) -> float:
        """Calculates the diameter of the car based on the distance between the front point and the base points"""
        if not self.valid():
            return 0.0
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
                self.center.x + self.radius * math.cos(self.start), #x0
                self.center.y + self.radius * math.sin(self.start)  #y0
            ),Point(
                self.center.x + self.radius * math.cos(self.end),   #x1
                self.center.y + self.radius * math.sin(self.end)    #y1
            )
        ]
    
    def _intersects(self, wall:Wall) -> bool:
        """Checks if the arc intersects with a given wall"""
        # Calculate the distance from the center of the arc to the line segment defined by the wall
        line = wall._asLine()
        
        polarradius = math.sqrt((line.start.x - self.center.x) ** 2 + (line.start.y - self.center.y) ** 2)
        if (line.angle() >= self.start and line.angle() <= self.end and polarradius < self.radius):
            return True  # returns True if the start is within the arc
        
        polarradius = math.sqrt((line.end.x - self.center.x) ** 2 + (line.end.y - self.center.y) ** 2)
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
        
        deltaX = line.end.x - line.start.x
        deltaY = line.end.y - line.start.y
        deltaR = math.sqrt(deltaX ** 2 + deltaY ** 2)
        delta = line.start.x * line.end.y - line.end.x * line.start.y
        
        # Check if the line segment intersects with the arc
        discriminant = self.radius ** 2 * deltaR ** 2 - delta ** 2
        
        if discriminant < 0:
            return False
        
        # Calculate the intersection points
        x0 = (delta * deltaY + __sgn(deltaY) * deltaX * math.sqrt(discriminant))/(deltaR ** 2)
        y0 = (-delta * deltaX + abs(deltaY) * math.sqrt(discriminant))/(deltaR ** 2)
        x1 = (delta * deltaY - __sgn(deltaY) * deltaX * math.sqrt(discriminant))/(deltaR ** 2)
        y1 = (-delta * deltaX - abs(deltaY) * math.sqrt(discriminant))/(deltaR ** 2)
        
        intersections = [Point(x0, y0).move(self.center), Point(x1, y1).move(self.center)]
        for intersection in intersections:
            # Check if the intersection point is within the arc's angle range
            angle = math.atan2(intersection.y - self.center.y, intersection.x - self.center.x)
            if self.start <= angle <= self.end:
                return True
        
        return False