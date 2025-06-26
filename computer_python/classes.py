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
        deliver class to represent a delivery action for the robot.
        Inherits from Movement and represents a forward push to deliver the ball.
    """
    def __init__(self):
        super().__init__(50)

class Wall:
    """
        Wall class to represent a wall in the environment
        wall is represented by two points: start and end
        can generate the angle of the wall
    """
    def __init__(self, wall:List[tuple[Union[int,float],int | float,int | float,int | float]]):
        """Initializes the Wall with a list of points
            correct form: [(x1, y1, x2, y2)]
        """
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
    

    def _intersects(self, wall: Wall, extend: int = 0, tolerance: float = 1e-3) -> bool:
        """Checks if this line segment intersects with a wall segment."""
        # Extend the current line by the specified amount
        extended_line = self.extend(extend, even=True) # Apply half extension to each side if even
        extended_wall = wall._asLine().extend(extend, even=True) # Apply half extension to each side if even

        # Convert both lines to their function representation
        a1, b1, c1 = extended_line._asFunction()
        a2, b2, c2 = extended_wall._asFunction()

        # Calculate the determinant
        det = a1 * b2 - a2 * b1

        # If determinant is close to zero, lines are parallel or coincident
        if abs(det) < tolerance:
            # Check for collinearity and overlap if parallel

            # A more robust collinearity check using cross product
            # (P2-P1) x (P3-P1) == 0 where P1,P2 are on one line, P3 on other
            collinear_check1 = (extended_line.end.x - extended_line.start.x) * (extended_wall.start.y - extended_line.start.y) - \
                               (extended_line.end.y - extended_line.start.y) * (extended_wall.start.x - extended_line.start.x)
            collinear_check2 = (extended_line.end.x - extended_line.start.x) * (extended_wall.end.y - extended_line.start.y) - \
                               (extended_line.end.y - extended_line.start.y) * (extended_wall.end.x - extended_line.start.x)

            if abs(collinear_check1) < tolerance and abs(collinear_check2) < tolerance:
                # Lines are collinear, check for overlap
                # Project onto the axis with larger extent
                
                # Sort points to simplify overlap check
                line1_min_x, line1_max_x = min(extended_line.start.x, extended_line.end.x), max(extended_line.start.x, extended_line.end.x)
                line1_min_y, line1_max_y = min(extended_line.start.y, extended_line.end.y), max(extended_line.start.y, extended_line.end.y)
                wall_min_x, wall_max_x = min(extended_wall.start.x, extended_wall.end.x), max(extended_wall.start.x, extended_wall.end.x)
                wall_min_y, wall_max_y = min(extended_wall.start.y, extended_wall.end.y), max(extended_wall.start.y, extended_wall.end.y)

                overlap_x = max(line1_min_x, wall_min_x) <= min(line1_max_x, wall_max_x) + tolerance
                overlap_y = max(line1_min_y, wall_min_y) <= min(line1_max_y, wall_max_y) + tolerance

                return overlap_x and overlap_y
            return False  # Parallel and non-collinear, no intersection

        # Calculate intersection point
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        intersection = Point(x, y)

        # Check if the intersection point lies within both line segments (including endpoints)
        within_line = (
            min(extended_line.start.x, extended_line.end.x) - tolerance <= intersection.x <= max(extended_line.start.x, extended_line.end.x) + tolerance and
            min(extended_line.start.y, extended_line.end.y) - tolerance <= intersection.y <= max(extended_line.start.y, extended_line.end.y) + tolerance
        )
        within_wall = (
            min(extended_wall.start.x, extended_wall.end.x) - tolerance <= intersection.x <= max(extended_wall.start.x, extended_wall.end.x) + tolerance and
            min(extended_wall.start.y, extended_wall.end.y) - tolerance <= intersection.y <= max(extended_wall.start.y, extended_wall.end.y) + tolerance
        )

        return within_line and within_wall
    
    def _asFunction(self) -> tuple[float, float, float]:
        """Converts the line to a function of ax + by + c = 0"""
        if self.start.y < self.end.y:
            self.start, self.end = self.end, self.start
        
        a = self.end.y - self.start.y  # a = (y2 - y1)
        b = -(self.end.x - self.start.x)  # b = -(x2 - x1)
        c = -(a * self.start.x + b * self.start.y)  # c = -(a * x1 + b * y1)
        
        return (a, b, c)  # returns [a, b, c] for the line equation ax + by + c = 0
    
    def Shift(self, offset:int | float, angle:Union[float,None] = None) -> tuple['Line','Line']:
        """Shifts the line by a given offset in both directions"""
        if angle is None:
            angle = self.angle()
        
        if offset == 0:
            return (self,self)  # returns the original line if offset is zero or negative
        
        offset = abs(offset)
        
        offset_y = offset * math.cos(angle)
        offset_x = offset * math.sin(angle)
        
        new_start = Point(self.start.y + offset_y, self.start.x + offset_x)
        new_end = Point(self.end.y + offset_y, self.end.x + offset_x)
        out1 = Line(new_start, new_end)
        new_start = Point(self.start.y - offset_y, self.start.x - offset_x)
        new_end = Point(self.end.y - offset_y, self.end.x - offset_x)
        out2 = Line(new_start, new_end)
        return (out1,out2)  # returns a list of two lines shifted by the offset in both directions
    
    def move(self, point:Point) -> 'Line':
        """Moves the line by a given point"""
        return Line(self.start.move(point), self.end.move(point))
    
    def negate(self) -> 'Line':
        """Negates the line by swapping the start and end points"""
        return Line(self.start.negate(), self.end.negate())
    
    def distanceTo(self, point: Point) -> float:
        """Returns the shortest distance from the line segment to a given point."""
        px = self.end.x - self.start.x
        py = self.end.y - self.start.y
        
        # If the segment is a point
        if px == 0 and py == 0:
            return point.distanceTo(self.start)

        # Calculate the parameter t for the closest point on the infinite line
        # t = ((P - A) . (B - A)) / |B - A|^2
        t = ((point.x - self.start.x) * px + (point.y - self.start.y) * py) / (px * px + py * py)

        # Clamp t to [0, 1] to find the closest point on the segment
        t = max(0, min(1, t))

        # Project the point onto the segment
        closest_x = self.start.x + t * px
        closest_y = self.start.y + t * py
        # Calculate the distance
        return math.hypot(point.x - closest_x, point.y - closest_y)

    def extend(self, length: float, even: bool = False) -> 'Line':
        """Extends the line by the specified length."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y

        line_length = math.sqrt(dx**2 + dy**2)
        if line_length == 0:
            return self  # Avoid division by zero for zero-length lines

        scale = length / line_length
        if even:
            start = Point(self.start.x - dx * scale / 2, self.start.y - dy * scale / 2)
            end = Point(self.end.x + dx * scale / 2, self.end.y + dy * scale / 2)
        else:
            start = self.start
            end = Point(self.end.x + dx * scale, self.end.y + dy * scale)

        return Line(start, end)


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
        bounds = self.getBoundingLines()
        corners:tuple[Point,Point,Point,Point] = (bounds[2].start, bounds[2].end, bounds[1].start, bounds[1].end)
        
        angle = bounds[0].angle()
        
        delta_X = corners[0].x
        delta_Y = corners[0].y
        
        for corner in corners:
            corner.x -= delta_X
            corner.y -= delta_Y
        
        
        tempCorners = [corners[0]]
        for i in range(1,len(corners)):
            tempCorners.append(corners[i].rotateAround(corners[0], angle))
        corners = (tempCorners[0], tempCorners[1], tempCorners[2], tempCorners[3])
        min_X = min([corner.x for corner in corners])
        max_X = max([corner.x for corner in corners])
        min_Y = min([corner.y for corner in corners])
        max_Y = max([corner.y for corner in corners])
        
        if( \
            abs(min_X) > 0.1 or\
            abs(min_Y) > 0.1 \
        ):
            printLog("ERROR","failed rotation when searching inbounds",producer="Class Car validTarget")
            return False
        
        if(target.x < min_X or target.x > max_X or target.y < min_Y or target.y > max_Y):
            printLog("ERROR","target out of bounds",producer="Class Car validTarget")
            return False
        return True
    @property
    def radius(self) -> float:
        """Returns the turning radius of the car (from rotation center to front)"""
        return Line(self.front, self.getRotationCenter()).length()

    def getBoundingLines(self:'Car') -> tuple['Line','Line','Line']:
        """generate 3 lines, one following the center of the car and two lines that are the previouse line offset to the wheels"""
        centerLine:Line = Line(self.getRotationCenter(),self.front)
        return (centerLine, *centerLine.Shift(self.getWidth() / 2))

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
        return [
            Point(self.center.x + self.radius * math.cos(self.start),
                self.center.y + self.radius * math.sin(self.start)),
            Point(self.center.x + self.radius * math.cos(self.end),
                self.center.y + self.radius * math.sin(self.end))
        ]
    
    def point_at(self, t: float) -> Point:
        """Returns a point on the arc at angle offset `t` from the start angle"""
        angle = self.start + t  # assumes counter-clockwise
        return Point(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle)
        )
    
    def _intersects(self, wall: Wall) -> bool:
        """Checks if the arc intersects with a given wall."""

        def angle_in_arc(angle, start, end):
            """Checks if a given angle is within the arc's angular span."""
            def norm(a): return a % (2 * math.pi)
            angle = norm(angle)
            start = norm(start)
            end = norm(end)

            # Adjust angles for cases crossing 0/2pi boundary
            if start > end:
                return angle >= start or angle <= end
            else:
                return start <= angle <= end

        # Convert the wall to a line
        line = wall._asLine()

        # 1. Check if wall endpoints are inside the arc’s circle and within arc angle
        for point in [line.start, line.end]:
            dx = point.x - self.center.x
            dy = point.y - self.center.y
            dist_sq = dx**2 + dy**2
            
            # Check if within radius (with tolerance for floating point)
            if dist_sq <= self.radius**2 + 1e-6:
                angle = math.atan2(dy, dx)
                if angle_in_arc(angle, self.start, self.end):
                    return True

        # 2. Check if the wall intersects the arc’s circle
        # Translate the wall line to the arc’s local space
        local_start = Point(line.start.x - self.center.x, line.start.y - self.center.y)
        local_end = Point(line.end.x - self.center.x, line.end.y - self.center.y)

        dx_line = local_end.x - local_start.x
        dy_line = local_end.y - local_start.y
        dr2 = dx_line**2 + dy_line**2
        
        # A = local_start.x, B = local_start.y, C = local_end.x, D = local_end.y
        # From line segment intersection with circle:
        # P = A + t * (C - A)
        # (Px - CenterX)^2 + (Py - CenterY)^2 = R^2
        # ( (Ax + t*dx_line) )^2 + ( (Ay + t*dy_line) )^2 = R^2
        # Where CenterX and CenterY are 0 in local space
        # (Ax + t*dx_line)^2 + (Ay + t*dy_line)^2 = R^2
        # (Ax^2 + 2*Ax*t*dx_line + t^2*dx_line^2) + (Ay^2 + 2*Ay*t*dy_line + t^2*dy_line^2) = R^2
        # t^2 * (dx_line^2 + dy_line^2) + 2*t*(Ax*dx_line + Ay*dy_line) + (Ax^2 + Ay^2 - R^2) = 0
        # t^2 * dr2 + 2*t*dot_product + (local_start.x^2 + local_start.y^2 - self.radius^2) = 0
        
        a = dr2
        b = 2 * (local_start.x * dx_line + local_start.y * dy_line)
        c = local_start.x**2 + local_start.y**2 - self.radius**2

        discriminant = b*b - 4*a*c

        if discriminant < -1e-6: # Allow for small negative due to floating point
            return False  # no intersection with circle

        sqrt_disc = math.sqrt(max(0, discriminant)) # Ensure non-negative argument to sqrt

        # 3. Compute the intersection points
        # t = (-b +/- sqrt(discriminant)) / 2a
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        
        # Check if intersection points are on the line segment [0, 1] and within the arc's angle
        for t in [t1, t2]:
            if -1e-6 <= t <= 1.0 + 1e-6: # Check if t is within [0, 1] range with tolerance
                intersection_x_local = local_start.x + t * dx_line
                intersection_y_local = local_start.y + t * dy_line
                
                intersection_world = Point(intersection_x_local + self.center.x, intersection_y_local + self.center.y)

                angle = math.atan2(intersection_world.y - self.center.y, intersection_world.x - self.center.x)
                
                if angle_in_arc(angle, self.start, self.end):
                    # Check if the intersection point is actually on the wall segment
                    # (this is implicitly handled by the 't' range check for the line segment)
                    # The more robust check from _intersects would be:
                    min_wx, max_wx = min(line.start.x, line.end.x), max(line.start.x, line.end.x)
                    min_wy, max_wy = min(line.start.y, line.end.y), max(line.start.y, line.end.y)
                    
                    if (min_wx - 1e-6 <= intersection_world.x <= max_wx + 1e-6 and
                        min_wy - 1e-6 <= intersection_world.y <= max_wy + 1e-6):
                        return True
        return False
