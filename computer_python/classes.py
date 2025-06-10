class Point:
    """Point class to represent a point in 2D space"""
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

class Movement:
    """Movement class to represent a movement in 2D space"""
    def __init__(self, distance:float, direction:float):
        self.distance = distance
        self.direction = direction

class Rotation:
    """Rotation class to represent an angle change in radians"""
    def __init__(self, angle:float):
        self.angle = angle
    def __init__(self, PointBase0:Point,PointBase1:Point, PointTarget:Point):
        self.angle = math.atan2(PointTarget[1] - PointBase0[1], PointTarget[0] - PointBase0[0]) - math.atan2(PointBase1[1] - PointBase0[1], PointBase1[0] - PointBase0[0])

class Wall:
    def __init__(self, wall:np.array):
        self.wall = wall
        self.start = Point(wall[0][1], wall[0][0])
        self.end = Point(wall[1][1], wall[1][0])