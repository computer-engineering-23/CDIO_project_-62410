import unittest
from classes import Point, Line, Wall, Movement, Rotation, deliver, Car, Arc

class TestLineIntersects(unittest.TestCase):
    def test_intersects_simple(self):
        """Test a simple intersection between two lines."""
        line1 = Line(Point(0, 0), Point(4, 4))
        wall = Wall([(2, 0, 2, 4)])  # Vertical wall crossing line1
        self.assertTrue(line1._intersects(wall), "Expected intersection but got False.")

    def test_no_intersection_parallel(self):
        """Test parallel lines that do not intersect."""
        line1 = Line(Point(0, 0), Point(4, 0))
        wall = Wall([(0, 1, 4, 1)])  # Parallel wall above line1
        self.assertFalse(line1._intersects(wall), "Expected no intersection but got True.")

    def test_no_intersection_disjoint(self):
        """Test lines that are disjoint and do not intersect."""
        line1 = Line(Point(0, 0), Point(1, 1))
        wall = Wall([(2, 2, 3, 3)])  # Wall far away from line1
        self.assertFalse(line1._intersects(wall), "Expected no intersection but got True.")

    def test_intersects_at_endpoint(self):
        """Test intersection at the endpoint of a line."""
        line1 = Line(Point(0, 0), Point(2, 2))
        wall = Wall([(2, 2, 4, 4)])  # Wall starting at line1's endpoint
        self.assertTrue(line1._intersects(wall), "Expected intersection at endpoint but got False.")

    def test_intersects_extended_line(self):
        """Test intersection when lines are extended."""
        line1 = Line(Point(0, 0), Point(1, 1))
        wall = Wall([(2, 2, 3, 3)])  # Wall intersects extended line1
        self.assertTrue(line1._intersects(wall, extend=2), "Expected intersection with extended line but got False.")

    def test_intersects_with_tolerance(self):
        """Test intersection with a small tolerance."""
        line1 = Line(Point(0, 0), Point(1, 1))
        wall = Wall([(1.001, 1.001, 2, 2)])  # Wall slightly off due to precision
        self.assertTrue(line1._intersects(wall, tolerance=1e-2), "Expected intersection with tolerance but got False.")

    def test_no_intersection_with_large_tolerance(self):
        """Test no intersection when tolerance is too large."""
        line1 = Line(Point(0, 0), Point(1, 1))
        wall = Wall([(2, 2, 3, 3)])  # Wall far away from line1
        self.assertFalse(line1._intersects(wall, tolerance=1), "Expected no intersection but got True.")

class TestPoint(unittest.TestCase):
    def test_point_equality(self):
        """Test equality of two points."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        self.assertTrue(p1 == p2, "Expected points to be equal.")

    def test_point_move(self):
        """Test moving a point."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        result = p1.move(p2)
        self.assertEqual(result, Point(4, 6), "Expected moved point to be (4, 6).")

    def test_point_negate(self):
        """Test negating a point."""
        p1 = Point(1, -2)
        result = p1.negate()
        self.assertEqual(result, Point(-1, 2), "Expected negated point to be (-1, 2).")

    def test_point_distance(self):
        """Test distance between two points."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        self.assertEqual(p1.distanceTo(p2), 5, "Expected distance to be 5.")

class TestWall(unittest.TestCase):
    def test_wall_as_line(self):
        """Test converting a wall to a line."""
        wall = Wall([(0, 0, 4, 4)])
        line = wall._asLine()
        self.assertEqual(line.start, Point(0, 0), "Expected line start to be (0, 0).")
        self.assertEqual(line.end, Point(4, 4), "Expected line end to be (4, 4).")

    def test_wall_intersect(self):
        """Test intersection between two walls."""
        wall1 = Wall([(0, 0, 4, 4)])
        wall2 = Wall([(0, 4, 4, 0)])
        result = wall1.intersect(wall2)
        self.assertEqual(result, Point(2, 2), "Expected intersection point to be (2, 2).")

class TestMovement(unittest.TestCase):
    def test_movement_initialization(self):
        """Test movement initialization."""
        movement = Movement(10)
        self.assertEqual(movement.distance, 10, "Expected movement distance to be 10.")

class TestRotation(unittest.TestCase):
    def test_rotation_initialization(self):
        """Test rotation initialization."""
        rotation = Rotation(1.57)
        self.assertEqual(rotation.angle, 1.57, "Expected rotation angle to be 1.57.")

class TestDeliver(unittest.TestCase):
    def test_deliver_initialization(self):
        """Test deliver action initialization."""
        deliver_action = deliver()
        self.assertEqual(deliver_action.distance, 50, "Expected deliver distance to be 50.")

class TestCar(unittest.TestCase):
    def test_car_area(self):
        """Test calculating the area of a car."""
        triangle = [Point(0, 0), Point(100, 0), Point(50, 40)]  # Larger triangle
        car = Car(triangle, Point(50, 40))
        self.assertAlmostEqual(car.area(), 2000, "Expected car area to be 2000.")

    def test_car_valid(self):
        """Test if the car is valid."""
        triangle = [Point(0, 0), Point(100, 0), Point(50, 40)]  # Larger triangle
        car = Car(triangle, Point(50, 40))
        self.assertTrue(car.valid(), "Expected car to be valid.")

class TestArc(unittest.TestCase):
    def test_arc_points(self):
        """Test generating points on an arc."""
        arc = Arc(Point(0, 0), 0, 3.14, 5)
        points = arc.points()
        self.assertGreater(len(points), 0, "Expected arc to generate points.")

    def test_arc_intersects(self):
        """Test intersection between an arc and walls."""
        arc = Arc(Point(0, 0), 0, 3.14, 5)
        wall = Wall([(0, 0, 5, 5)])
        self.assertTrue(arc.Intersects([wall]), "Expected arc to intersect with wall.")

if __name__ == "__main__":
    unittest.main()