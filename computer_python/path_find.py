import cv2
import numpy as np
from typing import List, Tuple, Union
from image_recognition import Camera, polygonArea
from classes import Point, Wall, Car, Movement, Rotation, deliver, Line, Arc
from Log import printLog
import math
from typing import Callable, Optional
from random import random

def deltaRotation(newAngle:float, currentAngle:float) -> float:
    """Generates the rotation needed to turn the car to the new angle"""
    rotation = newAngle - currentAngle
    if rotation > math.pi:
        rotation -= 2 * math.pi
    elif rotation < -math.pi:
        rotation += 2 * math.pi
    return rotation

class track:
    def __init__(self, cam:Camera,
            walls:List[List[tuple[int | float, int | float, int | float, int | float]]] |None = None, 
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
        self.approach_point: Point | None = None
        self.delivery_goal: Point | None = None
        self.extra_obstacles: List[Wall] = []
    
    def update(self, walls:bool | int = False, goals:bool = False, targets:bool = False, obsticles:bool = False, car:bool = False, frame:np.ndarray | None= None):
        if(frame is None):
            frame = self.cam.getFrame()
        
        corners = None
        
        if(frame is None):
            printLog("error","No frame received from camera.",producer="update track")
            return
        
        if(walls):
            printLog("DEBUG", "updating walls",producer="update track")
            oldWallStyle = self.cam.generateWall(40 if type(walls) is bool else walls)
            corners = self.cam.findCorners(oldWallStyle)
            newWallStyle = self.cam.makeWalls(corners)
            if(newWallStyle is not None and len(newWallStyle) > 0):
                self.walls = newWallStyle
        
        if(goals):
            if(corners is None or len(corners) == 0):
                corners = self.cam.findCorners(self.cam.generateWall(40 if type(goals) is bool else goals))
            goalsPoints = self.cam.makeGoals(corners)
            if(goalsPoints is not None):
                printLog("DEBUG", f"found {len(goalsPoints)} goals",producer="update track")
                self.goals = [*goalsPoints]
            else:
                printLog("DEBUG", "no goals found",producer="update track")
                self.goals = []
                self.goals = self.formatGoals(self.cam.midpointWalls(self.cam.shape[1], self.cam.walls))
        
        if(targets):
            printLog("DEBUG", "updating targets",producer="update track")
            detected = self.cam.findCircle(np.copy(frame))
            if detected:
                all_targets = self.formatTargets(detected)
                self.targets = [t for t in all_targets if not self.is_target_too_close(t)]
            else:
                self.targets = []
        
        if(obsticles is not None):
            printLog("DEBUG", "updating obsticles",producer="update track")
            self.obsticles = self.formatObsticles(self.cam.findEgg(np.copy(frame)))
        
        if(car):
            printLog("DEBUG", "updating car",producer="update track")
            tempCar:Tuple[List[Tuple[List[int | float], str]], Tuple[List[int | float], str]] | None = self.cam.findCar(frame)
            fails = 0
            while(tempCar is None or len(tempCar[0]) <= 2): 
                if(tempCar is not None):
                    printLog("DEBUG", "falied car length",len(tempCar[0]),producer="update track")
                else:
                    printLog("DEBUG", "no car found",producer="update track")
                fails += 1
                tempCar = self.cam.findCar(frame)
                if(fails == 5):
                    return None
            car_,front_ = tempCar
            car__ = self.formatCar(car_, front_)
            if(car__ is None or not car__.valid()):
                printLog("DEBUG", "car is not valid",producer="update track")
                return None
            self.car = car__
            printLog("DEBUG", f"car updated with {len(self.car.triangle)} points",producer="update track")

            cross_lines = self.cam.findCross(self.cam.walls)
            if cross_lines:
                self.extra_obstacles = list(cross_lines)
            else:
                self.extra_obstacles = []
        
        return 1
    
    def is_target_too_close(self, target: Point, min_distance: float = 25) -> bool:
        """Returns True if the target is too close to any wall or extra obstacle (like the cross)."""
        for wall in self.walls + self.extra_obstacles:
            if Line(wall.start, wall.end).distanceTo(target) < min_distance:
                return True
        if self.car:
            for pt in self.car.triangle:
                if pt.distanceTo(target) < min_distance:
                    return True
            if self.car.front.distanceTo(target) < min_distance:
                return True
        return False
    
    def formatWalls(self, walls:List[List[tuple[int | float, int | float, int | float, int | float]]] | None) -> List[Wall]:
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
            realGoals.append(Point(goal[0], goal[1]))
        return realGoals
    def formatTargets(self, targets:Union[List[Tuple[List[int | float],str]],None]) -> List[Point]:
        realTargets = []
        if(targets is None or len(targets) == 0):
            return []
        for target in targets:
            realTargets.append(Point(target[0][0], target[0][1]))
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
                triangle.append(Point(point[0][0], point[0][1]))
            if(len(triangle) < 3):
                return self.car
            if(len(triangle) > 3):
                triangle = triangle[0:3]
            front_point:Point = Point(0, 0)
            if(front is not None):
                front_point = Point(front[0][0], front[0][1])
            return Car(triangle,front_point)
        return Car([Point(0,0),Point(0,1),Point(1,0)], Point(0, 0))  # Default car if no car is provided
    
    
    def is_path_safe(self, car: 'Car', target: 'Point', walls: list['Wall'], buffer: float | None = None) -> bool:
        """
        Check if the car can safely move straight toward the target.
        Extend the car’s bounding box lines and ensure they don’t intersect walls.
        """
        if buffer is None:
            buffer = target.distanceTo(car.front)
        
        buffer += 40  # ⬅️ Add safety margin

        bounding_lines = car.getBoundingBox()
        for line in bounding_lines:
            extended = line.extend(buffer)
            for wall in walls:
                if extended.intersects([wall]):
                    return False
        
        # Also check center point distance from any wall
        for wall in walls:
            if Line(wall.start, wall.end).distanceTo(car.getRotationCenter()) < car.radius + 10:
                return False

        return True


        
    def arc_intersects_wall(self, arc: Arc, walls: list[Wall], robot_radius: float = 0.0) -> bool:
        return arc.Intersects(walls)
    
    def find_detour_target(self, original_target: Point, car: Car, walls, is_path_clear_fn: Callable, robot_radius: float) -> Optional[Point]:
        """Try points in circles around the target to find a reachable detour point"""
        for radius in range(30, 121, 30):  # Try 30, 60, 90, 120 pixels away
            for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):  # 16 directions
                dx = radius * math.cos(angle)
                dy = radius * math.sin(angle)
                detour = Point(original_target.x + dx, original_target.y + dy)
                if is_path_clear_fn(car, detour, walls, robot_radius):
                    return detour
        return None
    
    def find_safe_arc(self,car, angle_to_target: float, walls) -> float | None:
        """Try nearby angles to find a safe turning arc around a wall"""
        direction = car.getRotation()
        center = car.getRotationCenter()
        radius = car.radius

        angle_offsets = [-0.5, 0.5, -1.0, 1.0]  # radians ~[-30°, 30°, -60°, 60°]
        for offset in angle_offsets:
            test_angle = angle_to_target + offset
            arc = Arc(center=center, startAngle=direction, endAngle=test_angle, radius=radius)
            if arc.Intersects(walls):
                continue
            return test_angle
        return None


    def generatepath(self, target:Point | None = None, checkTarget:bool = True, attempt:int = 0, car:Car | None = None) -> tuple[List[Movement | Rotation | deliver],Point |None]:
        """Generates a path from the car to the closest target"""
        MAX_ATTEMPTS = 15
        walls = self.walls + self.extra_obstacles
        if attempt > MAX_ATTEMPTS:
            printLog("ERROR", "Pathfinding recursion limit reached: ", attempt, producer="pathGenerator")
            return [Movement(-10) if random() > 0.3 else Movement(10)], target
        path: List[deliver | Movement | Rotation] = []
        
        # Copy car to simulate forward steps
        if(car is None):
            car = self.car.copy()
        car_center = car.getRotationCenter()
        direction = car.getRotation()
        front = car.front
        if target is None:
            if self.targets is None or len(self.targets) == 0 or self.car.front is None:
                printLog("DEBUG", "no targets or no car",producer="pathGenerator")
                return path,None
            # Find the closest target
            self.targets.sort(key=lambda t: front.distanceTo(t))
            target = self.targets[0]
            
            for i in range(len(self.targets)):
                if car.validTarget(self.targets[i]):
                    target = self.targets[i]
                    printLog(f"DEBUG", "found destination",producer="pathGenerator")
                    break
            else:
                printLog("DEBUG", "no valid targets",producer="pathGenerator")
        elif checkTarget:
            self.targets.sort(key=lambda t: target.distanceTo(t)) # type: ignore
            for i in range(len(self.targets)):
                if car.validTarget(self.targets[i]):
                    target = self.targets[i]
                    printLog("DEBUG", "adjusted target",producer="pathGenerator")
                    break
            else:
                printLog(f"DEBUG", "failed to adjust target",producer="pathGenerator")
        else:
            printLog("DEBUG", f"using provided target: ({target.x:.2f}, {target.y:.2f})",producer="pathGenerator")
        
        while True:
            if(attempt > MAX_ATTEMPTS):
                break
            # Calculate vector to target
            dy = target.y - car_center.y
            dx = target.x - car_center.x
            angle_to_target = math.atan2(dy, dx)
            car_center = car.getRotationCenter()
            direction = car.getRotation()
            front = car.front
            # Compute required rotation
            rotation_amount = deltaRotation(direction, angle_to_target)

            arc = Arc(center=car_center, startAngle=direction, endAngle=angle_to_target, radius=car.radius)
            if arc.Intersects(walls):
                safe_angle = self.find_safe_arc(car, angle_to_target, self.walls)
                if arc.Intersects(walls):
                    safe_angle = self.find_safe_arc(car, angle_to_target, walls)
                    if safe_angle is None:
                        detour = self.find_detour_target(target, car.copy(), walls, self.is_path_safe, car.radius)
                        if detour:
                            printLog("DEBUG", f"Detouring around arc-blocked wall to ({detour.x:.2f}, {detour.y:.2f})", producer="pathGenerator")
                            return path + self.generatepath(target=detour, checkTarget=False, attempt=attempt+1, car=car.copy())[0], target
                        else:
                            printLog("DEBUG", "No valid detour found after arc failed, backing up", producer="pathGenerator")
                            backup_distance = -30
                            path.append(Movement(backup_distance))
                            car.applySelf(path[-1])
                            morePath, _ = self.generatepath(target=target, checkTarget=checkTarget, attempt=attempt+1, car=car.copy())
                            return path + morePath, target
                else:
                    printLog("DEBUG", f"Adjusted rotation angle to avoid wall: {safe_angle:.2f}", producer="pathGenerator")
                    angle_to_target = safe_angle
                    if direction is None or safe_angle is None:
                        printLog("ERROR", "Cannot compute rotation: direction or angle is None", producer="pathGenerator")
                        return [Movement(-20)], target
                    rotation_amount = deltaRotation(direction, safe_angle)
            
            if(abs(rotation_amount) > 0.1):
                path.append(Rotation(rotation_amount))
                # Proactively try detour if too close to wall, even if technically safe
                if not self.is_path_safe(car, target, self.walls, buffer=car.radius + 10):
                    printLog("DEBUG", "Path unsafe or tight — finding detour", producer="pathGenerator")
                    detour = self.find_detour_target(target, car, self.walls, self.is_path_safe, car.radius)
                    if detour:
                        printLog("DEBUG", f"Using detour to ({detour.x:.2f}, {detour.y:.2f})", producer="pathGenerator")
                        morePath, _ = self.generatepath(target=detour, checkTarget=False, attempt=attempt+1, car=car.copy())
                        return path + morePath, target
                    else:
                        printLog("DEBUG", "No valid detour found, backing up", producer="pathGenerator")
                        path.append(Movement(-30))
                        car.applySelf(path[-1])
                        morePath, _ = self.generatepath(target=target, checkTarget=checkTarget, attempt=attempt+1, car=car.copy())
                        return path + morePath, target

            
            # Compute forward movement
            distance = car.front.distanceTo(target)
            
            #check if car hits something with a boundning box
            if not self.is_path_safe(car, target, self.walls, buffer=car.radius):
                detour = self.find_detour_target(target, car, walls, self.is_path_safe, car.radius)
                if detour:
                    printLog("DEBUG", f"Using detour to ({detour.x:.2f}, {detour.y:.2f})", producer="pathGenerator")
                    morePath = self.generatepath(target=detour, checkTarget=False, attempt=attempt+1,car = car)
                    return path + morePath[0], target
                else:
                    printLog("DEBUG", "No valid detour found, backing up", producer="pathGenerator")
                    # Add a backup movement and try again
                    distance = -30
            
            distance = min(distance, 75)  # Limit to a maximum of 75 pixels per step
            
            path.append(Movement(distance))
            car.applySelf(path[-1])  # apply movement to simulate robot state
            
            if(abs(distance) < 10):
                printLog("DEBUG", "Target is too close, stopping", producer="pathGenerator")
                break            
            
            # Debug info
            printLog("DEBUG", f"Target: ({target.x:.2f}, {target.y:.2f})",producer="pathGenerator")
            printLog("DEBUG", f"Front:   ({front.x:.2f}, {front.y:.2f})",producer="pathGenerator")
            printLog("DEBUG", f"Angle to target: {angle_to_target:.2f} rad",producer="pathGenerator")
            printLog("DEBUG", f"Rotation applied: {rotation_amount:.2f} rad",producer="pathGenerator")
            printLog("DEBUG", f"Movement: {distance:.2f} px",producer="pathGenerator")
            if(car.front.distanceTo(target) < 10):
                break
            attempt += 1
        
        return (path,target)


    def drawCar(self, frame:np.ndarray,car:Car) -> np.ndarray:
        """Draws the car on the frame"""
        for i in range(len(car.triangle)):
            p0 = car.triangle[i]
            p1 = car.triangle[(i + 1) % 3]
            cv2.line(frame, (int(p0.x), int(p0.y)), (int(p1.x), int(p1.y)), (255, 255, 0), 2)
        return frame
    def Draw(self, frame:np.ndarray,path = None, target = None):
        """Draws the track on the frame"""
        for wall in self.walls:
            cv2.line(frame, (int(wall.start.x), int(wall.start.y)), (int(wall.end.x), int(wall.end.y)), (0, 0, 255), 1)

        for wall in self.extra_obstacles:
            cv2.line(frame, (int(wall.start.x), int(wall.start.y)), (int(wall.end.x), int(wall.end.y)), (255, 0, 255), 2)

        for goal in self.goals:
            cv2.circle(frame, (int(goal.x), int(goal.y)), 5, (255, 0, 0), -1)
        
        for target_ in self.targets:
            cv2.circle(frame, (int(target_.x), int(target_.y)), 5, (0, 255, 0), -1)
        
        for obsticle in self.obsticles:
            cv2.circle(frame, (int(obsticle.x), int(obsticle.y)), 5, (0, 255, 255), -1)
        
        frame =self.drawCar(frame,self.car)
        
        car: Car = self.car.copy()

        #draw approach point if it exists
        if self.approach_point is not None and target is not None:
            cv2.circle(frame, (int(self.approach_point.x), int(self.approach_point.y)), 5, (150, 0, 255), -1)
            cv2.line(frame,
                    (int(self.approach_point.x), int(self.approach_point.y)),
                    (int(target.x), int(target.y)),
                    (150, 0, 255), 1)
            cv2.putText(frame, "Approach Point", (int(self.approach_point.x)+5, int(self.approach_point.y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 0, 255), 1)


# Compute path once
        if(path is None):
            path,target_ = self.generatepath()
        if(target is None):
            target = Point(0, 0)
# Draw the path step-by-step
        for step in path:
            prev_front = car.front.copy()
            prev_center = car.getRotationCenter().copy()
            
            car.applySelf(step)
            
            new_front = car.front
            new_center = car.getRotationCenter()

            if isinstance(step, Movement):
                # Movement is visualized from the old front to the new front
                cv2.arrowedLine(frame,
                                (int(prev_front.x), int(prev_front.y)),
                                (int(new_front.x), int(new_front.y)),
                                (0, 255, 0), 1)

            if isinstance(step, Rotation):
                # Rotation is visualized from center before to front after
                cv2.arrowedLine(frame,
                                (int(prev_center.x), int(prev_center.y)),
                                (int(new_center.x), int(new_center.y)),
                                (255, 255, 255), 1)
                cv2.putText(frame,
                            f"Rotate: {step.angle:.2f} rad",
                            (int(new_front.x), int(new_front.y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame = self.drawCar(frame, car)

        cv2.putText(frame, "walls: red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "goals: blue", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "targets: green", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "obsticles: yellow", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "car: cyan", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Press 'q' to exit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if(target is not None):
            cv2.circle(frame, (int(target.x), int(target.y)), 5, (0,0,0), -1)
        
        return frame
    
    def test(self):
        """Test function to show the track"""
        while True:
            self.update(walls=False, goals=False, targets=True, obsticles=True, car=True)
            frame:np.ndarray | None = self.cam.getFrame()
            if(frame is None):
                printLog("error","No frame received from camera.",producer="test track")
                break
            self.Draw(frame)
            self.cam.displayFrame(frame,"Track")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def testPath(self):
        """Test function to show the path"""
        return [Movement(10), Rotation(math.pi), Movement(5)]
