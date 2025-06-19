import cv2
import numpy as np
from typing import List, Tuple, Union
from image_recognition import Camera
from classes import Point, Wall, Car, Pickup, Movement, Rotation
import math

def deltaRotation(newAngle:float, currentAngle:float) -> float:
    """Generates the rotation needed to turn the car to the new angle"""
    if(newAngle < 0):
        newAngle += 2 * math.pi
    if(currentAngle < 0):
        currentAngle += 2 * math.pi
    rotation = newAngle - currentAngle
    if(rotation > math.pi):
        rotation -= 2 * math.pi
    elif(rotation < -math.pi):
        rotation += 2 * math.pi
    return rotation

class track:
    def __init__(self,cam:Camera ,
            walls:Union[List[List[List[Union[int,float]]]],None] = None, 
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
    
    def update(self, walls:bool = False, goals:bool = False, targets:bool = False, obsticles:bool = False, car:bool = False, frame:np.ndarray | None= None):
        if(frame is None):
            frame = self.cam.getFrame()
        
        if(frame is None):
            print("No frame received from camera.")
            return
        
        if(walls):
            self.walls = self.formatWalls(self.cam.generateWall(40))
        
        if(goals):
            self.goals = self.formatGoals(self.cam.midpointWalls(self.cam.shape[1], self.cam.walls))
        
        if(targets):
            detected = self.cam.findCircle(np.copy(frame))
            self.targets = self.formatTargets(detected) if detected else []
        
        if(obsticles is not None):
            self.obsticles = self.formatObsticles(self.cam.findEgg(np.copy(frame)))
        
        if(car):
            tempCar:Tuple[List[Tuple[List[int | float], str]], Tuple[List[int | float], str]] | None = self.cam.findCar(frame)
            fails = 0
            while(tempCar is None or len(tempCar[0]) <= 2): 
                if(tempCar is not None):
                    print("[DEBUG] falied car length:",len(tempCar[0]))
                else:
                    print("[DEBUG] no car found")
                fails += 1
                tempCar = self.cam.findCar(frame)
                if(fails == 5):
                    return None
            car_,front_ = tempCar
            self.car = self.formatCar(car_, front_)
        
        return 1
    
    def formatWalls(self, walls:Union[List[List[List[int | float]]],None]) -> List[Wall]:
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
            realGoals.append(Point(goal[1], goal[0]))
        return realGoals
    def formatTargets(self, targets:Union[List[Tuple[List[int | float],str]],None]) -> List[Point]:
        realTargets = []
        if(targets is None or len(targets) == 0):
            return []
        for target in targets:
            realTargets.append(Point(target[0][1], target[0][0]))
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
                triangle.append(Point(point[0][1], point[0][0]))
            if(len(triangle) < 3):
                return self.car
            if(len(triangle) > 3):
                triangle = triangle[0:3]
            front_point:Point = Point(0, 0)
            if(front is not None):
                front_point = Point(front[0][1], front[0][0])
            return Car(triangle,front_point)
        return Car([Point(0,0),Point(0,1),Point(1,0)], Point(0, 0))  # Default car if no car is provided

    def generatepath(self, target:Point | None = None, checkTarget:bool = True) -> tuple[List[Pickup | Movement | Rotation],Point |None]:
        """Generates a path from the car to the closest target"""
        path: List[Pickup | Movement | Rotation] = []
        
        # Copy car to simulate forward steps
        car = self.car.copy()
        car_center = car.getRotationCenter()
        direction = car.getRotation()
        front = car.front
        
        if target is None:
            if self.targets is None or len(self.targets) == 0 or self.car.front is None:
                print("[DEBUG] no targets or no car")
                return path,None
            # Find the closest target
            self.targets.sort(key=lambda t: front.distanceTo(t))
            target = self.targets[0]
            
            for i in range(len(self.targets)):
                if car.validTarget(self.targets[i]):
                    target = self.targets[i]
                    print(f"[DEBUG] found destination")
                    break
            else:
                print(f"[DEBUG] no valid targets")
        elif checkTarget:
            self.targets.sort(key=lambda t: target.distanceTo(t)) # type: ignore
            for i in range(len(self.targets)):
                if car.validTarget(self.targets[i]):
                    target = self.targets[i]
                    print(f"[DEBUG] adjusted target")
                    break
            else:
                print(f"[DEBUG] failed to adjust target")
        else:
            print(f"[DEBUG] using provided target: ({target.x:.2f}, {target.y:.2f})")
        # Calculate vector to target
        dy = target.y - car_center.y
        dx = target.x - car_center.x
        angle_to_target = math.atan2(dy, dx)
        # Compute required rotation
        rotation_amount = deltaRotation(direction, angle_to_target)
        if(abs(rotation_amount) > 0.1):
            path.append(Rotation(rotation_amount))
            car.applySelf(path[-1])  # apply rotation to simulate robot state
        
        # Compute forward movement
        distance = car.front.distanceTo(target)
        if(distance < 15):
            distance = 35
        path.append(Movement(distance))
        car.applySelf(path[-1])  # apply movement to simulate robot state
        
        # Debug info
        print(f"[DEBUG] Target: ({target.x:.2f}, {target.y:.2f})")
        print(f"[DEBUG] From:   ({front.x:.2f}, {front.y:.2f})")
        print(f"[DEBUG] Angle to target: {angle_to_target:.2f} rad")
        print(f"[DEBUG] Rotation applied: {rotation_amount:.2f} rad")
        print(f"[DEBUG] Movement: {distance:.2f} px")
        
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
        
        for goal in self.goals:
            cv2.circle(frame, (int(goal.x), int(goal.y)), 5, (255, 0, 0), -1)
        
        for target in self.targets:
            cv2.circle(frame, (int(target.x), int(target.y)), 5, (0, 255, 0), -1)
        
        for obsticle in self.obsticles:
            cv2.circle(frame, (int(obsticle.x), int(obsticle.y)), 5, (0, 255, 255), -1)
        
        frame =self.drawCar(frame,self.car)
        
        car: Car = self.car.copy()

# Compute path once
        if(path is None):
            path,target = self.generatepath()
        if(target is None):
            target = Point(0,0)
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
                print("No frame received from camera.")
                break
            self.Draw(frame)
            self.cam.displayFrame(frame,"Track")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        self.cam.close()
        cv2.destroyAllWindows()
    
    def testPath(self):
        """Test function to show the path"""
        return [Movement(10), Rotation(math.pi), Movement(5), Pickup()]
