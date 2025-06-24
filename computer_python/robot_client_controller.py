# Full update to add smarter delivery with wall safety and left-goal prioritization

import math
import time
import socket
from classes import Movement, Rotation, Point, deliver
from path_find import track, deltaRotation
from image_recognition import Camera
from Log import enableLog, printLog, closeLog
from typing import List
from playsound import playsound

host = '0.0.0.0'
port = 12345

enableLog()
printLog("INFO", "Logging enabled", producer="init Client")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

printLog("status", f"Waiting on {host}:{port}...", producer="init Client")
playsound("ready.mp3")

client_socket, client_address = server_socket.accept()
printLog("status", f"Connected to: {client_address}", producer="init Client")

data = client_socket.recv(1024)
printLog("status", "Received", data.decode(), producer="init Client")

cam = Camera(debug=True)
robot_track = track(cam)
target: Point | None = None
t = time.time()
hasBall = False
ballFrames = 0
frameNumber = 0
path: List[Movement | Rotation | deliver] = []
delivering = False
approach_point = None
rotating_to_deliver = False

try:
    while True:
        if time.time() != t:
            printLog("FPS", str(1 / (time.time() - t)), producer="client Loop")
            printLog("time", "frame number:", frameNumber, producer="client loop")
            t = time.time()
            frameNumber += 1

        printLog("STATUS", "generating frame", producer="client Loop")
        frame = robot_track.cam.getFrame()
        if frame is None:
            break

        robot_track.cam.displayFrame(frame, "success", False)
        response = robot_track.update(walls=(10 + (ballFrames // 10)) if frameNumber % 10 == 0 else False, goals=True if frameNumber % 10 == 0 else False, targets=True, obsticles=False, car=True, frame=frame)

        if response is None:
            printLog("RETRY", "no car", producer="client Loop")
            robot_track.cam.displayFrame(frame, "fail", True)
            step = Movement(-10)
            continue

        if not hasBall:
            path, target = robot_track.generatepath(target)
            delivering = False
            rotating_to_deliver = False
            robot_track.approach_point = None
            robot_track.delivery_goal = None

        else:
            car_front = robot_track.car.front

            if not delivering:
                if not robot_track.goals:
                    printLog("ERROR", "No goals available", producer="client loop")
                    continue

                # Select left-most goal
                goal = min(robot_track.goals, key=lambda g: g.x)
                robot_track.delivery_goal = goal

                # Compute safe approach point
                dx = goal.x - car_front.x
                dy = goal.y - car_front.y
                angle = math.atan2(dy, dx)
                approach_distance = 80
                approach_x = goal.x - approach_distance * math.cos(angle)
                approach_y = goal.y - approach_distance * math.sin(angle)
                approach_point = Point(approach_x, approach_y)

                # Check for wall safety
                if not robot_track.is_path_clear(car_front, approach_point, robot_track.walls, robot_track.car.radius):
                    printLog("WARN", "Approach blocked, adjusting", producer="client loop")
                    approach_x = goal.x - 100 * math.cos(angle + 0.3)
                    approach_y = goal.y - 100 * math.sin(angle + 0.3)
                    approach_point = Point(approach_x, approach_y)

                robot_track.approach_point = approach_point

                path, _ = robot_track.generatepath(approach_point, checkTarget=False)

                if car_front.distanceTo(approach_point) < 15:
                    delivering = True
                    path = []

            elif rotating_to_deliver:
                rotating_to_deliver = False
                path = [deliver()]

            else:
                goal = robot_track.delivery_goal
                if goal is None:
                    continue

                car = robot_track.car.copy()
                center = car.getRotationCenter()
                direction = car.getRotation()

                dx = goal.x - center.x
                dy = goal.y - center.y
                angle_to_goal = math.atan2(dy, dx)
                rot = deltaRotation(direction, angle_to_goal)

                if abs(rot) > 0.1:
                    path = [Rotation(rot)]
                    rotating_to_deliver = True
                else:
                    rotating_to_deliver = True
                    path = []

        robot_track.Draw(frame, path, target)
        robot_track.cam.displayFrame(frame, "Track")

        if not path:
            printLog("RETRY", "failed to follow path (empty)", producer="client Loop")
            continue

        step = path[0]

        # COMMAND GENERATION
        if isinstance(step, deliver):
            cmd = "deliver 1"
        elif isinstance(step, Movement):
            if step.distance > 0:
                cmd = f"drive {step.distance / 200}"
            elif step.distance < 0:
                cmd = f"backward {abs(step.distance) / 200}"
            else:
                continue
        elif isinstance(step, Rotation):
            if abs(step.angle) < 0.05:
                continue
            angle_degrees = math.degrees(step.angle)
            cmd = f"rotate {-angle_degrees / 3}"
        else:
            continue

        printLog("command", "sending command", cmd, producer="client sender")
        client_socket.sendall(cmd.encode())

        response = client_socket.recv(1024).decode()
        while response.startswith("OKOK"):
            response = response[2:]
        while response.endswith("OK") and len(response) > 2:
            response = response[:-2]

        if not response.startswith("OK"):
            continue
        elif response == "OK ball caught":
            robot_track.update(goals=True)
            robot_track.update(walls=True, goals=True, targets=False, obsticles=False, car=False, frame=frame)
            target = robot_track.goals[0].move(Point(-100, 0))
            hasBall = True
            ballFrames = 0
            time.sleep(2.3)
        elif response == "OK ball lost":
            target = None
            hasBall = False

except Exception as e:
    import traceback
    printLog("ERROR", f"Exception: {e}", producer="client cleanup")
    printLog("ERROR", traceback.format_exc(), producer="client cleanup")
finally:
    try:
        client_socket.close()
        server_socket.close()
    except Exception as e:
        printLog("ERROR", f"Failed to close socket: {e}", producer="client cleanup")
    closeLog()
