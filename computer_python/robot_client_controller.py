# -*- coding: utf-8 -*-
import socket
import time
from classes import Movement, Rotation, Point, deliver
from path_find import track, deltaRotation
from image_recognition import Camera
from Log import enableLog, printLog, closeLog, blockTag
import math
import traceback
from typing import List
from playsound import playsound

host = '0.0.0.0'  # Lyt på alle interfaces
port = 12345      # Samme port som EV3-klienten bruger
enableLog()
printLog("INFO", "Logging enabled",producer="init Client")

# Opret TCP-socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

printLog("status",f"Venter på forbindelse på {host}:{port}...",producer="init Client")
playsound("ready.mp3")

# Accepter forbindelse fra klient
client_socket, client_address = server_socket.accept()
printLog("status", f"Forbundet til: {client_address}",producer="init Client")

# Modtag data fra klienten
data = client_socket.recv(1024)
printLog("status","Modtaget", data.decode(),producer="init Client")

cam = Camera(debug=True)
robot_track = track(cam)
target:Point | None = None
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
            printLog("FPS",str(1 / (time.time()-t)),producer="client Loop")
            printLog("time", "frame number:",frameNumber,producer="client loop")
            t = time.time()
            frameNumber += 1

        printLog("STATUS", "generating frame",producer="client Loop")
        frame = robot_track.cam.getFrame()

        cross = robot_track.cam.findCross(robot_track.cam.walls)
        if cross is not None:
            wall1, wall2 = cross
            robot_track.extra_obstacles = [wall1, wall2]
        else:
            robot_track.extra_obstacles = []

        if frame is None:
            break

        robot_track.cam.displayFrame(frame,"success",False)

        response = robot_track.update(
            walls=(10 + (ballFrames//10)) if frameNumber % 10 == 0 else False,
            goals=True if frameNumber % 10 == 0 else False,
            targets=True,
            obsticles=False,
            car=True,
            frame=frame
        )

        if response is None:
            printLog("RETRY","no car",producer="client Loop")
            robot_track.cam.displayFrame(frame,"fail",True)
            step = Movement(-10)
        elif not hasBall:
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

                # Select right-most goal
                goal = max(robot_track.goals, key=lambda g: g.x)
                robot_track.delivery_goal = goal

                # Compute horizontal approach point
                approach_distance = 70
                approach_x = goal.x - approach_distance
                approach_y = goal.y  # Same Y to stay horizontal
                approach_point = Point(approach_x, approach_y)
                robot_track.approach_point = approach_point

                # Plan path to the approach point
                path, _ = robot_track.generatepath(approach_point, checkTarget=False)

                if car_front.distanceTo(approach_point) < 10:
                    delivering = True
                    path = []
            elif rotating_to_deliver:
                rotating_to_deliver = False
                path = [deliver()]
            else:
                goal = robot_track.delivery_goal
                if goal is None:
                    printLog("ERROR", "Missing delivery goal", producer="client loop")
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

        robot_track.Draw(frame,path,target)
        robot_track.cam.displayFrame(frame,"Track")

        if path is None or len(path) < 1:
            printLog("RETRY", "failed to follow path (empty)",producer="client Loop")
            continue

        step = path[0]

        # Command formatting
        if isinstance(step, deliver):
            cmd = "deliver 1"
            printLog("status", "create deliver command", producer="client sender")

        elif isinstance(step, Movement):
            if step.distance > 0:
                cmd = f"drive {step.distance / 200}"
            elif step.distance < 0:
                cmd = f"backward {abs(step.distance) / 200}"
            else:
                printLog("ERROR", "zero movement ignored", step.distance, producer="client sender")
                continue
            printLog("status", "create movement", step.distance, producer="client sender")

        elif isinstance(step, Rotation):
            if abs(step.angle) < 0.05:
                printLog("DEBUG", f"Skipping small rotation: {step.angle:.4f}", producer="client sender")
                continue
            angle_degrees = math.degrees(step.angle)
            cmd = f"rotate {-angle_degrees / 3:.2f}"
            printLog("status", "create rotate", step.angle, producer="client sender")

        else:
            printLog("ERROR", "Unknown step type", str(step), producer="client sender")
            continue

        printLog("command","sending command", cmd,producer="client sender")
        printLog("STATUS", "has ball",hasBall,producer="client sender")
        client_socket.sendall(cmd.encode())

        response = client_socket.recv(1024).decode()
        printLog("RESPONSE","modified",response,producer="client reciever")
        printLog("Raw_response",f"{repr(response)}",producer="client reciever")

        while response.startswith("OKOK"): response = response[2:]
        while response.endswith("OK") and len(response) > 2: response = response[:-2]

        if not response.startswith("OK"):
            printLog("ERROR", "at", cmd,producer="client reciever")
            printLog("ERROR", "received unexpected response", response,producer="client reciever")
            continue

        elif response == "OK ball caught":
            robot_track.update(goals=True)
            robot_track.update(walls=True, goals=True, targets=False, obsticles=False, car=False, frame=frame)
            target = robot_track.goals[0].move(Point(-100, 0))
            hasBall = True
            ballFrames = 0
            time.sleep(2.3)

        elif response == "OK ball lost":
            printLog("STATUS", "Ball lost",producer="client reciever")
            target = None
            hasBall = False

except Exception as e:
    printLog("ERROR", "An error occurred:", traceback.format_exception_only(e),producer="client cleanup")
    printLog("ERROR", "Stack trace:", traceback.format_stack(),producer="client cleanup")

finally:
    printLog("STATUS", "Closing connection without error",producer="client cleanup")

try:
    client_socket.close()
except Exception as e:
    printLog("ERROR", "Failed to close client socket", str(e),producer="client cleanup")
finally:
    printLog("STATUS", "Client socket closed",producer="client cleanup")

try:
    server_socket.close()
except Exception as e:
    printLog("ERROR", "Failed to close server socket", str(e),producer="client cleanup")
finally:
    printLog("STATUS", "Server socket closed",producer="client cleanup")

printLog("STATUS", "closing log",producer="client cleanup")
closeLog()
