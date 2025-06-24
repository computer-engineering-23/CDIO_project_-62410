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
    while(1):
        #client loop
        if time.time() != t:
            printLog("FPS",str(1 / (time.time()-t)),producer="client Loop")
            printLog("time", "frame number:",frameNumber,producer="client loop")
            t = time.time()
            frameNumber += 1
        printLog("STATUS", "generating frame",producer="client Loop")
        
        # Update the track to get latest car and targets
        frame = robot_track.cam.getFrame()
        
        if(frame is None):
            break
        robot_track.cam.displayFrame(frame,"success",False)
        
        response = robot_track.update(walls=(10 + (ballFrames//10)) if frameNumber % 10 == 0 else False ,goals=True if frameNumber % 10 == 0 else False, targets=True, obsticles=False, car=True, frame=frame)
        
        if(response is None): 
            printLog("RETRY","no car",producer="client Loop")
            robot_track.cam.displayFrame(frame,"fail",True)
            step = Movement(-10)
        else:
            if not hasBall:
                # Standard path to ball
                path, target = robot_track.generatepath(target)
                delivering = False
                rotating_to_deliver = False
                robot_track.approach_point = None
                robot_track.delivery_goal = None
            else:
                car = robot_track.car.copy()
                car_front = car.front
                car_center = car.getRotationCenter()
                car_angle = car.getRotation()

                # Ensure a goal exists
                if not robot_track.goals:
                    printLog("ERROR", "No goals available", producer="client loop")
                    continue

                # Pick closest goal
                goal = min(robot_track.goals, key=lambda g: car_front.distanceTo(g))
                robot_track.delivery_goal = goal

                # Calculate approach point
                dx = goal.x - car_front.x
                dy = goal.y - car_front.y
                angle_to_goal = math.atan2(dy, dx)
                approach_distance = 60
                approach_point = Point(goal.x - approach_distance * math.cos(angle_to_goal),
                                       goal.y - approach_distance * math.sin(angle_to_goal))
                robot_track.approach_point = approach_point

                dist_to_approach = car_front.distanceTo(approach_point)
                angle_diff = abs(deltaRotation(car_angle, angle_to_goal))

                printLog("DEBUG", f"Dist to approach: {dist_to_approach:.2f}", producer="client loop")
                printLog("DEBUG", f"Angle diff to goal: {angle_diff:.2f}", producer="client loop")

                if not delivering:
                    # Not near enough yet — keep going
                    if dist_to_approach > 12:
                        path, _ = robot_track.generatepath(approach_point, checkTarget=False)
                    elif angle_diff > 0.15:
                        # Rotate to face goal
                        path = [Rotation(deltaRotation(car_angle, angle_to_goal))]
                        rotating_to_deliver = True
                    else:
                        # Aligned & in position → ready to deliver next frame
                        delivering = True
                        path = []
                elif rotating_to_deliver:
                    # Just finished rotating → now deliver
                    rotating_to_deliver = False
                    path = [deliver()]
                else:
                    # Already aligned & ready → deliver
                    path = [deliver()]



                        
        robot_track.Draw(frame,path,target)
        robot_track.cam.displayFrame(frame,"Track")
        if(path is None or len(path) < 1):
            printLog("RETRY", "failed to follow path (empty)",producer="client Loop")
            continue
        step = path[0]
        
        #client sender
        # client sender
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
            cmd = f"rotate {-angle_degrees / 3}"  # Assuming EV3 expects this scale
            printLog("status", "create rotate", step.angle, producer="client sender")

        else:
            printLog("ERROR", "Unknown step type", str(step), producer="client sender")
            continue
        
        printLog("command","sending command", cmd,producer="client sender")
        printLog("STATUS", "has ball",hasBall,producer="client sender")
        client_socket.sendall(cmd.encode())
        
        #client reciever
        response = client_socket.recv(1024).decode()
        printLog("RESPONSE","modified",response,producer="client reciever")
        printLog("Raw_response",f"{repr(response)}",producer="client reciever")
        while(response.startswith("OKOK")): response = response[2:len(response)]
        while(response.endswith("OK") and len(response) > 2): response = response[0:len(response) - 2]
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
    printLog("ERROR", "An error occurred:\n","\t", traceback.format_exception_only(e),producer="client cleanup")
    printLog("ERROR", "Stack trace:", traceback.format_stack() ,producer="client cleanup")
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