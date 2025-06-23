# -*- coding: utf-8 -*-
import socket
import time
from classes import Movement, Rotation, Point, deliver
from path_find import track
from image_recognition import Camera
from Log import enableLog, printLog, closeLog, blockTag
import math
import traceback

host = '0.0.0.0'  # Lyt på alle interfaces
port = 12345      # Samme port som EV3-klienten bruger
enableLog()
blockTag("Raw_response")
printLog("INFO", "Logging enabled",producer="init Client")

# Opret TCP-socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

printLog("status",f"Venter på forbindelse på {host}:{port}...",producer="init Client")

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
                path, target = robot_track.generatepath(target)
            else:
                goal = robot_track.goals[0]
                distance_to_goal = robot_track.car.front.distanceTo(goal)

                if distance_to_goal > 30:
                    printLog("DELIVERY", f"Getting closer to goal ({distance_to_goal:.1f}px)", producer="client Loop")
                    target = goal
                    path, target = robot_track.generatepath(target, checkTarget=False)
                else:
                    printLog("DELIVERY", "Close enough to deliver", producer="client Loop")
                    path = [deliver(0.2)]
            
            robot_track.Draw(frame,path,target)
            robot_track.cam.displayFrame(frame,"Track")
            if(path is None or len(path) < 1):
                printLog("RETRY", "failed to follow path (empty)",producer="client Loop")
                continue
            step = path[0]
        
        #client sender
        if isinstance(step, Movement):
            if step.distance > 0:
                if(step.distance < 100 and hasBall):
                    cmd = f"deliver {step.distance / 200}"
                else:
                    cmd = f"drive {step.distance / 200}"
            elif step.distance < 0:
                cmd = f"backward {0-step.distance / 200}"
            else:
                printLog("ERROR","no movement", step.distance,producer="client sender")
                continue
            printLog("status","create movement",step.distance,producer="client sender")
        elif isinstance(step, Rotation):
            angle_degrees = math.degrees(step.angle)
            cmd = f"rotate {0-angle_degrees/3}"
            printLog("status","create rotate",step.angle,producer="client sender")
        else:
            printLog("error","Unknown step", step,producer="client sender")
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
            target = robot_track.goals[0].move(Point(0, -50))
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