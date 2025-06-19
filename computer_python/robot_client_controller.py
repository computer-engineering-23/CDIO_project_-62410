# -*- coding: utf-8 -*-
import socket
import time
from classes import Movement, Rotation, Point
from path_find import track
from image_recognition import Camera
from Log import enableLog, printLog, closeLog, blockTag
import math
import cv2
host = '0.0.0.0'  # Lyt på alle interfaces
port = 12345      # Samme port som EV3-klienten bruger

if(input("enable log (y/n): ") == "y"):
    enableLog()
    blockTag("Raw_response")
    printLog("INFO", "Logging enabled")

# Opret TCP-socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

printLog("status",f"Venter på forbindelse på {host}:{port}...")

# Accepter forbindelse fra klient
client_socket, client_address = server_socket.accept()
printLog("status", f"Forbundet til: {client_address}")

# Modtag data fra klienten
data = client_socket.recv(1024)
printLog("status","Modtaget:", data.decode())

    
cam = Camera(debug=True)
robot_track = track(cam)
target:Point | None = None
t = time.time()
hasBall = False
ballFrames = 0

while(1):
    if time.time() != t:
        printLog("FPS",str(1 / (time.time()-t)))
        t = time.time()
    printLog("STATUS", "generating frame")
    
    # Update the track to get latest car and targets
    frame = robot_track.cam.getFrame()
    
    if(frame is None):
        break
    robot_track.cam.displayFrame(frame,"success",False)
    
    response = robot_track.update(walls=False, goals=True, targets=True, obsticles=False, car=True, frame=frame)
    
    if(response is None): 
        printLog("RETRY","no car")
        robot_track.cam.displayFrame(frame,"fail",True)
        step = Movement(-10)
    else:
        if not hasBall:
            # Get the path
            path,target = robot_track.generatepath(target)
    
        else:
            if(ballFrames % 10 == 0):
                robot_track.update(walls=10,goals=True)
            ballFrames += 1
            path,target = robot_track.generatepath(target,False)
        
        robot_track.Draw(frame,path,target)
        robot_track.cam.displayFrame(frame,"Track")
        if(path is None or len(path) < 1):
            printLog("RETRY", "failed to follow path (empty)")
            continue
        step = path[0]
    if isinstance(step, Movement):
        if step.distance > 0:
            cmd = f"drive {step.distance / 100}"
        elif step.distance < 0:
            cmd = f"backward {0-step.distance / 100}"
        else:
            printLog("ERROR","no movement:", step.distance)
            continue
    
    elif isinstance(step, Rotation):
        angle_degrees = math.degrees(step.angle)
        cmd = f"rotate {0-angle_degrees/3}"
    
    else:
        printLog("error","Unknown step:", step)
        continue

    printLog("STATUS"," sending")
    printLog("STATUS", "has ball:",hasBall)
    client_socket.sendall(cmd.encode())
    response = client_socket.recv(1024).decode()
    printLog("RESPONSE","modified",response)
    printLog("Raw_response",f"{repr(response)}")
    if not response.startswith("OK"):
        printLog("ERROR", "at:", cmd)
        continue
    elif response == "OK ball caught":
        robot_track.update(goals=True)
        robot_track.update(walls=True, goals=True, targets=False, obsticles=False, car=False, frame=frame)
        target = robot_track.goals[0].move(Point(-30, 0))
        hasBall = True
        ballFrames = 0
        time.sleep(2.3)
    elif response == "OK ball lost":
        printLog("STATUS", "Ball lost")
        target = None
        hasBall = False
# Luk forbindelsen
client_socket.close()
server_socket.close()
closeLog()