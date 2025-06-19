# -*- coding: utf-8 -*-
import socket
import time
from classes import Movement, Rotation, Pickup, Point, Dropoff
from path_find import track
from image_recognition import Camera
import math
import cv2
host = '0.0.0.0'  # Lyt på alle interfaces
port = 12345      # Samme port som EV3-klienten bruger


# Opret TCP-socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Venter på forbindelse på {host}:{port}...")

# Accepter forbindelse fra klient
client_socket, client_address = server_socket.accept()
print(f"Forbundet til: {client_address}")

# Modtag data fra klienten
data = client_socket.recv(1024)
print("Modtaget:", data.decode())

    
cam = Camera(debug=True)
robot_track = track(cam)
target:Point | None = None
t = time.time()
hasBall = False

while(1):
    if time.time() != t:
        print("[FPS]:", 1 / (time.time()-t))
        t = time.time()
    print("generating frame")
    
    # Update the track to get latest car and targets
    frame = robot_track.cam.getFrame()
    
    if(frame is None):
        break
    robot_track.cam.displayFrame(frame,"success",False)
    
    response = robot_track.update(walls=False, goals=False, targets=True, obsticles=False, car=True, frame=frame)
    
    if(response is None): 
        print("noCar")
        robot_track.cam.displayFrame(frame,"fail",True)
        step = Movement(-10)
    else:
        if not hasBall:
            # Get the path
            path,target = robot_track.generatepath(target)
    
        else:
            path,target = robot_track.generatepath(target,False)
        
        robot_track.Draw(frame,path,target)
        robot_track.cam.displayFrame(frame,"Track")
        
        step = path[0]
    if isinstance(step, Movement):
        if step.distance > 0:
            cmd = f"drive {step.distance / 100}"
        elif step.distance < 0:
            cmd = f"backward {0-step.distance / 100}"
        else:
            print("error movement == ", step.distance)
            continue
    
    elif isinstance(step, Rotation):
        angle_degrees = math.degrees(step.angle)
        cmd = f"rotate {0-angle_degrees/3}"
    
    elif isinstance(step, Pickup):
        cmd = "grab"
    elif isinstance(step, Dropoff):
        cmd = "open"
    else:
        print("Unknown step:", step)
        continue

    print("sending")
    print("has ball:",hasBall)
    client_socket.sendall(cmd.encode())
    response = client_socket.recv(1024).decode()
    print("[RESPONSE]:", response)
    if not response.startswith("OK"):
        print("[ERROR] at:", cmd)
        continue
    elif response == "OK ball caught":
        temp = time.time()
        timeToWait = 2.3 
        robot_track.update(goals=True)
        target = robot_track.goals[0].move(Point(-20, 0))
        hasBall = True
        time.sleep(timeToWait)

    elif response == "OK ball lost":
        print("Ball lost")
        target = None
        hasBall = False
# Luk forbindelsen
client_socket.close()
server_socket.close()
