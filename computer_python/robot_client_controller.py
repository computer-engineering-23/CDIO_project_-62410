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

    
cam = Camera()
robot_track = track(cam)
while(1):

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
        time.sleep(1)
        continue
    
    # Get the path
    path = robot_track.generatepath()

    robot_track.Draw(frame)
    robot_track.cam.displayFrame(frame,"Track")

    print("waiting 1 sec")
    time.sleep(1)
    print("moving")

    for step in path:
        if isinstance(step, Movement):
            if step.distance > 0:
                cmd = f"drive {step.distance / 200}"
            elif step.distance < 0:
                cmd = f"backward {0-step.distance / 200}"
            else:
                print("error movement == ", step.distance)
                continue

        elif isinstance(step, Rotation):
            angle_degrees = math.degrees(step.angle)
            cmd = f"rotate {0-angle_degrees}"

        elif isinstance(step, Pickup):
            cmd = "grab"
        elif isinstance(step, Dropoff):
            cmd = "open"
        else:
            print("Unknown step:", step)
            continue

        client_socket.sendall(cmd.encode())
        response = client_socket.recv(1024).decode()
        if response != "OK":
            print("Error at:", cmd)
            print("response: ", response)
            break
        print("response: OK")
# Luk forbindelsen
client_socket.close()
server_socket.close()

