#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket
import time
from classes import Movement, Rotation, Pickup, Point, Dropoff
from path_find import track
from image_recognition import Camera
import math

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

# Update the track to get latest car and targets
robot_track.update(walls=False, goals=False, targets=True, obsticles=False, car=True)

# Get the path
path = robot_track.generatepath()

for step in path:
    if isinstance(step, Movement):
        if math.isclose(step.distance, 0, abs_tol=0.1):
            cmd = "drive"
        elif math.isclose(step.distance, math.pi, abs_tol=0.1):
            cmd = "backward"
        else:
            print("Unsupported movement direction:", step.distance)
            continue

    elif isinstance(step, Rotation):
        angle_degrees = math.degrees(step.angle)
        cmd = f"rotate {angle_degrees}"

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
        break

# Luk forbindelsen
client_socket.close()
server_socket.close()

