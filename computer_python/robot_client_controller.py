#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket
import time

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

commands = ["drive", "grab", "turn right", "open", "stop"]

for cmd in commands:
    client_socket.sendall(cmd.encode())
    response = client_socket.recv(1024).decode()
    if response != "OK":
        print("Error at:", cmd)
        break

# Luk forbindelsen
client_socket.close()
server_socket.close()
