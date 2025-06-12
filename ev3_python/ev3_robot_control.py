#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import ev3dev.ev3 as ev3
import time

host = '192.168.245.134' # IP-address to computer
port = 12345            # Port the server is listening to

s = socket.socket()
s.connect((host, port)) 

s.send("EV3 is ready")

leftMotor = ev3.LargeMotor('outA')
rightMotor = ev3.LargeMotor('outB')
smallMotor = ev3.MediumMotor('outC')

while(True):
    data = s.recv(1024)
    command = data.strip()

    print ("recieved:", command)

    if command == "drive":
        print "Driving forward for 2 seconds"
        leftMotor.run_forever(speed_sp=-250)
        rightMotor.run_forever(speed_sp=-250)
        time.sleep(2)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())

    elif command == "grab":
        smallMotor.run_forever(speed_sp=250)
        time.sleep(2.5)
        smallMotor.stop()
        s.sendall("OK".encode())

    elif command == "open":
        smallMotor.run_forever(speed_sp=-250)
        time.sleep(2.5)
        smallMotor.stop()
        s.sendall("OK".encode())

    elif command == "stop":
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())

    elif command == "turn right":
        rightMotor.run_forever(speed_sp=-100)
        leftMotor.run_forever(speed_sp=100)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())


    elif command == "turn left":
        rightMotor.run_forever(speed_sp=100)
        leftMotor.run_forever(speed_sp=-100)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())

    elif command == "backward":
        print "Driving backward for 2 seconds"
        leftMotor.run_forever(speed_sp=250)
        rightMotor.run_forever(speed_sp=250)
        time.sleep(2)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())
        
    else:
        print "Unknown command:", command
        s.sendall("OK".encode())


