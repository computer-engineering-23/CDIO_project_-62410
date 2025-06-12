#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import ev3dev.ev3 as ev3
import time

host = '192.168.28.119' # IP-address to computer
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

    elif command.startswith("rotate "):
        try:
            angle_deg = float(command.split()[1])
            duration_per_degree = 0.01  # Adjust this value based on real robot testing
            turn_time = abs(angle_deg) * duration_per_degree

            if angle_deg > 0:
                # Turn left (counter-clockwise)
                leftMotor.run_forever(speed_sp=-100)
                rightMotor.run_forever(speed_sp=100)
            else:
                # Turn right (clockwise)
                leftMotor.run_forever(speed_sp=100)
                rightMotor.run_forever(speed_sp=-100)

            time.sleep(turn_time)
            leftMotor.stop()
            rightMotor.stop()
            s.sendall("OK".encode())

        except (IndexError, ValueError):
            print("Invalid rotate command:", command)
            s.sendall("ERR".encode())

    elif command == "backward":
        print "Driving backward for 2 seconds"
        leftMotor.run_forever(speed_sp=250)
        rightMotor.run_forever(speed_sp=250)
        time.sleep(2)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())

    elif command == "deliver":
        smallMotor.run_forever(speed_sp=250)
        time.sleep(1.5)
        smallMotor.stop()
        leftMotor.run_forever(speed_sp=-250)
        rightMotor.run_forever(speed_sp=-250)
        time.sleep(2)
        leftMotor.stop()
        rightMotor.stop()
        s.sendall("OK".encode())
        
    else:
        print "Unknown command:", command
        s.sendall("OK".encode())


