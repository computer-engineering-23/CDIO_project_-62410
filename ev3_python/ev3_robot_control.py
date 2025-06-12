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
degrees_per_robot_degree = 617 / 180 # 617 motor degrees correspond to 180 degrees robot rotation

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
        rightMotor.stop()        s.sendall("OK".encode())


    elif command.startswith("rotate "):
        try:
            angle_deg = float(command.split()[1])
            motor_degrees = abs(angle_deg) * degrees_per_robot_degree
            speed = 30

            if angle_deg > 0:
                # Turn left (counter-clockwise)
                leftMotor.on_for_degrees(speed=-speed, degrees=motor_degrees, block=False)
                rightMotor.on_for_degrees(speed=speed, degrees=motor_degrees)
            else:
                # Turn right (clockwise)
                leftMotor.on_for_degrees(speed=speed, degrees=motor_degrees, block=False)
                rightMotor.on_for_degrees(speed=-speed, degrees=motor_degrees)

            
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


