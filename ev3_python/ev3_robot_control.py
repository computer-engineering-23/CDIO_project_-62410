#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import ev3dev.ev3 as ev3
import time
import threading

host = '192.168.28.172' # IP-address to computer
port = 12345            # Port the server is listening to

s = socket.socket()
s.connect((host, port)) 

s.send("EV3 is ready".encode())

leftMotor = ev3.LargeMotor('outA')
rightMotor = ev3.LargeMotor('outB')
smallMotor = ev3.MediumMotor('outC')
colorSensor = ev3.ColorSensor('in1')
colorSensor.mode = 'COL-REFLECT'

has_ball = True
color_limit = 4 # Adjust according to testing

degrees_per_robot_degree = 500 / 180 # 617 motor degrees correspond to 180 degrees robot rotation


def sensor_loop():
    global has_ball
    while True:
        light = colorSensor.value()
        print("sensor active", light)
        if light > color_limit and not has_ball:
            print("Ball detected - closing grabber")
            smallMotor.run_forever(speed_sp=350)
            time.sleep(2)
            smallMotor.stop()
            has_ball = True
            try:
                s.sendall("OK ball caught\n".encode())
            except:
                print("error with send ball caught")
        elif light < 5 and has_ball:
            print("No ball")
            has_ball = False
            smallMotor.run_forever(speed_sp=-350)
            time.sleep(2)
            smallMotor.stop()
            try:
                s.sendall("OK ball lost\n".encode())
            except:
                print("error with send ball lost")

        time.sleep(0.1)

t = threading.Thread(target=sensor_loop)
t.daemon = True
t.start()

while(True):
    data = s.recv(1024)
    command = data.strip()

    print ("recieved:", command)
    
    
    if command.startswith("drive "):
        try:    
            run_time = float(command.split()[1])
            leftMotor.run_forever(speed_sp=-250)
            rightMotor.run_forever(speed_sp=-250)
            time.sleep(run_time)
            leftMotor.stop()
            rightMotor.stop()
            s.sendall("OK".encode())

       
        except (IndexError, ValueError):
            print("Invalid rotate command:", command)
            s.sendall("ERR".encode())

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
        smallMotor.stop()
        s.sendall("OK".encode())


    elif command.startswith("rotate "):
        try:
            angle_deg = float(command.split()[1])
            motor_degrees = abs(angle_deg) * degrees_per_robot_degree

            if angle_deg > 0:
                # Turn left (counter-clockwise)
                leftMotor.run_to_rel_pos(position_sp=-motor_degrees, speed_sp=200, stop_action='brake')
                rightMotor.run_to_rel_pos(position_sp=motor_degrees, speed_sp=200, stop_action='brake')
            else:
                # Turn right (clockwise)
                leftMotor.run_to_rel_pos(position_sp=motor_degrees, speed_sp=200, stop_action='brake')
                rightMotor.run_to_rel_pos(position_sp=-motor_degrees, speed_sp=200, stop_action='brake')

            while 'running' in leftMotor.state or 'running' in rightMotor.state:
                time.sleep(0.01)

            s.sendall("OK".encode())

        except (IndexError, ValueError):
            print("Invalid rotate command:", command)
            s.sendall("ERR".encode())

   
    elif command.startswith("backward "):
        try:    
            run_time = float(command.split()[1])
            leftMotor.run_forever(speed_sp=250)
            rightMotor.run_forever(speed_sp=250)
            time.sleep(run_time)
            leftMotor.stop()
            rightMotor.stop()
            s.sendall("OK".encode())

       
        except (IndexError, ValueError):
            print("Invalid rotate command:", command)
            s.sendall("ERR".encode())



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
