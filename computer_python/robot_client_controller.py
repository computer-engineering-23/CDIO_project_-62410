#!/usr/bin/env python3

import asyncio
import websockets
import json
import cv2
import numpy as np
import threading
import time

class RobotClient:
    def __init__(self, robot_ip="192.168.1.100", robot_port=8765):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.websocket = None
        self.robot_status = {}
        self.running = True
        
        # Ball detection parameters
        self.balls_detected = []
        self.egg_detected = None
        self.walls_detected = []
        
        # Course dimensions (cm)
        self.course_width = 180
        self.course_height = 120
        self.goal_a_width = 20  # 150 points
        self.goal_b_width = 8   # 100 points
        
    async def connect_to_robot(self):
        """Connect to the EV3 robot via WebSocket"""
        try:
            uri = f"ws://{self.robot_ip}:{self.robot_port}"
            print(f"Connecting to robot at {uri}")
            self.websocket = await websockets.connect(uri)
            print("Connected to robot!")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False
    
    async def send_command(self, command):
        """Send command to robot"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(command))
                response = await self.websocket.recv()
                return json.loads(response)
            except Exception as e:
                print(f"Error sending command: {e}")
                return None
        else:
            print("Not connected to robot")
            return None
    
    async def listen_for_status(self):
        """Listen for status updates from robot"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get('type') == 'status':
                    self.robot_status = data
                    self.print_status()
        except websockets.exceptions.ConnectionClosed:
            print("Connection to robot lost")
        except Exception as e:
            print(f"Error receiving status: {e}")
    
    def print_status(self):
        """Print current robot status"""
        pos = self.robot_status.get('position', {})
        print(f"\rRobot Status - Pos: ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}) "
              f"Angle: {pos.get('angle', 0):.1f}° "
              f"Balls: {self.robot_status.get('balls_collected', 0)} "
              f"Moving: {self.robot_status.get('is_moving', False)}", end='')
    
    def detect_objects_from_camera(self):
        """Use your existing image recognition code to detect objects"""
        # Initialize camera
        cap = cv2.VideoCapture(1)  # Use your camera index
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Could not get image from camera")
                break
            
            # Your existing detection code (adapted)
            output = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Orange color range (balls)
            lower_orange = np.array([10, 150, 150])
            upper_orange = np.array([25, 255, 255])
            
            # White color range (balls)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 80, 255])
            
            # Create masks
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask = cv2.bitwise_or(mask_orange, mask_white)
            
            # Apply mask and detect circles
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)
            
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=60
            )
            
            # Process detected circles
            self.balls_detected = []
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Determine if ball is orange or white
                    roi = hsv[y-r:y+r, x-r:x+r]
                    orange_pixels = cv2.inRange(roi, lower_orange, upper_orange)
                    white_pixels = cv2.inRange(roi, lower_white, upper_white)
                    
                    if np.sum(orange_pixels) > np.sum(white_pixels):
                        ball_type = "orange"
                        color = (0, 165, 255)  # Orange in BGR
                    else:
                        ball_type = "white"
                        color = (255, 255, 255)  # White
                    
                    # Convert pixel coordinates to real-world coordinates
                    # This is a simplified conversion - you'll need to calibrate
                    real_x = (x / frame.shape[1]) * self.course_width
                    real_y = (y / frame.shape[0]) * self.course_height
                    
                    self.balls_detected.append({
                        'type': ball_type,
                        'x': real_x,
                        'y': real_y,
                        'pixel_x': x,
                        'pixel_y': y,
                        'radius': r
                    })
                    
                    # Draw detection
                    cv2.circle(output, (x, y), r, color, 4)
                    cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(output, f"{ball_type.capitalize()} Ball", 
                              (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show detection result
            cv2.imshow("Ball Detection", output)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    async def autonomous_ball_collection(self):
        """Autonomous ball collection strategy"""
        print("Starting autonomous ball collection...")
        
        # Strategy: Collect orange ball first for bonus points
        orange_balls = [ball for ball in self.balls_detected if ball['type'] == 'orange']
        white_balls = [ball for ball in self.balls_detected if ball['type'] == 'white']
        
        # Prioritize orange balls
        for ball in orange_balls:
            print(f"Navigating to orange ball at ({ball['x']:.1f}, {ball['y']:.1f})")
            
            # Navigate to ball
            nav_cmd = {
                'type': 'navigate_to',
                'x': ball['x'],
                'y': ball['y']
            }
            await self.send_command(nav_cmd)
            
            # Wait for movement to complete
            await asyncio.sleep(3)
            
            # Grab the ball
            await self.send_command({'type': 'grab_ball'})
            
            # Navigate to appropriate goal (Goal A for higher points)
            goal_x = self.course_width  # Right side goal
            goal_y = self.course_height / 2
            
            nav_cmd = {
                'type': 'navigate_to',
                'x': goal_x - 10,  # Stop before goal
                'y': goal_y
            }
            await self.send_command(nav_cmd)
            
            # Release ball
            await self.send_command({'type': 'release_ball', 'is_orange': True})
            
            print("Orange ball delivered!")
            break
        
        # Collect white balls
        for ball in white_balls:
            print(f"Navigating to white ball at ({ball['x']:.1f}, {ball['y']:.1f})")
            
            # Navigate to ball
            nav_cmd = {
                'type': 'navigate_to',
                'x': ball['x'],
                'y': ball['y']
            }
            await self.send_command(nav_cmd)
            
            await asyncio.sleep(3)
            
            # Grab the ball
            await self.send_command({'type': 'grab_ball'})
            
            # Navigate to goal
            goal_x = self.course_width
            goal_y = self.course_height / 2
            
            nav_cmd = {
                'type': 'navigate_to',
                'x': goal_x - 10,
                'y': goal_y
            }
            await self.send_command(nav_cmd)
            
            # Release ball
            await self.send_command({'type': 'release_ball', 'is_orange': False})
            
            print("White ball delivered!")
    
    async def manual_control(self):
        """Manual control interface"""
        print("\nManual Control Commands:")
        print("w/s - Move forward/backward")
        print("a/d - Turn left/right")
        print("o/c - Open/close claw")
        print("g - Grab ball")
        print("r - Release ball")
        print("q - Quit")
        print("auto - Start autonomous mode")
        
        while self.running:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'q':
                    break
                elif command == 'w':
                    await self.send_command({'type': 'move_forward', 'distance': 10})
                elif command == 's':
                    await self.send_command({'type': 'move_backward', 'distance': 10})
                elif command == 'a':
                    await self.send_command({'type': 'turn_left', 'angle': 45})
                elif command == 'd':
                    await self.send_command({'type': 'turn_right', 'angle': 45})
                elif command == 'o':
                    await self.send_command({'type': 'open_claw'})
                elif command == 'c':
                    await self.send_command({'type': 'close_claw'})
                elif command == 'g':
                    await self.send_command({'type': 'grab_ball'})
                elif command == 'r':
                    await self.send_command({'type': 'release_ball'})
                elif command == 'auto':
                    await self.autonomous_ball_collection()
                elif command == 'stop':
                    await self.send_command({'type': 'stop'})
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
        
        self.running = False
    
    async def run(self):
        """Main run function"""
        # Connect to robot
        if not await self.connect_to_robot():
            return
        
        # Start camera detection in separate thread
        camera_thread = threading.Thread(target=self.detect_objects_from_camera)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Start status listener
        status_task = asyncio.create_task(self.listen_for_status())
        
        # Start manual control
        await self.manual_control()
        
        # Cleanup
        if self.websocket:
            await self.websocket.close()

if __name__ == "__main__":
    # Replace with your EV3's IP address
    robot_ip = input("Enter EV3 IP address (default: 192.168.1.100): ").strip()
    if not robot_ip:
        robot_ip = "192.168.1.100"
    
    client = RobotClient(robot_ip)
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nShutting down client...")
    except Exception as e:
        print(f"Error: {e}")