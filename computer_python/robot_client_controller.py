import websockets
import json

class RobotClient:
    def __init__(self, robot_ip="192.168.1.100", robot_port=8765):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.websocket = None
    
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
    
    async def autonomous_ball_collection(self):#startover most likely
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
        
        while 1:#rewrite
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