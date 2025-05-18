import math
import keyboard
import mss.tools
import numpy as np
import serial
import torch
import time

# Configuration for the MAKUC device
makuc_port = 'COM3'
makuc_baudrate = 4000000  # 4 Mbps
makuc_timeout = 0.01  # Short timeout for responsiveness

# Movement step size for arrow keys
move_step = 5

# Current vertical offset
vertical_offset = 0

try:
    # Initialize serial communication with the MAKUC device
    makuc = serial.Serial(makuc_port, makuc_baudrate, timeout=makuc_timeout)
    print(f"Successfully connected to MAKUC on {makuc_port} at {makuc_baudrate} bps.")

    # Load the YOLOv5 model
    model = torch.hub.load('C:/yolov5', 'custom', path='C:/yolov5.pt', source='local', force_reload=True)

    with mss.mss() as sct:
        # Use the first monitor, change to desired monitor number
        dimensions = sct.monitors[1]
        SQUARE_SIZE = 200

        # Part of the screen to capture
        monitor = {"top": int((dimensions['height'] / 2) - (SQUARE_SIZE / 2)),
                   "left": int((dimensions['width'] / 2) - (SQUARE_SIZE / 2)),
                   "width": SQUARE_SIZE,
                   "height": SQUARE_SIZE}

        while True:
            # Check for arrow key presses to adjust vertical offset
            if keyboard.is_pressed('up'):
                vertical_offset -= move_step
                time.sleep(0.02)  # Small delay for smoother movement
            elif keyboard.is_pressed('down'):
                vertical_offset += move_step
                time.sleep(0.02)  # Small delay for smoother movement

            # SCREEN CAPTURE AND OBJECT DETECTION
            BRGframe = np.array(sct.grab(monitor))
            RGBframe = BRGframe[:, :, [2, 1, 0]]
            results = model(RGBframe, size=600)
            model.conf = 0.6

            enemyNum = results.xyxy[0].shape[0]

            if enemyNum > 0:
                distances = []
                closest = float('inf')
                closestEnemy = -1

                for i in range(enemyNum):
                    x1 = float(results.xyxy[0][i, 0])
                    x2 = float(results.xyxy[0][i, 2])
                    y1 = float(results.xyxy[0][i, 1])
                    y2 = float(results.xyxy[0][i, 3])

                    centerX = (x2 - x1) / 2 + x1
                    centerY = (y2 - y1) / 2 + y1

                    distance = math.sqrt(((centerX - (SQUARE_SIZE / 2)) ** 2) + ((centerY - (SQUARE_SIZE / 2)) ** 2))
                    distances.append(distance)

                    if distance < closest:
                        closest = distance
                        closestEnemy = i

                if closestEnemy != -1 and keyboard.is_pressed('f'):
                    x1_closest = float(results.xyxy[0][closestEnemy, 0])
                    x2_closest = float(results.xyxy[0][closestEnemy, 2])
                    y1_closest = float(results.xyxy[0][closestEnemy, 1])
                    y2_closest = float(results.xyxy[0][closestEnemy, 3])

                    Xenemycoord_closest = int((x2_closest - x1_closest) / 2 + x1_closest)
                    Yenemycoord_closest = int((y2_closest - y1_closest) / 2 + y1_closest)

                    difx = int(Xenemycoord_closest - (SQUARE_SIZE / 2))
                    dify = int(Yenemycoord_closest - (SQUARE_SIZE / 2)) + vertical_offset

                    # Send the movement command to MAKUC
                    move_command = f"km.move({difx},{dify})\r".encode()
                    makuc.write(move_command)
                #    print(f"Sent move command: {move_command.decode().strip()}")

            # Optional: Add a small delay to control the loop speed
            # time.sleep(0.001)

except serial.SerialException as e:
    print(f"Error communicating with MAKUC on {makuc_port}: {e}")
except FileNotFoundError:
    print("Error: YOLOv5 model file not found at the specified path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if 'makuc' in locals() and makuc.is_open:
        makuc.close()
        print(f"Closed serial port {makuc_port}.")
