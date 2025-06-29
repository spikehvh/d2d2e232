import math
import numpy as np
import serial
import torch
from mss import mss as mss_module
import time
import sys
from ctypes import WinDLL, c_int, WINFUNCTYPE, windll
import keyboard # Ensure keyboard is imported for key registration
import cv2 # Import OpenCV for video recording and drawing
import threading
import os
import glob
from serial.tools import list_ports # Import list_ports to find serial devices
import onnxruntime as ort

# --- Configuration ---
# You might need to adjust MAKCU_COM_PORT and MAKCU_BAUD_RATE
MAKCU_BAUD_RATE = 4000000

# Device PID and VID to search for
TARGET_VID = 0x1A86 # Replace with your device's Vendor ID
TARGET_PID = 0x55D3 # Replace with your device's Product ID

# Model directories
YOLOV5_MODEL_DIR = 'C:/yolov5'
MODELS_DIR = 'C:/models'

# IMPORTANT: Define priority for class IDs.
# Class 1 (enemy_head) is the main priority.
# Class 0 (another relevant enemy part, e.g., enemy_body) is the secondary priority.
# If class 1 is detected, it will be targeted. If not, class 0 will be targeted.
# By setting PRIORITY_CLASSES = [1], the script will only consider class ID 1.
# Class ID 0 detections will be ignored.
PRIORITY_CLASSES = [1,6] #PRIORITY_CLASSES = [1, 0]

# --- WindMouse specific settings ---
DEFAULT_WINDMOUSE_G0 = 6
DEFAULT_WINDMOUSE_W0 = 1.5
DEFAULT_WINDMOUSE_M0 = 5
DEFAULT_WINDMOUSE_D0 = 6


mouse4_active = False # Start with Mouse4 inactive (held down for instant flick)
MOUSE4_COOLDOWN_SECONDS = 0.30 # 1 second delay

MOUSE4_X_OFFSET = 0 # Example: +5 to flick 5 pixels right from detected center
MOUSE4_Y_OFFSET = 0 # Example: -10 to flick 10 pixels up from detected center

# Global variable for aimbot toggle state
aimbot_active = False # Start with aimbot inactive
last_toggle_time = 0
toggle_cooldown_seconds = 0.3 # Cooldown to prevent rapid toggling
CIRCULAR_FOV_RADIUS = 90 # Half of the previous MOVEMENT_TRIGGER_RANGE for similar effective size

# Global variable for movement scaling/smoothing
# Adjust this value to control how much the mouse moves per detection.
# Lower values mean more smoothing (slower, finer adjustments).
# Higher values mean less smoothing (faster, larger adjustments).
MOVEMENT_SCALE = 1.6 # Default scaling factor

# Global variable for raw movement commands toggle state
raw_movement_active = False # Start with raw movement inactive

# --- Demo Video Recording Settings ---
# Set to True to enable video recording of the AI's view with detections
RECORD_DEMO_VIDEO = False # True False
DEMO_VIDEO_FILENAME = "aimbot_demo.mp4"
DEMO_VIDEO_FPS = 30 # Frames per second for the demo video
DEMO_VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 (e.g., 'mp4v', 'XVID', 'MJPG')

# Global variables for model
model = None
model_type = None  # 'yolov5' or 'yolov8'
CLASS_NAMES = []

# --- WindMouse Toggle Function ---
def toggle_windmouse():
    global windmouse_enabled
    windmouse_enabled = not windmouse_enabled
    status = "ENABLED" if windmouse_enabled else "DISABLED"
    print(f"⚙️ WindMouse is now: {status}")

def windmouse(start_x, start_y, end_x, end_y, G0=DEFAULT_WINDMOUSE_G0, W0=DEFAULT_WINDMOUSE_W0, M0=DEFAULT_WINDMOUSE_M0, D0=DEFAULT_WINDMOUSE_D0):
    """
    Generates a series of natural-looking mouse movements from (start_x, start_y) to (end_x, end_y)
    using the WindMouse algorithm.

    Args:
        start_x (int): Starting X coordinate.
        start_y (int): Starting Y coordinate.
        end_x (int): Ending X coordinate.
        end_y (int): Ending Y coordinate.
        G0 (float): Gravity factor.
        W0 (float): Wind factor.
        M0 (float): Max step length.
        D0 (float): A threshold for when to stop.
    """
    current_x, current_y = start_x, start_y
    # Distance to target
    dist = math.hypot(end_x - current_x, end_y - current_y)

    # Initial random velocity
    v_x, v_y = 0.0, 0.0

    while dist >= D0:
        W = W0 * math.sqrt(dist) # Apply wind based on distance
        
        # Calculate angle of wind, random component + direction towards target
        wind_x = W * math.sin(math.pi * np.random.uniform(0, 1) * 2) # Random wind angle
        wind_y = W * math.sin(math.pi * np.random.uniform(0, 1) * 2)

        # Apply gravity (pull towards target)
        G = G0 * dist
        gravity_x = (end_x - current_x) / dist * G
        gravity_y = (end_y - current_y) / dist * G

        # Update velocity with wind and gravity
        v_x = (v_x + wind_x + gravity_x) / 1.5 # Dampen velocity
        v_y = (v_y + wind_y + gravity_y) / 1.5

        # Limit velocity to M0
        speed = math.hypot(v_x, v_y)
        if speed > M0:
            v_x = v_x / speed * M0
            v_y = v_y / speed * M0

        # Update current position
        new_x = current_x + v_x
        new_y = current_y + v_y

        # Clamp to integers for mouse movement
        move_x = int(round(new_x - current_x))
        move_y = int(round(new_y - current_y))

        if move_x != 0 or move_y != 0:
            makcu_move(move_x, move_y)

        current_x += move_x
        current_y += move_y
        dist = math.hypot(end_x - current_x, end_y - current_y)

        # Small sleep to simulate realistic human delay
        time.sleep(0.001)

    # Final direct move to target if not exactly there
    if int(round(end_x)) != int(round(current_x)) or int(round(end_y)) != int(round(current_y)):
        makcu_move(int(round(end_x - current_x)), int(round(end_y - current_y)))


# --- Model Selection Functions ---
def list_models(model_type):
    """List available models based on type"""
    if model_type == 'yolov5':
        # Look for .pt files in the models directory and yolov5 directory
        pt_files = []
        if os.path.exists(MODELS_DIR):
            pt_files.extend(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
            pt_files.extend(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
        if os.path.exists(YOLOV5_MODEL_DIR):
            pt_files.extend(glob.glob(os.path.join(YOLOV5_MODEL_DIR, "*.pt")))
            pt_files.extend(glob.glob(os.path.join(YOLOV5_MODEL_DIR, "*.onnx")))
        # Also check for the default yolov5.pt in C:/
        if os.path.exists('C:/yolov5.pt'):
            pt_files.append('C:/yolov5.pt')
        return pt_files
    elif model_type == 'yolov8':
        # Look for .onnx files in the models directory
        onnx_files = []
        if os.path.exists(MODELS_DIR):
            onnx_files.extend(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
        return onnx_files

def select_model():
    """Interactive model selection"""
    print("\n--- Model Selection ---")
    print("1. YOLOv5 (.pt files)")
    print("2. YOLOv8 (.onnx files)")

    while True:
        try:
            choice = input("\nSelect model type (1 or 2): ").strip()
            if choice == '1':
                model_type = 'yolov5'
                break
            elif choice == '2':
                model_type = 'yolov8'
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    # List available models
    available_models = list_models(model_type)

    if not available_models:
        print(f"\n❗ No {model_type} models found!")
        if model_type == 'yolov5':
            print(f"Please place .pt files in: {MODELS_DIR} or {YOLOV5_MODEL_DIR}")
        else:
            print(f"Please place .onnx files in: {MODELS_DIR}")
        sys.exit(1)

    print(f"\n--- Available {model_type.upper()} Models ---")
    for i, model_path in enumerate(available_models, 1):
        model_name = os.path.basename(model_path)
        print(f"{i}. {model_name} ({model_path})")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}): ").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(available_models):
                selected_model = available_models[model_index]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    return model_type, selected_model

def load_yolov5_model(model_path):
    """Load YOLOv5 model"""
    try:
        # Always use local YOLOv5 installation to prevent downloading
        model = torch.hub.load('C:/yolov5', 'custom', path=model_path, source='local', force_reload=True)

        model.conf = 0.30  # Set confidence threshold for detections
        print(f"✅ YOLOv5 model loaded: {os.path.basename(model_path)}")

        if torch.cuda.is_available():
            model.cuda()  # Move model to GPU if CUDA is available
            print(f"✅ YOLOv5 model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available. YOLOv5 model running on CPU.")

        # Get model class names
        if hasattr(model, 'names'):
            class_names = model.names
            print(f"Model classes: {class_names}")
        else:
            class_names = [str(i) for i in range(100)]  # Fallback
            print("Could not retrieve model class names. Using generic class IDs.")

        return model, class_names

    except Exception as e:
        print(f"❗ Error loading YOLOv5 model: {e}")
        print("Make sure your YOLOv5 installation is at C:/yolov5")
        sys.exit(1)

def load_yolov8_model(model_path):
    """Load YOLOv8 ONNX model"""
    try:
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_available_providers() else ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        print(f"✅ YOLOv8 ONNX model loaded: {os.path.basename(model_path)}")

        # Check if CUDA is available
        if 'CUDAExecutionProvider' in session.get_providers():
            print("✅ YOLOv8 model running on GPU (CUDA)")
        else:
            print("⚠️ YOLOv8 model running on CPU")

        # Get input details
        input_details = session.get_inputs()[0]
        input_shape = input_details.shape
        print(f"Model input shape: {input_shape}")

        # Use generic class names since each model has different classes
        class_names = [str(i) for i in range(100)]  # Generic class IDs

        return session, class_names

    except Exception as e:
        print(f"❗ Error loading YOLOv8 ONNX model: {e}")
        sys.exit(1)

def yolov8_inference(session, image, conf_threshold=0.6):
    """Run inference with YOLOv8 ONNX model"""
    # Preprocess image
    input_image = cv2.resize(image, (640, 640))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})

    # Post-process outputs
    predictions = outputs[0][0]  # Remove batch dimension

    # Filter predictions by confidence
    valid_detections = []
    for detection in predictions.T:  # Transpose to get (num_detections, 6)
        if len(detection) >= 6:
            confidence = detection[4]
            if confidence >= conf_threshold:
                # Scale coordinates back to original image size
                x_center, y_center, width, height = detection[0:4]
                x1 = (x_center - width/2) * (640/640)  # Assuming 640x640 input
                y1 = (y_center - height/2) * (640/640)
                x2 = (x_center + width/2) * (640/640)
                y2 = (y_center + height/2) * (640/640)
                class_id = int(detection[5])

                valid_detections.append([x1, y1, x2, y2, confidence, class_id])

    return np.array(valid_detections) if valid_detections else np.array([]).reshape(0, 6)


# Function to find the COM port by VID and PID
def find_com_port_by_vid_pid(target_vid, target_pid):
    """
    Searches for a serial port with the given Vendor ID (VID) and Product ID (PID).
    Returns the port name (e.g., 'COM3') if found, otherwise returns None.
    """
    print(f"Searching for device with VID: {hex(target_vid)} and PID: {hex(target_pid)}...")
    ports = list_ports.comports() # This line lists all available serial ports
    for port in ports:
        print(f"Found port: {port.device}, Description: {port.description}, HWID: {port.hwid}, VID: {hex(port.vid) if port.vid else 'N/A'}, PID: {hex(port.pid) if port.pid else 'N/A'}")
        if port.vid == target_vid and port.pid == target_pid: # Checks if the port's VID and PID match
            print(f"Device found on port: {port.device}")
            return port.device # Returns the COM port name
    print("Device not found.")
    return None

# Attempt to find the MAKCU COM port
MAKCU_COM_PORT = find_com_port_by_vid_pid(TARGET_VID, TARGET_PID)

if MAKCU_COM_PORT is None:
    print("Error: MAKCU device not found. Please ensure it's connected and the correct VID/PID are set.")
    sys.exit(1)
# --- Makcu Setup ---1
makcu = None
try:
    makcu = serial.Serial(MAKCU_COM_PORT, MAKCU_BAUD_RATE, timeout=0.1)
    time.sleep(0.1) # Give a short moment for the serial connection to establish
    makcu.reset_input_buffer()
    print(f"✅ Makcu connected on {MAKCU_COM_PORT} at {MAKCU_BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"❗ Error opening Makcu serial port {MAKCU_COM_PORT}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❗ Unexpected error during Makcu setup: {e}")
    sys.exit(1)

def makcu_move(x, y):
    """Sends a mouse movement command via Makcu serial."""
    if not makcu or not makcu.is_open:
        return
    try:
        command = f"km.move({int(x)},{int(y)})\r\n"
        makcu.write(command.encode('utf-8'))
        time.sleep(0.0) # Small delay to prevent overwhelming the serial buffer
    except serial.SerialException:
        pass
    except Exception:
        pass

# --- Windows API Setup for Mouse Input ---
try:
    user32 = WinDLL("user32", use_last_error=True)
    # Virtual key code for Mouse 5 button (XBUTTON2)
    VK_XBUTTON2 = 0x06 # Mouse 5
    VK_XBUTTON1 = 0x05 # Mouse 4
    # For beeps on toggle
    kernel32 = WinDLL("kernel32", use_last_error=True)

    # Setup for low-level mouse hook (not used in the final version, but kept for reference)
    WH_MOUSE_LL = 14
    WM_XBUTTONDOWN = 0x020B
    WM_XBUTTONUP = 0x020C

    # Hook procedure type
    HOOKPROC = WINFUNCTYPE(c_int, c_int, c_int, c_int)

except Exception as e:
    print(f"❗ Error loading Windows DLLs for mouse input: {e}")
    sys.exit(1)

# --- Mouse Hook for Mouse4/Mouse5 Detection ---
def mouse_monitor_thread():
    """Thread function to monitor Mouse4 (instant flick) and Mouse5 (aimbot) buttons."""
    global aimbot_active, mouse4_active
    while True:
        try:
            # Check Mouse5 button state (XBUTTON2 for aimbot)
            aimbot_state = windll.user32.GetAsyncKeyState(VK_XBUTTON2) & 0x8000
            if aimbot_state:
                if not aimbot_active:
                    aimbot_active = True
            else:
                if aimbot_active:
                    aimbot_active = False
            
            # Check Mouse4 button state (XBUTTON1 for instant flick)
            instant_flick_state = windll.user32.GetAsyncKeyState(VK_XBUTTON1) & 0x8000
            if instant_flick_state:
                if not mouse4_active:
                    mouse4_active = True
            else:
                if mouse4_active:
                    mouse4_active = False
            
            time.sleep(0.001)   # Check every 1ms
        except Exception:
            pass

# --- Movement Scale Adjustment Functions ---
def increase_movement_scale():
    global MOVEMENT_SCALE
    MOVEMENT_SCALE = min(MOVEMENT_SCALE + 0.05, 3.5) # Cap at 3.50
    print(f"⬆️ Movement Scale: {MOVEMENT_SCALE:.2f}")

def decrease_movement_scale():
    global MOVEMENT_SCALE
    MOVEMENT_SCALE = max(MOVEMENT_SCALE - 0.05, 0.05) # Min value to prevent zero movement
    print(f"⬇️ Movement Scale: {MOVEMENT_SCALE:.2f}")

def toggle_raw_movement():
    """Toggles the raw movement command state."""
    global raw_movement_active
    raw_movement_active = not raw_movement_active
    status = "ENABLED" if raw_movement_active else "DISABLED"
    print(f"🔄 Raw Movement Commands: {status}")

def makcu_click_left(state):
    """Sends a left click command via Makcu serial.
    state: 1 for press down, 0 for release.
    """
    if not makcu or not makcu.is_open:
        return
    try:
        command = f"km.left({int(state)})\r\n"
        makcu.write(command.encode('utf-8'))
    except serial.SerialException:
        pass
    except Exception:
        pass


# --- Main Aimbot Logic ---
if __name__ == "__main__":
    print("\n--- YOLOv5/v8 Aimbot Initializing ---")

    # Model Selection
    model_type, selected_model_path = select_model()

    # Load the selected model
    if model_type == 'yolov5':
        model, CLASS_NAMES = load_yolov5_model(selected_model_path)
    else:  # yolov8
        model, CLASS_NAMES = load_yolov8_model(selected_model_path)

    print(f"🔌 Makcu connected on {MAKCU_COM_PORT}")
    print(f"🎯 Targeting priority (highest to lowest): {PRIORITY_CLASSES}")
    print(f"🖱️ HOLD 'Mouse5' (side button) to activate regular aimbot movement (uses WindMouse/smoothing).")
    print(f"🖱️ HOLD 'Mouse4' (side button) for instant flick to target, shoot, and immediately back with a {MOUSE4_COOLDOWN_SECONDS}s delay.")
    print(f"⬆️ Press 'up arrow' to increase smoothing (decrease movement scale).")
    print(f"⬇️ Press 'down arrow' to decrease smoothing (increase movement scale).")
    print(f"⬅️ Press 'left arrow' to toggle RAW movement commands (bypasses movement scale).")
    print(f"Current Movement Scale: {MOVEMENT_SCALE:.2f}")
    if RECORD_DEMO_VIDEO:
        print(f"🎥 Demo video recording is ENABLED. Output: {DEMO_VIDEO_FILENAME}")
    else:
        print("🚫 Demo video recording is DISABLED.")
    print("---------------------------------------------")

    # Register hotkeys (keyboard only now)
    try:
        keyboard.add_hotkey('up', decrease_movement_scale, trigger_on_release=False) # Up arrow for more smoothing (less scale)
        keyboard.add_hotkey('down', increase_movement_scale, trigger_on_release=False) # Down arrow for less smoothing (more scale)
        keyboard.add_hotkey('left', toggle_raw_movement, trigger_on_release=False) # Left arrow to toggle raw movement
        print("✅ Keyboard hotkeys 'up', 'down', 'left' registered.")
    except Exception as e:
        print(f"\n❗ Error setting up keyboard hotkeys: {e}")
        print("Try running the script as administrator.")

    # Start Mouse5 detection thread
    try:
        mouse_thread = threading.Thread(target=mouse_monitor_thread, daemon=True)
        mouse_thread.start()
        print("✅ Mouse5 button monitoring started.")
    except Exception as e:
        print(f"\n❗ Error setting up Mouse5 detection: {e}")
        print("Try running the script as administrator.")

    video_writer = None # Initialize video_writer to None
    try:
        with mss_module() as sct:
            # Use the first monitor, change to desired monitor number
            dimensions = sct.monitors[1] # Assuming monitor 1 is the primary game monitor

            # Define the SCAN_RANGE (what YOLO sees)
            SCAN_RANGE = 640
            # Define the MOVEMENT_TRIGGER_RANGE (the square within which the mouse will move)
            MOVEMENT_TRIGGER_RANGE = 200

            # Monitor configuration for screen capture (SCAN_RANGE)
            monitor = {"top": int((dimensions['height'] / 2) - (SCAN_RANGE / 2)),
                       "left": int((dimensions['width'] / 2) - (SCAN_RANGE / 2)),
                       "width": SCAN_RANGE,
                       "height": SCAN_RANGE}

            # Calculate the center of the scan range
            scan_center_x = SCAN_RANGE / 2
            scan_center_y = SCAN_RANGE / 2

            # Setup video writer if enabled
            if RECORD_DEMO_VIDEO:
                try:
                    video_writer = cv2.VideoWriter(DEMO_VIDEO_FILENAME, DEMO_VIDEO_FOURCC, DEMO_VIDEO_FPS, (SCAN_RANGE, SCAN_RANGE))
                    if not video_writer.isOpened():
                        raise IOError("Could not open video writer.")
                except Exception as e:
                    print(f"❗ Error initializing video writer: {e}")
                    print("Video recording will be disabled.")
                    RECORD_DEMO_VIDEO = False # Disable recording if setup fails

            while True:
                # SCREEN CAPTURE AND CONVERTING TO MODEL'S SUPPORTED FORMAT
                BRGframe = np.array(sct.grab(monitor))
                # Convert to format model can read (RGB) for inference and drawing
                RGBframe = BRGframe[:, :, [2, 1, 0]]

                # PASSING CONVERTED SCREENSHOT INTO MODEL
                if model_type == 'yolov5':
                    results = model(RGBframe, size=SCAN_RANGE)
                    all_detections = results.xyxy[0] if results.xyxy[0].shape[0] > 0 else []
                else:  # yolov8
                    detections = yolov8_inference(model, cv2.cvtColor(RGBframe, cv2.COLOR_RGB2BGR))
                    all_detections = detections

                # Prepare frame for video recording (if enabled)
                display_frame = RGBframe.copy() # Create a copy to draw on for display/recording
                # Convert to BGR for OpenCV drawing functions
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # Filter detections based on priority and find target
                target_detection = None

                for class_id_priority in PRIORITY_CLASSES:
                    current_priority_detections = []
                    for detection in all_detections:
                        if model_type == 'yolov5':
                            class_id = int(detection[5]) # Class ID is typically the 6th element (index 5)
                        else:  # yolov8
                            class_id = int(detection[5]) if len(detection) > 5 else 0

                        if class_id == class_id_priority:
                            current_priority_detections.append(detection)

                    if current_priority_detections:
                        # Find the closest target within the current priority class
                        closest_distance = float('inf')
                        for detection in current_priority_detections:
                            if model_type == 'yolov5':
                                x1, y1, x2, y2, conf, cls = detection.tolist()
                            else:  # yolov8
                                x1, y1, x2, y2, conf, cls = detection[:6]

                            centerX = (x2 - x1) / 2 + x1
                            centerY = (y2 - y1) / 2 + y1
                            distance = math.sqrt(((centerX - scan_center_x) ** 2) + ((centerY - scan_center_y) ** 2))

                            if distance < closest_distance:
                                closest_distance = distance
                                target_detection = detection
                        break # Found a target in a priority class, stop checking lower priorities

                # --- Drawing Detections on the display_frame for video ---
                if RECORD_DEMO_VIDEO:
                    # Draw all detected bounding boxes
                    for detection in all_detections:
                        if model_type == 'yolov5':
                            x1, y1, x2, y2, conf, cls_id = detection.tolist()
                        else:  # yolov8
                            x1, y1, x2, y2, conf, cls_id = detection[:6]

                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Get class name safely
                        class_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"class_{int(cls_id)}"
                        label = f"{class_name} {conf:.2f}"
                        color = (0, 255, 0) # Green for general detections
                        thickness = 1

                        # Highlight the target detection
                        if target_detection is not None and np.array_equal(detection, target_detection):
                            color = (0, 0, 255) # Red for the targeted detection
                            thickness = 2

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Draw the MOVEMENT_TRIGGER_RANGE square
                    trigger_x1 = int(scan_center_x - (MOVEMENT_TRIGGER_RANGE / 2))
                    trigger_y1 = int(scan_center_y - (MOVEMENT_TRIGGER_RANGE / 2))
                    trigger_x2 = int(scan_center_x + (MOVEMENT_TRIGGER_RANGE / 2))
                    trigger_y2 = int(scan_center_y + (MOVEMENT_TRIGGER_RANGE / 2))
                    cv2.rectangle(display_frame, (trigger_x1, trigger_y1), (trigger_x2, trigger_y2), (255, 255, 0), 1) # Yellow square

                     # Write the frame to the video file
                    video_writer.write(display_frame)

                # --- Mouse4 Instant Movement Logic ---
                if mouse4_active: # If Mouse4 is held down
                    current_time = time.time()
                    if target_detection is not None and (current_time - last_mouse4_action_time) >= MOUSE4_COOLDOWN_SECONDS:
                        x1_target = float(target_detection[0])
                        x2_target = float(target_detection[2])
                        y1_target = float(target_detection[1])
                        y2_target = float(target_detection[3])

                        Xenemycoord = (x2_target - x1_target) / 2 + x1_target
                        Yenemycoord = (y2_target - y1_target) / 2 + y1_target

                        distance_to_target = math.sqrt(((Xenemycoord - scan_center_x) ** 2) + ((Yenemycoord - scan_center_y) ** 2))

                        if distance_to_target <= CIRCULAR_FOV_RADIUS:
                            difx = Xenemycoord - scan_center_x
                            dify = Yenemycoord - scan_center_y

                            # Apply offsets for precise flicking
                            difx += MOUSE4_X_OFFSET
                            dify += MOUSE4_Y_OFFSET

                            # Move instantly to target
                            target_move_x = int(round(difx))
                            target_move_y = int(round(dify))
                            
                            makcu_move(target_move_x, target_move_y)

                            # --- Shooting action ---
                            makcu_click_left(1) # Press left mouse button down
                            # Optional: Add a very small delay if the click isn't registering consistently
                            # time.sleep(0.01) 
                            makcu_click_left(0) # Release left mouse button

                            # Immediately send move back
                            makcu_move(-target_move_x, -target_move_y)
                            # No sleep or smoothing here for instant effect as requested
                            
                            last_mouse4_action_time = current_time # Update last action time

                # --- Regular Aimbot Movement Logic (Mouse5) ---
                elif aimbot_active: # If Mouse5 is held down (and Mouse4 is not)
                    if target_detection is not None:
                        x1_target = float(target_detection[0])
                        x2_target = float(target_detection[2])
                        y1_target = float(target_detection[1])
                        y2_target = float(target_detection[3])

                        Xenemycoord = (x2_target - x1_target) / 2 + x1_target
                        Yenemycoord = (y2_target - y1_target) / 2 + y1_target

                        distance_to_target = math.sqrt(((Xenemycoord - scan_center_x) ** 2) + ((Yenemycoord - scan_center_y) ** 2))

                        if distance_to_target <= CIRCULAR_FOV_RADIUS:
                            difx = Xenemycoord - scan_center_x
                            dify = Yenemycoord - scan_center_y

                            if windmouse_enabled:
                                scaled_difx = difx * MOVEMENT_SCALE
                                scaled_dify = dify * MOVEMENT_SCALE
                                windmouse(0, 0, scaled_difx, scaled_dify)
                            else:
                                # Direct movement with scaling
                                scaled_difx = difx * MOVEMENT_SCALE
                                scaled_dify = dify * MOVEMENT_SCALE
                                makcu_move(int(round(scaled_difx)), int(round(scaled_dify)))
                                time.sleep(0.001) # Small delay for direct movement

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if video_writer is not None:
            print(f"Releasing video writer and saving {DEMO_VIDEO_FILENAME}...")
            video_writer.release()
            print("Video saved.")
        if makcu and makcu.is_open:
            print("Closing Makcu serial connection...")
            makcu.close()
            print("Makcu connection closed.")
        print("Exiting script.")

