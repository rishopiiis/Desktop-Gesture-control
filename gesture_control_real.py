# gesture_control_real.py - WORKING gesture control with real data
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import tensorflow as tf
from tensorflow import keras
import pickle
import time
import os
import sys

print("="*60)
print("WORKING GESTURE CONTROL (REAL DATA)")
print("="*60)

class WorkingGestureControl:
    def __init__(self):
        print("Initializing...")
        
        # Try to load real model first, then fallback
        model_files = [
            ('real_gesture_model.h5', 'real_model_info.pkl'),
            ('gesture_model.h5', 'model_info.pkl')
        ]
        
        self.model = None
        for model_file, info_file in model_files:
            if os.path.exists(model_file) and os.path.exists(info_file):
                print(f"Loading {model_file}...")
                try:
                    self.model = keras.models.load_model(model_file)
                    with open(info_file, 'rb') as f:
                        info = pickle.load(f)
                    self.class_names = info['class_names']
                    print(f"✓ Model loaded: {len(self.class_names)} gestures")
                    print(f"✓ Accuracy: {info.get('accuracy', 'unknown'):.1%}")
                    break
                except Exception as e:
                    print(f"✗ Failed to load {model_file}: {e}")
        
        if self.model is None:
            print("✗ No model found!")
            print("\nPlease:")
            print("1. Collect real data: python collect_real_gestures.py")
            print("2. Train model: python train_real_model.py")
            sys.exit(1)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Screen info
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            print("✗ No webcam!")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Control settings
        self.prev_x, self.prev_y = 0, 0
        self.confidence_threshold = 0.7  # Start with 70% confidence
        self.last_action_time = 0
        self.action_delay = 0.3  # Prevent rapid actions
        
        print(f"\nGestures configured:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        
        print(f"\nControls:")
        print(f"  Q - Quit")
        print(f"  H - Help")
        print(f"  + - Increase confidence threshold")
        print(f"  - - Decrease confidence threshold")
        print(f"  R - Reset mouse position")
        print("="*60)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize landmarks"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)
    
    def predict_with_confidence(self, landmarks):
        """Predict gesture with confidence check"""
        landmarks = landmarks.reshape(1, -1)
        predictions = self.model.predict(landmarks, verbose=0)
        
        gesture_id = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get top 3 for debugging
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_conf = predictions[0][top_3_idx]
        
        return gesture_id, confidence, top_3_idx, top_3_conf
    
    def execute_action(self, gesture_name, hand_landmarks):
        """Execute desktop action"""
        current_time = time.time()
        
        # Action cooldown
        if current_time - self.last_action_time < self.action_delay:
            return gesture_name
        
        if gesture_name == "mouse_move":
            # Get index finger position
            index_tip = hand_landmarks.landmark[8]
            screen_x = int(index_tip.x * self.screen_width)
            screen_y = int(index_tip.y * self.screen_height)
            
            # Smooth movement
            smooth_x = int(self.prev_x + (screen_x - self.prev_x) * 0.5)
            smooth_y = int(self.prev_y + (screen_y - self.prev_y) * 0.5)
            
            pyautogui.moveTo(smooth_x, smooth_y)
            self.prev_x, self.prev_y = smooth_x, smooth_y
            
        elif gesture_name == "left_click":
            pyautogui.click()
            self.last_action_time = current_time
            
        elif gesture_name == "right_click":
            pyautogui.rightClick()
            self.last_action_time = current_time
            
        elif gesture_name == "scroll_up":
            pyautogui.scroll(50)
            self.last_action_time = current_time
            
        elif gesture_name == "scroll_down":
            pyautogui.scroll(-50)
            self.last_action_time = current_time
            
        elif gesture_name == "drag":
            pyautogui.mouseDown()
            # Move while dragging
            index_tip = hand_landmarks.landmark[8]
            screen_x = int(index_tip.x * self.screen_width)
            screen_y = int(index_tip.y * self.screen_height)
            pyautogui.moveTo(screen_x, screen_y)
            time.sleep(0.1)
            pyautogui.mouseUp()
            self.last_action_time = current_time
        
        # "none" gesture does nothing
        
        return gesture_name
    
    def run(self):
        """Main control loop"""
        print("\nStarting gesture control...")
        print("Show your hand to the camera!")
        print("Move hand naturally for mouse control")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand
            results = self.hands.process(rgb)
            
            gesture_text = "No hand detected"
            confidence_text = ""
            action_text = ""
            debug_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Extract and predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    gesture_id, confidence, top_3_idx, top_3_conf = self.predict_with_confidence(landmarks)
                    
                    gesture_name = self.class_names[gesture_id]
                    gesture_text = f"Gesture: {gesture_name}"
                    confidence_text = f"Confidence: {confidence:.1%}"
                    
                    # Debug info
                    debug_items = []
                    for idx, conf in zip(top_3_idx, top_3_conf):
                        debug_items.append(f"{self.class_names[idx]}:{conf:.0%}")
                    debug_text = " | ".join(debug_items)
                    
                    # Execute action if confident enough
                    if confidence > self.confidence_threshold:
                        action = self.execute_action(gesture_name, hand_landmarks)
                        action_text = f"Action: {action}"
                        
                        # Highlight if action was executed
                        cv2.rectangle(frame, (5, 5), (635, 475), (0, 255, 0), 3)
            
            # Display info
            y_pos = 30
            cv2.putText(frame, gesture_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            
            cv2.putText(frame, confidence_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            
            cv2.putText(frame, action_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_pos += 30
            
            cv2.putText(frame, debug_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls overlay
            cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, "Q: Quit  H: Help  +/-: Sensitivity", 
                       (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Threshold: {self.confidence_threshold:.0%}", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Gesture Control (REAL DATA)", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("\n" + "="*60)
                print("HELP - GESTURE GUIDE:")
                for i, name in enumerate(self.class_names):
                    print(f"  {i}: {name}")
                print(f"\nCurrent threshold: {self.confidence_threshold:.0%}")
                print("  Increase (+) if too sensitive (false positives)")
                print("  Decrease (-) if missing gestures")
                print("="*60)
            elif key == ord('+'):
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Threshold increased: {self.confidence_threshold:.0%}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.3, self.confidence_threshold - 0.05)
                print(f"Threshold decreased: {self.confidence_threshold:.0%}")
            elif key == ord('r'):
                self.prev_x, self.prev_y = 0, 0
                print("Mouse position reset")
            elif key == ord('d'):  # Debug mode
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        print(f"\nDebug - Landmarks shape: {landmarks.shape}")
                        print(f"First 10 values: {landmarks[:10]}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nGesture control stopped.")

def main():
    # Check requirements
    try:
        import mediapipe as mp
        import tensorflow as tf
    except ImportError as e:
        print(f"Missing: {e}")
        print("\nInstalling requirements...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "mediapipe", "tensorflow", "opencv-python", "pyautogui", "numpy"])
    
    # Run
    try:
        control = WorkingGestureControl()
        control.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()