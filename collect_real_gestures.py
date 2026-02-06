# collect_real_gestures_fixed.py - FIXED SAVE FUNCTION
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
import sys

print("="*60)
print("REAL GESTURE DATA COLLECTION (FIXED SAVE)")
print("="*60)

class RealGestureCollector:
    def __init__(self):
        # 7 gestures for desktop control
        self.gestures = [
            ("mouse_move", "👆 Index finger pointing"),
            ("left_click", "✌️ Index+Middle fingers (peace sign)"),
            ("right_click", "🫎 Three fingers up"),
            ("scroll_up", "🖐️ Open hand, move UP slightly"),
            ("scroll_down", "✊ Fist, move DOWN slightly"),
            ("drag", "👌 Thumb+Index pinched"),
            ("none", "🤚 Relaxed hand")
        ]
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            print("ERROR: No webcam found!")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Data storage
        self.dataset = {
            'features': [],
            'labels': [],
            'class_names': [g[0] for g in self.gestures]
        }
        
        self.current_gesture = 0
        self.collecting = False
        self.samples_collected = {i: 0 for i in range(len(self.gestures))}
        self.total_needed = 100  # Samples per gesture
        
        # Show current working directory
        print(f"\n📁 Current directory: {os.getcwd()}")
        print(f"\nWill collect {self.total_needed} samples for each gesture:")
        for i, (name, desc) in enumerate(self.gestures):
            print(f"  {i}: {name} - {desc}")
        
        print(f"\nCONTROLS:")
        print(f"  0-6: Select gesture")
        print(f"  SPACE: Start/stop collecting")
        print(f"  S: Save dataset NOW")
        print(f"  Q: Quit and auto-save")
        print("="*60)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmarks as feature vector"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)
    
    def save_dataset(self):
        """FIXED: Save collected dataset to file"""
        if len(self.dataset['features']) == 0:
            print("\n⚠ No data collected yet!")
            return
        
        try:
            # Convert to numpy arrays
            features = np.array(self.dataset['features'])
            labels = np.array(self.dataset['labels'])
            
            # Create dataset dict
            dataset = {
                'features': features,
                'labels': labels,
                'class_names': self.dataset['class_names']
            }
            
            # Save with proper path
            filename = 'real_gesture_dataset.pkl'
            save_path = os.path.abspath(filename)
            
            print(f"\n" + "="*60)
            print("💾 SAVING DATASET...")
            print(f"  File: {filename}")
            print(f"  Path: {save_path}")
            
            with open(filename, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"\n✅ SUCCESS! Dataset saved.")
            print(f"  Total samples: {len(features)}")
            print(f"  File size: {os.path.getsize(filename) / 1024:.1f} KB")
            
            # Show class distribution
            print(f"\n📊 Class distribution:")
            for i, name in enumerate(self.dataset['class_names']):
                count = np.sum(labels == i)
                print(f"  {name}: {count} samples")
            
            # Create a backup copy
            backup_name = f"backup_{int(time.time())}.pkl"
            with open(backup_name, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"  Backup: {backup_name}")
            
            # Verify file exists
            if os.path.exists(filename):
                print(f"\n✅ Verification: File exists!")
                return True
            else:
                print(f"\n❌ ERROR: File not created!")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR saving dataset: {e}")
            return False
    
    def run(self):
        """Main collection loop"""
        print(f"\nReady! Current gesture: {self.gestures[self.current_gesture][0]}")
        print(f"Press SPACE to start collecting...")
        
        last_save_time = time.time()
        collection_started = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw UI overlay
            cv2.rectangle(frame, (0, 0), (640, 160), (50, 50, 50), -1)
            
            # Current status
            gesture_name, gesture_desc = self.gestures[self.current_gesture]
            status = "COLLECTING" if self.collecting else "READY"
            status_color = (0, 255, 0) if self.collecting else (0, 0, 255)
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Collection progress
            collected = self.samples_collected[self.current_gesture]
            progress = f"Collected: {collected}/{self.total_needed}"
            cv2.putText(frame, progress, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Total progress
            total_collected = sum(self.samples_collected.values())
            total_needed = self.total_needed * len(self.gestures)
            total_progress = f"Total: {total_collected}/{total_needed}"
            cv2.putText(frame, total_progress, (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw hand if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Auto-collect if enabled
                    if self.collecting:
                        current_time = time.time()
                        if current_time - last_save_time > 0.1:  # 10 FPS collection
                            landmarks = self.extract_landmarks(hand_landmarks)
                            
                            if len(landmarks) == 63:
                                self.dataset['features'].append(landmarks)
                                self.dataset['labels'].append(self.current_gesture)
                                self.samples_collected[self.current_gesture] += 1
                                last_save_time = current_time
                                
                                # Visual feedback
                                cv2.circle(frame, (600, 50), 20, (0, 255, 0), -1)
                                
                                # Auto-save every 50 samples
                                if total_collected % 50 == 0 and total_collected > 0:
                                    self.save_dataset()
                                
                                # Move to next gesture if enough samples
                                if self.samples_collected[self.current_gesture] >= self.total_needed:
                                    print(f"\n✓ Completed {gesture_name} ({self.total_needed} samples)")
                                    if self.current_gesture < len(self.gestures) - 1:
                                        self.current_gesture += 1
                                        print(f"→ Now collecting: {self.gestures[self.current_gesture][0]}")
                                    else:
                                        print(f"\n🎉 All gestures completed!")
                                        self.collecting = False
            
            # Show gesture visualization
            self.draw_gesture_guide(frame, self.current_gesture)
            
            # Display frame
            cv2.imshow("Collect Real Gestures (FIXED)", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord(' '):  # SPACE to toggle collection
                self.collecting = not self.collecting
                status = "STARTED" if self.collecting else "PAUSED"
                print(f"\nCollection {status} for {gesture_name}")
            elif ord('0') <= key <= ord('6'):
                gesture_num = key - ord('0')
                if gesture_num < len(self.gestures):
                    self.current_gesture = gesture_num
                    print(f"\nSwitched to: {self.gestures[self.current_gesture][0]}")
            elif key == ord('s'):
                print(f"\nManual save requested...")
                success = self.save_dataset()
                if success:
                    print(f"✅ Save successful!")
                else:
                    print(f"❌ Save failed!")
            elif key == ord('c'):  # Check current data
                print(f"\n📊 Current data status:")
                print(f"  Total samples: {len(self.dataset['features'])}")
                for i, name in enumerate(self.dataset['class_names']):
                    count = sum(1 for label in self.dataset['labels'] if label == i)
                    print(f"  {name}: {count} samples")
        
        # Auto-save on exit
        print(f"\nAuto-saving on exit...")
        success = self.save_dataset()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if success:
            print(f"\n✅ Data collection COMPLETED!")
            print(f"✅ Dataset saved successfully!")
            print(f"\nNext steps:")
            print(f"  1. Train model: python train_real_model.py")
            print(f"  2. Run control: python gesture_control_real.py")
        else:
            print(f"\n❌ ERROR: Data NOT saved!")
    
    def draw_gesture_guide(self, frame, gesture_id):
        """Draw visual guide for current gesture"""
        guides = [
            "Show: 👆 ONE finger pointing",
            "Show: ✌️ TWO fingers (peace sign)",
            "Show: 🫎 THREE fingers up",
            "Show: 🖐️ OPEN hand, palm forward",
            "Show: ✊ CLOSED fist",
            "Show: 👌 THUMB+INDEX pinched",
            "Show: 🤚 RELAXED hand"
        ]
        
        cv2.putText(frame, guides[gesture_id], (10, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

def main():
    # Check requirements
    try:
        import mediapipe as mp
    except ImportError:
        print("Installing MediaPipe...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe", "opencv-python", "numpy"])
    
    # Start collection
    print("\n" + "="*60)
    print("IMPORTANT: Your data will be saved in:")
    print(f"  {os.getcwd()}")
    print("="*60)
    
    collector = RealGestureCollector()
    collector.run()

if __name__ == "__main__":
    main()