@echo off
echo ========================================
echo   REAL GESTURE CONTROL - COMPLETE SETUP
echo ========================================
echo.

echo Step 1: Installing requirements...
pip install mediapipe tensorflow opencv-python pyautogui numpy

echo.
echo Step 2: Collect REAL gesture data...
echo This will open your webcam and collect hand gestures.
echo Follow the on-screen instructions.
echo.
pause

python collect_real_gestures.py

echo.
echo Step 3: Train model with REAL data...
python train_real_model.py

echo.
echo Step 4: Run gesture control...
echo Now you can run the working gesture control:
echo   python gesture_control_real.py
echo.
pause