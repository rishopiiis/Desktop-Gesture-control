# batch_train.py - Train multiple faces from images
import cv2
import os
import numpy as np
import pickle

def batch_train_from_images():
    """Train from existing image folders"""
    image_dir = "face_images"  # Create this folder
    models_dir = "trained_models"
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Folder structure: face_images/John_Doe/*.jpg
    #                   face_images/Jane_Smith/*.jpg
    
    face_samples = []
    labels = []
    label_dict = {}
    next_id = 0
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print("🔍 Looking for face images...")
    
    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"Processing: {person_name}")
        
        if person_name not in label_dict:
            label_dict[person_name] = next_id
            next_id += 1
        
        label_id = label_dict[person_name]
        
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    face_equalized = cv2.equalizeHist(face_resized)
                    
                    face_samples.append(face_equalized)
                    labels.append(label_id)
    
    if len(face_samples) == 0:
        print("❌ No faces found for training")
        return
    
    print(f"\nFound {len(face_samples)} face samples from {len(label_dict)} people")
    
    # Create and train recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(labels))
    
    # Save model
    recognizer.write(os.path.join(models_dir, "face_model.yml"))
    with open(os.path.join(models_dir, "labels.pickle"), 'wb') as f:
        pickle.dump(label_dict, f)
    
    print("✅ Batch training complete!")
    print(f"   Trained on {len(label_dict)} people")
    print(f"   Total samples: {len(face_samples)}")

if __name__ == "__main__":
    batch_train_from_images()