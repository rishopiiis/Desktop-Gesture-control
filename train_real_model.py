# train_real_model.py - Train with REAL collected data
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os
import sys

print("="*60)
print("TRAIN WITH REAL GESTURE DATA")
print("="*60)

def create_simple_model(input_shape, num_classes):
    """Create a simple but effective model"""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def manual_train_val_test_split(X, y, test_size=0.2, val_size=0.1):
    """Manual data splitting"""
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int((n_samples - n_test) * val_size)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    # Split data
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Check if dataset exists
    dataset_files = ['real_gesture_dataset.pkl', 'gesture_dataset.pkl']
    dataset = None
    
    for filename in dataset_files:
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            with open(filename, 'rb') as f:
                dataset = pickle.load(f)
            break
    
    if dataset is None:
        print("✗ No dataset found!")
        print("\nPlease collect real data first:")
        print("  python collect_real_gestures.py")
        sys.exit(1)
    
    X = dataset['features']
    y = dataset['labels']
    class_names = dataset['class_names']
    
    print(f"\nDataset loaded:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(class_names)}")
    
    # Show class distribution
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {class_names[label]}: {count} samples")
    
    # Check for class imbalance
    min_samples = np.min(counts)
    max_samples = np.max(counts)
    
    if max_samples > min_samples * 3:
        print(f"\n⚠ WARNING: Class imbalance detected!")
        print(f"  Max: {max_samples}, Min: {min_samples}")
        print(f"  Model may favor classes with more samples")
    
    # Split data
    print(f"\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = manual_train_val_test_split(X, y)
    
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Create and train model
    print(f"\nCreating model...")
    model = create_simple_model((X.shape[1],), len(class_names))
    model.summary()
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train
    print(f"\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\nEvaluating...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Test Accuracy: {test_accuracy:.2%}")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    for i, name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
            print(f"  {name}: {class_acc:.2%}")
    
    # Save model
    model.save('real_gesture_model.h5')
    print(f"\n✓ Model saved: real_gesture_model.h5")
    
    # Save model info
    model_info = {
        'class_names': class_names,
        'accuracy': float(test_accuracy),
        'input_shape': X.shape[1],
        'dataset': 'real_collected',
        'samples': X.shape[0]
    }
    
    with open('real_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✓ Model info saved")
    
    # Test predictions
    print(f"\nTesting sample predictions:")
    
    # Create test samples
    test_gestures = []
    for i in range(len(class_names)):
        # Find a sample of this class
        idx = np.where(y_test == i)[0]
        if len(idx) > 0:
            sample = X_test[idx[0]]
            true_label = y_test[idx[0]]
            
            prediction = model.predict(sample.reshape(1, -1), verbose=0)
            pred_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            status = "✓" if pred_class == true_label else "✗"
            print(f"{status} {class_names[true_label]} → {class_names[pred_class]} ({confidence:.1%})")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        print("Installing TensorFlow...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    
    main()