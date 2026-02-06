# check_save.py - Test if saving works
import pickle
import numpy as np
import os

print("="*60)
print("TESTING SAVE FUNCTION")
print("="*60)

# Test 1: Create test data
test_data = {
    'features': np.random.randn(10, 63),
    'labels': np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]),
    'class_names': ['mouse_move', 'left_click', 'right_click', 
                   'scroll_up', 'scroll_down', 'drag', 'none']
}

# Test 2: Save
filename = 'test_save.pkl'
current_dir = os.getcwd()
save_path = os.path.join(current_dir, filename)

print(f"\n📁 Current directory: {current_dir}")
print(f"💾 Attempting to save: {save_path}")

try:
    with open(filename, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"✅ Save attempt completed")
    
    # Check if file exists
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"✅ File created successfully!")
        print(f"   Size: {file_size} bytes")
        print(f"   Path: {os.path.abspath(filename)}")
        
        # Try to load it back
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"✅ File can be loaded back")
        print(f"   Samples: {len(loaded_data['features'])}")
        print(f"   Classes: {len(loaded_data['class_names'])}")
        
        # Clean up
        os.remove(filename)
        print(f"✅ Test file cleaned up")
        
    else:
        print(f"❌ ERROR: File not created!")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "="*60)
print("PERMISSION CHECK:")
print("="*60)

# Check write permissions
test_write_file = 'write_test.txt'
try:
    with open(test_write_file, 'w') as f:
        f.write('test')
    os.remove(test_write_file)
    print("✅ Has write permission in this folder")
except:
    print("❌ NO write permission in this folder!")
    print("   Try running as Administrator")
    print("   Or move to a different folder (Desktop, Documents)")

print(f"\n" + "="*60)