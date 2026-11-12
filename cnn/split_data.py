import os
import shutil
import random

# 1. Define paths using '..' to go up one level
# '../data' means "Go out of CNN folder, then into data folder"
base_dir = os.path.join('..', 'data')
source_path = os.path.join(base_dir, 'PetImages')

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Define split ratio
split_ratio = 0.8 

# 2. Safety Check
if not os.path.exists(source_path):
    print(f"❌ Error: Could not find {source_path}")
    print(f"Current working directory is: {os.getcwd()}")
    print("Are you sure 'data' is right next to the 'CNN' folder?")
else:
    print(f"✅ Found source data at: {source_path}")

    # 3. Create folders (Overwrite if exists)
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # 4. Split the data
    classes = ['Cat', 'Dog']
    for class_name in classes:
        class_source = os.path.join(source_path, class_name)
        
        # Make subfolders (e.g., ../data/train/Cat)
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        if os.path.exists(class_source):
            # Get and shuffle files
            files = [f for f in os.listdir(class_source) if f.endswith('.jpg')]
            random.shuffle(files)
            
            split_point = int(len(files) * split_ratio)
            train_files = files[:split_point]
            test_files = files[split_point:]
            
            print(f"Processing {class_name}: {len(train_files)} training, {len(test_files)} testing...")
            
            # Copy files
            for f in train_files:
                shutil.copy(os.path.join(class_source, f), os.path.join(train_dir, class_name, f))
            for f in test_files:
                shutil.copy(os.path.join(class_source, f), os.path.join(test_dir, class_name, f))
                
    print("\n🎉 Success! Data is now in '../data/train' and '../data/test'")