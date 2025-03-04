import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
WINDOW_SIZE = 18  # Number of frames per sequence
TARGET_SIZE = (64, 64)  # Frame size
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001

def create_3d_cnn_model(input_shape):
    """
    Create a 3D CNN model for fall detection
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        # First 3D CNN layer
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Second 3D CNN layer
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Third 3D CNN layer
        Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Fourth 3D CNN layer
        Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def process_video(video_path, window_size=WINDOW_SIZE, target_size=TARGET_SIZE):
    """
    Process video for fall detection
    
    Args:
        video_path (str): Path to video file
        window_size (int): Number of frames to sample
        target_size (tuple): Target size for frames (height, width)
        
    Returns:
        np.array: Processed frames
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < window_size:
        print(f"Warning: Video {video_path} has fewer frames ({total_frames}) than required ({window_size})")
        cap.release()
        return None
    
    # Calculate frames to sample
    sample_indices = np.linspace(0, total_frames - 1, window_size, dtype=int)
    
    # Initialize array for frames
    frames = np.zeros((window_size, target_size[0], target_size[1], 2))
    
    # Extract and process frames
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            cap.release()
            return None
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Convert to grayscale for first channel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to edge detection for second channel
        edges = cv2.Canny(gray, 100, 200)
        
        # Store in frames array
        frames[i, :, :, 0] = gray / 255.0
        frames[i, :, :, 1] = edges / 255.0
    
    cap.release()
    return frames

def load_up_fall_dataset(base_path):
    """
    Load the UP-Fall dataset
    
    Args:
        base_path (str): Path to UP-Fall dataset
        
    Returns:
        X (np.array): Video sequences
        y (np.array): Labels (1 for fall, 0 for non-fall)
    """
    # Placeholder for data and labels
    X = []
    y = []
    
    # Find all video files
    fall_videos = glob(os.path.join(base_path, 'Fall/**/*.avi'), recursive=True)
    non_fall_videos = glob(os.path.join(base_path, 'NonFall/**/*.avi'), recursive=True)
    
    print(f"Found {len(fall_videos)} fall videos and {len(non_fall_videos)} non-fall videos")
    
    # Process fall videos
    print("Processing fall videos...")
    for video_path in tqdm(fall_videos):
        frames = process_video(video_path)
        if frames is not None:
            X.append(frames)
            y.append(1)  # Fall label
    
    # Process non-fall videos
    print("Processing non-fall videos...")
    for video_path in tqdm(non_fall_videos):
        frames = process_video(video_path)
        if frames is not None:
            X.append(frames)
            y.append(0)  # Non-fall label
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")
    return X, y

def create_dataset_from_raw_videos(raw_video_path, output_path):
    """
    Create a structured dataset from raw videos for fall detection
    
    Args:
        raw_video_path (str): Path to raw videos folder
        output_path (str): Path to save processed dataset
    """
    # Create output directories
    os.makedirs(os.path.join(output_path, 'Fall'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'NonFall'), exist_ok=True)
    
    # Process raw videos
    if os.path.exists(raw_video_path):
        print(f"Processing raw videos from {raw_video_path}")
        
        # This function should be customized based on your actual raw data structure
        # Here's a skeleton implementation assuming certain folder structure
        
        # Find all fall videos
        fall_videos = glob(os.path.join(raw_video_path, '*fall*.mp4'))
        fall_videos.extend(glob(os.path.join(raw_video_path, '*Fall*.mp4')))
        
        # Find all non-fall videos
        non_fall_videos = glob(os.path.join(raw_video_path, '*ADL*.mp4'))  # Activities of Daily Living
        non_fall_videos.extend(glob(os.path.join(raw_video_path, '*walking*.mp4')))
        non_fall_videos.extend(glob(os.path.join(raw_video_path, '*sitting*.mp4')))
        
        # Process and save fall videos
        for i, video_path in enumerate(fall_videos):
            dest_path = os.path.join(output_path, 'Fall', f'fall_{i}.avi')
            # Convert/process video if needed
            os.system(f'ffmpeg -i "{video_path}" -c:v mjpeg -q:v 3 "{dest_path}"')
        
        # Process and save non-fall videos
        for i, video_path in enumerate(non_fall_videos):
            dest_path = os.path.join(output_path, 'NonFall', f'nonfall_{i}.avi')
            # Convert/process video if needed
            os.system(f'ffmpeg -i "{video_path}" -c:v mjpeg -q:v 3 "{dest_path}"')
        
        print(f"Processed {len(fall_videos)} fall videos and {len(non_fall_videos)} non-fall videos")
    else:
        print(f"Raw video path {raw_video_path} not found")

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Keras training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    """
    Train the fall detection model
    """
    # Data paths - adjust these based on your setup
    RAW_VIDEO_PATH = "raw_videos"  # Path to raw videos if you need to process them
    DATASET_PATH = "UP-Fall-Dataset"  # Path to the UP-Fall dataset or your processed dataset
    
    # Create dataset from raw videos if needed
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        create_dataset_from_raw_videos(RAW_VIDEO_PATH, DATASET_PATH)
    
    # Load dataset
    X, y = load_up_fall_dataset(DATASET_PATH)
    
    if len(X) == 0:
        print("No data loaded. Make sure your dataset is correctly structured.")
        return
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create model
    model = create_3d_cnn_model(input_shape=(WINDOW_SIZE, TARGET_SIZE[0], TARGET_SIZE[1], 2))
    model.summary()
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'fall_detection_3d_cnn.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('fall_detection_3d_cnn.h5')
    print("Model saved to 'fall_detection_3d_cnn.h5'")

if __name__ == "__main__":
    train_model()