import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import os

# --- CONFIGURATION ---
DATASET_PATH = r"D:\project\new plant disease"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def train_model():
    print(f"Checking dataset at: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset path not found. Please configure the correct path in train_model.py")
        return

    # --- DATA AUGMENTATION & LOADING ---
    print("🚀 Initializing Indices & Data Generators...")
    
    # Advanced Augmentation for Robustness (addressing lighting, rotation etc.)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Using valid split from train dir if structure allows, or assume separate folders
    )

    # Assuming 'train' and 'valid' folders exist inside DATASET_PATH
    # If not, we fall back to splitting the single directory
    train_dir = os.path.join(DATASET_PATH, 'train')
    valid_dir = os.path.join(DATASET_PATH, 'valid')
    
    if os.path.exists(train_dir):
        print("Found train/valid structure.")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
    else:
        print("Using automatic validation split on root directory.")
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

    # Save Class Names
    class_names = list(train_generator.class_indices.keys())
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print(f"✅ Saved {len(class_names)} classes to class_names.json")

    # --- MODEL ARCHITECTURE (EfficientNetB3) ---
    # Using B3 as a sweet spot between B0 (fast) and B4 (heavy) for accuracy
    print("🏗️ Building EfficientNet Model...")
    
    base_model = applications.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    base_model.trainable = False # Freeze base for first pass

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- CALLBACKS ---
    checkpoint = ModelCheckpoint(
        'best_plant_disease_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    # --- TRAINING ---
    print("🔥 Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # --- FINE TUNING (Optional but recommended) ---
    print("🔧 Fine-tuning top layers...")
    base_model.trainable = True
    # Freeze all except last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🔥 Starting Fine-Tuning...")
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    model.save('trained_plant_disease_model.keras')
    print("🎉 Training Complete. Model saved.")

if __name__ == "__main__":
    train_model()
