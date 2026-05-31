import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

# =========================
# IMAGE SETTINGS
# =========================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3

# =========================
# DATA PREPROCESSING
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# =========================
# LOAD TRAINING DATA
# =========================

train_data = train_datagen.flow_from_directory(
    'archive (10)/PlantVillage',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# =========================
# LOAD VALIDATION DATA
# =========================

val_data = train_datagen.flow_from_directory(
    'archive (10)/PlantVillage',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =========================
# PRINT CLASS LABELS
# =========================

print("\nClass Labels:")
print(train_data.class_indices)

# =========================
# BUILD CNN MODEL
# =========================

model = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # First Convolution Block
    Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    # Second Convolution Block
    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    # Third Convolution Block
    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    # Reduce Parameters
    GlobalAveragePooling2D(),

    # Dense Layer
    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.3),

    # Output Layer
    Dense(
        train_data.num_classes,
        activation='softmax'
    )
])

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

print("\nStarting Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    verbose=2
)

print("\nTraining Finished")

# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(
    val_data,
    verbose=0
)

print(f"\nValidation Accuracy: {accuracy:.2f}")

# =========================
# SAVE MODEL
# =========================

model.save("plant_disease_model.h5")

print("\nModel Saved Successfully")