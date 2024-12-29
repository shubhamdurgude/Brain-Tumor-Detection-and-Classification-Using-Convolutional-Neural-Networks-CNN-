# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

"""### **Define Directory Path**"""

# Define the paths to training and testing directories
train_dir = '/content/drive/MyDrive/Datasets/Brain_Tumor_MRI_dataset/Training'
test_dir = '/content/drive/MyDrive/Datasets/Brain_Tumor_MRI_dataset/Testing'

"""### **Load Dataset**"""

# Load the training dataset with a validation split
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42
)

# Load the validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=42
)

# Load the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32,
    shuffle=False
)

"""### **Classes in Dataset**"""

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

"""### **Sample images from Training dataset**"""

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")

plt.tight_layout()
plt.show()

"""### **Normalize Images**"""

# Normalize the datasets
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

"""### **Class Weights**"""

# Calculate class weights to handle class imbalance
y_train = np.concatenate([y for _, y in train_dataset], axis=0)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weights)

"""### **Define Model**"""

# Define the CNN model
#num_classes = len(train_dataset.class_names)  # Adjust according to your number of classes
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Use early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with class weights
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,  # Use validation dataset here
    epochs=5,
    class_weight=class_weights,  # Use class weights here
    callbacks=[early_stopping],
    verbose=2
)

# Evaluate the model on train dataset
train_loss, train_accuracy = model.evaluate(train_dataset)

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Save the model architecture and weights
model.save('My_model.h5')
print("Model saved!")

# Evaluate model on test images
test_loss, test_accuracy = model.evaluate(test_dataset)

# Generate predictions on the test dataset
predictions = model.predict(test_dataset)

# Get the predicted class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Display confusion matrix
conf_matrix = confusion_matrix(y_true, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display the classification report
report = classification_report(y_true, predicted_labels, target_names=class_names)
print(report)

# predictions on test dataset
def show_sample_predictions(dataset, model, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"True: {class_names[labels[i]]}, Pred: {class_names[predicted_labels[i]]}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_predictions(test_dataset, model, class_names)

# sample predictions for a single input image
img_path = '/content/drive/MyDrive/Datasets/Brain_Tumor_MRI_dataset/sample_image.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Display the image with the prediction
plt.figure(figsize=(5, 5))
plt.imshow(tf.keras.utils.load_img(img_path))
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis('off')
plt.show()

# Visualize the training history (loss and accuracy)
def plot_training_history(history):
    # Plot training & validation accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

