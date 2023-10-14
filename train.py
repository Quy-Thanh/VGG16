import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def trainning(data_dir, output_file):
	# Prepare training data
	data = [] # List of saved data
	labels = [] # List of stored corresponding labels

	# Loop through all folders in the root directory
	for class_name in os.listdir(data_dir):
	    class_dir = os.path.join(data_dir, class_name)
	    if os.path.isdir(class_dir): 								# Make sure the subfolder is a folder and not a file
	       	# Loop through all image files in subfolders
	        for image_name in os.listdir(class_dir):
	            if image_name.endswith(".jpg"):						# Only use images in .jpg format
	                image_path = os.path.join(class_dir, image_name)
	                img = cv2.imread(image_path)
	                
	                # Resize the image to the desired size (In this case: 224x224)
	                img = cv2.resize(img, (224, 224))
	                
	                # Normalize pixels to the range [0, 1]
	                img = img / 255.0
	                
	                # Add photos and labels to the corresponding list
	                data.append(img)
	                labels.append(class_name)

	# Convert list of data and labels to NumPy array
	data = np.array(data)
	labels = np.array(labels)

	# Create a mapping from label strings to integers
	label_to_int = {label: index for index, label in enumerate(np.unique(labels))}

	# Convert label string to integer
	labels_as_int = [label_to_int[label] for label in labels]

	# Convert the label array to one-hot encoding
	one_hot_labels = to_categorical(labels_as_int)

	# Divide the data into training set and test set
	train_data, val_data, train_labels, val_labels = train_test_split(data, one_hot_labels, test_size=0.2, random_state=42)

	# Build the model
	model = tf.keras.Sequential([
	    layers.Input(shape=(224, 224, 3)),  			# Input with 224 x 224 size and 3 color channels (RGB)
	    layers.Conv2D(64, (3, 3), activation='relu'),
	    layers.MaxPooling2D((2, 2)),
	    layers.Conv2D(128, (3, 3), activation='relu'),
	    layers.MaxPooling2D((2, 2)),
	    layers.Flatten(),
	    layers.Dense(128, activation='relu'),
	    layers.Dense(8, activation='softmax') 			# 8 output layers (number of layers corresponds to number of diseases)
	])

	# Define ModelCheckpoint callback
	checkpoint_dir = 'models/checkpoints'  				# Directory to save checkpoints
	os.makedirs(checkpoint_dir, exist_ok=True)  		# Create the folder if it doesn't already exist
	checkpoint_path = os.path.join(checkpoint_dir, "model_weights.h5")

	checkpoint_callback = ModelCheckpoint(
	    filepath=checkpoint_path,
	    save_best_only=True,  							# Save only the best model based on monitor criteria (default is validation loss)
	    monitor='val_loss',  							# Check according to validation loss
	    verbose=1,  									# Show a notification when the model is saved
	)

	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Model training
	model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels), callbacks=[checkpoint_callback])

	# Evaluate the model
	model.evaluate(val_data, val_labels)

	# Export model
	model.save(output_file)

if __name__ == "__main__":
	data_dir = "data/"
	output_file = "models/savedmodels/version1.keras"
	os.makedirs("models/savedmodels/", exist_ok=True)  		# Create the folder if it doesn't already exist
	trainning(data_dir, output_file)
