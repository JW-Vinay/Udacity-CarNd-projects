import csv
import numpy as np
import cv2
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
images = []
measurements = []
correction = 0.2
for i, line in enumerate(lines):
	if i == 0:
		continue
	for index in range(3):
		path = line[index]	
		filename = path.split('/')[-1]
		updated_path = "data/IMG/" + filename
		image = cv2.imread(updated_path)
		images.append(image)
	center_steering_angle = float(line[3])
	left_steering_angle = center_steering_angle + correction
	right_steering_angle = center_steering_angle  - correction
	measurements.extend([center_steering_angle, left_steering_angle, right_steering_angle])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_img = cv2.flip(image,1)
	augmented_images.append(flipped_img)
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_v1.h5')
