import csv
import numpy as np
import cv2
import time
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam

EPOCHS = 5
MAX_IMAGE_SAVE = 2

# read data from csv files
def read_data(path, skip_first=False):
  lines = []
  with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for i,line in enumerate(reader):
      if skip_first and i == 0:
        continue
      lines.append(line)
  return lines

# load & read driving data from csvs.
def load_data():
  lines = []
  l1 = read_data('data/driving_log.csv', True)
  l2 = read_data('data/dl2.csv')
  lines.extend(l1)
  lines.extend(l2)
  print(len(l1))
  print(len(l2))
  return lines

# save augmented image to disk
random_count = 0
def save_adjusted_img(original, brightness_adjusted, flipped):
  global random_count
  if random_count < 2:
    timestamp = int(round(time.time()))
    orig_path = 'augmented/' + str(timestamp)
    write_image(original, orig_path)
    write_image(brightness_adjusted, orig_path + '_brightened')
    write_image(flipped, orig_path + '_flipped')
    random_count += 1

# save image to disk
def write_image(image, path):
  path += '.jpg'
  print(path)
  cv2.imwrite(path, image)

def generator(lines, batch_size=32):
  num_samples = len(lines)
  print('num samples', num_samples)
  correction = 0.2
  while 1:
    shuffle(lines)
    for offset in range(0, num_samples, batch_size):
        batch_samples = lines[offset:offset+batch_size]
        images = []
        measurements = []
        for batch_sample in batch_samples:
          for index in range(3):
              path = batch_sample[index]
              tokens = path.split('/')
              filename = tokens[-1].strip()
              folder = tokens[-2].strip()
              updated_path = "data/" + folder + "/" + filename
              image = cv2.imread(updated_path)
 #             print('updated path', updated_path)
              rgb_image = convert_bgr_to_rgb(image)
              images.append(rgb_image)
          center_steering_angle = float(batch_sample[3])
          left_steering_angle = center_steering_angle + correction
          right_steering_angle = center_steering_angle  - correction
          measurements.extend([center_steering_angle, left_steering_angle, right_steering_angle])

        augmented_images, augmented_measurements = [], []
        i1  = random.randint(0, MAX_IMAGE_SAVE)
        i2  = random.randint(0, MAX_IMAGE_SAVE)
        for index, (image, measurement) in enumerate(zip(images, measurements)):
            brightness_adjusted = augment_brightness(image)
            augmented_images.append(brightness_adjusted)
            augmented_measurements.append(measurement)
            flipped_img = cv2.flip(image,1)
            if index == i1 or index == i2:
              save_adjusted_img(image, brightness_adjusted, flipped_img)
            augmented_images.append(augment_brightness(flipped_img))
            augmented_measurements.append(measurement*-1.0)

        X_train = np.array(augmented_images)
        y_train = np.array(augmented_measurements)
        yield X_train, y_train

# convert a valid bgr image to rgb
def convert_bgr_to_rgb(image):
  b,g,r = cv2.split(image)       # get b,g,r
  rgb_img = cv2.merge([r,g,b])     # switch it to rgb
  return rgb_img

def augment_brightness(image):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  brightness = 0.25 +  np.random.uniform()
  hsv_image[:,:,2] = hsv_image[:,:,2]*brightness
  rgb_image = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)
  return rgb_image

def get_model():
  model = Sequential()
  model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((60,20),(0,0))))
  model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
  model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
  model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu', W_regularizer=l2(0.001)))
  model.add(Convolution2D(64,3,3, activation='elu', W_regularizer=l2(0.001)))
  model.add(Convolution2D(64,3,3, activation='elu', W_regularizer=l2(0.001)))
  model.add(Flatten())
  #model.add(Dense(1164))
  #model.add(ELU())
  #model.add(Dropout(0.5))
  model.add(Dense(100, W_regularizer=l2(0.001)))
  model.add(ELU())
  #model.add(Dropout(0.5))
  model.add(Dense(50, W_regularizer=l2(0.001)))
  model.add(ELU())
  #model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer=l2(0.001)))
  model.add(ELU())
  #model.add(Dropout(0.5))
  #model.add(Dense(1164,W_regularizer=l2(0.01)))
  #model.add(Dropout(0.5))
  #model.add(Dense(100, W_regularizer=l2(0.01)))
  #model.add(Dropout(0.5))
  #model.add(Dense(50,W_regularizer=l2(0.01)))
  #model.add(Dropout(0.5))
  #model.add(Dense(10,W_regularizer=l2(0.01)))
  #model.add(Dropout(0.5))
  model.add(Dense(1))
  return model

# data length is muliplied by 6 to account for 3 camera angles per sample and then each flipped image (3 * 2 = 6)
def get_samples_per_epoch(data):
  return len(data) * 6

def main():
  data = load_data()
  train_data, validation_data = train_test_split(data, test_size=0.2)
  training_generator = generator(train_data)
  validation_generator = generator(validation_data)
  print('Training Data Len', len(train_data))
  model = get_model()
  adam  = Adam(lr=0.0001)
  model.compile(optimizer=adam, loss='mse')
  samples_per_epoch = get_samples_per_epoch(train_data)
  validations_per_epoch = get_samples_per_epoch(validation_data)
  model.fit_generator(generator=training_generator, samples_per_epoch= samples_per_epoch, nb_val_samples=validations_per_epoch, validation_data=validation_generator, nb_epoch=EPOCHS, verbose=1)
  model.save('model_final.h5')

if __name__ == '__main__':
	main()
