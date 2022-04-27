# https://www.geeksforgeeks.org/python-image-classification-using-keras/
# Note: Keras is slower than Pytorch but more intuitive
# Should be ok for our needs

# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import matplotlib.pyplot as plt
import numpy as np


# TODO -- set img height/weight according to preprocessing
img_width, img_height = 64, 64

# TODO - Training data dir. with sub-dirs with classifier names
# i.e. train_data_dir/sunsets -> all training sunset pictures
train_data_dir = 'data/train'

# TODO - Validation (basically test) data dir. with sub-dirs with classifier names
validation_data_dir = 'data/test'

# TODO - set number of samples accordingly
nb_train_samples = 400
nb_validation_samples = 100

epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.compile(loss ='binary_crossentropy', 
					optimizer ='rmsprop', 
				metrics =['accuracy']) 

train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
			horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='binary') 

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='binary') 

model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size) 

x_vals = range(epochs)
val_loss = [0.6459, 0.6098, 0.4486, 0.6026, 0.4921, 0.5549, 
			0.5889, 0.5723, 0.5780, 0.4571]
val_acc = [0.6875, 0.6875, 0.7188, 0.7188, 0.6875, 0.7812, 
			0.6562, 0.7812, 0.5625, 0.8438]

plt.scatter(x_vals, val_loss,c="red")
plt.plot(np.unique(x_vals), np.poly1d(np.polyfit(x_vals, val_loss, 1))(np.unique(x_vals)))
plt.title("Testing Loss across iterations")
plt.xlabel("Epoch Iterations")
plt.ylabel("Loss")
plt.savefig("Binary-Classification-Plot-Loss-1.jpg")
plt.clf()

plt.scatter(x_vals, val_acc,c="green")
plt.plot(np.unique(x_vals), np.poly1d(np.polyfit(x_vals, val_acc, 1))(np.unique(x_vals)))
plt.title("Testing Accuracy across iterations")
plt.xlabel("Epoch Iterations")
plt.ylabel("Accuracy Rate")
plt.savefig("Binary-Classification-Plot-Accuracy-1.jpg")

model.save_weights('model_saved.h5') 
