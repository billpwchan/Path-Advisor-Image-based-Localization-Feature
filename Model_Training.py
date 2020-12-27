import argparse
import json
import os

import keras
import matplotlib.pyplot as plt
from keras import backend as k
from keras.applications import *
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.engine.sequential import Sequential
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Sample Calling Script => python Model_Training.py -n 116 -td ./dataset/training_set -iw 299 -ih 299 -b 64 -e 150 -f 132 (Success with GTX 1080TI)
parser = argparse.ArgumentParser(
    description='HKUST Path Advisor Image-Classification Model Training Script')

# Required arguments
parser.add_argument('-n', '--nb_classes', required=True,
                    help="<Required> number of classes", type=int)
parser.add_argument('-td', '--train_dir', required=True,
                    help="<Required> trainin data file path", type=str)
parser.add_argument('-vd', '--validate_dir', required=False,
                    help="<Optional> validation data file path (if not provided, 20/80 split will be used)")
parser.add_argument('-lm', '--load_model', required=False,
                    help="<Optional> load trained model")
parser.add_argument('-lw', '--load_weights', required=False,
                    help="<Optional> load model weights")
parser.add_argument('-iw', '--width', required=False,
                    default=299, help="<Optional> image width")
parser.add_argument('-ih', '--height', required=False,
                    default=299, help="<Optional> image height")
parser.add_argument('-b', '--batch_size', required=False,
                    default=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], help="<Optional> training batch size", type=int)
parser.add_argument('-t', '--train_top', required=False, default=False,
                    help="<Optional> train top layer only")
parser.add_argument('-e', '--epoch', required=False,
                    default=3000, help="<Optional> training # epoch")
parser.add_argument('-f', '--freeze', required=False, default=65,
                    help="<Optional> # layers freezed during training")
args = parser.parse_args()

# Hyper parameters for model
nb_classes = int(args.nb_classes)
img_width, img_height = int(args.width), int(args.height)
batch_size = int(args.batch_size)
if args.train_top:
    print("## Training Top Layers Only")
else:
    print("## Train All Layers")
nb_epoch = int(args.epoch)
freeze_layers = int(args.freeze)
if os.path.exists(args.train_dir):
    train_data_dir = args.train_dir
else:
    raise IOError("Invalid Training Path Provided")
if args.validate_dir and os.path.exists(args.validate_dir):
    validation_data_dir = args.validate_dir

# Import Xception Model
base_model = Xception(input_shape=(img_width, img_height, 3),
                      weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = Dropout(0.85)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# Add top layer block to your base model
model = Model(base_model.input, predictions)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
#                  input_shape=(img_width, img_height, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(Dropout(0.4))

# model.add(GlobalAveragePooling2D())

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(nb_classes, activation='softmax'))

# save weights of best training epoch: monitor either val_loss or val_acc
if args.load_model and os.path.exists(args.load_model):
    model = load_model(args.load_model)
elif args.load_weights and os.path.exists(args.load_weights):
    model.load_weights(args.load_weights)

# Read Data and Augment it
if args.validate_dir and os.path.exists(args.validate_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       zoom_range=0.1,
                                       brightness_range=[0.9, 1.1],
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True
                                       )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(
                                                            img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                            target_size=(
                                                                img_width, img_height),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')
else:
    print("NO VALIDATED DIR FOUND")
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       zoom_range=0.1,
                                       brightness_range=[0.9, 1.1],
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(
                                                            img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        subset='training')
    validation_generator = train_datagen.flow_from_directory(train_data_dir,
                                                             target_size=(
                                                                 img_width, img_height),
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             subset='validation')

for layer in model.layers[:freeze_layers]:
    layer.trainable = False
for layer in model.layers[freeze_layers:]:
    layer.trainable = True

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

final_model_path = os.path.join(
    os.path.abspath('model'), 'model.h5')
callbacks_list = [
    ModelCheckpoint(final_model_path, monitor='val_categorical_accuracy',
                    verbose=0, save_best_only=True, mode='auto', period=1),
    TensorBoard(log_dir='./Graph', histogram_freq=0,
                write_graph=True, write_images=True),
    EarlyStopping(monitor='val_loss',
                  mode='min', verbose=1, patience=700),
    CSVLogger('./logs/training.log')
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)

# fine-tune the model
print("### Started Training ###")
predict_history = model.fit_generator(train_generator,
                                      steps_per_epoch=train_generator.samples/batch_size,
                                      epochs=nb_epoch,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.samples/batch_size,
                                      callbacks=callbacks_list)


# # summarize history for accuracy
# plt.plot(predict_history.history['categorical_accuracy'])
# plt.plot(predict_history.history['val_categorical_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize histor for Top-5 accuracy
# plt.plot(predict_history.history['top_k_categorical_accuracy'])
# plt.plot(predict_history.history['val_top_k_categorical_accuracy'])
# plt.title('model top 5 accuracy')
# plt.ylabel('top 5 accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(predict_history.history['loss'])
# plt.plot(predict_history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
