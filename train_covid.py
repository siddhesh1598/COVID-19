# import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imutils import paths

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# argparser 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input data")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to model")
args = vars(ap.parse_args())

# initialize parameters
lr = 1e-3
epochs = 25
batch_size = 8

# load dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# iterate over all images and append
# images to data list
# labels to labels list
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)

# encoding the dataset
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print("Labels: \n", labels)

# partition dataset into training and testing
(X_train, X_test, y_train, y_test) = train_test_split(data,
										labels,
										test_size=0.2,
										stratify=labels,
										random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

'''
X_train = X_train.reshape(-1, 
						X_train.shape[0], 
						X_train.shape[1],
						X_train.shape[2])

X_test = X_test.reshape(-1, 
						X_test.shape[0], 
						X_test.shape[1],
						X_test.shape[2])
'''

# data augmentation
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# model
vgg = VGG16(weights="imagenet", include_top=False,
			input_tensor=Input(shape=(224, 224, 3)))

model = vgg.output
model = AveragePooling2D(pool_size=(4, 4))(model)
model = Flatten()(model)
model = Dense(64, activation="relu")(model)
model = Dropout(0.5)(model)
model = Dense(2, activation="softmax")(model)

model = Model(inputs=vgg.input, outputs=model)

for layer in vgg.layers:
	layer.trainable = False

# compile
print("[INFO] compiling model...")
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy",
				optimizer=opt,
				metrics=["accuracy"])

# train
print("[INFO] training model...")
H = model.fit(
	x=X_train,
	y=y_train,
	batch_size=batch_size,
	steps_per_epoch=len(X_train)//batch_size,
	validation_data=(X_test, y_test),
	validation_steps=len(X_test)//batch_size,
	epochs=epochs)

# test
print("[INFO] evaluating model...")
predIdxs = model.predict(X_test, batch_size=batch_size)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(y_test.argmax(axis=1),
						predIdxs,
						target_names=lb.classes_))

# confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# save model
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")