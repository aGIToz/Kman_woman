# import the necessary packages
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from myUtils import ImageToArrayPreprocessor
from myUtils import SimplePreprocessor
from myUtils import SimpleDatasetLoader
from conf import myConfig as config
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# grab the images.
print("[INFO] loading images...")
p = Path(args["dataset"])
print(p)
imagePaths = list(p.glob('./**/*.jpg'))
imagePaths = [str(names) for names in imagePaths]
print(imagePaths[1])
classNames=[os.path.split(os.path.split((names))[0])[1] for names in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print(classNames)

# initialize the image preprocessors
sap = SimplePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=config.split, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print('trainY[0]',trainY[0])
print('testY[0]',testY[0])

# load the VGG16 network, ensuring the head FC layer sets are left
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

print("[INFO] extracting features...")
features_train = baseModel.predict(trainX)
features_test = baseModel.predict(testX)

print("[INFO] saving features...")
np.save(config.path_feature_train,features_train)
np.save(config.path_feature_test,features_test)

print("[INFO] load the features")
features_train = np.load(config.path_feature_train)
features_test = np.load(config.path_feature_test)

print("[INFO] training the head")
model = Sequential()
model.add(Flatten(input_shape=features_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(config.param_dropout))
model.add(Dense(1, activation='sigmoid'))

print("[INFO] compile the model")
opt = RMSprop(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit(features_train,trainY,class_weight=config.classWeights,epochs=config.epochs,
	batch_size=config.batch_size,validation_data=(features_test, testY))

print("[INFO] saving model...")
model.save(args["model"])

print("[INFO] evaluating network on test set...")
predictions = model.predict(features_test, batch_size=config.batch_size)
threshold, upper, lower = 0.5, 1, 0
y_pred = np.where(predictions>threshold, upper, lower) 
print("y_pred", y_pred)
print(classification_report(testY,
	y_pred, target_names=classNames))

# plot loss/accuracy vs epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, config.epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, config.epochs), H.history["acc"], label="acc")
plt.plot(np.arange(0, config.epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
