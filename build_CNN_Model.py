import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Dropout, Flatten, Activation, Dense, \
    BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import json


data = np.load('train_dataset.npy')
target = np.load('train_targetset.npy')
train_data, valid_data, train_target, valid_target = train_test_split(data, target, test_size=0.1)
print(train_data.shape)
data_generator = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
test_data = np.load('valid_dataset.npy')
test_target = np.load('valid_targetset.npy')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=data.shape[1:]))  # start:Dense layers
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(Conv2D(250, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2, 2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2, 2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2, 2))

model.add(Conv2D(256, (2, 2)))
model.add(Activation("relu"))
model.add(MaxPool2D(2, 2))

model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# metrics=[metrics.AUC(name='auc'), 'accuracy']
# Checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')


BATCH_SIZE = 32
history = model.fit(
    data_generator.flow(train_data, train_target, batch_size=BATCH_SIZE),
    steps_per_epoch=int(train_data.shape[0] / BATCH_SIZE),
    epochs=16,
    validation_data=(valid_data, valid_target),
    callbacks=[checkpoint]
)

model.evaluate(valid_data, valid_target)

# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
model.save("model.h5")
print("Saved model to disk")

with open('history.json', 'w') as f:
    json.dump(str(history.history), f)

plt.plot(history.history['loss'], 'r', label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('# epoches')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], 'r', label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('# epoches')
plt.ylabel('accuracy')
plt.legend()
plt.show()


print("evaluating network")
model.load_weights("weights.best.hdf5")
Y_val_pred = model.predict(valid_data)
acc_score = accuracy_score(np.argmax(valid_target, axis=1), np.argmax(Y_val_pred, axis=1))
print("Accuracy score: ", acc_score)
Y_pred = model.predict(test_data)


# checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
# history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.1)

# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(train_data, train_target, batch_size=32, epochs=1, steps_per_epoch=len(train_data)//32, validation_split=0.1)
# model.fit(data_generator.flow(train_data, train_target, batch_size=32), steps_per_epoch=len(train_data) / 32,epochs=10)
# print(model.summary())
# test_loss, test_accuracy = model.evaluate(test_data, test_target)
# print("test loss", test_loss)
# print("accuracy", test_accuracy)
