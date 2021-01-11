from tensorflow.keras.models import load_model
from keras.backend import backend as k
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# path to image to visualize
img_path = '0a7faa2a.jpg'
img = cv2.imread(img_path)
print(img)
img_size = 256

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# -- resize  the images--
resized = cv2.resize(gray, (img_size, img_size))
resized = np.array(resized) / 255.0
resized = np.reshape(resized, (1, img_size, img_size, 1))
# predict on the image
preds1 = model.predict_classes(resized)[0]
print(preds1)

preds = model.predict(resized)[0]
print(preds)

# begin visualization
covid_output = model.output[:, 0]
# Output feature map from the deepest convolutional layer
last_conv_layer = model.get_layer('conv2d_7')
# compute the Gradient of the expected class with regard to the output feature map of block5_conv3 or the deepst
# convolutional layer)
grads = k.gradients(covid_output, last_conv_layer.output)[0]
pooled_grads = k.mean(grads, axis=(0, 1, 2))
iterate = k.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([resized])
for i in range(256):  # number of filters in the deepest convolutional layer
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# For visualization purposes, we normalize the heatmap between 0 and 1.
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
img = cv2.imread(img_path)  # path to the image
print(img.shape)
# Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# Converts the heatmap to RGB
heatmap = np.uint8(255 * heatmap)
# Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img  # 0.4 here is a heatmap intensity factor.
# Saves the image to disk
print(superimposed_img)
cv2.imwrite('cam_image.png', superimposed_img)

