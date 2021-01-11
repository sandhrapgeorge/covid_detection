import cv2
import os
import numpy as np
from keras.utils import np_utils


# --setting data path--
data_path = 'dataset/train'
# -- finding the categories of data--
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
# --assigning different categories with numerical class labels--
label_dict = dict(zip(categories, labels))
print(label_dict)
print(categories)
print(labels)

# -- setting the target image size --
img_size = 256
data = []
target = []
for category in categories:
    # -- getting folder path of images --
    folder_path = os.path.join(data_path, category)
    # -- listing the image names in the folder path --
    img_names = os.listdir(folder_path)
    # -- reading each image in the form of array
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # -- resize  the images--
            resized = cv2.resize(gray, (img_size, img_size))
            # -- appending resized image into the previous array image list--
            data.append(resized)
            # -- appending the class label of image to previous images class label list--
            target.append(label_dict[category])
        except Exception as e:
            print("Exception : ", e)

# --converting image list into array and Normalizing it--
data = np.array(data)/255.0
# -- Reshaping the image data--
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
# --converting class label list into array--
target = np.array(target)
new_target = np_utils.to_categorical(target)
# --permentitly storing the image array and class label array--
np.save('train_dataset', data)
np.save('train_targetset', new_target)

data_path = 'dataset/valid'
data = []
target = []
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print("Exception : ", e)

data = np.array(data)/255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)
new_target = np_utils.to_categorical(target)
np.save('valid_dataset', data)
np.save('valid_targetset', new_target)



