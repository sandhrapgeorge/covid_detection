import pandas as pd
import os
import shutil
import random

# --setting path of source dataset1 (covid-19 positive)--
data_path1 = "source_datasets/covidgit/images"
# --reading metedata excel file inorder to extract covid positive data from source dataset--
dataset = pd.read_csv(os.path.join("source_datasets/covidgit", "metadata.csv"))
# --setting destination directory for covid-19 positive data--
target_dir = "sourcedata/Covid19 Positive"

# --copying covid-19 positive images from source directory to destination directory--
cnt = 0
for (i, raw) in dataset.iterrows():
    if raw['finding'] == 'Pneumonia/Viral/COVID-19' or raw['finding'] == 'Pneumonia/Viral/SARS':
        file_name = raw['filename']
        try:
            image_path = os.path.join(data_path1, file_name)
            image_copy_path = os.path.join(target_dir, file_name)
            shutil.copy2(image_path, image_copy_path)
            cnt += 1
        except Exception as e:
            i+=1

# --printing total number of covid-19 positive images--
print("no. of positive images", cnt)


# --reading metedata excel file inorder to extract covid negative data from source dataset--
file_path = "Chest_xray_Corona_Metadata.csv"
dataset = pd.read_csv(file_path)
# --setting  path of source dataset2 (covid19 negative)--
images_path1 = "source_datasets/kaggle_coronahack/train"
images_path2 = "source_datasets/kaggle_coronahack/test"
# --setting destination directory for covid-19 negative data--
target_dir = "sourcedata/Covid19 Negative"
# --copying covid-19 negative images from source directory to destination directory--
cnt = 0
i=0
for (i, raw) in dataset.iterrows():
    if raw['Label'] == "Normal":
        file_name = raw["X_ray_image_name"]
        try:
            if raw["Dataset_type"] == "TRAIN":
                image_path = os.path.join(images_path1, file_name)
                image_copy_path = os.path.join(target_dir, file_name)
                shutil.copy2(image_path, image_copy_path)
            elif raw["Dataset_type"] == "TEST":
                image_path = os.path.join(images_path2, file_name)
                image_copy_path = os.path.join(target_dir, file_name)
                shutil.copy2(image_path, image_copy_path)
            cnt += 1
        except Exception as e:
            i+=1

# --printing total number of covid-19 negative images--
print("no.of negative images", cnt)

# --splitting covid positive images to 3 categories - train, valid, test--
source_dir = "sourcedata/Covid19 Positive"
destination_dir1 = "dataset/train/Covid19 Positive"
destination_dir2 = "dataset/valid/Covid19 Positive"
destination_dir3 = "dataset/test/Covid19 Positive"

image_names1 = os.listdir(source_dir)
random.shuffle(image_names1)
for i in range(579):
    image_name = image_names1[i]
    image_path = os.path.join(source_dir, image_name)
    if i < 510:
        target_path = os.path.join(destination_dir1, image_name)
        shutil.copy2(image_path, target_path)
    elif i < 567:
        target_path = os.path.join(destination_dir2, image_name)
        shutil.copy2(image_path, target_path)
    else:
        target_path = os.path.join(destination_dir3, image_name)
        shutil.copy2(image_path, target_path)


# --splitting covid negative images to 3 categories - train, valid, test--
source_dir = "sourcedata/Covid19 Negative"
destination_dir1 = "dataset/train/Covid19 Negative"
destination_dir2 = "dataset/valid/Covid19 Negative"
destination_dir3 = "dataset/test/Covid19 Negative"

image_names1 = os.listdir(source_dir)
random.shuffle(image_names1)
for i in range(1576):
    image_name = image_names1[i]
    image_path = os.path.join(source_dir, image_name)
    if i < 1410:
        target_path = os.path.join(destination_dir1, image_name)
        shutil.copy2(image_path, target_path)
    elif i < 1567:
        target_path = os.path.join(destination_dir2, image_name)
        shutil.copy2(image_path, target_path)
    else:
        target_path = os.path.join(destination_dir3, image_name)
        shutil.copy2(image_path, target_path)