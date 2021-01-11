from sklearn.metrics import accuracy_score
import gc
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model


data = np.load('train_dataset.npy')
target = np.load('train_targetset.npy')
train_data, valid_data, train_target, valid_target = train_test_split(data, target, test_size=0.1)
data_generator = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

test_data = np.load('valid_dataset.npy')
test_target = np.load('valid_targetset.npy')

BATCH_SIZE = 32
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("model.h5")
#print("Loaded model from disk")

# load model
model = load_model('model.h5')
#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#   model = load_model('model.h5')
# summarize model.
model.summary()

print("evaluating network")
model = load_model("weights.best.hdf5")
Y_val_pred = model.predict(valid_data)
acc_score = accuracy_score(np.argmax(valid_target, axis=1), np.argmax(Y_val_pred, axis=1))
print("Accuracy score: ", acc_score)
Y_pred = model.predict(test_data)


def plot_confusion_matrix(cm1, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap1=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm1)

    plt.imshow(cm1, interpolation='nearest', cmap=cmap1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, format(cm1[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


cm = confusion_matrix(np.argmax(test_target, axis=1), np.argmax(Y_pred, axis=1))

cm_plot_label = ['normal', 'covid-19']
plot_confusion_matrix(cm, cm_plot_label, title='Confusion Metrix for Covid-19')

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict(data_generator.flow(test_data, batch_size=BATCH_SIZE, shuffle=False),
                          steps=len(test_data) / BATCH_SIZE)

    predictions.append(preds)
    gc.collect()


Y_pred_tta = np.mean(predictions, axis=0)
#cm = confusion_matrix(np.argmax(test_target, axis=1), np.argmax(Y_pred_tta, axis=1))
#cm_plot_label = ['normal', 'covid-19']
#plot_confusion_matrix(cm, cm_plot_label, title='Confusion Metrix for Covid-19')


cl = classification_report(np.argmax(test_target, axis=1), np.argmax(Y_pred_tta, axis=1))
print("classification report:")
print(cl)

roc_log = roc_auc_score(np.argmax(test_target, axis=1), np.argmax(Y_pred_tta, axis=1))
false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(test_target, axis=1), np.argmax(Y_pred_tta, axis=1))
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'r--', label="Base line")
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()
plt.close()




