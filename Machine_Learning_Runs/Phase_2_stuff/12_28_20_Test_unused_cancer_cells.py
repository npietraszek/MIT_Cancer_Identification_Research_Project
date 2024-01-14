import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import numpy as np
# import tensorflow as tf
import math
from keras import optimizers

from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md

# Code imported from internet
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory():
     def on_train_begin(self, logs={}):
        self.losses = []

     def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

shape_aux= (3,20,50,50)
model = md.list_model_2(input_shape=shape_aux)
model.summary()

nb_epochs = 100
batch_sz=16
loss_function='categorical_crossentropy'


adam=optimizers.Adam(clipvalue=1)
model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge

'''
int_eph=0
csv_logger = CSVLogger(filename)
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [lrate, checkpoint,csv_logger]
'''
config.gpu_options.allow_growth = True

final_testing_data = []
final_testing_labels = []
starting_directory = r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\New_20X_images\Macro V5 images\20X (no standard) Images Macro V5\Unused_Cancer_Cells"


for root, dirs, files in os.walk(starting_directory):
    for dir in dirs:
        final_testing_data.append(os.path.join(starting_directory, dir, "Final_5D_array.npy"))
        final_testing_labels.append(os.path.join(starting_directory, dir, "Label_matrix.npy"))



number_testing_examples = 2090

model.load_weights(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\12_25_20_new_run_1_(list_model_2,no-standard-harsh_bal(20-50-50),0.001_learning,rotate)\Superloop_run_0", r"12_25_20_Test1_balanced_0.001learning_DNN.26.hdf5"))
correct_counter = 0
wrong_cancer_cells = 0
wrong_fibroblasts = 0
cells_that_were_wrong_list = []
cells_that_were_wrong_fibroblasts = []
cells_that_were_wrong_cancer_cells = []
# Testing examples * 4 due to the rotated matrices...
for i in range(number_testing_examples):
    is_correct = False
    current_example = np.expand_dims((np.array(np.load(final_testing_data[i]))),axis = 0)
    current_prediction = model.predict(current_example)
    testing_answer = np.array(np.load(final_testing_labels[i]))
    current_prediction_index = 0
    testing_answer_index = 0
    for x in range(2):
        if current_prediction[0][x] == max(current_prediction[0]):
            current_prediction_index = x
        if testing_answer[x] == max(testing_answer):
            testing_answer_index = x
    if current_prediction_index == testing_answer_index:
        correct_counter += 1
        is_correct = True
    else:
        cells_that_were_wrong_list.append(final_testing_data[i])
        if testing_answer[0] == 0:
            # It's a cancer cell!
            wrong_cancer_cells = wrong_cancer_cells + 1
        else:
            # It's a fibroblast!
            wrong_fibroblasts = wrong_fibroblasts + 1


    print(model.predict(current_example), testing_answer, is_correct, correct_counter, final_testing_data[i])
#test_acc = model.evaluate_generator(generator = test_gen, verbose=1)
#print('\nTest loss:', test_acc)

model.summary()
print("The maximum accuracy was " + str(correct_counter))
#      + " with testing loss " + str(test_acc))
print("Total number of wrong cancer cells = " + str(wrong_cancer_cells))
print("Total number of wrong fibroblasts = " + str(wrong_fibroblasts))
#print("List of wrong cells: " + str(cells_that_were_wrong_list))
#print("Total number of epochs = "+ str(nb_epochs))

#print("Model running is +" + )

# print("Model running is +" + )
