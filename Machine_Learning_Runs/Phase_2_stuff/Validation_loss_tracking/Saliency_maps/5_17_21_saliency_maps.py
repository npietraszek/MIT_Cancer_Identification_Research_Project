'''
  Visualizing how layers represent classes with keras-vis saliency maps

  Features are not obvious but it seems clear that it's analyzing the perimeter of the cell
  and also how many z stacks are around.
'''

# =============================================
# Model to be visualized
# =============================================
from keras import activations
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import numpy as np
# import tensorflow as tf
from keras import optimizers
import vis.visualization

from MIT_Tumor_Identifcation_Project.Machine_learning_runs.Phase_2_stuff import nicholas_models_phase_2_new_testing as md

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

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


import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

shape_aux= (3,20,50,50)
model = md.new_20X_model_4(input_shape=shape_aux)
model.summary()


number_training_examples = 10728
number_validation_examples = 28
number_testing_examples = 28
nb_epochs = 40
batch_sz=16
loss_function='categorical_crossentropy'

adam=optimizers.Adam(clipvalue=1)
model.compile(loss=loss_function, optimizer=adam) #binary_cross_entropy, hinge
#model.load_weights(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_2_intermission_stuff\Machine_Learning\Weights\9_19_20_new_run_1_(model_4,50-90-harsh_bal(20-30-30),0.001_learning,rotate)\Superloop_run_0\9_19_20_Test1_balanced_0.001learning_DNN.30.hdf5")
#model.load_weights(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_Cell_Processing\Machine learning stuff\Weights\7_6_20_new_run_4_(model_3,dataset_4)\7_6_20_Test6_xy_DNN.18.hdf5")
model.load_weights(os.path.join(r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_2_stuff\Machine Learning\Weights\3_25_21_new_run_1_(new_model_4,no_standard-harsh_bal(20-50-50),tile_keep_1600_seperate)\Superloop_run_4",r"3_25_21_Test1_balanced_0.001learning_DNN.13.hdf5"))
# =============================================
# Activation Maximization code
# =============================================
from vis.utils import utils
import matplotlib.pyplot as plt

example_matrix = np.load(r"D:\MIT_Tumor_Identifcation_Project_Stuff\May_Cutting_Image_Types\step9 Rotated_ROI_1600_cells\device 1 chip 1 and 2 ROI5_02.oib\device 1 chip 1 and 2 ROI5_02.oib - Series 1-1 0000_cell1_0Fb1Tc_accuracy55.4942index7\Final_5D_array_2.npy")

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'activation_3')


# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)
model.summary()

# Numbers to visualize
numbers_to_visualize = [[0,1],[1,0]]

filter_index_to_visualize = 1

# Visualize

visualization = vis.visualization.visualize_saliency(model, layer_index,seed_input = example_matrix,keepdims=True, filter_indices=filter_index_to_visualize)
#visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize, input_range= (0., 1.))
for x in range(20):
    plt.figure(1)
    plt.imshow(example_matrix[0][x])
    plt.title(f'Target = DAPI image: Cell slice {x+1}')

    plt.figure(2)
    plt.imshow(visualization[0][x])
    if filter_index_to_visualize == 1:
        plt.title(f'Target = DAPI image: Cancer Map slice {x+1}')
    else:
        plt.title(f'Target = DAPI image: Fibroblast Map slice {x+1}')

    plt.show()

for x in range(20):
    plt.figure(1)
    plt.imshow(example_matrix[1][x])
    plt.title(f'Target = Reflective image: Cell slice {x+1}')

    plt.figure(2)
    plt.imshow(visualization[1][x])
    if filter_index_to_visualize == 1:
        plt.title(f'Target = Reflective image: Cancer Map slice {x+1}')
    else:
        plt.title(f'Target = Reflective image: Fibroblast Map slice {x+1}')
    plt.show()

for x in range(20):
    plt.figure(1)
    plt.imshow(example_matrix[2][x])
    plt.title(f'Target = Brightfield image: Cell slice {x+1}')

    plt.figure(2)
    plt.imshow(visualization[2][x])
    if filter_index_to_visualize == 1:
        plt.title(f'Target = Brightfield image: Cancer Map slice {x+1}')
    else:
        plt.title(f'Target = Brightfield image: Fibroblast Map slice {x+1}')
    plt.show()