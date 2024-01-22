'''
3/19/21
Full machine learning run program. Runs a "superloop" of multiple machine learning epoch runs.
Each superloop iteration has a different randomization of training dataset and validation dataset and testing dataset,
'''

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
from keras import optimizers


import Common_Utils.machine_learning.models.nicholas_models_phase_2 as md
from Common_Utils.machine_learning.loss_function_utils import step_decay_creator
from Common_Utils.machine_learning.machine_learning_runners import super_train_loop


if __name__ == "__main__":
    weight_save_folder=r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\Extra_runs_with_best_model\1_1_22_Test29"
    weight_file_name = "1_1_22_Test29_DNN.{epoch:02d}.hdf5"
    text_results_file_name = "1_1_22_test29_set.txt"
    model_name_to_use = "new_20X_model_4"
    shape_aux = (3, 20, 50, 50)
    model = md.new_20X_model_4(input_shape=shape_aux)
    nb_epochs = 30
    batch_sz=16
    loss_function='categorical_crossentropy'
    directory_to_write_to = r"D:\MIT_Tumor_Identifcation_Project_Stuff\June_Finishing_Project\Runs\Extra_runs_with_best_model"
    epoch_name_part_1 = r"1_1_22_Test29_DNN."
    step_decay_parameters = (0.0001, 0.5, 5.0)
    super_train_loop(weight_save_folder=weight_save_folder, epoch_name_part_1=epoch_name_part_1,
                     weight_file_name=weight_file_name, model=model, model_name_to_use=model_name_to_use, text_results_file_name=text_results_file_name, directory_to_write_to=directory_to_write_to,
                     superloop_rounds=1, batch_sz=batch_sz, nb_epochs=nb_epochs, 
                     loss_function=loss_function, 
                     step_decay_model = step_decay_creator(initial_lrate=step_decay_parameters[0],drop=step_decay_parameters[1],epochs_drop=step_decay_parameters[2]),
                     optimizer = optimizers.Adam(clipvalue=1),
                     print_results = True)