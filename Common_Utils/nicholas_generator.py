import numpy as np
from keras.utils import Sequence
# Generator in testing. STILL NEEDS VERIFICATION
class nicholas_generator(Sequence):
    def __init__(self, matrix_filenames, label_filenames, batch_size):
        self.matrix_filenames = matrix_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.matrix_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        matrix_batch = self.matrix_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        label_batch = self.label_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        # return np.array([np.resize(np.squeeze(np.load(file_name)[0]),(1,64,64,64)) for file_name in batch]), np.array([np.resize(np.squeeze(np.load(file_name)[1]),(1,64,64,64)) for file_name in batch])
        # return np.array([np.reshape(np.load(file_name)[0,0],(64,64,64,2)) for file_name in batch]), np.array([np.reshape(np.load(file_name)[0,1],(3,64,64,64)) for file_name in batch])
        return np.array([np.load(file_name) for file_name in matrix_batch]), np.array(
            [np.load(file_name) for file_name in label_batch])
