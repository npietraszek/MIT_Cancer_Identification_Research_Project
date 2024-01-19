import math

def step_decay_modify(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    if epoch < 10:
        return initial_lrate
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
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
