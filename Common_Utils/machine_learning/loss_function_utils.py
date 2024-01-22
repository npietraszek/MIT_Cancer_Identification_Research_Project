import math


class LossHistory():
     def on_train_begin(self, logs={}):
        self.losses = []

     def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))




'''
Creates a step decay function for reducing the learning rate of the machine learning function
alongside a tuple describing its parameters.
Parameters
----------
initial_lrate: float
    The initial learning rate of the machine learning model
drop: float
    The amount to drop the learning rate by
epochs_drop: float
    The rate at which the learning rate will accelerate in decay
Returns
-------
tuple
    A tuple containing the parameters of the step decay function
function
    The step decay function
'''
def step_decay_creator(initial_lrate=0.01, drop=0.5, epochs_drop=5.0):
    def step_decay(epoch):
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    return (initial_lrate, drop, epochs_drop), step_decay


'''
Slight modification to step_decay that waits a few epochs before starting to decay.
Creates a step decay function for reducing the learning rate of the machine learning function
alongside a tuple describing its parameters.
Parameters
----------
initial_lrate: float
    The initial learning rate of the machine learning model
drop: float
    The amount to drop the learning rate by
epochs_drop: float
    The rate at which the learning rate will accelerate in decay
Returns
-------
tuple
    A tuple containing the parameters of the step decay function
function
    The step decay function

'''
def step_decay_modify_creator(initial_lrate=0.01, drop=0.5, epochs_drop=5.0) :
    def step_decay_modify(epoch):
        if epoch < 10:
            return initial_lrate
        else:
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
    return (initial_lrate, drop, epochs_drop), step_decay_modify

class LossHistory():
     def on_train_begin(self, logs={}):
        self.losses = []

     def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# Parameters used for machine learning runs in the past...
'''
def old_step_decay_modify_6(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def old_step_decay_modify_5(epoch):
    initial_lrate = 0.001
    drop = 0.33
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def old_step_decay_modify_4(epoch):
    initial_lrate = 0.00001
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def old_step_decay_modify_3(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def old_step_decay_modify_2(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay_modify(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    if epoch < 10:
        return initial_lrate
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

def old_step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
'''