# intialize the batch size and number of epochs for training
batch_size = 32
epochs = 40

# test-train split ratio
split = 0.25

# path to save the features
path_feature_train = 'output/features_train.npy'
path_feature_test = 'output/features_test.npy'

# class weights
classWeights = [1.66, 1]

# Dropout param
param_dropout = 0.9
