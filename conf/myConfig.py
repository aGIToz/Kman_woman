# intialize the batch size and number of epochs for training
batch_size = 32
epochs = 40

# test-train split ratio
split = 0.25

# path to save the features
path_feature_train = 'output/features_train.npy'
path_feature_test = 'output/features_test.npy'

# class weights, this is because there is class
# imbalance between man and woman. For 1 image of man
# there are 1.66 image of woman
classWeights = [1.66, 1]

# Dropout param, decreasing this will increase the 
# overfitting(gap between val_acc and acc), althouh val_acc increases
param_dropout = 0.9
