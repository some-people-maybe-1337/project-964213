# stds:
# x_train[0].std()=0.45000416
# x_train[1].std()=0.28936166
# y_train.std()=0.3794828
# x_test[0].std()=0.44997844
# x_test[1].std()=0.28936166
# y_test.std()=0.3792028

# shapes:
# x_train[0].shape=(1024, 421, 421)
# x_train[1].shape=(421, 421, 2)
# y_train.shape=(1024, 421, 421)
# x_test[0].shape=(256, 421, 421)
# x_test[1].shape=(421, 421, 2)
# y_test.shape=(256, 421, 421)

# 0.1
sz3 -i unflattened_data/x_train_branch_unflattened.bin -z sz_compressed/x_train_branch_0.1.out -f -M ABS 0.045000416 -3 421 421 1024
sz3 -i unflattened_data/x_train_trunk_unflattened.bin -z sz_compressed/x_train_trunk_0.1.out -f -M ABS 0.028936166 -3 2 421 421
sz3 -i unflattened_data/y_train_unflattened.bin -z sz_compressed/y_train_0.1.out -f -M ABS 0.03794828 -3 421 421 1024
sz3 -i unflattened_data/x_test_branch_unflattened.bin -z sz_compressed/x_test_branch_0.1.out -f -M ABS 0.044997844 -3 421 421 256
sz3 -i unflattened_data/x_test_trunk_unflattened.bin -z sz_compressed/x_test_trunk_0.1.out -f -M ABS 0.028936166 -3 2 421 421
sz3 -i unflattened_data/y_test_unflattened.bin -z sz_compressed/y_test_0.1.out -f -M ABS 0.03792028 -3 421 421 256

# 0.01
sz3 -i unflattened_data/x_train_branch_unflattened.bin -z sz_compressed/x_train_branch_0.01.out -f -M ABS 0.0045000416 -3 421 421 1024
sz3 -i unflattened_data/x_train_trunk_unflattened.bin -z sz_compressed/x_train_trunk_0.01.out -f -M ABS 0.0028936166 -3 2 421 421
sz3 -i unflattened_data/y_train_unflattened.bin -z sz_compressed/y_train_0.01.out -f -M ABS 0.003794828 -3 421 421 1024
sz3 -i unflattened_data/x_test_branch_unflattened.bin -z sz_compressed/x_test_branch_0.01.out -f -M ABS 0.0044997844 -3 421 421 256
sz3 -i unflattened_data/x_test_trunk_unflattened.bin -z sz_compressed/x_test_trunk_0.01.out -f -M ABS 0.0028936166 -3 2 421 421
sz3 -i unflattened_data/y_test_unflattened.bin -z sz_compressed/y_test_0.01.out -f -M ABS 0.003792028 -3 421 421 256

# 0.001
sz3 -i unflattened_data/x_train_branch_unflattened.bin -z sz_compressed/x_train_branch_0.001.out -f -M ABS 0.00045000416 -3 421 421 1024
sz3 -i unflattened_data/x_train_trunk_unflattened.bin -z sz_compressed/x_train_trunk_0.001.out -f -M ABS 0.00028936166 -3 2 421 421
sz3 -i unflattened_data/y_train_unflattened.bin -z sz_compressed/y_train_0.001.out -f -M ABS 0.0003794828 -3 421 421 1024
sz3 -i unflattened_data/x_test_branch_unflattened.bin -z sz_compressed/x_test_branch_0.001.out -f -M ABS 0.00044997844 -3 421 421 256
sz3 -i unflattened_data/x_test_trunk_unflattened.bin -z sz_compressed/x_test_trunk_0.001.out -f -M ABS 0.00028936166 -3 2 421 421
sz3 -i unflattened_data/y_test_unflattened.bin -z sz_compressed/y_test_0.001.out -f -M ABS 0.0003792028 -3 421 421 256

# 0.0001
sz3 -i unflattened_data/x_train_branch_unflattened.bin -z sz_compressed/x_train_branch_0.0001.out -f -M ABS 0.000045000416 -3 421 421 1024
sz3 -i unflattened_data/x_train_trunk_unflattened.bin -z sz_compressed/x_train_trunk_0.0001.out -f -M ABS 0.000028936166 -3 2 421 421
sz3 -i unflattened_data/y_train_unflattened.bin -z sz_compressed/y_train_0.0001.out -f -M ABS 0.00003794828 -3 421 421 1024
sz3 -i unflattened_data/x_test_branch_unflattened.bin -z sz_compressed/x_test_branch_0.0001.out -f -M ABS 0.000044997844 -3 421 421 256
sz3 -i unflattened_data/x_test_trunk_unflattened.bin -z sz_compressed/x_test_trunk_0.0001.out -f -M ABS 0.000028936166 -3 2 421 421
sz3 -i unflattened_data/y_test_unflattened.bin -z sz_compressed/y_test_0.0001.out -f -M ABS 0.00003792028 -3 421 421 256

# compression ratio = 146.87 
# compression time = 4.382025
# compressed data file = sz_compressed/x_train_branch_0.1.out
# compression ratio = 97.84 
# compression time = 0.014665
# compressed data file = sz_compressed/x_train_trunk_0.1.out
# compression ratio = 98.61 
# compression time = 4.629392
# compressed data file = sz_compressed/y_train_0.1.out
# compression ratio = 148.01 
# compression time = 1.108443
# compressed data file = sz_compressed/x_test_branch_0.1.out
# compression ratio = 97.84 
# compression time = 0.014488
# compressed data file = sz_compressed/x_test_trunk_0.1.out
# compression ratio = 98.63 
# compression time = 1.172287
# compressed data file = sz_compressed/y_test_0.1.out
# compression ratio = 146.87 
# compression time = 4.395685
# compressed data file = sz_compressed/x_train_branch_0.01.out
# compression ratio = 30.65 
# compression time = 0.015609
# compressed data file = sz_compressed/x_train_trunk_0.01.out
# compression ratio = 29.07 
# compression time = 4.971724
# compressed data file = sz_compressed/y_train_0.01.out
# compression ratio = 148.01 
# compression time = 1.137419
# compressed data file = sz_compressed/x_test_branch_0.01.out
# compression ratio = 30.65 
# compression time = 0.015859
# compressed data file = sz_compressed/x_test_trunk_0.01.out
# compression ratio = 28.79 
# compression time = 1.256845
# compressed data file = sz_compressed/y_test_0.01.out
# compression ratio = 146.87 
# compression time = 4.531368
# compressed data file = sz_compressed/x_train_branch_0.001.out
# compression ratio = 23.43 
# compression time = 0.012707
# compressed data file = sz_compressed/x_train_trunk_0.001.out
# compression ratio = 16.57 
# compression time = 5.461247
# compressed data file = sz_compressed/y_train_0.001.out
# compression ratio = 148.01 
# compression time = 1.174230
# compressed data file = sz_compressed/x_test_branch_0.001.out
# compression ratio = 23.43 
# compression time = 0.016680
# compressed data file = sz_compressed/x_test_trunk_0.001.out
# compression ratio = 17.14 
# compression time = 1.641236
# compressed data file = sz_compressed/y_test_0.001.out
# compression ratio = 146.87 
# compression time = 4.406871
# compressed data file = sz_compressed/x_train_branch_0.0001.out
# compression ratio = 23.45 
# compression time = 0.017139
# compressed data file = sz_compressed/x_train_trunk_0.0001.out
# compression ratio = 11.49 
# compression time = 5.928379
# compressed data file = sz_compressed/y_train_0.0001.out
# compression ratio = 148.01 
# compression time = 1.130226
# compressed data file = sz_compressed/x_test_branch_0.0001.out
# compression ratio = 23.45 
# compression time = 0.017684
# compressed data file = sz_compressed/x_test_trunk_0.0001.out
# compression ratio = 11.37 
# compression time = 1.679863
# compressed data file = sz_compressed/y_test_0.0001.out