# shapes:
# x_train[0].shape=(1024, 421, 421)
# x_train[1].shape=(421, 421, 2)
# y_train.shape=(1024, 421, 421)
# x_test[0].shape=(256, 421, 421)
# x_test[1].shape=(421, 421, 2)
# y_test.shape=(256, 421, 421)

sz3 -z sz_compressed/x_train_branch_0.1.out -o sz_decompressed/x_train_branch_0.1.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_train_trunk_0.1.out -o sz_decompressed/x_train_trunk_0.1.bin -f -3 2 421 421
sz3 -z sz_compressed/y_train_0.1.out -o sz_decompressed/y_train_0.1.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_test_branch_0.1.out -o sz_decompressed/x_test_branch_0.1.bin -f -3 421 421 256
sz3 -z sz_compressed/x_test_trunk_0.1.out -o sz_decompressed/x_test_trunk_0.1.bin -f -3 2 421 421
sz3 -z sz_compressed/y_test_0.1.out -o sz_decompressed/y_test_0.1.bin -f -3 421 421 256

sz3 -z sz_compressed/x_train_branch_0.01.out -o sz_decompressed/x_train_branch_0.01.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_train_trunk_0.01.out -o sz_decompressed/x_train_trunk_0.01.bin -f -3 2 421 421
sz3 -z sz_compressed/y_train_0.01.out -o sz_decompressed/y_train_0.01.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_test_branch_0.01.out -o sz_decompressed/x_test_branch_0.01.bin -f -3 421 421 256
sz3 -z sz_compressed/x_test_trunk_0.01.out -o sz_decompressed/x_test_trunk_0.01.bin -f -3 2 421 421
sz3 -z sz_compressed/y_test_0.01.out -o sz_decompressed/y_test_0.01.bin -f -3 421 421 256

sz3 -z sz_compressed/x_train_branch_0.001.out -o sz_decompressed/x_train_branch_0.001.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_train_trunk_0.001.out -o sz_decompressed/x_train_trunk_0.001.bin -f -3 2 421 421
sz3 -z sz_compressed/y_train_0.001.out -o sz_decompressed/y_train_0.001.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_test_branch_0.001.out -o sz_decompressed/x_test_branch_0.001.bin -f -3 421 421 256
sz3 -z sz_compressed/x_test_trunk_0.001.out -o sz_decompressed/x_test_trunk_0.001.bin -f -3 2 421 421
sz3 -z sz_compressed/y_test_0.001.out -o sz_decompressed/y_test_0.001.bin -f -3 421 421 256

sz3 -z sz_compressed/x_train_branch_0.0001.out -o sz_decompressed/x_train_branch_0.0001.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_train_trunk_0.0001.out -o sz_decompressed/x_train_trunk_0.0001.bin -f -3 2 421 421
sz3 -z sz_compressed/y_train_0.0001.out -o sz_decompressed/y_train_0.0001.bin -f -3 421 421 1024
sz3 -z sz_compressed/x_test_branch_0.0001.out -o sz_decompressed/x_test_branch_0.0001.bin -f -3 421 421 256
sz3 -z sz_compressed/x_test_trunk_0.0001.out -o sz_decompressed/x_test_trunk_0.0001.bin -f -3 2 421 421
sz3 -z sz_compressed/y_test_0.0001.out -o sz_decompressed/y_test_0.0001.bin -f -3 421 421 256

# compression ratio = 146.866819
# decompression time = 1.810072 seconds.
# decompressed file = sz_decompressed/x_train_branch_0.1.bin
# compression ratio = 97.835369
# decompression time = 0.010929 seconds.
# decompressed file = sz_decompressed/x_train_trunk_0.1.bin
# compression ratio = 98.614526
# decompression time = 1.826221 seconds.
# decompressed file = sz_decompressed/y_train_0.1.bin
# compression ratio = 148.010996
# decompression time = 0.441601 seconds.
# decompressed file = sz_decompressed/x_test_branch_0.1.bin
# compression ratio = 97.835369
# decompression time = 0.011408 seconds.
# decompressed file = sz_decompressed/x_test_trunk_0.1.bin
# compression ratio = 98.625069
# decompression time = 0.483494 seconds.
# decompressed file = sz_decompressed/y_test_0.1.bin
# compression ratio = 146.867473
# decompression time = 1.758977 seconds.
# decompressed file = sz_decompressed/x_train_branch_0.01.bin
# compression ratio = 30.650613
# decompression time = 0.012903 seconds.
# decompressed file = sz_decompressed/x_train_trunk_0.01.bin
# compression ratio = 29.067159
# decompression time = 1.999512 seconds.
# decompressed file = sz_decompressed/y_train_0.01.bin
# compression ratio = 148.009186
# decompression time = 0.441361 seconds.
# decompressed file = sz_decompressed/x_test_branch_0.01.bin
# compression ratio = 30.650613
# decompression time = 0.009645 seconds.
# decompressed file = sz_decompressed/x_test_trunk_0.01.bin
# compression ratio = 28.794079
# decompression time = 0.504948 seconds.
# decompressed file = sz_decompressed/y_test_0.01.bin
# compression ratio = 146.867265
# decompression time = 1.736681 seconds.
# decompressed file = sz_decompressed/x_train_branch_0.001.bin
# compression ratio = 23.428694
# decompression time = 0.014806 seconds.
# decompressed file = sz_decompressed/x_train_trunk_0.001.bin
# compression ratio = 16.574924
# decompression time = 2.559212 seconds.
# decompressed file = sz_decompressed/y_train_0.001.bin
# compression ratio = 148.010151
# decompression time = 0.445570 seconds.
# decompressed file = sz_decompressed/x_test_branch_0.001.bin
# compression ratio = 23.428694
# decompression time = 0.014804 seconds.
# decompressed file = sz_decompressed/x_test_trunk_0.001.bin
# compression ratio = 17.142637
# decompression time = 1.065887 seconds.
# decompressed file = sz_decompressed/y_test_0.001.bin
# compression ratio = 146.866492
# decompression time = 1.795165 seconds.
# decompressed file = sz_decompressed/x_train_branch_0.0001.bin
# compression ratio = 23.448066
# decompression time = 0.014800 seconds.
# decompressed file = sz_decompressed/x_train_trunk_0.0001.bin
# compression ratio = 11.488258
# decompression time = 3.795983 seconds.
# decompressed file = sz_decompressed/y_train_0.0001.bin
# compression ratio = 148.010151
# decompression time = 0.484435 seconds.
# decompressed file = sz_decompressed/x_test_branch_0.0001.bin
# compression ratio = 23.448066
# decompression time = 0.012130 seconds.
# decompressed file = sz_decompressed/x_test_trunk_0.0001.bin
# compression ratio = 11.370646
# decompression time = 0.920753 seconds.
# decompressed file = sz_decompressed/y_test_0.0001.bin