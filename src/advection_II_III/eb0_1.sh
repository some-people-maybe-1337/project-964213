
./sz3 -f -i ../../../../original_dat_128X128_size1000/x_train_0_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/x_train_0_128X128_size1000.dat.sz3 -2 128 1000 -M ABS 0.1*0.64982635

./sz3 -f -i ../../../../original_dat_128X128_size1000/x_train_1_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/x_train_1_128X128_size1000.dat.sz3 -2 2 16384 -M ABS 0.1*0.29093927

./sz3 -f -i ../../../../original_dat_128X128_size1000/y_train_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/y_train_128X128_size1000.dat.sz3 -2  16384 1000 -M ABS 0.1*0.6505351

./sz3 -f -i ../../../../original_dat_128X128_size1000/x_test_0_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/x_test_0_128X128_size1000.dat.sz3 -2 1000 128 -M ABS 0.1*0.6573201

./sz3 -f -i ../../../../original_dat_128X128_size1000/x_test_1_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/x_test_1_128X128_size1000.dat.sz3 -2 2 16384 -M ABS 0.1*0.29093927

./sz3 -f -i ../../../../original_dat_128X128_size1000/y_test_128X128_size1000.dat -z ../../../../original_dat_128X128_size1000/y_test_128X128_size1000.dat.sz3 -2 1000 16384 -M ABS 0.1*0.6580172



./sz3 -f -z ../../../../original_dat_128X128_size1000/x_train_0_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/SZ/eb0.1/decompressed_dat/x_train_0_128X128_size1000.dat.sz3.out -2 128 1000 -M ABS 0.1*0.64982635

./sz3 -f -z ../../../../original_dat_128X128_size1000/x_train_1_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/SZ/eb0.1/decompressed_dat/x_train_1_128X128_size1000.dat.sz3.out -2 2 16384 -M ABS 0.1*0.29093927

./sz3 -f -z ../../../../original_dat_128X128_size1000/y_train_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/SZ/eb0.1/decompressed_dat/y_train_128X128_size1000.dat.sz3.out -2  16384 1000 -M ABS 0.1*0.6505351

./sz3 -f -z ../../../../original_dat_128X128_size1000/x_test_0_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/SZ/eb0.1/decompressed_dat/x_test_0_128X128_size1000.dat.sz3.out -2 1000 128 -M ABS 0.1*0.6573201

./sz3 -f -z ../../../../original_dat_128X128_size1000/x_test_1_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/SZ/eb0.1/decompressed_dat/x_test_1_128X128_size1000.dat.sz3.out -2 2 16384 -M ABS 0.1*0.29093927

./sz3 -f -z ../../../../original_dat_128X128_size1000/y_test_128X128_size1000.dat.sz3 -o ../../../../original_dat_128X128_size1000/eb0.1/decompressed_dat/y_test_128X128_size1000.dat.sz3.out -2 1000 16384 -M ABS 0.1*0.6580172
