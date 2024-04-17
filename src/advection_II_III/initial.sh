mkdir -p original_dat_128X128_size1000
cd ../../data/advection
python3 main.py
cp advection_ic2.npz advection_ic2_test_128X128_size1000.npz
rm advection_ic2.npz
python3 main.py
cp advection_ic2.npz advection_ic2_train_128X128_size1000.npz
rm advection_ic2.npz

mv advection_ic2_test_128X128_size1000.npz ../../src/advection_II_III/original_dat_128X128_size1000
mv advection_ic2_train_128X128_size1000.npz ../../src/advection_II_III/original_dat_128X128_size1000

cd ../../src/advection_II_III
python3 deeponet.py

cd original_dat_128X128_size1000
mkdir -p pyBlaz
mkdir -p SZ
mkdir -p ZFP

cd pyBlaz
mkdir -p blocksize_4_index_8/decompressed_dat
mkdir -p blocksize_4_index_16/decompressed_dat
mkdir -p blocksize_8_index_8/decompressed_dat
mkdir -p blocksize_8_index_16/decompressed_dat

cd ../SZ
mkdir -p eb0.1/decompressed_dat
mkdir -p eb0.01/decompressed_dat
mkdir -p eb0.001/decompressed_dat
mkdir -p eb0.0001/decompressed_dat

cd ../ZFP
mkdir -p eb0.1/decompressed_dat
mkdir -p eb0.01/decompressed_dat
mkdir -p eb0.001/decompressed_dat
mkdir -p eb0.0001/decompressed_dat

