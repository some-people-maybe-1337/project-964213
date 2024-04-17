#download the data
#produce results in the appropraite folder using the scripts
git clone https://github.com/szcompressor/SZ3.git
cd SZ3
mkdir build
cd build
cmake ..
make -j
cd tools/sz3
cp ../../../../eb0_1.sh .
cp ../../../../eb0_01.sh .
cp ../../../../eb0_001.sh .
cp ../../../../eb0_0001.sh .
chmod +x eb*.sh
./eb0_1.sh
./eb0_01.sh
./eb0_001.sh
./eb0_0001.sh


#plotting the graph