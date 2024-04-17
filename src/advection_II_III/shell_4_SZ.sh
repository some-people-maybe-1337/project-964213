python3 deeponet_SZ.py --augment full --set train --subset_train both --compressor sz --multiplier 0.1 > case4_0_1.txt
python3 deeponet_SZ.py --augment full --set train --subset_train both --compressor sz --multiplier 0.01 > case4_0_01.txt
python3 deeponet_SZ.py --augment full --set train --subset_train both --compressor sz --multiplier 0.001 > case4_0_001.txt
python3 deeponet_SZ.py --augment full --set train --subset_train both --compressor sz --multiplier 0.0001 > case4_0_0001.txt
