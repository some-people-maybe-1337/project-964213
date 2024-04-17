
python3 deeponet_ZFP.py --augment half --set train --subset_train both --compressor zfp --multiplier 0.1 > case5_0_1.txt
python3 deeponet_ZFP.py --augment half --set train --subset_train both --compressor zfp --multiplier 0.01 > case5_0_01.txt
python3 deeponet_ZFP.py --augment half --set train --subset_train both --compressor zfp --multiplier 0.001 > case5_0_001.txt
python3 deeponet_ZFP.py --augment half --set train --subset_train both --compressor zfp --multiplier 0.0001 > case5_0_0001.txt

