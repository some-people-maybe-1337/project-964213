
python3 deeponet_pyBlaz.py --augment half --set train --subset_train both --compressor pyBlaz --blocksize 4 --index_type 8 > case5_blocksize_4_index_8.txt
python3 deeponet_pyBlaz.py --augment half --set train --subset_train both --compressor pyBlaz --blocksize 4 --index_type 16 > case5_blocksize_4_index_16.txt
python3 deeponet_pyBlaz.py --augment half --set train --subset_train both --compressor pyBlaz --blocksize 8 --index_type 8 > case5_blocksize_8_index_8.txt
python3 deeponet_pyBlaz.py --augment half --set train --subset_train both --compressor pyBlaz --blocksize 8 --index_type 16 > case5_blocksize_8_index_16.txt

