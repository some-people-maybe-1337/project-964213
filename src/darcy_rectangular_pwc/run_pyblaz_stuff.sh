# with resolution 1 you have to run at most 3 at a time.

# half
# bs4 int16
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset train --trials 2 --half

# bs8 int16
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset train --trials 3 --half

# bs4 int8
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset train --trials 2 --half

# bs8 int8
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset train --trials 2 --half

# # replace
# # bs4 int16
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int16 --subset train --trials 2

# bs8 int16
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int16 --subset train --trials 3

bs4 int8
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 4 --float_type float32 --index_dtype int8 --subset train --trials 2

# bs8 int8
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --pyblaz --block_size 8 --float_type float32 --index_dtype int8 --subset train --trials 2