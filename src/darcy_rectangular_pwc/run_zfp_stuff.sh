# with resolution 1 you have to run at most 3 at a time.

# half
# 0.1

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset train --trials 2 --half

# 0.01
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset train --trials 2 --half

# 0.001
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset train --trials 2 --half

# 0.0001
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset train --trials 3 --half
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset train --trials 2 --half

# replace
# augment
# 0.1
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.1 --subset train --trials 2

# 0.01
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.01 --subset train --trials 2

# 0.001
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.001 --subset train --trials 2

# 0.0001
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset both --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset both --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset test --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset test --trials 2

DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset train --trials 3
DDE_BACKEND=tensorflow python deeponet.py --results_folder results --resolution 1 --sz --std_multiplier 0.0001 --subset train --trials 2
