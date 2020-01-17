'''
Hao
This code reproduces plots from output of STEEL (Grylls P. et al 2019).
Some functions are taken from previous scripts.
Follow the instruction in STEEL/ReadMe.md to install Functions.
'''

with open ('../default.conf') as conf_file:
    input_params = conf_file.readlines()
for i in range(0, len(input_params)):
    if 'path_to_STEEL' in input_params[i]:
        for j in range(0, len(input_params[i])):
            if input_params[i][len(input_params[i])-1-j] == "\'":
                if input_params[i][len(input_params[i])-2-j] == '/':
                    exec(input_params[i])
                    break
                else:
                    exec( input_params[i][0:len(input_params[i])-1-j] + "/\'")
                    break
        break

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(path_to_STEEL+'Functions')
import Read_data as Rd
