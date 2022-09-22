__author__ = 'michaelrosenthal'

import drag_functions_turb as fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from time_TvsR import *
from matplotlib.colors import LogNorm

m_range = [-20,-2]
r_range = [-4,8]
m_arr = np.logspace(m_range[0],m_range[1],num=1000)
r_arr = np.logspace(r_range[0],r_range[1],num=300)
time_arr = np.zeros((len(m_arr),len(r_arr)))
master_arr = []

# time_arr = np.zeros()

m_star = 1
semi_major = 30.
alph = 1e-4

param_arr = [m_range[0],m_range[1],r_range[0],r_range[1],m_star,semi_major,alph]
#Print output from TvsR
verbose = 0

for i,M in enumerate(m_arr):
    for j,rad in enumerate(r_arr):
        t,tmp = TvsR_sng(alph=alph,a_au=semi_major,m_suns=m_star,m_earths=M,verbose=0,r=rad)
        time_arr[i,j]=t
#
# time_arr[0,len(r_arr)] = m_range[0]
# time_arr[1,len(r_arr)] = m_range[1]

param_str = 'a_%.1f_M_*_%.1f_alph_%.5f_long' % (semi_major,m_star,alph)
# param_str = r'$a = %.1f$ AU, $M_* = %.1fM_{\odot}$, $\alpha = %.5f$'\
# % (semi_major,m_star,alph)

master_arr.append(np.asarray(m_range))
# master_arr.append(r_arr)
master_arr.append(time_arr)

np.savetxt('Data/' + param_str + '_grid_3.txt',time_arr)
f = open('Data/' + param_str + '_grid_3_params.txt','w')
for num in param_arr:
    f.write(str(num) +'\n')
f.close()