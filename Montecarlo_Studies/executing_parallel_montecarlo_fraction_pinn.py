# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:55:55 2025

@author: MongoloPawa
"""

import multiprocessing
import numpy as np
import pandas as pd
from pinn_parallel_montecarlo_func import paralel_montecarlo_pinn_schrodinger
from pinn_parallel_montecarlo_func import paralel_montecarlo_pinn_pendulum
from pinn_parallel_montecarlo_func import paralel_montecarlo_pinn_fraction

def wrapper(args):
    
    j, max_epochs, n_neurons_1, n_neurons_2, seed = args
    
    np.random.seed(seed)
    
    print(j, flush=True)
    
    return paralel_montecarlo_pinn_schrodinger(j, max_epochs, n_neurons_1, n_neurons_2, seed)

if __name__ == "__main__":
    
    
    total_runs = 4*multiprocessing.cpu_count()
    
    num_workers = multiprocessing.cpu_count()//4
    
    max_epochs=2000
    n_neurons_1 = 3
    n_neurons_2 = 6
    seed=128
    
    args_list = [(_, max_epochs, n_neurons_1, n_neurons_2, seed) for _ in range(total_runs)]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        
        results = pool.map(wrapper, args_list)
        
    first_error_arr = []
    max_epoch_arr = []
    last_error_arr = []
    val_error_arr = []

    for i in range(total_runs): first_error_arr.append(results[i][2][0]), max_epoch_arr.append(results[i][2][1]), last_error_arr.append(results[i][2][2]), val_error_arr.append(results[i][2][3]),
    
    final_array = np.array([first_error_arr, max_epoch_arr, last_error_arr, val_error_arr])
    
    df = pd.DataFrame(final_array)
    
    filepath = "montecarlo_results_schr_8_3_6_v2.xlsx"
    
    df.to_excel(filepath)

