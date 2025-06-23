# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:55:55 2025

@author: MongoloPawa
"""

import multiprocessing
import numpy as np
import pandas as pd
from piqnn_parallel_montecarlo_func_v2 import paralel_montecarlo_piqnn_schrodinger
from piqnn_parallel_montecarlo_func_v2 import paralel_montecarlo_piqnn_pendulum
from piqnn_parallel_montecarlo_func_v2 import paralel_montecarlo_piqnn_fraction

def wrapper(args):
    
    j, max_epochs, n_qubits, seed = args
    
    np.random.seed(seed)
    
    print(j, flush=True)
    
    return paralel_montecarlo_piqnn_schrodinger(j, max_epochs, n_qubits, seed)

if __name__ == "__main__":
    
    
    total_runs = 4*multiprocessing.cpu_count()
    
    num_workers = multiprocessing.cpu_count()//4
    
    max_epochs=2000
    n_qubits=3
    seed=128
    
    args_list = [(_, max_epochs, n_qubits, seed) for _ in range(total_runs)]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        
        results = pool.map(wrapper, args_list)
        
    first_error_arr = []
    max_epoch_arr = []
    last_error_arr = []
    val_error_arr = []

    for i in range(total_runs): first_error_arr.append(results[i][2][0]), max_epoch_arr.append(results[i][2][1]), last_error_arr.append(results[i][2][2]), val_error_arr.append(results[i][2][3]),
    
    final_array = np.array([first_error_arr, max_epoch_arr, last_error_arr, val_error_arr])
    
    df = pd.DataFrame(final_array)
    
    filepath = "montecarlo_results_schr_3q.xlsx"
    
    df.to_excel(filepath)

