#!/usr/bin/env python3

# %%
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(1, './')
from scg_simulate import scg_simulate
import numpy as np
import random
from scipy import signal
from tqdm import tqdm

if __name__ == '__main__':

    if(len(sys.argv) >= 6):
        N = int(sys.argv[1])
        noise = float(sys.argv[2])
        S_min = int(sys.argv[3])
        S_max = int(sys.argv[4])
        random_seed = int(sys.argv[5])
        data_file = sys.argv[6]

    else:
        print(f"Usage: {sys.argv[0]} num_rows noise_level S_min S_max random_seed path_file \n where noise level (amplitude of the laplace noise).")
        print(f"Example: {sys.argv[0]} 100 0.5 90 180 43 ../data/simu.1000_6.npy")       
        exit()

    fs = 100
    duration = 10 # 10 seconds
    simulated_data = []

    np.random.seed(random_seed)
    random.seed(random_seed)

    heart_rates = [random.randint(50, 150) for _ in range(N)]
    respiratory_rates = [random.randint(10, 30) for _ in range(N)]
    systolics = [random.randint(S_min, S_max) for _ in range(N)]
    diastolics = [random.randint(60,100) for _ in range(N)] #+ systolic

    for ind in tqdm(range(N)):
        heart_rate = heart_rates[ind]
        respiratory_rate = respiratory_rates[ind]
        systolic = systolics[ind]
        diastolic = diastolics[ind]

        print('hr:', heart_rate, 'rr:', respiratory_rate, 'sp:', systolic, 'dp:', diastolic)
        
        data = scg_simulate(duration=duration, sampling_rate=fs, noise=noise, heart_rate=heart_rate, respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic, random_state=random_seed)
        ## N + 6 size. 6 are [mat_int(here 0 for synthetic data), time_stamp, hr, rr, sbp, dbp]
        simulated_data.append(list(data)+[0]+[ind]+[heart_rate]+[respiratory_rate]+[systolic]+[diastolic])

    simulated_data = np.asarray(simulated_data)
    np.save(data_file,simulated_data)
    print(f'{data_file} is generated and saved!')
    # import pdb; pdb.set_trace()
