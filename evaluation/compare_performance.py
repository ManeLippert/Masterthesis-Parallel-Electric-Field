import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_perform(data):
    content = []

    with open(data, 'r') as file:
        for row in file.readlines():
            content.append([r.replace('\n', '') for r in row.split('  ') if r != ''])
        
        data = {
            'name'    : [n[0]        for n in content],
            'calls'   : [int(c[1])   for c in content],
            'time'    : [float(t[2]) for t in content],
            'percent' : [float(p[3]) for p in content]
        }
    
    return data

def read_perfloop(data):
    time = []

    with open(data, 'r') as file:
        for row in file.readlines():
            time.append(float([r.replace('\n', '') for r in row.split('  ') if r != ''][1]))
    
    time = time[:-1]
    
    return np.array(time)

# time_f = []
# time_g = []

for b in range(4,10,2):

    time_f = read_perfloop(f"../data/performance/f-version/linear/kthrho0.300/beta0.00{b}/perfloop.dat")
    time_g = read_perfloop(f"../data/performance/g-version/linear/kthrho0.300/beta0.00{b}/perfloop.dat")
    
    print(f'beta = 0.{b}%')
    
    print(time_f.mean(), time_g.mean())
    print(time_f.mean()/time_g.mean())