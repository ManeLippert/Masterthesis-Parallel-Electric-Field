import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_perform_f = "../data/performance/f-version/linear/kthrho0.300/beta0.004/perform.dat"
data_perform_g = "../data/performance/g-version/linear/kthrho0.300/beta0.004/perform.dat"

def read_performance(data):
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

time_f = []
time_g = []

for b in range(4,10,2):

    df_f = pd.DataFrame(read_performance(f"../data/performance/f-version/linear/kthrho0.300/beta0.00{b}/perform.dat"))
    df_g = pd.DataFrame(read_performance(f"../data/performance/g-version/linear/kthrho0.300/beta0.00{b}/perform.dat"))

    time_f.append(df_f.iloc[1]['time'])#/df_f.iloc[1]['calls'])
    time_g.append(df_g.iloc[1]['time'])#/df_f.iloc[1]['calls'])

    print(f'beta = 0.{b}%')
    
    print(df_g.iloc[1]['time']/df_g.iloc[1]['calls'])
    print(df_f.iloc[1]['time']/df_f.iloc[1]['calls'])
    
    print(df_f.iloc[1]['time']/df_g.iloc[1]['time'])
    

time_f = np.array(time_f)
time_g = np.array(time_g)

print(time_f.mean()/time_g.mean())

# print('perform.dat')
# print(f'{(time_f)/(time_g) * 100:.2f} %')
# print('perform.dat + perflib / other')
# print(f'{(time_f - 0.19370E+05)/(time_g - 0.90772E+04):.2f} %')
# print('output.dat (main loop time)')
# print(f'{909.755150053184/730.764887287747:.2f} %')
# print('output.dat (run time)')
# print(f'{910.650430012029/731.632210198324:.2f} %')

print("\n")

timesetps = [25600, 31700, 44200]
time_f    = [7725, 16805, 19961]
time_g    = [6115, 10212, 10541]

plt.plot(timesetps, time_f, 'o')
plt.plot(timesetps, time_g, 'o')

plt.ylim(0,20000)
plt.xlim(0,50000)

plt.show()