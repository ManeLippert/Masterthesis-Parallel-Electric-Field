import pandas as pd

data_perform_f = "../data/benchmark/f-version/linear/kthrho0.300/beta0.008/perform.dat"
data_perform_g = "../data/benchmark/g-version/linear/kthrho0.300/beta0.008/perform.dat"

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

df_f = pd.DataFrame(read_performance(data_perform_f))
df_g = pd.DataFrame(read_performance(data_perform_g))

time_f = df_f['time'].sum()
time_g = df_g['time'].sum()

print('perform.dat')
print(f'{(time_f)/(time_g):.2f} %')
# print('perform.dat + perflib / other')
# print(f'{(time_f - 926.62)/(time_g - 638.24):.2f} %')
# print('output.dat (main loop time)')
# print(f'{909.755150053184/730.764887287747:.2f} %')
# print('output.dat (run time)')
# print(f'{910.650430012029/731.632210198324:.2f} %')