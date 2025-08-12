'''
file = open('ABIDEII_Composite_Phenotypic.csv', "r")
sites = {}
for line in file.readlines():
    id = line[line.index('-')+1:line.index(',')]
    if id in sites:
        sites[id] += 1
    else:
        sites[id] = 1
print(sites)
'''

import pandas as pd
df = pd.read_csv('Phenotypic_V1_0b.csv')
print(df['SITE_ID'].value_counts())

'''
file = open('ids2', 'r')
sites = {}
for line in file.readlines():
    if line in sites:
        sites[line] += 1
    else:
        sites[line] = 1
print(sites)
'''
