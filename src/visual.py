import matplotlib.pyplot as plt 
import pandas as pd 

df_d2a2 = pd.read_csv('D2A2.txt')
print(df_d2a2)
df_d2a2_root = df_d2a2[df_d2a2['device']=='root']

plt.