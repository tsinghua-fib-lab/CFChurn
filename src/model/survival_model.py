# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:36:53 2018

@author: zgz
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test

dataset = np.load('sample12_dataset_norm.npy', allow_pickle=True)

discrete_x = np.array(dataset[0])
continous_x = np.array(dataset[1])




duration = [[] for _ in range(6)]
event = [[] for _ in range(6)]
with open('../data/quasi_survival_analysis', 'r') as file:
    for line in file:
        tem = eval(line)
        interval_index = tem[0]
        duration_list = tem[2]
        event_list = tem[3]
        for i in range(len(duration_list)):
            duration[interval_index] += [i]*duration_list[i]
            event[interval_index] += [1]*event_list[i] + [0]*(duration_list[i]-event_list[i])

################################ survival curve ################################
# 调整x，y轴的label距离坐标轴的距离
mpl.rcParams['xtick.major.pad'] = 10
# 调整字体为type 1 font（字体版权问题要求）
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 作图尺寸
plt.figure(figsize=(12,9))
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
# 调整坐标轴边框的粗细
plt.rcParams['axes.linewidth'] = 2

kmf = KaplanMeierFitter()
for i in range(6):
    if i!=5:
        kmf.fit(np.array(duration[i]), np.array(event[i]), label='{}-{}'.format(i*200, (i+1)*200))
    else:
        kmf.fit(np.array(duration[i]), np.array(event[i]), label='{}+'.format(1000))
    print(kmf.median_survival_time_)
    print(median_survival_times(kmf.confidence_interval_))
    kmf.plot(lw=3)

ax=plt.gca()
# ax.set_xlabel(r'\textbf{Spatial Error/Km}',fontsize=28)
# ax.set_ylabel(r'\textbf{CDF of Check-ins/\%}',fontsize=28)
ax.set_xlabel(r'(a) #Days',fontsize=40)
ax.set_ylabel(r'Propotion Still Inviting',fontsize=40)
ax.set_xlim(0,520)
ax.set_ylim(0.4,1)
ax.tick_params(labelsize=40)
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(name_list)
ax.legend(fontsize=24, loc='lower left')
#ax.semilogx()
plt.tight_layout()
plt.show()

################################ log-rank test ################################
T_exp, E_exp = duration[1], event[1]
T_con, E_con = duration[5], event[5]

results = logrank_test(T_exp, T_con, event_observed_A=E_exp, event_observed_B=E_con)
results.print_summary()

print(results.p_value)
print(results.test_statistic)


################################ 90th percentile survival time ################################
kmf = KaplanMeierFitter()
for i in range(6):
    kmf.fit(np.array(duration[i]), np.array(event[i]), label='{}+'.format(i*200))
    print(qth_survival_times(0.9, kmf.survival_function_))
    print(qth_survival_times(0.9, kmf.confidence_interval_))

################################ RMST ################################
time_limit = 500
model = {}
for i in range(6):
        kmf = KaplanMeierFitter()
        model[i] = kmf.fit(np.array(duration[i]), np.array(event[i]), label='{}+'.format(i*200))
        print(restricted_mean_survival_time(model[i], t=time_limit))

plt.figure()
rmst_plot(model[1], model2=model[5], t=time_limit)

plt.show()
