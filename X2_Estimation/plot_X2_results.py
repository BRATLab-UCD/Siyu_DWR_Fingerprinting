#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# please change root_path to the folder you save the pickle files on your own computer
root_path = '/Users/siyuqi/Downloads/drive-download-20230621T000337Z-001'




path_formatter = 'X2_i18_d8_d2_o1_first_chronological_%s_full_results.pkl'

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

eval_metrics = ['MSE', 'Bias', 'NSE'] # , 'R'

def load_results(filepath):
    # read pickle files
    with open(filepath, 'rb') as f:
        temp = pickle.load(f)
    return temp

def plot_box_and_whisker(results_dict, results_labels,legends,
                         metrics, fig_name_prefix,xlabel=None):
    # plot train and test results in two rows of figures

    datasets_to_plot = ['_train', '_test']
    
    nrows = len(datasets_to_plot)
    ncols=len(metrics)
        
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols,
                           figsize=(max(ncols*(len(results_labels)/2+2),15),nrows*3.5))
    plt.subplots_adjust(wspace=min(1.4/len(results_labels),0.35),hspace=0.5)
    min_vals = defaultdict(lambda:float('inf'))
    max_vals = defaultdict(float)
    for mm,label_suffix in enumerate(datasets_to_plot):
        
        for nn, metric in enumerate(metrics):
            row = (mm*len(metrics)+nn)//ncols
            col = (mm*len(metrics)+nn)-(mm*len(metrics)+nn)//ncols*ncols

            plot_metric = 'NSE' if metric == 'RSR' else metric
            all_data_to_plot = []
            for ii, label in enumerate(results_labels):
                to_plot = results_dict[label]['X2'+label_suffix][plot_metric]
                if metric == 'R':
                    data_to_plot = (np.asarray(to_plot).reshape(-1))**2
                elif metric == 'RSR':
                    data_to_plot=np.sqrt(1 - np.asarray(to_plot).reshape(-1))
                else:
                    data_to_plot = np.asarray(to_plot).reshape(-1)
                all_data_to_plot.append(data_to_plot)
                
            try:
                ax[row][col].plot(legends, all_data_to_plot)
            except ValueError:
                print('here')
                print(all_data_to_plot)


            label_text = metric
            if metric =='Bias':
                label_text = 'Bias (%)'
            elif metric == 'R':
                label_text = 'r' + r'${}^2$'
                ax[row][col].set_ylim(ax[row][col].get_ylim()[0],min(ax[row][col].get_ylim()[1],1))
            elif metric == 'NSE':
                ax[row][col].set_ylim(ax[row][col].get_ylim()[0],min(ax[row][col].get_ylim()[1],1))
                
            ax[row][col].set_ylabel(label_text)
            
            min_vals[metric] = min(min_vals[metric], ax[row][col].get_ylim()[0])
            max_vals[metric] = max(max_vals[metric], ax[row][col].get_ylim()[1])

    for mm,label_suffix in enumerate(datasets_to_plot):
        if 'train' in label_suffix:
            title_prefix = 'Training'
        elif 'test' in label_suffix:
            title_prefix = 'Test'
        else:
            title_prefix = ''
        for nn, metric in enumerate(metrics):
            row = (mm*len(metrics)+nn)//ncols
            col = (mm*len(metrics)+nn)-(mm*len(metrics)+nn)//ncols*ncols
            ax[row][col].set_ylim(min_vals[metric],max_vals[metric])
            if row == nrows-1:
                ax[row][col].set_xlabel(xlabel)
                shift_ratio = 0.35
            else:
                shift_ratio = 0.2
            label_text = metric
            if metric =='Bias':
                label_text = 'Bias (%)'
            elif metric == 'R':
                label_text = 'r' + r'${}^2$'
            
            mid_pos =ax[row][col].get_xlim()[1] - (ax[row][col].get_xlim()[1] - ax[row][col].get_xlim()[0])*0.5
            ax[row][col].text(mid_pos,
                              ax[row][col].get_ylim()[0]*(1+shift_ratio)-ax[row][col].get_ylim()[1]*shift_ratio,
                              "(%s)" % chr(97+col+row*ncols) + ' %s %s' % (title_prefix, label_text),
                              weight='bold',
                              verticalalignment='center',
                              horizontalalignment='center')
            
    plt.savefig('%s.png'% (fig_name_prefix),bbox_inches='tight',dpi=300)

baseline_nday = 8
baseline_window_size = 11
baseline_nwindow = 10


####################################
##### Varying window lengths #######
####################################

all_results = {}
labels = []
window_sizes = [0] + list(np.arange(1,baseline_window_size+2,2))
for window_size in window_sizes:
    label = '%dx%d_and_%dday' % (window_size,baseline_nwindow,baseline_nday)
    all_results[label] = load_results(os.path.join(root_path,path_formatter % label))
    labels.append(label)
    
plot_box_and_whisker(all_results, labels,window_sizes,
                      metrics=eval_metrics, fig_name_prefix='X2_plots/vary_window_size',
                      xlabel='Length of moving windows')


####################################
## Varying number of daily values ##
####################################
all_results = {}
labels = []
ndays = list(np.arange(0,10,2))
path_formatter = 'X2_i%d_d8_d2_o1_first_chronological_%s_full_results.pkl'
for nday in ndays:
    label = '%dx%d_and_%dday' % (baseline_window_size,baseline_nwindow,nday)
    all_results[label] = load_results(os.path.join(root_path,path_formatter % (nday+baseline_nwindow, label)))
    labels.append(label)
    
plot_box_and_whisker(all_results, labels, ndays,
                          metrics=eval_metrics, fig_name_prefix='X2_plots/vary_nday',
                          xlabel='Number of daily values')

####################################
#### Varying number of windows #####
####################################

all_results = {}
labels = []
nwindows = [0] + list(np.arange(0,baseline_nwindow+2,2))
path_formatter = 'X2_i%d_d8_d2_o1_first_chronological_%s_full_results.pkl'
for nwindow in nwindows:
    label = '%dx%d_and_%dday' % (baseline_window_size,nwindow,baseline_nday)
    all_results[label] = load_results(os.path.join(root_path,path_formatter % (baseline_nday+nwindow, label)))
    labels.append(label)
    
plot_box_and_whisker(all_results, labels,nwindows,
                      metrics=eval_metrics, fig_name_prefix='X2_plots/vary_nwindow',
                      xlabel='Number of 11-day moving windows')


#####################################
#### Varying downsampling ratios ####
#####################################

path_formatter = 'X2_i18_d8_d2_o1_first_chronological_%s_full_results.pkl'
all_results = {}
labels = []
keep_1_over_n = list(np.arange(2,8))
for n in keep_1_over_n:
    label = 'downsample_1over%d' % (n)
    all_results[label] = load_results(os.path.join(root_path,path_formatter % label))
    labels.append(label)
    
plot_box_and_whisker(all_results, labels,np.arange(2,8),
                          metrics=eval_metrics, fig_name_prefix='X2_plots/downsample_outflow',
                          xlabel='Downsampling ratio')


#####################################
###### Using a single variable ######
#####################################

# root_path = '/Users/siyuqi/Downloads/drive-download-20230613T221202Z-001'
# paths = ['X2_i18_d8_d2_o1_first_chronological_only_Outflow_full_results.pkl',
#          'X2_i18_d8_d2_o1_first_chronological_only_SMSC_Gate_full_results.pkl',
#          'X2_i18_d8_d2_o1_first_chronological_only_Tide_full_results.pkl']

# for p in paths:
#     print(p)
#     file = open(os.path.join(root_path,p), 'rb')
#     data = pickle.load(file)
#     file.close()
#     print(data)
    