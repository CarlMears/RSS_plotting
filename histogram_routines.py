# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:02:53 2018

@author: mears
"""

def averages_from_histograms(hist,edges):
    import numpy as np
    
    num_hist = hist.size
    num_edges = edges.size
    
    if ((num_hist+1) == num_edges):
        tot_num = np.sum(hist)
        median_num = tot_num/2.0
        cumulative_sum = np.cumsum(hist)
        past_median = np.where(cumulative_sum > median_num)
        j = np.min(past_median)
        median_value = edges[j] + ((median_num - cumulative_sum[j-1])/hist[j])*(edges[j+1]-edges[j])
        centers = 0.5*(edges[0:num_edges-1]+edges[1:num_edges])
        mean_value = np.sum(centers*hist)/tot_num
        return mean_value,median_value