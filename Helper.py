#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,x,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

class ComparisonPlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average reward') 
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,x,y,label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x 
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

    # Test Performance plot
    PerfTest = ComparisonPlot(title="Test Comparison")
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 1')
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 2')
    PerfTest.save(name='comparison_test.png') 