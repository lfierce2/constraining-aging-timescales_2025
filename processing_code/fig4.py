#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:26:26 2019

@author: fiercenator
"""

import numpy as np
import os
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from matplotlib.table import table
from matplotlib import rc

from general_tools import *
from gaussian_kde_weighted import *

use_basemap = False
if use_basemap:
    from mpl_toolkits.basemap import Basemap

scl = 'lin'
surface_only = True


read_dir = '/Users/fier887/Downloads/02_aging_timescale/gcm_data/CAM_monthly/'

ncfiles = []
ncfiles.extend([read_dir + x for x in os.listdir(read_dir) if x.endswith('.nc')])

base_monolayer = 8
monolayers = [1,2,4,8]
months = [1,7]

month_strs = [
    'January',
    'Februrary',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December']

bc_unit = 'ug'

confidence = 0.99
lower_percentile = (1 - confidence) / 2 * 100  # Lower bound
upper_percentile = (1 + confidence) / 2 * 100  # Upper bound


if use_basemap:
    m = Basemap(projection='kav7',lon_0=0)

fig,axs = plt.subplots(len(months),3,squeeze=True,sharey=False)#,sharex=False)
fig.set_size_inches(9.5,7.5)

dx_axis = 0.05
base_monolayer = 8
for jj,month in enumerate(months):
    ncfile = read_dir + 'aging_L' + str(int(base_monolayer)) + '__2010_' + str(int(month)).zfill(2) + '.nc'
    
    f = netCDF4.Dataset(ncfile) 
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    lev = f.variables['lev'][:]
    
    tau_m = f.variables['total_m'][:]
    tau_p = f.variables['total_p'][:]
    mbc_fresh_base = f.variables['bc_pc'][:]
    mbc_total_base = f.variables['bc_pc'][:] + f.variables['bc_accu'][:]
    
    mean_bc = np.zeros(len(lat))
    lower_bound = np.zeros(len(lat))
    upper_bound = np.zeros(len(lat))
    if scl == 'log':
        if surface_only:
            for ii in range(mbc_total_base.shape[1]):
                mean_bc[ii] = np.mean(np.log10(mbc_total_base[55,ii,:].ravel()))
                lower_bound[ii] = np.percentile(np.log10(mbc_total_base[55,ii,:].ravel()), lower_percentile)
                upper_bound[ii] = np.percentile(np.log10(mbc_total_base[55,ii,:].ravel()), upper_percentile)
        else:
            for ii in range(mbc_total_base.shape[1]):
                mean_bc[ii] = np.mean(np.log10(mbc_total_base[:,ii,:].ravel()))
                lower_bound[ii] = np.percentile(np.log10(mbc_total_base[:,ii,:].ravel()), lower_percentile)
                upper_bound[ii] = np.percentile(np.log10(mbc_total_base[:,ii,:].ravel()), upper_percentile)
    else:
        if surface_only:
            for ii in range(mbc_total_base.shape[1]):
                mean_bc[ii] = np.mean(mbc_total_base[55,ii,:].ravel())
                lower_bound[ii] = np.percentile(mbc_total_base[55,ii,:].ravel(), lower_percentile)
                upper_bound[ii] = np.percentile(mbc_total_base[55,ii,:].ravel(), upper_percentile)
        else:
            for ii in range(mbc_total_base.shape[1]):
                mean_bc[ii] = np.mean(mbc_total_base[:,ii,:].ravel())
                lower_bound[ii] = np.percentile(mbc_total_base[:,ii,:].ravel(), lower_percentile)
                upper_bound[ii] = np.percentile(mbc_total_base[:,ii,:].ravel(), upper_percentile)
        
    axs[jj,1].fill_betweenx(lat,lower_bound,upper_bound,alpha=0.1,color='C0')
    axs[jj,1].plot(mean_bc,lat,color='C0')
    
    if jj == 0:
        axs[jj,1].set_xticklabels('')

    if scl == 'log':
        mbc_xticks = np.linspace(-18,-12,3)
    else:
        mbc_xticks = np.array([0.,1.,2.,3.,4.])*1e-9
    axs[jj,1].set_xticks(mbc_xticks)
    if jj == len(months) - 1:
        if scl == 'log':
            if bc_unit == 'kg':
                if surface_only:
                    lab_str = r'geometric mean of $m_{\text{BC},' + str(int(base_monolayer)) + '}$ at the' + '\nsurface [kg/kg]'
                else:
                    lab_str = r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ [kg-BC/kg-air]'    
                ticklabs = [r'$10^{' + str(int(one_tick)) + '}$' for one_tick in mbc_xticks]
            elif bc_unit == 'ug':
                if surface_only:
                    lab_str = 'geometric mean of\n' + r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ [$\mu$g/kg]'    
                else:
                    lab_str = r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ [$\mu$g-BC/kg-air]'
                ticklabs = [r'$10^{' + str(int(one_tick+9)) + '}$' for one_tick in mbc_xticks]
        else:
            if bc_unit == 'kg':
                if surface_only:
                    lab_str = r'mean $m_{\text{BC},' + str(int(base_monolayer)) + '}$ at the' + '\nsurface [kg/kg]'
                else:
                    lab_str = r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ [kg-BC/kg-air]'
                ticklabs = mbc_xticks
            elif bc_unit == 'ug':
                if surface_only:
                    lab_str = r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ at the' + '\n' + 'surface $\mu$g/kg]'    
                else:
                    lab_str = r'$m_{\text{BC},' + str(int(base_monolayer)) + '}$ [$\mu$g/kg]'    

                    
                ticklabs = [int(xtick) for xtick in mbc_xticks*1e9]
        axs[jj,1].set_xticklabels(ticklabs)
        
        xlab = axs[jj,1].set_xlabel(lab_str,fontsize=15)
        xlab.set_verticalalignment('top')
        axs[jj,1].set_xlabel(axs[jj,1].get_xlabel(), labelpad=10) 

    
    for mm,monolayer_thickness in enumerate(monolayers[:-1]):
        ncfile = read_dir + 'aging_L' + str(int(monolayer_thickness)) + '__2010_' + str(int(month)).zfill(2) + '.nc'
        
        f = netCDF4.Dataset(ncfile) 
        lon = f.variables['lon'][:]
        lat = f.variables['lat'][:]
        lev = f.variables['lev'][:]
        
        tau_m = f.variables['total_m'][:]
        tau_p = f.variables['total_p'][:]
        mbc_fresh = f.variables['bc_pc'][:]
        mbc_total = f.variables['bc_pc'][:] + f.variables['bc_accu'][:]
        
        LEV = np.zeros(mbc_fresh.shape);
        ii = 0
        for lev in f.variables['lev']:
            LEV[ii,:,:] = lev
            ii = ii + 1
        idx1,idx2,idx3 = np.where((mbc_fresh[:]>0.)&(LEV>200))
        mean_ratio = np.zeros(mbc_total_base.shape[1])
        std_ratio = np.zeros(mbc_total_base.shape[1])
        if surface_only:
            for ii in range(mbc_total.shape[1]):
                mean_ratio[ii] = np.sum((mbc_total[55,ii,:].ravel())/np.sum(mbc_total_base[55,ii,:]).ravel())
                std_ratio[ii] = np.sum(
                    mbc_total_base[55,ii,:].ravel() * ((mbc_total[55,ii,:]/mbc_total_base[55,ii,:]).ravel() - mean_ratio[ii])**2) / np.sum(mbc_total_base[55,ii,:].ravel())
    
        else:
            for ii in range(mbc_total.shape[1]):
                mean_ratio[ii] = np.sum((mbc_total[:,ii,:].ravel())/np.sum(mbc_total_base[:,ii,:]).ravel())
                std_ratio[ii] = np.sum(
                    mbc_total_base[:,ii,:].ravel() * ((mbc_total[:,ii,:]/mbc_total_base[:,ii,:]).ravel() - mean_ratio[ii])**2) / np.sum(mbc_total_base[55,ii,:].ravel())
        
        axs[jj,0].text(0.8,85,month_strs[months[jj]-1],horizontalalignment='right',verticalalignment='top',fontsize=17)
        axs[jj,0].set_axis_off()
        lab_str = r'$m_{\text{BC},' + str(int(monolayer_thickness)) + r'}/m_{\text{BC},' + str(int(base_monolayer)) + '}$'
        axs[jj,2].plot(mean_ratio,lat,color='C' + str(int(3-mm)),label=lab_str)
        print(jj,mm)
        if jj<len(months)-1:
            axs[jj,mm].set_xticklabels('')
        axs[jj,2].set_yticklabels('')
        
        if jj == len(months) - 1:
            xlab = axs[jj,2].set_xlabel('mixing ratio of BC' + '\n' + r'relative to $m_{\text{BC},8}$',fontsize=15)
            xlab.set_verticalalignment('top')
            axs[jj,2].set_xlabel(axs[jj,2].get_xlabel(), labelpad=10) 
        
        if jj == 0:
            hleg = axs[jj,2].legend(
                loc='center left',fontsize=11.5,frameon=False,
                labelspacing=0.,bbox_to_anchor=(0., 0.65))
        
        ratio_xticks = np.linspace(0.,1.,5)
        axs[jj,2].set_xticks(ratio_xticks)
        if jj == 0:
            axs[jj,2].set_xticklabels('')
        elif len(ratio_xticks == 5):
            axs[jj,2].set_xticklabels([str('0.0'),str('0.25'),str('0.5'),str('0.75'),str('1.0')])
        
        axs[jj,2].set_xlim([0.,1.])
        axs[jj,1].set_xlim([0.,4.5e-9])
        
for ii in range(axs.shape[0]):
    axs[ii,1].set_ylabel('latitude [degrees]',fontsize=15)
    for jj in range(axs.shape[1]):
        axs[ii,jj].set_ylim([min(lat),max(lat)])
        
        axs[ii,jj].tick_params(axis='both', which='major', labelsize=14)
        axs[ii,jj].tick_params(axis='both', which='minor', labelsize=12)

fig.savefig('figs/fig4.png',dpi=2000,bbox_inches='tight')
        