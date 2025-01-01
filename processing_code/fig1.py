#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:28:21 2019

@author: fiercenator

Figure 1: PartMC aging figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from read_partmc import *
from general_tools import *

import os

N_times = 25
sa_flux = np.zeros(N_times-1)
time = np.zeros(N_times-1)
num_tot = np.zeros(N_times-1)
for tt in range(1,N_times):
    aero_id_t = get_particle_variable_all_ensembles('aero_id', partmc_dir, tt)
    aero_id_tPlus = get_particle_variable_all_ensembles('aero_id', partmc_dir, tt + 1)    
    aero_removed_id = get_particle_variable_all_ensembles('aero_removed_id', partmc_dir, tt + 1)
    aero_removed_other_id = get_particle_variable_all_ensembles('aero_removed_other_id', partmc_dir, tt + 1)


    idx_still_there, = np.where((~np.in1d(aero_id_t,aero_removed_id)*1. + ~np.in1d(aero_id_t,aero_removed_other_id)*1.)>1) 
    idx_was_there, = np.where((~np.in1d(aero_id_tPlus,aero_removed_id)*1. + ~np.in1d(aero_id_tPlus,aero_removed_other_id)*1. + np.in1d(aero_id_tPlus,aero_id_t)*1.)>2)
    
    aero_comp_vol = get_particle_variable_all_ensembles('aero_comp_vol', partmc_dir, tt)    
    
    dry_diameter_t = get_particle_variable_all_ensembles('dry_diameter', partmc_dir, tt)
    dry_diameter_tPlus = get_particle_variable_all_ensembles('dry_diameter', partmc_dir, tt + 1)        
    wet_diameter_t = get_particle_variable_all_ensembles('wet_diameter', partmc_dir, tt)    
    
    time_t = get_particle_variable_all_ensembles('time', partmc_dir, tt)
    time_tPlus = get_particle_variable_all_ensembles('time', partmc_dir, tt + 1)
    
    tkappa = get_particle_variable_all_ensembles('tkappa', partmc_dir, tt)
    
    time[tt-1] = time_t
    sa_flux[tt-1] = 3600*1e9*sum(np.pi/6*(dry_diameter_tPlus[idx_was_there]**3 - dry_diameter_t[idx_still_there]**3))/sum((time_tPlus - time_t)*np.pi*wet_diameter_t[idx_still_there]**2)
    num_tot[tt-1] = sum(1./aero_comp_vol)/100.**3

fig = plt.figure()
ax1a = plt.subplot(2,1,1)
ax2a = plt.subplot(2,1,2)
plt.subplots_adjust(left=0.15,right=0.85)

ax1b = ax1a.twinx();
ax2b = ax2a.twinx();
ax2c = ax2b.twinx();

col_cond = '#0033cc'
col_coag = '#009999'

ax1a.plot(time,sa_flux,color=col_cond);
ax1b.plot(time,num_tot,color=col_coag);


sa_flux_input = sa_flux; sa_flux_input[sa_flux<0] = 0;
hln_cond, = ax2a.plot(time,1./(0.1*sa_flux_input),color=col_cond); 
hln_coag, = ax2a.plot(time,1./(6e-6*num_tot),color=col_coag); 
hln_tot, = ax2a.plot(time,1./(0.1*sa_flux + 6e-6*num_tot),color='k',linewidth=2.); 

xticks = np.array([0,6,12,18,24])*3600
hrs = np.remainder(xticks/3600+6,24)
time_fun = lambda h: str(int(h)) + ':00'
hrs_str = [time_fun(h) for h in hrs]

ax1a.set_xticks(xticks)
ax1a.set_xticklabels('')
ax2a.set_xticks(xticks)
ax2a.set_xticklabels(hrs_str)

ax1a.set_xlim([0,3600*23])
ax1b.set_xlim([0,3600*23])
ax1b.set_ylim([0,ax1b.get_ylim()[1]*1.1])
ax2a.set_xlim([0,3600*23])

ax2a.set_yscale('log')
ax2a.set_yticks([1.,24.,24.*7.,24*7*4])
ax2a.set_yticklabels(['1 hour', '1 day', '1 week', '1 month'])
ax2b.set_yscale(ax2a.get_yscale())
ax2b.set_ylim(ax2a.set_ylim())
ax2b.set_yticklabels('')

ax1a.tick_params(axis='y', colors=col_cond)
ax1b.tick_params(axis='y', colors=col_coag)

ax1a.set_ylabel('secondary aerosol\ngrowth rate [nm/h]',color=col_cond)
ax1b.set_ylabel('number conc. [cm$^3$]',color=col_coag)
ax2a.set_ylabel('timescale for internal mixing')

ax2c.set_yticks([])


plt.legend([hln_tot,hln_cond,hln_coag],['overall','by condensation','by coagulation'],loc=4)
fig.set_size_inches([6.,6.])

fig.savefig('figs/fig1_partmc_timeseries.png',dpi=1200)