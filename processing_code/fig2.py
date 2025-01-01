#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:28:21 2019

@author: fiercenator

Figure 2: timescale parameterization in GCM
"""


import numpy as np
import os
import netCDF4
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from gaussian_kde_weighted import *
from general_tools import *


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


read_dir = '/Users/fiercenator/stuff/projects/02_aging_timescale/gcm_data/CAM_aging/'

ncfiles = []
ncfiles.extend([read_dir + x for x in os.listdir(read_dir) if x.endswith('.nc')])

monolayer_thicknesses = [1.,2.,4.,8]
m = Basemap(projection='kav7',lon_0=0)


monolayer_thickness = 1.
idx, =np.where([ncfile.endswith(str(int(monolayer_thickness)) + '.nc') for ncfile in ncfiles])
ncfile = ncfiles[idx[0]]

f = netCDF4.Dataset(ncfile) 
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]


LON,LAT = np.meshgrid(lon,lat)
mbc_fresh = f.variables['bc_pc'][:]
con_p = f.variables['con_p'][:]
coag_p = f.variables['coag_p'][:]
tau_p = f.variables['total_p'][:]

fig = plt.figure()
ax1 = plt.subplot(2,2,1)
ax3 = plt.subplot(2,2,2)
ax2 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)

pcol1 = m.pcolor(LON, LAT, np.log10(1./tau_p[55,:,:]),latlon=True,clim=[0.,np.log10(24.*7.)],ax=ax1); 
m.drawcoastlines(ax=ax1);
pcol1.set_clim(vmin=np.log10(1./(30.*24.)),vmax=np.log10(1./6.))

ax1.set_title(r'$\tau_{\mathrm{mix}}$',fontweight='bold')

cbar1 = m.colorbar(pcol1,location='right',pad="10%",ticks=np.log10(1./np.array([24.,24.*7.,24.*30.])),ax=ax1)
cbar1.set_ticklabels(['1 day','1 week','1 month'])

pcol2 = m.pcolor(LON, LAT, (1./con_p[55,:,:])/(1./tau_p[55,:,:]),latlon=True,clim=[0.,1.],ax=ax2); 
m.drawcoastlines(ax=ax2);
pcol2.set_cmap('Blues')
pcol2.set_clim(vmin=0.,vmax=1.)

ax2.set_title(r'$f_{\mathrm{cond}}$')

ax3.set_ylabel('frequency dist., weighted\nby mixing ratio of fresh BC')

cbar2 = m.colorbar(pcol2,location='right',pad="10%",ax=ax2,ticks=(0.,0.5,1.))
cbar2.set_ticklabels(['0%','50%','100%'])

LEV = np.zeros(mbc_fresh.shape);
ii = 0
for lev in f.variables['lev']:
    LEV[ii,:,:] = lev
    ii = ii + 1

idx1,idx2,idx3 = np.where((mbc_fresh[:]>0.)&(LEV>200))
f_cond = tau_p[idx1,idx2,idx3]/con_p[idx1,idx2,idx3].ravel()
f_coag = tau_p[idx1,idx2,idx3]/coag_p[idx1,idx2,idx3].ravel()
mbc = mbc_fresh[idx1,idx2,idx3]
tau_bins = np.logspace(-1,3,80)
contribution_bins = np.linspace(0.,1.,40)

idx = np.digitize(tau_p[idx1,idx2,idx3],tau_bins)
hist1d = np.zeros(len(tau_bins))
fcond_mean = np.zeros(len(tau_bins))
fcond_std = np.zeros(len(tau_bins))
fcoag_mean = np.zeros(len(tau_bins))
fcoag_std = np.zeros(len(tau_bins))
dtau = np.log10(tau_bins[1]) - np.log10(tau_bins[0])
for ii in range(len(tau_bins)):
    hist1d[ii] = sum(mbc[idx==ii])/(sum(mbc)*dtau)
    if hist1d[ii]>0:
        fcond_mean[ii],fcond_std[ii] = weighted_avg_and_std(f_cond[idx==ii], mbc[idx==ii])
        fcoag_mean[ii],fcoag_std[ii] = weighted_avg_and_std(f_coag[idx==ii], mbc[idx==ii])

kernel_1d = gaussian_kde_weighted(np.log10(tau_p[idx1,idx2,idx3]),weights=mbc_fresh[idx1,idx2,idx3],bw_method=0.1)
ax3.plot(np.log10(tau_bins),kernel_1d(np.log10(tau_bins)),color='k')

col_cond = '#0033cc'
col_coag = '#009999'

ax4.errorbar(np.log10(tau_bins),savgol_filter(fcond_mean,5,1),savgol_filter(fcond_std,5,1),color=col_cond); 
ax4.errorbar(np.log10(tau_bins),savgol_filter(fcoag_mean,5,1),savgol_filter(fcoag_std,5,1),color=col_coag); 
ax4.set_xticks(np.log10([1,24,24*7,24*30])); 
ax4.set_xticklabels(['1 hour', '1 day', '1 week', '1 month'],rotation=45)
ax4.set_xlabel(r'$\tau_{\mathrm{mix}}$')

ax3.set_xticks(np.log10([1,24,24*7,24*30])); 
ax3.set_xticklabels([])

ybox1 = TextArea(r'$f_{\mathrm{cond}}$', textprops=dict(color=col_cond, rotation=90,ha='left',va='bottom'))
ybox2 = TextArea(', ', textprops=dict(color='k', rotation=90,ha='left',va='bottom'))
ybox3 = TextArea(r'$f_{\mathrm{coag}}$', textprops=dict(color=col_coag, rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=0)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.295, 0.09), 
                                  bbox_transform=ax4.transAxes, borderpad=0.)
ax4.add_artist(anchored_ybox)

 
ax4.set_xlim(np.log10([0.5,24*30*1.01]))
ax3.set_xlim(np.log10([0.5,24*30*1.01]))

ax3.set_position([0.75,0.47,0.3,0.36])
ax4.set_position([0.75,0.16,0.3,0.26])

fig.savefig('figs/fig2_combined.png',dpi=1000,bbox_inches='tight')
