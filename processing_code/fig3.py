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
from mpl_toolkits.basemap import Basemap
from matplotlib.table import table
from matplotlib import rc
from general_tools import *
from gaussian_kde_weighted import *

surface_only = True

read_dir = '/Users/fier887/Downloads/02_aging_timescale/gcm_data/CAM_monthly/'

ncfiles = []
ncfiles.extend([read_dir + x for x in os.listdir(read_dir) if x.endswith('.nc')])

month = 7
monolayer_thicknesses = [1.,2.,4.,8]
m = Basemap(projection='kav7',lon_0=0)


monolayer_thickness = 1.

ncfile = read_dir + 'aging_L' + str(int(monolayer_thickness)) + '__2010_' + str(int(month)).zfill(2) + '.nc'

f = netCDF4.Dataset(ncfile) 
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
lev = f.variables['lev'][:]

LON,LAT = np.meshgrid(lon,lat)
con_p = f.variables['con_p'][:]
coag_p = f.variables['coag_p'][:]
tau_p = f.variables['total_p'][:]

con_1 = f.variables['con_m'][:] 
coag_1 = f.variables['coag_m'][:]
tau_1 = f.variables['total_m'][:]
mbc_1 = f.variables['bc_pc'][:] + f.variables['bc_accu'][:]


monolayer_thickness = 8.
ncfile = read_dir + 'aging_L' + str(int(monolayer_thickness)) + '__2010_' + str(int(month)).zfill(2) + '.nc'


f = netCDF4.Dataset(ncfile) 
lon = f.variables['lon'][:]

con_8 = f.variables['con_m'][:]
coag_8 = f.variables['coag_m'][:]
tau_8 = f.variables['total_m'][:]
mbc_8 = f.variables['bc_pc'][:] + f.variables['bc_accu'][:]

fig = plt.figure()
fig.set_size_inches(8.,4.)
ax1 = plt.subplot(1,4,1)
pcol1 = m.pcolor(LON, LAT, np.log10(1./tau_1[55,:,:]),latlon=True,clim=[0.,np.log10(24.*7.)],ax=ax1); 

m.drawcoastlines();
pcol1.set_clim(vmin=np.log10(1/(30.*24.)),vmax=0)

ax2 = plt.subplot(1,4,2)
bbox_vals = ax2.get_position()
ax2.set_position([bbox_vals.x0-0.025,bbox_vals.y0,bbox_vals.width,bbox_vals.height])

pcol2 = m.pcolor(LON, LAT, np.log10(1/tau_8[55,:,:]),latlon=True,clim=[0.,np.log10(24.*7.)],ax=ax2); 
m.drawcoastlines();
pcol2.set_clim(vmin=np.log10(1/(30.*24.)),vmax=0)

cbar = m.colorbar(pcol2,location='right',ticks=np.log10(1./np.array([1.,24.,24.*7.,24.*30.])))#,ticks=np.log10([1.,24.,24.*7.,24.*30.]))
cbar.set_ticklabels(['1 hour','1 day','1 week','1 month'])


ax3 = plt.subplot(1,4,3)
if surface_only:
    pcol3 = m.pcolor(LON, LAT, np.log10(mbc_8[55,:,:]*1e9),latlon=True,ax=ax3,clim=[-5.5,0.5]);

else:
    pcol3 = m.pcolor(LON, LAT, np.log10(np.mean(mbc_8,axis=0)*1e9),latlon=True,ax=ax3,clim=[-5.5,0.5]);
pcol3.set_cmap('Greys')
m.drawcoastlines();
cbar3 = m.colorbar(pcol3,location='right')

if surface_only:
    cbar3.set_ticks([-4,-2,0])
    cbar3.set_ticklabels([r'$10^{-4}$',r'$10^{-2}$',r'$10^{0}$'])
cbar3.ax.get_yaxis().label.set_position((0., 1.7)) 

ax4 = plt.subplot(1,4,4)
if surface_only:
    pcol4 = m.pcolor(LON, LAT, mbc_1[55,:,:]/mbc_8[55,:,:],latlon=True,ax=ax4,clim=[0.,1.])#,clim=[0.,3.],ax=ax4);
else:
    pcol4 = m.pcolor(LON, LAT, np.sum(mbc_1,axis=0)/np.sum(mbc_8,axis=0),latlon=True,ax=ax4,clim=[0.,1.])#,clim=[0.,3.],ax=ax4);

pcol4.set_cmap('Purples_r')
m.drawcoastlines();
cbar4 = m.colorbar(pcol4,location='right')

ax1.set_title(r'$\tau_{\mathrm{age}}$' + '\n1 monolayer')
ax2.set_title(r'$\tau_{\mathrm{age}}$'+ '\n8 monolayers')
ax3.set_title(r'$m_{\mathrm{BC},8}$ [$\mu$g/kg]')#$,fontweight='bold')
ax4.set_title(r'$m_{\mathrm{BC},1}/m_{\mathrm{BC},8}$')#,fontweight='bold')

bbox_vals = ax3.get_position()
ax3.set_position([bbox_vals.x0+0.04,bbox_vals.y0,bbox_vals.width,bbox_vals.height])

bbox_vals = ax4.get_position()
ax4.set_position([bbox_vals.x0+0.07,bbox_vals.y0,bbox_vals.width,bbox_vals.height])

fig.savefig('figs/fig3_sens-to-monolayer_month' + str(month) + '.png',dpi=2000,bbox_inches='tight')