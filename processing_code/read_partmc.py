#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:58:09 2018

@author: fiercenator
"""

import netCDF4
import numpy as np
import os
from general_tools import *



def get_ncfile(partmc_dir, timestep, ensemble_number):
    barlist = list() 
    for root, dirs, files in os.walk(partmc_dir):
        f = files[0]
    if f.startswith('urban_plume_wc_'):
        preface_string = 'urban_plume_wc_' #''.join([c for idx,c in enumerate(f) if idx<f.find('0')])
    elif f.startswith('urban_plume_'):
        preface_string = 'urban_plume_'
    else:
        preface_string = 'YOU_NEED_TO_CHANGE_preface_string_'
    ncfile = partmc_dir + preface_string + str(int(ensemble_number)).zfill(4) + '_' + str(int(timestep)).zfill(8) + '.nc'
    return ncfile

def get_N_ensembles(partmc_dir):
    barlist = list() 
    ensemble_nums = []
    for root, dirs, files in os.walk(partmc_dir):
        for f in files:
            if f[0] != '.':
                after_preface_str = ''.join([c for idx,c in enumerate(f) if idx>=f.find('0')])
                ensemble_str = ''.join([c for idx,c in enumerate(after_preface_str) if idx<(after_preface_str.find('_'))])
                ensemble_nums.append(int(ensemble_str))
    N_ensembles = max(ensemble_nums)
    return N_ensembles
    
def get_N_timesteps(partmc_dir):
    barlist = list() 
    timestep_nums = []
    for root, dirs, files in os.walk(partmc_dir):
        for f in files:
            if f[0] != '.':
                after_preface_str = ''.join([c for idx,c in enumerate(f) if idx>f.rfind('_')])
                timestep_str = ''.join([c for idx,c in enumerate(after_preface_str) if idx<(after_preface_str.find('.'))])
                timestep_nums.append(int(timestep_str))
    N_times = max(timestep_nums)
    return N_times

def get_particle_variable_all_ensembles(varname, partmc_dir, timestep, wavelength=550*1e-9):
    N_ensembles = get_N_ensembles(partmc_dir)
    vardat = np.array([])
    for ensemble_number in range(1,N_ensembles+1):
        ncfile = get_ncfile(partmc_dir, timestep, ensemble_number)
        vardat_one = get_particle_variable(ncfile, varname, wavelength=wavelength)
        if len(np.shape(vardat_one)) == 2:
            if ensemble_number == 1:
                vardat = vardat_one
            else:
                vardat = np.append(vardat,vardat_one,axis=1)
        else:
            vardat = np.append(vardat,vardat_one)
    return vardat

def get_particle_variable(ncfile, varname, wavelength=550*1e-9,
          kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65,
          ri_soa = 1.45+0j, ri_poa=1.45+0j, ri_inorg = 1.5+0j):
    f = netCDF4.Dataset(ncfile)
    if varname.startswith('aero_removed_id'):
        vardat = f.variables['aero_removed_id'][:]        
    elif varname.startswith('aero_removed_action'):
        vardat = f.variables['aero_removed_action'][:]
        vardat_sigma = 0.;        
    elif varname.startswith('aero_removed_other_id'):
        vardat = f.variables['aero_removed_other_id'][:]
        vardat_sigma = 0.;
    elif varname.startswith('aero_id'):
        vardat = f.variables['aero_id'][:]
        vardat_sigma = 0.;
    elif varname.startswith('aero_particle_mass_uniform'):
        part_mass = f.variables['aero_particle_mass'][:]
        vardat = get_partmass_uniform(part_mass)
        vardat_sigma = 0.;
    elif varname.startswith('aero_particle_mass'):
        vardat = f.variables['aero_particle_mass'][:]        
        vardat_sigma = 0;
    elif varname.startswith('aero_particle_volume'):
        part_mass = f.variables['aero_particle_mass'][:]
        aero_density = f.variables['aero_density'][:]        
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0
        vardat = np.zeros(part_mass.shape)
        for kk in range(N_species):
            vardat[kk,:] = part_mass[kk,:]/aero_density[kk]
        vardat_sigma = 0;
    elif varname.startswith('dry_mass'):
        mass_comp = f.variables['aero_particle_mass'][:]     
        vardat = np.sum(mass_comp[:19,:],axis=0)
        vardat_sigma = 0;
    elif varname.startswith('dry_vol'):
        vol_comp = get_particle_variable(ncfile,'aero_particle_volume')
        vardat = np.sum(vol_comp[:19,:],axis=0)
        vardat_sigma = 0;
    elif varname.startswith('time'):
        vardat = f.variables['time'][:]        
        vardat_sigma = 0;
    elif varname.startswith('aero_spec'):
        vardat = f.variables['aero_species'].names.split(',')
    elif varname.startswith('aero_comp_vol'):
        vardat = f.variables['aero_comp_vol'][:]        
        vardat_sigma = 0;
    elif varname.startswith('dry_dia_uniform'):
#        part_mass_uniform,sig = get_particle_variable(ncfile, 'aero_particle_mass_uniform')
        part_mass_uniform = get_particle_variable(ncfile, 'aero_particle_mass_uniform')        
        part_density = f.variables['aero_density'][:]
        part_vol = np.zeros(part_mass_uniform.shape)
        for kk in range(len(part_density)-1):
            part_vol[kk,:] = part_mass_uniform[kk,:]/part_density[kk]
        vardat = (np.sum(part_vol, axis=0)*6/np.pi)**(1/3)
        vardat_sigma = 0.
    elif varname.startswith('dry_dia'):
        part_mass = f.variables['aero_particle_mass'][:]
        part_density = f.variables['aero_density'][:]
        part_vol = np.zeros(part_mass.shape)
        kk = 0
        for kk in range(len(part_density)-1):
            part_vol[kk,:] = part_mass[kk,:]/part_density[kk]
        vardat = (np.sum(part_vol, axis=0)*6/np.pi)**(1./3.)
        vardat_sigma = 0.
    elif varname.startswith('wet_dia'):
        part_mass = f.variables['aero_particle_mass'][:]
        part_density = f.variables['aero_density'][:]
        part_vol = np.zeros(part_mass.shape)
        kk = 0
        for kk in range(len(part_density)):
            part_vol[kk,:] = part_mass[kk,:]/part_density[kk]
        vardat = (np.sum(part_vol, axis=0)*6/np.pi)**(1./3.)
        vardat_sigma = 0.        
    elif varname.startswith('core_dia_withOC'):
        part_mass = f.variables['aero_particle_mass'][:]
        part_density = f.variables['aero_density'][:]
        idx_core = [part_density.shape[0]-3,part_density.shape[0]-2]
        core_vol = np.sum([part_mass[i,:]/part_density[i] for i in idx_core],axis=0)
        vardat = (core_vol*6/np.pi)**(1/3)
        vardat_sigma = 0.        
    elif varname.startswith('core_dia'):
        part_mass = f.variables['aero_particle_mass'][:]
        part_density = f.variables['aero_density'][:]
        idx_bc = part_density.shape[0]-2
        core_vol = part_mass[idx_bc,:]/part_density[idx_bc]
        vardat = (core_vol*6/np.pi)**(1/3)
        vardat_sigma = 0.
    elif varname.startswith('m_core'):
        if abs(wavelength-630*1e-9)<3e-9:
            vardat = (1.95+0.79*1j)*np.ones(np.shape(f.variables['aero_refract_core_real'][:]))
            vardat_sigma = 0.
        elif abs(wavelength-532e-9)<2e-9:
            vardat = (1.88+0.8*1j)*np.ones(np.shape(f.variables['aero_refract_core_real'][:]))
            vardat_sigma = 0.            
        else:            
            from scipy.interpolate import interp1d
            wvl_all = 1e-9*np.array([300.,350.,400.,500.,530.,600.,690.,800.,860.,900.,1060.,1200.,1400.,1670.,1800.,2000.,2350.,2500.,3000.,3390.,3510.,3750.,4000.,4500.,5000.,5500.,6000.,6500.,7000.,7500.,8000.,8500.,9000.,9500.,10000.,10500.,11000.,11500.,12000.,12500.,13000.,13500.,14000.,14500.]);
            real_ri = [1.8,1.8,1.8,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.91,1.92,1.94,1.97,1.98,2.02,2.07,2.08,2.1,2.13,2.14,2.16,2.18,2.23,2.26,2.28,2.31,2.39,2.34,2.4,2.36,2.32,2.34,2.36,2.4,2.42,2.43,2.44,2.5,2.61,2.65,2.64,2.62,2.6];
            imag_ri = [0.74,0.74,0.74,0.74,0.74,0.72,0.7,0.69,0.69,0.68,0.68,0.68,0.68,0.68,0.68,0.72,0.72,0.72,0.72,0.72,0.735,0.74,0.77,0.77,0.77,0.75,0.7,0.71,0.72,0.76,0.72,0.73,0.82,0.85,1,1.02,1.03,1.04,1.05,1.05,1.06,1.07,1.07,1.08];
            interpolation_real = interp1d(wvl_all,real_ri)
            interpolation_imag = interp1d(wvl_all,imag_ri)
            vardat = (interpolation_imag(wavelength)*1j + interpolation_real(wavelength))**np.ones(np.shape(f.variables['aero_refract_core_real'][:]))
            vardat_sigma = 0.
    elif varname.startswith('m_shell_uniform'):
        m_shell = get_particle_variable(ncfile,'m_shell')
        dry_dia = get_particle_variable(ncfile,'dry_diameter'); #dry_dia = dry_dia[0]
        core_dia = get_particle_variable(ncfile,'core_diameter'); #core_dia = core_dia[0]
        idx = core_dia.nonzero(); idx = idx[0]
        vardat = np.zeros(len(core_dia))
        these = np.nonzero(dry_dia[idx]**3-core_dia[idx]**3);
        idx = idx[these]; 
        if len(idx)>0:
            vardat[idx] = np.ones(len(idx))*np.sum((dry_dia[idx]**3-core_dia[idx]**3)*m_shell[idx])/np.sum(dry_dia[idx]**3-core_dia[idx]**3)
        
        idx = core_dia.nonzero(); idx = idx[0]
        these = np.nonzero(dry_dia[idx]**3>core_dia[idx]**3);
        idx = idx[these]; 
        if len(idx)>0:
            vardat[idx] = np.ones(len(idx))*np.sum((dry_dia[idx]**3)*m_shell[idx])/np.sum(dry_dia[idx]**3)
        vardat_sigma = 0            
    elif varname.startswith('m_shell'):
        var = get_particle_variable(ncfile, 'aero_particle_mass')
        mass_comp = np.vstack([var[:-2,:],var[-1,:]])
        var = get_particle_variable(ncfile, 'density');
        density = np.hstack([var[:-2],var[-1]])
        var = get_particle_variable(ncfile, 'aero_spec')
        spec_names = np.hstack([var[:-2],var[-1]])
        shell_spec = get_spec_names('shell')
        idx_shell = np.hstack([np.where(ismember(spec_names, spec))[0] for spec in shell_spec])                
        vardat = get_effective_ri(mass_comp[idx_shell,:],density[idx_shell], shell_spec, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)    
        vardat_sigma = 0.
    elif varname.startswith('R_bc_uniform'):
        part_mass = f.variables['aero_particle_mass'][:]
        part_mass_dry = part_mass[range(part_mass.shape[0]-1),:]
        part_mass = get_partmass_uniform(part_mass_dry, averaging_method = 'bc')
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0;
        vardat = sum(part_mass[0:(N_species-2),these])/part_mass[N_species-1,these]
        vardat_sigma = 0.        
    elif varname.startswith('R_bc') or varname.startswith('Rbc'):
        part_mass = f.variables['aero_particle_mass'][:]
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0;
        vardat = sum(part_mass[0:(N_species-2),these])/part_mass[N_species-1,these]
        vardat_sigma = 0.
    elif varname.startswith('m_bc'):
        part_mass = f.variables['aero_particle_mass'][:]
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0;
        vardat = part_mass[N_species-1,these];
        vardat_sigma = 0.
    elif varname.startswith('m_BrC'):
        part_mass = f.variables['aero_particle_mass'][:]
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0
        vardat = np.sum(part_mass[:-2,these],axis=0);
        vardat_sigma = 0.        
    elif varname.startswith('R_bc_vol'):
        part_mass = f.variables['aero_particle_mass'][:]
        aero_density = f.variables['aero_density'][:]        
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0
        part_vol = np.zeros(part_mass.shape)
        for kk in range(N_species):
            part_vol[kk,:] = part_mass[kk,:]/aero_density[kk]
        vardat = sum(part_vol[0:(N_species-2),these])/part_vol[N_species-1,these]
        vardat_sigma = 0.
    elif varname.startswith('v_bc'):
        part_mass = f.variables['aero_particle_mass'][:]
        N_species = part_mass.shape[0]-1
        these = part_mass[N_species-1,:]>0;
        aero_density = f.variables['aero_density'][:]        
        vardat = part_mass[N_species-1,these]/aero_density[N_species-1];
        vardat_sigma = 0.        
    elif varname.startswith('aero_absorb_cross_sect'):
        vardat = f.variables['aero_absorb_cross_sect'][:]
    elif varname.startswith('aero_scatter_cross_sect'):
        vardat = f.variables['aero_scatter_cross_sect'][:]        
    elif varname.startswith('abs_crossect_cs_uniform'): 
        import PyMieScatt
        core_dia = get_particle_variable(ncfile, 'core_dia') 
        dry_dia = get_particle_variable(ncfile, 'dry_dia_uniform') 
        m_shell = get_particle_variable(ncfile, 'm_shell_uniform') 
        m_core = get_particle_variable(ncfile, 'm_core',wavelength)         
        vardat = np.zeros(dry_dia.shape)
        for ii in range(m_core.shape[0]):
            if core_dia[ii]>0:
                output = PyMieScatt.MieQCoreShell(m_core[ii], m_shell[ii], wavelength, core_dia[ii], dry_dia[ii])
                vardat[ii] = output[2]*np.pi*dry_dia[ii]**2/4
            else:
                vardat[ii] = 0.  
        vardat_sigma = 0.
    elif varname.startswith('abs_crossect_cs'):        
        import PyMieScatt
        core_dia = get_particle_variable(ncfile, 'core_dia'); #core_dia = core_dia[0]
        dry_dia = get_particle_variable(ncfile, 'dry_dia'); #dry_dia = dry_dia[0]
        m_shell = get_particle_variable(ncfile, 'm_shell'); #m_shell = m_shell[0]
        m_core = get_particle_variable(ncfile, 'm_core',wavelength); #m_core = m_core[0]        
        
        vardat = np.zeros(dry_dia.shape)
        for ii in range(m_core.shape[0]):
            if core_dia[ii]>0:
                output = PyMieScatt.MieQCoreShell(m_core[ii], m_shell[ii], wavelength, core_dia[ii], dry_dia[ii])
                vardat[ii] = output[2]*np.pi*dry_dia[ii]**2/4
            else:
                vardat[ii] = 0.  
        vardat_sigma = 0.
    elif varname.startswith('abs_crossect_uncoated'):        
        import PyMieScatt
        core_dia = get_particle_variable(ncfile, 'core_dia');# core_dia = core_dia[0]
        m_core = get_particle_variable(ncfile, 'm_core',wavelength); #m_core = m_core[0]        
        
        vardat = np.zeros(core_dia.shape)
        if any(core_dia>0):
            for ii in range(m_core.shape[0]):
                if core_dia[ii]>0:
                    output = PyMieScatt.MieQCoreShell(m_core[ii], m_core[ii], wavelength, core_dia[ii], core_dia[ii])
                    vardat[ii] = output[2]*np.pi*core_dia[ii]**2/4
                else:
                    vardat[ii] = 0.  
        else:
            vardat = np.zeros(np.shape(core_dia))
        vardat_sigma = 0.
    elif varname.startswith('density'):
        vardat = f.variables['aero_density'][:]
        vardat_sigma = 0.
    elif varname.startswith('aero_kappa'):
        vardat = f.variables['aero_kappa'][:]
        vardat_sigma = 0.
    elif varname.startswith('tkappa_coreWithOC'):
        mass_comp = get_particle_variable(ncfile, 'aero_particle_mass');
        density = get_particle_variable(ncfile, 'density');
        spec_names = get_particle_variable(ncfile, 'aero_spec')
        vardat = get_effective_kappa(mass_comp[-3:-2,:], density[-3:-2], spec_names[-3:-2], kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)
    elif varname.startswith('tkappa_coatWithoutOC'):
        mass_comp = get_particle_variable(ncfile, 'aero_particle_mass');
        density = get_particle_variable(ncfile, 'density');
        spec_names = get_particle_variable(ncfile, 'aero_spec')
        vardat = get_effective_kappa(mass_comp[:-3,:], density[:-3], spec_names[:-3], kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)
        
    elif varname.startswith('tkappa_shell'):
        mass_comp = get_particle_variable(ncfile, 'aero_particle_mass');
        density = get_particle_variable(ncfile, 'density');
        spec_names = get_particle_variable(ncfile, 'aero_spec')
        
        vardat = get_effective_kappa(mass_comp[:-2,:], density[:-2], spec_names[:-2], kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)  
        vardat_sigma = 0.                
    elif varname.startswith('tkappa'):
        mass_comp = get_particle_variable(ncfile, 'aero_particle_mass');
        density = get_particle_variable(ncfile, 'density');
        spec_names = get_particle_variable(ncfile, 'aero_spec')
        
        vardat = get_effective_kappa(mass_comp[:-1,:], density[:-1], spec_names[:-1], kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)  
        vardat_sigma = 0.

    elif varname.startswith('ssat_uniform'):
        temp = 293.15
        dia = get_particle_variable(ncfile,'dry_diameter')        
        tkappa = np.ones(len(dia))*sum(dia**3*get_particle_variable(ncfile,'tkappa')/dia**3)
        
        vardat = process_compute_Sc(temp, dia, tkappa)
        vardat_sigma = 0.
    elif varname.startswith('ssat_bconly'):
        temp = 293.15
        tkappa = get_particle_variable(ncfile,'tkappa', kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)
        dia = get_particle_variable(ncfile,'dry_diameter');
        idx_bc, = np.where(get_particle_variable(ncfile,'core_dia')>0);
        vardat = np.zeros(len(dia))
        vardat[idx_bc] = process_compute_Sc(temp, dia[idx_bc], tkappa[idx_bc])
        vardat_sigma = 0
    elif varname.startswith('ssat'):
        temp = 293.15
        tkappa = get_particle_variable(ncfile,'tkappa', kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)
        dia = get_particle_variable(ncfile,'dry_diameter')
        vardat = process_compute_Sc(temp, dia, tkappa)
        vardat_sigma = 0

    elif varname.startswith('aero_n_orig_part') or varname.startswith('n_orig_part'):
        vardat = f.variables['aero_n_orig_part'][:]
    elif varname.startswith('gas_mixing_ratio'):
        vardat = f.variables['gas_mixing_ratio'][:]
    elif varname.startswith('gas_species'):
        vardat = f.variables['gas_species'][:]

                             
    return vardat

def get_partmass_uniform(part_mass, averaging_method = 'bc'):
    bc_mass = part_mass[18,:]
    part_mass_uniform = np.zeros(part_mass.shape)
    idx, = bc_mass.nonzero(); 
    for kk in range(part_mass.shape[0]):
        part_mass_uniform[kk,idx] = bc_mass[idx]*sum(part_mass[kk,idx])/sum(bc_mass[idx])
        
    return part_mass_uniform
    
def get_Dwet(Ddry, kappa, RH, temp):
    from scipy.optimize import brentq
    if RH>0 and kappa>0:
        sigma_w = 0.072; rho_w = 1000; M_w = 18/1e3; R=8.314;
        A = 4*sigma_w*M_w/(R*temp*rho_w)
        zero_this = lambda gf: RH/np.exp(A/(Ddry*gf))-(gf**3-1)/(gf**3-(1-kappa))
        return Ddry*brentq(zero_this,1,10000)
    else:
        return Ddry


def get_mass_h2o(Ddry, effective_kappa, effective_density, RH, temp):
    Dwet = get_Dwet(Ddry, effective_kappa, RH, temp)
    mass_h2o = (np.pi/6*Dwet**3 - np.pi/6*Ddry**3)*effective_density
    return mass_h2o
    
def process_compute_Sc(temp, dia, tkappa): # % numActivated/cm^2
    import numpy as np
    import scipy.optimize as opt
    sigma_w = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2
    m_w = 18/1000.; #kg/mol
    R = 8.314; # J/mol*K
    rho_w = 1000.; # kg/m^3
    A = 4.*sigma_w*m_w/(R*temp*rho_w);
    
    ssat = np.zeros(len(dia));
    crit_diam = np.zeros(len(dia));
    for i in range(len(dia)):
        if tkappa[i]>0.2: # equation 10 from Petters and Kreidenweis, 2007
            ssat[i] = (np.exp((4.*A**3./(27.*dia[i]**3.*tkappa[i]))**(0.5))-1.)*100.
        else:
            d = dia[i]
            f = lambda x: compute_Sc_funsixdeg(x,A,tkappa[i],d)
            soln = opt.root(f,d*10);
            x = soln.x
            crit_diam[i]=x 
            ssat[i]=(x**3.0-d**3.0)/(x**3-d**3*(1.0-tkappa[i]))*np.exp(A/x);
            ssat[i]=(ssat[i]-1.0)*100
    return ssat 

        
def compute_Sc_funsixdeg(diam,A,tkappa,dry_diam):
    c6=1.0;
    c4=-(3.0*(dry_diam**3)*tkappa/A); 
    c3=-(2.0-tkappa)*(dry_diam**3); 
    c0=(dry_diam**6.0)*(1.0-tkappa);
    
    z = c6*(diam**6.0) + c4*(diam**4.0) + c3*(diam**3.0) + c0;
    return z

def get_spec_names(spec_group):
    if spec_group == 'all':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'BC', 'H2O']
    elif spec_group == 'dry':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'BC']
    elif spec_group == 'inorg':
        spec_names = ['SO4','NO3','Cl','NH4']
    elif spec_group == 'soa':
        spec_names = ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']
    elif spec_group == 'poa':
        spec_names = ['OC']
    elif spec_group == 'total_org':
        spec_names = ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2','OC']
    elif spec_group == 'bc' or spec_group == 'core' :
        spec_names = ['BC']
    elif spec_group == 'dry_shell':
        spec_names = ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC']
    elif spec_group == 'shell' or spec_group == 'wet_shell':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'H2O']
    return spec_names
    
def get_aero_kappa(spec_names, kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65):
    kap_list = {
        'SO4': kap_inorg,
        'NO3': kap_inorg,
        'Cl': kap_inorg,
        'NH4': kap_inorg,
        'MSA': 0.53,
        'ARO1': kap_soa,
        'ARO2': kap_soa,
        'ALK1': kap_soa,
        'OLE1': kap_soa,
        'API1': kap_soa,
        'API2': kap_soa,
        'LIM1': kap_soa,
        'LIM2': kap_soa,
        'CO3': 0.53,
        'Na': 0.53,
        'Ca': 0.53,
        'OIN': 0.1,
        'OC':  kap_poa,
        'BC': 0.,
        'H2O': 0.}
    
    aero_kappa = np.array([])
    for spec_name in spec_names:
        aero_kappa = np.append(aero_kappa, kap_list.get(spec_name))
    
    return aero_kappa
def get_crossection_Mie(ncfile,  wvl, temp, RH, 
                    return_idx_bc = False,
                    return_this = 'Mie', # other options: BrC_core, BrC_shell, uncoated
                    kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65, 
                    ri_soa = 1.45+0j, ri_poa=1.45+0j, ri_inorg = 1.5+0j, return_Dwet=False, wet_dia=False, return_coatRI=False):    
    if return_this == 'Mie':
        return_Mie = True
        return_BrC_core = False    
        return_BrC_shell = False
        return_uncoated = False
    elif return_this == 'BrC_core':
        return_BrC_core = True        
        return_Mie = False
        return_BrC_shell = False
        return_uncoated = False        
    elif return_this == 'BrC_shell':
        return_BrC_shell = True
        return_Mie = False
        return_BrC_core = False    
        return_uncoated = False        
    elif return_this == 'uncoated':
        return_uncoated = True
        return_Mie = False
        return_BrC_core = False    
        return_BrC_shell = False
    
    mass_comp = get_particle_variable(ncfile, 'aero_particle_mass')    
    density = get_particle_variable(ncfile, 'density')
    
    spec_names = get_spec_names('all')    
    core_dia = get_particle_variable(ncfile,'core_diameter')
    core_spec = get_spec_names('core')    
    idx_core = [np.where(ismember(spec_names, spec))[0][0] for spec in core_spec]
    m_core = get_effective_ri(mass_comp[idx_core,:],density[idx_core], core_spec, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)
    idx_bc, = np.where(core_dia>0)
    optical_crossect = np.zeros([len(core_dia),2])
    if return_uncoated:

        if len(idx_bc)>0:
            optical_crossect[idx_bc,:] = np.vstack([compute_crossect_cs(core_dia[i], core_dia[i], m_core[i], m_core[i], wvl) for i in idx_bc])
            
            
    
    if return_Mie or return_BrC_core or return_BrC_shell:
        spec_names = get_spec_names('dry')        
        tkappa = get_effective_kappa(mass_comp[:-1], density[:-1], spec_names, kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)    
        dry_dia = get_particle_variable(ncfile,'dry_diameter')
        if np.any(wet_dia == False):
            wet_dia = np.vstack([get_Dwet(dry_dia[i], tkappa[i], RH, temp) for i in range(len(dry_dia))])[:,0]
        mass_comp[-1,:] = np.pi/6*(wet_dia**3 - dry_dia**3)*density[-1]
        
        spec_names = get_spec_names('all')
        shell_spec = get_spec_names('shell')
        idx_shell = np.hstack([np.where(ismember(spec_names, spec))[0] for spec in shell_spec])
        m_shell = get_effective_ri(mass_comp[idx_shell,:],density[idx_shell], shell_spec, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)    
        if return_Mie:
            optical_crossect = np.vstack([compute_crossect_cs(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in range(len(dry_dia))])
        if return_BrC_shell:
            optical_crossect = np.vstack([compute_crossect_cs(wet_dia[i], wet_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])            
        if return_BrC_core:
            optical_crossect = np.vstack([compute_crossect_cs(core_dia[i], core_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])            
            
    abs_crossect = optical_crossect[:,0]
    scat_crossect = optical_crossect[:,1]
    
    # need to add in parameterization and uncoated
    if return_idx_bc:
        if return_Dwet:
            if return_coatRI:
                return abs_crossect, scat_crossect, wet_dia, idx_bc
            else:
                return abs_crossect, scat_crossect, wet_dia, m_shell, idx_bc                
        else:
            return abs_crossect, scat_crossect, idx_bc
    else:
        if return_Dwet:
            if return_coatRI:            
                return abs_crossect, scat_crossect, wet_dia, m_shell
            else:
                return abs_crossect, scat_crossect, wet_dia
        else:
            return abs_crossect, scat_crossect
        

def get_crossection_Mie_uniform(ncfile,  wvl, temp, RH, 
                    return_idx_bc = False,
                    return_this = 'Mie', # other options: BrC_core, BrC_shell, uncoated
                    kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65, 
                    ri_soa = 1.45+0j, ri_poa=1.45+0j, ri_inorg = 1.5+0j, return_Dwet=False, wet_dia=False):
    if return_this == 'Mie':
        return_Mie = True
        return_BrC_core = False    
        return_BrC_shell = False
        return_uncoated = False
    elif return_this == 'BrC_core':
        return_BrC_core = True        
        return_Mie = False
        return_BrC_shell = False
        return_uncoated = False        
    elif return_this == 'BrC_shell':
        return_BrC_shell = True
        return_Mie = False
        return_BrC_core = False    
        return_uncoated = False        
    elif return_this == 'uncoated':
        return_uncoated = True
        return_Mie = False
        return_BrC_core = False    
        return_BrC_shell = False
        
    mass_comp = get_particle_variable(ncfile, 'aero_particle_mass_uniform')    
    density = get_particle_variable(ncfile, 'density')
    
    spec_names = get_spec_names('all')    
    core_dia = get_particle_variable(ncfile,'core_diameter')
    core_spec = get_spec_names('core')    
    idx_core = [np.where(ismember(spec_names, spec))[0][0] for spec in core_spec]
    m_core = get_effective_ri(mass_comp[idx_core,:],density[idx_core], core_spec, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)    
    idx_bc, = np.where(core_dia>0)
    
    optical_crossect = np.zeros([len(core_dia),2])
    if return_uncoated:

        if len(idx_bc)>0:
            optical_crossect[idx_bc,:] = np.vstack([compute_crossect_cs(core_dia[i], core_dia[i], m_core[i], m_core[i], wvl) for i in idx_bc])
            
    
    if return_Mie or return_BrC_core or return_BrC_shell:
        spec_names = get_spec_names('dry')        
        tkappa = get_effective_kappa(mass_comp[:-1], density[:-1], spec_names, kap_soa = kap_soa, kap_poa = kap_poa, kap_inorg = kap_inorg)    
        dry_dia = get_particle_variable(ncfile,'dry_dia_uniform')
        if np.any(wet_dia == False):
            wet_dia = np.vstack([get_Dwet(dry_dia[i], tkappa[i], RH, temp) for i in range(len(dry_dia))])[:,0]
        mass_comp[-1,:] = np.pi/6*(wet_dia**3 - dry_dia**3)*density[-1]
        
        spec_names = get_spec_names('all')
        shell_spec = get_spec_names('shell')
        idx_shell = np.hstack([np.where(ismember(spec_names, spec))[0] for spec in shell_spec])
        m_shell = get_effective_ri(mass_comp[idx_shell,:],density[idx_shell], shell_spec, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)    
        
        if return_Mie:
            optical_crossect = np.vstack([compute_crossect_cs(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in range(len(dry_dia))])
        if return_BrC_shell:
            optical_crossect = np.vstack([compute_crossect_cs(wet_dia[i], wet_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])            
        if return_BrC_core:
            optical_crossect = np.vstack([compute_crossect_cs(core_dia[i], core_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])            
            
    abs_crossect = optical_crossect[:,0]
    scat_crossect = optical_crossect[:,1]
    
    # need to add in parameterization and uncoated
    if return_idx_bc:
        if return_Dwet:
            return abs_crossect, scat_crossect, wet_dia, idx_bc
        else:
            return abs_crossect, scat_crossect, idx_bc
    else:
        if return_Dwet:
            return abs_crossect, scat_crossect, wet_dia
        else:
            return abs_crossect, scat_crossect
        

def compute_crossect_cs(core_dia, wet_dia, m_core, m_shell, wvl):
    import PyMieScatt
    if (core_dia>0) and (wet_dia>core_dia):
        output = PyMieScatt.MieQCoreShell(m_core, m_shell, wvl, core_dia, wet_dia)#, asCrossSection=True)
        scat_crossect = output[1]*np.pi*wet_dia**2/4.
        abs_crossect = output[2]*np.pi*wet_dia**2/4.
    elif core_dia == 0:
        output = PyMieScatt.MieQ(m_shell, wvl, wet_dia)#, asCrossSection = True)
        scat_crossect = output[1]
        if np.imag(m_shell) == 0:
            abs_crossect = 0.
        else:
            abs_crossect = output[2]*np.pi*wet_dia**2/4.
    else:
        output = PyMieScatt.MieQ(m_core, wvl, core_dia)#, asCrossSection = True)
        scat_crossect = output[1]*np.pi*core_dia**2/4.
        abs_crossect = output[2]*np.pi*core_dia**2/4.
        
    return abs_crossect, scat_crossect

def get_effective_kappa(mass_comp, density, spec_names, kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65):
    aero_kappa = get_aero_kappa(spec_names, kap_soa = 0.1, kap_poa = 1e-3, kap_inorg = 0.65);
    effective_kappa = sum([aero_kappa[k]*mass_comp[k,:]/density[k] for k in range(len(spec_names))])/sum([mass_comp[k,:]/density[k] for k in range(len(spec_names))])    
    return effective_kappa

def get_effective_ri(mass_comp, density, spec_names, wvl, ri_soa = 1.45 + 0j, ri_poa = 1.45 + 0j, ri_inorg = 1.5 + 0j):
    aero_ri = get_spec_ri(spec_names, wvl, ri_soa = ri_soa, ri_poa = ri_poa, ri_inorg = ri_inorg)    
    effective_ri = (
            sum([np.real(aero_ri[k])*mass_comp[k,:]/density[k] for k in range(len(spec_names))])/sum([mass_comp[k,:]/density[k] for k in range(len(spec_names))])    
            + 1j*sum([np.imag(aero_ri[k])*mass_comp[k,:]/density[k] for k in range(len(spec_names))])/sum([mass_comp[k,:]/density[k] for k in range(len(spec_names))]))
    return effective_ri

def get_spec_ri(spec_names, wvl, ri_oin=1.5 + 0j, ri_soa = 1.45 + 0j, ri_poa = 1.45 + 0j, ri_inorg = 1.5 + 0j):
    # soot
    wvl_all = 1e-9*np.array([300,350,400,500,530,600,690,800,860,900,1060,1200,1400,1670,1800,2000,2350,2500,3000,3390,3510,3750,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500]);
    real_ri = np.array([1.8,1.8,1.8,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.91,1.92,1.94,1.97,1.98,2.02,2.07,2.08,2.1,2.13,2.14,2.16,2.18,2.23,2.26,2.28,2.31,2.39,2.34,2.4,2.36,2.32,2.34,2.36,2.4,2.42,2.43,2.44,2.5,2.61,2.65,2.64,2.62,2.6]);
    imag_ri = np.array([0.74,0.74,0.74,0.74,0.74,0.72,0.7,0.69,0.69,0.68,0.68,0.68,0.68,0.68,0.68,0.72,0.72,0.72,0.72,0.72,0.735,0.74,0.77,0.77,0.77,0.75,0.7,0.71,0.72,0.76,0.72,0.73,0.82,0.85,1,1.02,1.03,1.04,1.05,1.05,1.06,1.07,1.07,1.08]);
    ri_bc = np.interp(wvl, wvl_all, real_ri) + 1j*np.interp(wvl, wvl_all, imag_ri)

    # h2o
    wvl_all = 1e-9*np.array([10,11,12,13,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,57.9,60,61.9,64,66.1,68.1,70,71.9,74,76,78,80,82,84,85.9,87.9,90,92,94,95.9,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,496,500,505,511,515,520,525,530,535,540,545,550,555,560,565,570,575,579,585,590,596,600,605,610,615,619,625,630,635,640,646,650,655,661,665,670,675,681,686,690,695,700,705,710,715,719,724,730,735,740,745,750,755,760,766,769,775,780,785,791,794,800,805,809,815,820,824,830,836,840,845,849,855,859,865,871,875,879,885,889,895,900,906,910,914,920,925,929,935,940,944,951,955,959,966,971,975,980,984,991,995,1000,1010,1020,1030,1040,1050,1060,1070,1080,1090,1100,1110,1120,1130,1140,1150,1160,1170,1180,1190,1200,1210,1220,1230,1240,1250,1260,1270,1280,1290,1300,1310,1320,1330,1340,1350,1360,1370,1380,1390,1400,1410,1420,1430,1440,1450,1460,1470,1480,1490,1500,1510,1520,1530,1540,1550,1560,1570,1580,1590,1600,1610,1620,1630,1640,1650,1660,1670,1680,1690,1700,1710,1720,1730,1740,1750,1760,1770,1780,1790,1800,1810,1820,1830,1840,1850,1860,1870,1880,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100,2110,2120,2130,2140,2150,2160,2170,2180,2190,2200,2210,2220,2230,2240,2250,2260,2270,2280,2290,2300,2310,2320,2330,2340,2350,2360,2370,2380,2390,2400,2410,2420,2430,2440,2450,2460,2470,2480,2490,2500,2510,2520,2530,2540,2550,2560,2570,2580,2580,2590,2590,2610,2610,2620,2620,2630,2640,2650,2660,2660,2670,2670,2680,2690,2700,2700,2710,2720,2720,2730,2740,2750,2750,2760,2770,2780,2790,2790,2800,2810,2820,2830,2830,2840,2850,2860,2860,2870,2880,2890,2900,2900,2920,2920,2930,2940,2950,2960,2970,2980,2990,3000,3050,3100,3150,3200,3250,3300,3350,3400,3450,3500,3550,3600,3650,3700,3750,3800,3850,3900,3950,4000,4050,4100,4150,4200,4250,4300,4350,4400,4450,4500,4550,4600,4650,4700,4750,4800,4850,4900,4960,5000,5050,5110,5150,5200,5250,5300,5350,5400,5450,5500,5550,5600,5650,5700,5750,5790,5850,5900,5960,6000,6050,6100,6150,6190,6250,6300,6350,6400,6460,6500,6550,6610,6650,6700,6750,6810,6860,6900,6950,7000,7050,7100,7150,7190,7240,7300,7350,7400,7450,7500,7550,7600,7660,7690,7750,7800,7850,7910,7940,8000,8050,8090,8150,8200,8240,8300,8360,8400,8450,8490,8550,8590,8650,8710,8750,8790,8850,8890,8950,9000,9060,9100,9140,9200,9250,9290,9350,9400,9440,9510,9550,9590,9660,9710,9750,9800,9840,9910,9950,10000,10100,10100,10100,10200,10300,10300,10400,10400,10500,10500,10500,10600,10600,10700,10700,10800,10800,10900,10900,11000,11000,11100,11100,11200,11300,11300,11400,11400,11500,11500,11600,11600,11600,11700,11800,11800,11900,11900,11900,12000,12100,12100,12200,12200,12300,12300,12400,12400,12500,12500,12600,12600,12700,12700,12800,12800,12900,12900,12900,13000,13100,13100,13200,13200,13200,13300,13300,13400,13500,13500,13600,13600,13700,13700,13700,13800,13800,13900,14000,14000,14100,14100,14200,14200,14300,14300,14400,14400,14500,14500,14600,14600,14700,14700,14800,14800,14900,14900,15000,15000,15100,15100,15100,15200,15200,15300,15400,15400,15500,15500,15600,15600,15700,15700,15700,15800,15900,15900,16000,16000,16000,16100,16100,16200,16300,16300,16400,16400,16400,16500,16600,16600,16600,16700,16800,16800,16900,16900,16900,17000,17100,17100,17100,17200,17300,17300,17300,17400,17500,17500,17500,17600,17700,17700,17700,17800,17900,17900,18000,18000,18000,18100,18200,18200,18200,18300,18400,18400,18500,18500,18500,18600,18700,18700,18800,18800,18800,18900,19000,19000,19100,19100,19100,19200,19200,19300,19400,19400,19500,19500,19500,19600,19600,19700,19800,19800,19900,19900,20000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000,43000,44000,45000,46000,47000,48000,49000,50000,52000,54000,56000,57900,60000,61900,64000,66100,68100,70000,71900,74000,76000,78000,80000,82000,84000,85900,87900,90000,92000,94000,95900,98000,1e+05,1.1e+05,1.2e+05,1.3e+05,1.4e+05,1.5e+05,1.6e+05,1.7e+05,1.8e+05,1.9e+05,2e+05,2.1e+05,2.2e+05,2.3e+05,2.4e+05,2.5e+05,2.6e+05,2.7e+05,2.8e+05,2.9e+05,3e+05,3.2e+05,3.4e+05,3.6e+05,3.8e+05,4e+05,4.2e+05,4.4e+05,4.6e+05,4.8e+05,5e+05,5.2e+05,5.4e+05,5.6e+05,5.79e+05,6e+05,6.5e+05,7e+05,7.5e+05,8e+05,8.49e+05,9e+05,9.51e+05,1e+06,1.1e+06,1.2e+06,1.3e+06,1.4e+06,1.5e+06,1.6e+06,1.7e+06,1.8e+06,1.9e+06,2e+06,2.1e+06,2.2e+06,2.3e+06,2.4e+06,2.5e+06,2.6e+06,2.7e+06,2.8e+06,2.9e+06,3e+06,3.1e+06,3.2e+06,3.3e+06,3.4e+06,3.5e+06,3.6e+06,3.7e+06,3.8e+06,3.9e+06,4e+06,4.1e+06,4.2e+06,4.3e+06,4.4e+06,4.5e+06,4.6e+06,4.7e+06,4.8e+06,4.9e+06,5e+06,5.1e+06,5.2e+06,5.3e+06,5.4e+06,5.5e+06,5.6e+06,5.7e+06,5.8e+06,5.9e+06,6e+06,6.1e+06,6.2e+06,6.3e+06,6.4e+06,6.5e+06,6.6e+06,6.7e+06,6.8e+06,6.9e+06,7e+06,7.1e+06,7.2e+06,7.3e+06,7.4e+06,7.5e+06,7.6e+06,7.7e+06,7.8e+06,7.9e+06,8e+06,8.1e+06,8.2e+06,8.3e+06,8.4e+06,8.5e+06,8.6e+06,8.7e+06,8.8e+06,8.9e+06,9e+06,9.1e+06,9.2e+06,9.3e+06,9.4e+06,9.5e+06,9.6e+06,9.7e+06,9.8e+06,9.9e+06,1e+07,1.1e+07,1.2e+07,1.3e+07,1.4e+07,1.5e+07,1.6e+07,1.7e+07,1.8e+07,1.9e+07,2e+07,2.1e+07,2.2e+07,2.3e+07,2.4e+07,2.5e+07,2.6e+07,2.7e+07,2.8e+07,2.9e+07,3e+07,3.1e+07,3.2e+07,3.3e+07,3.4e+07,3.5e+07,3.6e+07,3.7e+07,3.8e+07,3.9e+07,4e+07,4.1e+07,4.2e+07,4.3e+07,4.4e+07,4.5e+07,4.6e+07,4.7e+07,4.8e+07,4.9e+07,5e+07,5.1e+07,5.2e+07,5.3e+07,5.4e+07,5.5e+07,5.6e+07,5.7e+07,5.8e+07,5.9e+07,6e+07,6.1e+07,6.2e+07,6.3e+07,6.4e+07,6.5e+07,6.6e+07,6.7e+07,6.8e+07,6.9e+07,7e+07,7.1e+07,7.2e+07,7.3e+07,7.4e+07,7.5e+07,7.6e+07,7.7e+07,7.8e+07,7.9e+07,8e+07,8.1e+07,8.2e+07,8.3e+07,8.4e+07,8.5e+07,8.6e+07,8.7e+07,8.8e+07,8.9e+07,9e+07,9.1e+07,9.2e+07,9.3e+07,9.4e+07,9.5e+07,9.6e+07,9.7e+07,9.8e+07,9.9e+07,1e+08,1.1e+08,1.2e+08,1.3e+08,1.4e+08,1.5e+08,1.6e+08,1.7e+08,1.8e+08,1.9e+08,2e+08,2.1e+08,2.2e+08,2.3e+08,2.4e+08,2.5e+08,2.6e+08,2.7e+08,2.8e+08,2.9e+08,3e+08,3.1e+08,3.2e+08,3.3e+08,3.4e+08,3.5e+08,3.6e+08,3.7e+08,3.8e+08,3.9e+08,4e+08,4.1e+08,4.2e+08,4.3e+08,4.4e+08,4.5e+08,4.6e+08,4.7e+08,4.8e+08,4.9e+08,5e+08,5.11e+08,5.2e+08,5.3e+08,5.4e+08,5.5e+08,5.6e+08,5.7e+08,5.79e+08,5.9e+08,6e+08,6.1e+08,6.19e+08,6.3e+08,6.4e+08,6.5e+08,6.61e+08,6.7e+08,6.81e+08,6.9e+08,7e+08,7.1e+08,7.19e+08,7.3e+08,7.4e+08,7.5e+08,7.6e+08,7.69e+08,7.8e+08,7.91e+08,8e+08,8.09e+08,8.2e+08,8.3e+08,8.4e+08,8.49e+08,8.59e+08,8.71e+08,8.79e+08,8.89e+08,9e+08,9.1e+08,9.2e+08,9.29e+08,9.4e+08,9.51e+08,9.59e+08,9.71e+08,9.8e+08,9.91e+08,1e+09,1.1e+09,1.2e+09,1.3e+09,1.4e+09,1.5e+09,1.6e+09,1.7e+09,1.8e+09,1.9e+09,2e+09,2.1e+09,2.2e+09,2.3e+09,2.4e+09,2.5e+09,2.6e+09,2.7e+09,2.8e+09,2.9e+09,3e+09,3.1e+09,3.2e+09,3.3e+09,3.4e+09,3.5e+09,3.6e+09,3.7e+09,3.8e+09,3.9e+09,4e+09,4.1e+09,4.2e+09,4.3e+09,4.4e+09,4.5e+09,4.6e+09,4.7e+09,4.8e+09,4.9e+09,5e+09,5.11e+09,5.2e+09,5.3e+09,5.4e+09,5.5e+09,5.6e+09,5.7e+09,5.79e+09,5.9e+09,6e+09,6.1e+09,6.19e+09,6.3e+09,6.4e+09,6.5e+09,6.61e+09,6.7e+09,6.81e+09,6.9e+09,7e+09,7.1e+09,7.19e+09,7.3e+09,7.4e+09,7.5e+09,7.6e+09,7.69e+09,7.8e+09,7.91e+09,8e+09,8.09e+09,8.2e+09,8.3e+09,8.4e+09,8.49e+09,8.59e+09,8.71e+09,8.79e+09,8.89e+09,9e+09,9.1e+09,9.2e+09,9.29e+09,9.4e+09,9.51e+09,9.59e+09,9.71e+09,9.8e+09,9.91e+09,1e+10]);
    real_ri = np.array([0.96842,0.96478,0.96095,0.95695,0.95279,0.94412,0.93474,0.92458,0.91397,0.90269,0.89084,0.87877,0.86649,0.85414,0.84217,0.83072,0.81975,0.81,0.80229,0.79774,0.79701,0.80558,0.82074,0.83096,0.83524,0.8353,0.83163,0.8309,0.84057,0.86699,0.90353,0.9418,0.98169,1.0209,1.0497,1.0687,1.0877,1.1117,1.1406,1.1734,1.215,1.2595,1.3027,1.3468,1.3876,1.4254,1.4559,1.4766,1.4935,1.5067,1.5163,1.5236,1.5289,1.5354,1.5432,1.5481,1.5534,1.5609,1.5703,1.5846,1.6061,1.6268,1.6338,1.6194,1.5863,1.5364,1.4963,1.4711,1.4615,1.461,1.4693,1.4896,1.5213,1.5599,1.5969,1.6204,1.6415,1.6502,1.6529,1.6531,1.6472,1.6351,1.6056,1.5682,1.5494,1.5431,1.5133,1.4919,1.4752,1.4625,1.4517,1.4423,1.4347,1.4278,1.4216,1.4159,1.4107,1.4064,1.4023,1.3985,1.395,1.3917,1.3889,1.3862,1.3837,1.3813,1.3791,1.3769,1.3751,1.3731,1.3714,1.3698,1.3683,1.3668,1.3654,1.364,1.3626,1.3615,1.3604,1.3592,1.3582,1.3572,1.3563,1.3554,1.3545,1.3536,1.3527,1.352,1.3512,1.3504,1.3498,1.3492,1.3484,1.3478,1.3472,1.3466,1.3461,1.3455,1.345,1.3444,1.3439,1.3434,1.3429,1.3425,1.342,1.3415,1.3411,1.3406,1.3402,1.3398,1.3394,1.3391,1.3386,1.3383,1.3379,1.3376,1.3373,1.3369,1.3366,1.3363,1.336,1.3357,1.3353,1.335,1.3347,1.3344,1.3342,1.3339,1.3336,1.3333,1.3331,1.3328,1.3326,1.3323,1.3321,1.3318,1.3316,1.3313,1.3311,1.3309,1.3307,1.3305,1.3302,1.3301,1.3299,1.3297,1.3295,1.3293,1.3291,1.3289,1.3288,1.3286,1.3284,1.3283,1.3281,1.328,1.3278,1.3277,1.3275,1.3274,1.3272,1.3271,1.3269,1.3268,1.3267,1.3265,1.3264,1.3262,1.3261,1.326,1.3259,1.3257,1.3256,1.3255,1.3254,1.3253,1.3252,1.325,1.3249,1.3248,1.3247,1.3246,1.3245,1.3244,1.3242,1.3242,1.3241,1.3239,1.3239,1.3237,1.3236,1.3235,1.3234,1.3234,1.3232,1.3231,1.3231,1.3229,1.3228,1.3228,1.3226,1.3225,1.3225,1.3223,1.3222,1.3222,1.3221,1.322,1.3219,1.3218,1.3217,1.3215,1.3213,1.3211,1.321,1.3208,1.3206,1.3204,1.3202,1.3201,1.3199,1.3197,1.3195,1.3193,1.3191,1.3189,1.3188,1.3186,1.3184,1.3182,1.318,1.3178,1.3176,1.3174,1.3173,1.317,1.3169,1.3166,1.3165,1.3162,1.3161,1.3159,1.3156,1.3154,1.3152,1.315,1.3148,1.3145,1.3143,1.3141,1.3139,1.3137,1.3135,1.3134,1.3132,1.3131,1.3129,1.3127,1.3125,1.3123,1.3121,1.3119,1.3116,1.3114,1.3111,1.3109,1.3107,1.3104,1.3101,1.3099,1.3096,1.3094,1.3091,1.3089,1.3085,1.3083,1.308,1.3077,1.3074,1.3071,1.3068,1.3065,1.3061,1.3058,1.3054,1.3051,1.3047,1.3044,1.3042,1.3037,1.3034,1.3029,1.3026,1.3023,1.3017,1.3013,1.3006,1.3002,1.2999,1.2995,1.299,1.2988,1.2988,1.2987,1.2986,1.2985,1.2983,1.2981,1.2976,1.2973,1.2969,1.2965,1.2961,1.2956,1.2949,1.2945,1.294,1.2935,1.293,1.2924,1.2919,1.2914,1.2908,1.2902,1.2896,1.289,1.2884,1.2878,1.2871,1.2865,1.2858,1.2851,1.2844,1.2836,1.2829,1.2821,1.2813,1.2804,1.2796,1.2787,1.2778,1.2768,1.2763,1.2753,1.2743,1.2732,1.2721,1.2709,1.2697,1.2691,1.2678,1.2665,1.2651,1.2636,1.2628,1.2613,1.2597,1.258,1.2571,1.2553,1.2535,1.2514,1.2504,1.2481,1.2457,1.243,1.2402,1.2387,1.2371,1.2354,1.2337,1.2318,1.2278,1.2255,1.2231,1.2205,1.2181,1.2157,1.21,1.2065,1.203,1.1993,1.1953,1.1914,1.1881,1.18,1.1746,1.1689,1.161,1.1529,1.1495,1.1421,1.1362,1.1329,1.1317,1.1328,1.1309,1.128,1.1276,1.1295,1.1284,1.1255,1.1254,1.1275,1.1333,1.1424,1.1455,1.1523,1.1624,1.1784,1.1854,1.1959,1.208,1.2297,1.24,1.2521,1.2639,1.2859,1.2978,1.3079,1.3263,1.3345,1.3529,1.4119,1.452,1.4668,1.4615,1.4494,1.4326,1.4171,1.4049,1.3933,1.3842,1.3761,1.3689,1.3625,1.3569,1.3519,1.3474,1.344,1.3402,1.3367,1.3339,1.3314,1.3285,1.3262,1.324,1.3219,1.3199,1.3181,1.3164,1.3149,1.3136,1.3125,1.3118,1.3116,1.3115,1.3111,1.3107,1.3097,1.3087,1.3072,1.3059,1.3043,1.302,1.2999,1.2975,1.2949,1.2921,1.289,1.2857,1.2822,1.2783,1.2739,1.2688,1.263,1.2566,1.2484,1.2422,1.2349,1.2293,1.2319,1.2429,1.2685,1.2953,1.3301,1.3416,1.3399,1.3358,1.3292,1.325,1.3205,1.3177,1.3148,1.3114,1.309,1.3067,1.3045,1.3019,1.3001,1.2984,1.2968,1.2952,1.2936,1.2921,1.2907,1.2893,1.2879,1.2866,1.2852,1.2839,1.2826,1.2812,1.2799,1.2785,1.2771,1.2762,1.2748,1.2734,1.272,1.2705,1.2696,1.2682,1.2667,1.2657,1.2641,1.2626,1.2615,1.2599,1.2582,1.2571,1.2554,1.2542,1.2524,1.2512,1.2494,1.2474,1.2461,1.2448,1.2428,1.2414,1.2393,1.2379,1.2357,1.2341,1.2327,1.2303,1.2286,1.227,1.2244,1.2227,1.2209,1.2181,1.2161,1.2141,1.2111,1.2089,1.2067,1.2045,1.2022,1.1986,1.1959,1.1932,1.1903,1.1874,1.1839,1.1809,1.1784,1.1742,1.1708,1.1674,1.164,1.1606,1.1572,1.1538,1.1504,1.147,1.1436,1.1403,1.1374,1.1344,1.1314,1.1286,1.1255,1.122,1.1188,1.1161,1.1133,1.1103,1.1077,1.1054,1.1031,1.1007,1.0975,1.0966,1.0961,1.0943,1.0923,1.0906,1.0891,1.0865,1.0862,1.0875,1.0879,1.088,1.0867,1.0872,1.0897,1.0909,1.0913,1.0924,1.0956,1.098,1.0996,1.1008,1.1046,1.1074,1.109,1.1103,1.1142,1.1168,1.1183,1.1221,1.1248,1.1265,1.1306,1.1338,1.1358,1.1395,1.1414,1.1459,1.1496,1.1516,1.1563,1.1602,1.1619,1.1658,1.1679,1.172,1.1741,1.1785,1.1825,1.1847,1.1891,1.1914,1.1956,1.1976,1.2016,1.2036,1.2075,1.2094,1.2136,1.2153,1.2188,1.221,1.2256,1.2276,1.2316,1.2336,1.2375,1.2394,1.2433,1.2453,1.2494,1.2517,1.2536,1.257,1.2589,1.2632,1.2651,1.2684,1.2704,1.2726,1.2765,1.2782,1.2826,1.2847,1.2866,1.2906,1.2927,1.2947,1.2989,1.3013,1.3031,1.3066,1.3085,1.3131,1.3153,1.3171,1.3209,1.3227,1.3246,1.3268,1.3309,1.3331,1.3349,1.3389,1.3411,1.3429,1.3475,1.3497,1.3512,1.3528,1.3568,1.3593,1.3611,1.3647,1.3672,1.3692,1.3708,1.3745,1.3768,1.3786,1.38,1.3837,1.3859,1.3877,1.3894,1.3908,1.3943,1.3964,1.3982,1.3998,1.4011,1.4046,1.4068,1.4087,1.4104,1.4121,1.4153,1.4175,1.4198,1.4216,1.4228,1.4262,1.4283,1.43,1.4312,1.4328,1.4346,1.4359,1.4396,1.4416,1.4428,1.4442,1.4455,1.4467,1.4475,1.4503,1.4522,1.4538,1.4556,1.4569,1.4577,1.4587,1.4597,1.4604,1.4633,1.4654,1.4665,1.467,1.4672,1.4676,1.4837,1.4994,1.5164,1.5293,1.538,1.5441,1.5467,1.5463,1.5427,1.5355,1.5272,1.5191,1.5119,1.5059,1.4989,1.493,1.4867,1.481,1.4782,1.4766,1.4756,1.4772,1.4807,1.4853,1.4915,1.4994,1.5088,1.5203,1.5315,1.5423,1.5675,1.5941,1.6192,1.6437,1.6691,1.6902,1.7098,1.7294,1.7473,1.7628,1.7772,1.7908,1.8055,1.8191,1.8309,1.8423,1.8519,1.8599,1.8673,1.8742,1.8805,1.8863,1.8914,1.8954,1.8991,1.9075,1.9117,1.92,1.9274,1.9342,1.9417,1.9484,1.9557,1.9652,1.9746,1.9834,1.9923,2.0014,2.0104,2.0203,2.0292,2.0372,2.0451,2.0525,2.0598,2.074,2.087,2.0995,2.1128,2.1257,2.1395,2.1532,2.1663,2.1773,2.1887,2.2003,2.2109,2.2204,2.2283,2.2367,2.2546,2.2701,2.2902,2.3126,2.3372,2.3639,2.3853,2.3991,2.4368,2.4812,2.5275,2.5773,2.6291,2.6791,2.7293,2.7819,2.832,2.8819,2.9339,2.9833,3.0324,3.084,3.1335,3.1799,3.229,3.2795,3.3266,3.3746,3.4225,3.4682,3.5169,3.5633,3.6071,3.6501,3.6952,3.7419,3.7851,3.8295,3.8736,3.917,3.9606,4.0036,4.0451,4.0849,4.1258,4.1665,4.2076,4.2484,4.2888,4.3283,4.3672,4.4037,4.4424,4.4826,4.5187,4.5558,4.5936,4.6311,4.6677,4.7045,4.7404,4.7765,4.8114,4.8441,4.881,4.9152,4.9464,4.9798,5.014,5.0478,5.0834,5.116,5.1462,5.1772,5.2095,5.2415,5.2732,5.3049,5.3363,5.3674,5.3983,5.4289,5.4592,5.4893,5.519,5.5485,5.5777,5.6066,5.6352,5.6635,5.6915,5.7193,5.7467,5.7738,5.8006,5.8272,5.8534,5.8794,6.1319,6.346,6.5381,6.7111,6.8672,7.008,7.1357,7.2524,7.3588,7.4559,7.5444,7.6256,7.7011,7.7689,7.8312,7.8896,7.9413,7.9894,8.0338,8.0745,8.1122,8.1471,8.1798,8.2098,8.2383,8.2646,8.2884,8.3113,8.3328,8.3527,8.372,8.3892,8.4052,8.4209,8.4357,8.4499,8.4625,8.4747,8.4865,8.4973,8.5078,8.5175,8.527,8.5359,8.5456,8.5547,8.5634,8.5716,8.5796,8.5872,8.5944,8.6013,8.6079,8.6144,8.6205,8.6263,8.632,8.6374,8.6426,8.6477,8.6526,8.6572,8.6617,8.666,8.6703,8.6743,8.6782,8.682,8.6856,8.6892,8.6926,8.6959,8.6992,8.7024,8.7055,8.7086,8.7115,8.7144,8.7171,8.7198,8.7224,8.725,8.7275,8.7299,8.7322,8.7345,8.7368,8.7389,8.741,8.7431,8.7614,8.7761,8.7881,8.7978,8.8043,8.8097,8.8141,8.8179,8.8211,8.8238,8.8262,8.8282,8.8301,8.8317,8.8331,8.8343,8.8354,8.8365,8.8374,8.8381,8.8388,8.8394,8.84,8.8405,8.841,8.8414,8.8418,8.8422,8.8425,8.8428,8.8431,8.8433,8.8436,8.8438,8.844,8.8443,8.8444,8.8446,8.8448,8.8449,8.8451,8.8452,8.8453,8.8455,8.8456,8.8457,8.8458,8.8459,8.846,8.8461,8.8462,8.8462,8.8463,8.8464,8.8465,8.8465,8.8466,8.8467,8.8467,8.8468,8.8468,8.8469,8.8469,8.847,8.847,8.8471,8.8471,8.8471,8.8472,8.8472,8.8472,8.8473,8.8473,8.8473,8.8474,8.8474,8.8474,8.8475,8.8475,8.8475,8.8475,8.8476,8.8476,8.8476,8.8476,8.8476,8.8477,8.8477,8.8477,8.8477,8.8479,8.848,8.8481,8.8482,8.8482,8.8483,8.8483,8.8484,8.8484,8.8484,8.8484,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8485,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486,8.8486]);
    imag_ri = np.array([0.00175,0.00237,0.00315,0.00407,0.00517,0.00796,0.0116,0.0164,0.0223,0.0295,0.0382,0.0485,0.0606,0.0746,0.0907,0.109,0.13,0.153,0.18,0.209,0.241,0.277,0.3,0.315,0.331,0.35,0.374,0.412,0.456,0.503,0.536,0.563,0.579,0.586,0.581,0.586,0.598,0.614,0.629,0.645,0.657,0.657,0.653,0.644,0.629,0.605,0.575,0.543,0.519,0.493,0.471,0.449,0.43,0.415,0.399,0.383,0.371,0.36,0.349,0.339,0.322,0.288,0.239,0.187,0.149,0.133,0.142,0.168,0.193,0.217,0.241,0.264,0.277,0.277,0.258,0.234,0.202,0.167,0.135,0.104,0.0724,0.04,0.004,0.002,0.00118,0.000839,6e-05,1.25e-06,3.62e-07,1.85e-07,1.1e-07,6.71e-08,3.84e-08,2e-08,1.27e-08,1.16e-08,1.1e-08,1.07e-08,1.05e-08,9.9e-09,9.31e-09,8.61e-09,7.99e-09,7.44e-09,6.85e-09,6.29e-09,5.79e-09,5.41e-09,4.8e-09,4.4e-09,4.15e-09,3.83e-09,3.55e-09,3.33e-09,3.19e-09,3.08e-09,2.98e-09,2.88e-09,2.77e-09,2.65e-09,2.53e-09,2.42e-09,2.32e-09,2.22e-09,2.12e-09,2.03e-09,1.94e-09,1.84e-09,1.76e-09,1.66e-09,1.58e-09,1.49e-09,1.42e-09,1.34e-09,1.26e-09,1.17e-09,1.09e-09,1.02e-09,9.39e-10,8.69e-10,8.09e-10,7.8e-10,7.6e-10,7.5e-10,7.29e-10,7.01e-10,7.09e-10,7.16e-10,7.34e-10,7.85e-10,9.24e-10,1.08e-09,1.27e-09,1.46e-09,1.57e-09,1.64e-09,1.76e-09,1.89e-09,2.1e-09,2.27e-09,2.44e-09,2.66e-09,2.87e-09,3.13e-09,3.43e-09,3.84e-09,4.43e-09,5.22e-09,6.37e-09,7.72e-09,9.63e-09,1.13e-08,1.24e-08,1.33e-08,1.4e-08,1.47e-08,1.5e-08,1.55e-08,1.57e-08,1.61e-08,1.67e-08,1.78e-08,1.94e-08,2.03e-08,2.1e-08,2.18e-08,2.3e-08,2.47e-08,2.65e-08,2.96e-08,3.35e-08,4.1e-08,5e-08,6e-08,7.29e-08,9.14e-08,1.15e-07,1.35e-07,1.46e-07,1.53e-07,1.56e-07,1.58e-07,1.58e-07,1.57e-07,1.53e-07,1.48e-07,1.41e-07,1.34e-07,1.28e-07,1.26e-07,1.25e-07,1.27e-07,1.33e-07,1.45e-07,1.62e-07,1.82e-07,2.04e-07,2.24e-07,2.46e-07,2.69e-07,2.93e-07,3.15e-07,3.35e-07,3.55e-07,3.75e-07,3.91e-07,4.05e-07,4.23e-07,4.4e-07,4.62e-07,4.86e-07,5.15e-07,5.7e-07,6.7e-07,8.3e-07,1.06e-06,1.37e-06,1.77e-06,2.17e-06,2.56e-06,2.93e-06,3.19e-06,3.36e-06,3.46e-06,3.5e-06,3.48e-06,3.42e-06,3.34e-06,3.25e-06,3.13e-06,3e-06,2.69e-06,2.35e-06,2e-06,1.69e-06,1.42e-06,1.3e-06,1.26e-06,1.33e-06,1.5e-06,1.71e-06,2.04e-06,2.63e-06,3.87e-06,5.95e-06,9.31e-06,1.07e-05,1.12e-05,1.16e-05,1.18e-05,1.2e-05,1.19e-05,1.18e-05,1.16e-05,1.14e-05,1.1e-05,1.08e-05,1.09e-05,1.14e-05,1.22e-05,1.4e-05,1.64e-05,1.91e-05,2.25e-05,2.85e-05,4.05e-05,4.51e-05,5.8e-05,7.8e-05,0.000106,0.000153,0.000254,0.00032,0.000354,0.000363,0.000364,0.00036,0.000339,0.000302,0.000266,0.000225,0.000196,0.000174,0.00016,0.000144,0.000135,0.000124,0.000114,0.000107,9.94e-05,9.35e-05,8.8e-05,8.31e-05,8.1e-05,7.9e-05,7.59e-05,7.4e-05,7.4e-05,7.5e-05,7.6e-05,7.74e-05,8.05e-05,8.41e-05,8.9e-05,9.51e-05,0.0001,0.000105,0.000112,0.000122,0.000133,0.000136,0.000137,0.000138,0.000142,0.000155,0.000186,0.000321,0.000521,0.000722,0.000922,0.00116,0.00168,0.00183,0.00192,0.00191,0.00185,0.00172,0.00155,0.0014,0.00125,0.0011,0.00099,0.000889,0.000805,0.000739,0.000674,0.000621,0.000573,0.000529,0.000488,0.000464,0.00044,0.000418,0.000397,0.000383,0.000371,0.000359,0.000351,0.000343,0.00034,0.000338,0.000339,0.000341,0.000346,0.000357,0.000374,0.00039,0.000408,0.000429,0.000451,0.000469,0.000492,0.000511,0.000543,0.0006,0.000637,0.000685,0.000743,0.000792,0.000849,0.00091,0.00099,0.00107,0.00115,0.00125,0.00135,0.00147,0.00158,0.00171,0.00181,0.0019,0.00195,0.00199,0.00202,0.00207,0.00214,0.00227,0.00231,0.00234,0.00239,0.00243,0.00248,0.00258,0.0027,0.00298,0.0033,0.00402,0.00436,0.00483,0.00537,0.00628,0.00733,0.00855,0.0105,0.0127,0.0145,0.0164,0.0186,0.0205,0.0282,0.038,0.0462,0.0548,0.0648,0.0744,0.0835,0.0929,0.102,0.112,0.121,0.131,0.142,0.154,0.167,0.18,0.194,0.206,0.218,0.229,0.239,0.249,0.258,0.265,0.272,0.276,0.28,0.28,0.282,0.282,0.279,0.276,0.272,0.272,0.24,0.192,0.135,0.0924,0.0611,0.0369,0.0261,0.0195,0.0132,0.00939,0.00679,0.00515,0.00423,0.0036,0.0034,0.0034,0.00353,0.0038,0.00416,0.0046,0.00507,0.00562,0.00622,0.00688,0.0076,0.00845,0.00931,0.0103,0.0114,0.0124,0.0136,0.0147,0.0155,0.0157,0.0155,0.015,0.0144,0.0137,0.0131,0.0124,0.0118,0.0111,0.0106,0.0101,0.0099,0.00979,0.00988,0.0103,0.0108,0.0116,0.0126,0.0142,0.0166,0.0203,0.0248,0.033,0.0432,0.0622,0.0865,0.107,0.125,0.131,0.117,0.0879,0.0695,0.057,0.0495,0.0449,0.0418,0.0393,0.0373,0.0356,0.0345,0.0337,0.0331,0.0327,0.0324,0.0322,0.0321,0.032,0.0319,0.032,0.0321,0.0321,0.0322,0.0322,0.0323,0.0324,0.0325,0.0326,0.0327,0.0328,0.033,0.0331,0.0333,0.0335,0.0337,0.034,0.0341,0.0343,0.0345,0.0347,0.0349,0.0351,0.0353,0.0356,0.0358,0.036,0.0364,0.0365,0.0369,0.0371,0.0375,0.0378,0.0381,0.0384,0.0388,0.0392,0.0395,0.0399,0.0404,0.0407,0.0411,0.0415,0.042,0.0423,0.0429,0.0433,0.0437,0.0443,0.0448,0.0454,0.046,0.0466,0.0472,0.0478,0.0485,0.0493,0.05,0.0508,0.0517,0.0527,0.0538,0.0581,0.0563,0.0585,0.06,0.0619,0.0639,0.0662,0.0685,0.0709,0.0736,0.0765,0.0796,0.0829,0.0865,0.0897,0.0933,0.0968,0.1,0.104,0.108,0.113,0.117,0.122,0.127,0.132,0.137,0.142,0.147,0.152,0.157,0.162,0.168,0.174,0.18,0.187,0.193,0.199,0.206,0.211,0.218,0.224,0.23,0.236,0.242,0.248,0.253,0.259,0.264,0.269,0.274,0.279,0.284,0.288,0.293,0.298,0.301,0.306,0.31,0.314,0.318,0.323,0.326,0.33,0.333,0.336,0.34,0.343,0.347,0.349,0.351,0.355,0.357,0.36,0.362,0.365,0.368,0.37,0.372,0.374,0.376,0.377,0.379,0.381,0.383,0.384,0.386,0.387,0.39,0.392,0.393,0.394,0.396,0.397,0.399,0.4,0.402,0.403,0.404,0.405,0.405,0.407,0.408,0.41,0.41,0.412,0.413,0.414,0.415,0.416,0.418,0.418,0.419,0.42,0.421,0.421,0.423,0.423,0.423,0.423,0.424,0.425,0.425,0.425,0.426,0.426,0.427,0.427,0.428,0.428,0.428,0.429,0.429,0.429,0.43,0.429,0.429,0.429,0.43,0.43,0.429,0.43,0.43,0.429,0.429,0.429,0.429,0.428,0.428,0.428,0.428,0.427,0.427,0.426,0.427,0.426,0.426,0.425,0.425,0.425,0.425,0.424,0.424,0.423,0.423,0.423,0.423,0.422,0.421,0.421,0.42,0.419,0.418,0.418,0.417,0.416,0.416,0.414,0.413,0.412,0.411,0.41,0.409,0.409,0.408,0.407,0.406,0.404,0.404,0.403,0.402,0.401,0.401,0.399,0.397,0.395,0.394,0.393,0.382,0.372,0.363,0.348,0.336,0.323,0.31,0.299,0.289,0.282,0.279,0.28,0.283,0.286,0.292,0.299,0.307,0.319,0.332,0.344,0.359,0.374,0.388,0.403,0.418,0.432,0.447,0.458,0.468,0.477,0.498,0.508,0.516,0.523,0.526,0.525,0.525,0.523,0.521,0.517,0.514,0.51,0.508,0.502,0.496,0.491,0.484,0.477,0.472,0.465,0.46,0.455,0.449,0.443,0.438,0.418,0.417,0.419,0.421,0.426,0.433,0.44,0.451,0.461,0.47,0.478,0.487,0.496,0.506,0.514,0.521,0.528,0.536,0.543,0.551,0.566,0.581,0.597,0.614,0.629,0.645,0.659,0.673,0.685,0.7,0.713,0.724,0.736,0.748,0.762,0.792,0.831,0.873,0.912,0.95,0.984,1.01,1.04,1.12,1.19,1.26,1.33,1.39,1.45,1.51,1.57,1.62,1.67,1.72,1.76,1.81,1.85,1.89,1.93,1.97,2.01,2.04,2.08,2.11,2.14,2.17,2.2,2.23,2.26,2.29,2.31,2.33,2.36,2.38,2.4,2.42,2.45,2.46,2.48,2.5,2.52,2.54,2.55,2.57,2.58,2.6,2.61,2.63,2.64,2.65,2.66,2.67,2.69,2.69,2.71,2.71,2.72,2.73,2.74,2.75,2.75,2.76,2.77,2.77,2.78,2.79,2.79,2.79,2.8,2.8,2.81,2.81,2.81,2.81,2.82,2.82,2.83,2.83,2.83,2.83,2.83,2.83,2.83,2.83,2.84,2.83,2.83,2.84,2.83,2.84,2.83,2.83,2.83,2.81,2.77,2.73,2.69,2.63,2.58,2.52,2.46,2.4,2.34,2.28,2.22,2.17,2.11,2.06,2.01,1.96,1.91,1.87,1.82,1.78,1.74,1.7,1.66,1.63,1.59,1.56,1.53,1.49,1.46,1.43,1.41,1.38,1.35,1.33,1.3,1.28,1.26,1.23,1.21,1.19,1.17,1.15,1.13,1.12,1.1,1.08,1.06,1.05,1.03,1.02,1,0.987,0.973,0.959,0.946,0.933,0.92,0.908,0.896,0.885,0.873,0.862,0.851,0.841,0.83,0.82,0.81,0.801,0.791,0.782,0.773,0.765,0.756,0.748,0.739,0.731,0.724,0.716,0.708,0.701,0.694,0.687,0.68,0.673,0.666,0.66,0.653,0.647,0.641,0.585,0.538,0.498,0.464,0.434,0.407,0.384,0.363,0.345,0.328,0.312,0.298,0.286,0.274,0.263,0.254,0.244,0.236,0.227,0.22,0.213,0.206,0.201,0.194,0.189,0.184,0.179,0.175,0.17,0.166,0.162,0.158,0.155,0.151,0.148,0.145,0.142,0.139,0.136,0.133,0.131,0.128,0.126,0.123,0.121,0.119,0.117,0.115,0.113,0.111,0.109,0.108,0.106,0.104,0.103,0.101,0.1,0.0986,0.097,0.0957,0.0944,0.0929,0.0918,0.0905,0.0893,0.0883,0.0871,0.0859,0.0849,0.0839,0.0828,0.0818,0.0809,0.0798,0.0789,0.078,0.0772,0.0762,0.0755,0.0746,0.0738,0.0731,0.0722,0.0714,0.0708,0.07,0.0693,0.0685,0.068,0.0673,0.0612,0.0562,0.052,0.0483,0.0451,0.0423,0.0398,0.0377,0.0357,0.034,0.0323,0.0309,0.0296,0.0284,0.0272,0.0262,0.0252,0.0244,0.0235,0.0227,0.022,0.0213,0.0207,0.0201,0.0195,0.019,0.0185,0.018,0.0176,0.0171,0.0167,0.0163,0.0159,0.0156,0.0152,0.0149,0.0146,0.0143,0.014,0.0137,0.0135,0.0132,0.013,0.0127,0.0125,0.0123,0.0121,0.0119,0.0117,0.0115,0.0113,0.0111,0.0109,0.0108,0.0106,0.0104,0.0103,0.0102,0.01,0.00986,0.0097,0.00957,0.00946,0.00933,0.0092,0.0091,0.00895,0.00885,0.00875,0.00863,0.00853,0.00843,0.00833,0.00822,0.00812,0.00803,0.00796,0.00785,0.00776,0.00769,0.0076,0.00753,0.00744,0.00736,0.00729,0.00721,0.00714,0.00706,0.00701,0.00693]);
    ri_h2o = np.interp(wvl, wvl_all, real_ri) + 1j*np.interp(wvl, wvl_all, imag_ri)
    
    ri_list = {
        'SO4': ri_inorg,
        'NO3': ri_inorg,
        'Cl': ri_inorg,
        'NH4': ri_inorg,
        'MSA': 1.5 + 0j,            
        'ARO1': ri_soa,
        'ARO2': ri_soa,
        'ALK1': ri_soa,
        'OLE1': ri_soa,
        'API1': ri_soa,
        'API2': ri_soa,
        'LIM1': ri_soa,
        'LIM2': ri_soa,
        'CO3': 1.5 + 0j,
        'Na': 1.5 + 0j,
        'Ca': 1.5 + 0j,
        'OIN': ri_oin,
        'OC':  ri_poa,
        'BC': ri_bc,
        'H2O': ri_h2o}
    aero_ri = np.array([])
    if type(spec_names) == 'str':
        aero_ri = ri_list.get(spec_names)
    else:
        for spec_name in spec_names:
            aero_ri = np.append(aero_ri, ri_list.get(spec_name))
    return aero_ri
    
