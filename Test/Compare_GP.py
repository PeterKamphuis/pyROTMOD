#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
from galpy.potential import MiyamotoNagaiPotential as MNP
from galpy.potential import RazorThinExponentialDiskPotential as EP
from astropy import units
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt


def read_table(filename,skip_line=0):
    counter =0.
    data= []
    with open(filename) as file:
        while counter < skip_line:
            counter +=1
            h = file.readline()
        counter = 0
        for line in file:
            print(line.split())
            tmp = [float(x.strip()) for x in line.split()]
            for i,val in enumerate(tmp):
                if counter == 0:
                    data.append([val])
                else:
                    data[i].append(val)
            counter +=1
    return data
def main():
    # Read the Gipsy rotmod file
    Bulge_rad,Bulge_Dens,Bulge_Vrot = read_table("Gipsy_Rotmod/Optical_Bulge_Rotmod.txt", skip_line=5)
    Bulge_Vrot = np.array(Bulge_Vrot,dtype=float)
    Bulge_rad  = np.array(Bulge_rad,dtype=float)
    Disk_rad,Disk_Dens,Disk_Vrot = read_table("Gipsy_Rotmod/Optical_Disk_Rotmod.txt", skip_line=10)
    Disk_Vrot = np.array(Disk_Vrot,dtype=float)
    Disk_rad  = np.array(Disk_rad,dtype=float)
    gas_rad,gas_Dens,gas_Vrot = read_table("Gipsy_Rotmod/gas_rotmod.txt", skip_line=6)
    gas_Vrot = np.array(gas_Vrot,dtype=float)
    gas_rad  = np.array(gas_rad,dtype=float)
    #gas4_rad,gas4_Dens,gas4_Vrot = read_table("Gipsy_Rotmod/gas_exp0.4_rotmod.txt", skip_line=8)
    #gas4_Vrot = np.array(gas4_Vrot,dtype=float)
    #gas4_rad  = np.array(gas4_rad,dtype=float)

    rad,disk,disk2,ser_disk,bulge,gas_disk,vobs,vobs_err = read_table("pyROTMOD_products/All_RCs_Thin.txt", skip_line=2)
    mp_disk = MNP(amp=5.28e+08*units.Msun,a=float(2.03)*units.kpc,b=float(0.)*units.kpc)
    razor = EP(amp=22.93175*units.Msun/units.pc**2,hr=float(2.03)*units.kpc)
    plt.figure()
    plt.plot(Bulge_rad,Bulge_Vrot,label='Gipsy Bulge', lw = 5, zorder=1, c='k')
    plt.plot(Disk_rad,Disk_Vrot,label='Gipsy Disk', lw = 5, zorder=1,c= 'b')
    plt.plot(gas_rad,gas_Vrot,label='Gipsy Gas Disk', lw = 5, zorder=1,c='g')
    plt.plot(rad,gas_disk,label='Gas Disk RC',zorder=3,c='r')
    plt.plot(rad,disk,label='Disk RC',zorder=3,c='orange')
    plt.plot(rad,bulge,label='Bulge RC',zorder=3,c= 'grey')
    plt.xlim(-0.01,40)
    plt.ylim(0,50.)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('V$_{\\rm rot}$ (km/s)')
    #plt.plot(rad,gas_disk,label='Gas Disk RC')
    plt.legend()
    plt.savefig('Compare_Gipsy_Infinitely_Thin.png')
    rad,disk,disk2,ser_disk,bulge,gas_disk,vobs,vobs_err = read_table("pyROTMOD_products/All_RCs.txt", skip_line=2)
    rad_OS,disk_OS,disk2_OS,ser_disk_OS,bulge_OS,gas_disk_OS,vobs_OS,vobs_err_OS = read_table("pyROTMOD_products/All_RCs_OS.txt", skip_line=2)
    Disk_rad,Disk_Dens,Disk_Vrot = read_table("Gipsy_Rotmod/disk_rotmod_sech_0.46.txt", skip_line=12)
    Disk_Vrot = np.array(Disk_Vrot,dtype=float)
    Disk_rad  = np.array(Disk_rad,dtype=float)
    gas_rad,gas_Dens,gas_Vrot = read_table("Gipsy_Rotmod/gas_rotmod_sech_0.4.txt", skip_line=8)
    gas_Vrot = np.array(gas_Vrot,dtype=float)
    gas_rad  = np.array(gas_rad,dtype=float)
    gas_rad_OS,gas_Dens_OS,gas_Vrot_OS = read_table("Gipsy_Rotmod/OS_rotmod.txt", skip_line=6)
    gas_Vrot_OS = np.array(gas_Vrot_OS,dtype=float)
    gas_rad_OS  = np.array(gas_rad_OS,dtype=float)
    plt.figure()
    plt.plot(Disk_rad,Disk_Vrot,label='Gipsy Disk sech 0.46', lw = 7, zorder=1,c= 'b')
    plt.plot(gas_rad,gas_Vrot,label='Gipsy Gas Disk sech 0.4', lw = 7, zorder=1,c='g')
    plt.plot(gas_rad_OS,gas_Vrot_OS,label='Gipsy OS Gas Disk thin', lw = 7, zorder=1,c='grey')
    plt.plot(rad,gas_disk,label='Gas Disk RC sech 0.46',zorder=3,c='r')
    plt.plot(rad_OS,gas_disk_OS,label='Gas Disk RC OS z0 = 0.',zorder=3,c='k')
    plt.plot(rad,disk2,label='Disk RC sech',zorder=3,c='orange')

    plt.xlim(-0.01,40)
    plt.ylim(-10,50.)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('V$_{\\rm rot}$ (km/s)')
    #plt.plot(rad,gas_disk,label='Gas Disk RC')
    plt.legend()
    plt.savefig('Compare_Gipsy_Thick.png')

if __name__ == "__main__":
    main()
