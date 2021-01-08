#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
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
    Bulge_rad,Bulge_Dens,Bulge_Vrot = read_table("Optical_Bulge_Rotmod.txt", skip_line=5)
    Bulge_Vrot = np.array(Bulge_Vrot,dtype=float)
    Bulge_rad  = np.array(Bulge_rad,dtype=float)
    Disk_rad,Disk_Dens,Disk_Vrot = read_table("Optical_Disk_Rotmod.txt", skip_line=6)
    Disk_Vrot = np.array(Disk_Vrot,dtype=float)
    Disk_rad  = np.array(Disk_rad,dtype=float)
    gas_rad,gas_Dens,gas_Vrot = read_table("gas_rotmod.txt", skip_line=6)
    gas_Vrot = np.array(gas_Vrot,dtype=float)
    gas_rad  = np.array(gas_rad,dtype=float)
    gas4_rad,gas4_Dens,gas4_Vrot = read_table("gas_exp0.4_rotmod.txt", skip_line=8)
    gas4_Vrot = np.array(gas4_Vrot,dtype=float)
    gas4_rad  = np.array(gas4_rad,dtype=float)

    rad,disk,bulge,gas_disk,vobs,vobs_err = read_table("pyROTMOD_Output/All_RCs.txt", skip_line=2)

    plt.figure()
    plt.plot(Bulge_rad,Bulge_Vrot,label='Gipsy Bulge')
    plt.plot(Disk_rad,Disk_Vrot,label='Gipsy Disk')
    #plt.plot(gas_rad,gas_Vrot,label='Gipsy Gas Disk')
    #plt.plot(gas4_rad,gas4_Vrot,label='Gipsy Gas Disk exp 0.4 Z')
    #plt.plot(Gas_Disk_rad,Gas_Disk_Vrot,label='Gipsy Gas Disk')
    #plt.plot(rad,vobs,label='Total RC')
    plt.plot(rad,disk,label='Disk RC')
    plt.plot(rad,bulge,label='Bulge RC')
    #plt.plot(rad,gas_disk,label='Gas Disk RC')
    plt.legend()
    plt.savefig('Compare_Gipsy.png')


if __name__ == "__main__":
    main()
