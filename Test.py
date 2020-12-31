#!/usr/bin/env python
# -*- coding: utf-8 -*-
from galpy.potential import plotRotcurve
from astropy import units
from galpy.potential import MiyamotoNagaiPotential
import matplotlib.pyplot as plt

mp= MiyamotoNagaiPotential(amp=5*10**10*units.Msun,a=3.*units.kpc,b=300.*units.pc)
plt.figure()
#plotRotcurve(mp,Rrange=[0.01,10.],grid=1001,yrange=[0.,1.2])
mp.plotRotcurve(Rrange=[0.01,10.],grid=1001,yrange=[0.,200])
plt.show()
