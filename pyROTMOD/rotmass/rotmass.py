# -*- coding: future_fstrings -*-




import numpy as np


def the_action_is_go(radii, Bulge_RC,Disk_RC,Total_RC,debug=False,interactive = False):

    MB = 1.
    MG = 1.
    MD = 1.

    Dark_RC = 100.
    





def composite_RC(MB,MD,MG,BRC,DRC,GRC,Dark):
    return np.sqrt(MB*BRC**2+MD*DRC**2+MG*GRC**2+Dark**2)
