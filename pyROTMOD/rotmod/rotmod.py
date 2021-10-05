# -*- coding: future_fstrings -*-

from galpy.potential import MN3ExponentialDiskPotential as MNP, RazorThinExponentialDiskPotential as EP,\
                            TriaxialHernquistPotential as THP
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import hypsecant
from scipy.integrate import quad
from astropy import units
from pyROTMOD.support import convertskyangle,integrate_surface_density
import pyROTMOD.constants as c
import numpy as np
import warnings
def convert_dens_rc(radii, optical_profiles, gas_profile,components,distance =1.,opt_h_z = 0., gas_scaleheight=0., galfit_file =False,vert_mode = None):

    kpc_radii = convertskyangle(radii[2:],distance=distance)
    optical_radii = convertskyangle(np.array(optical_profiles['RADI'][2:]),distance=distance)
    #print(components, [optical_profiles[i][0] for i in range(len(optical_profiles))])
    RCs = [['RADI','KPC',]+list(convertskyangle(radii[2:],distance=distance))]


    ########################### First we convert the optical values and produce a RC at locations of the gas ###################
    for x,type in enumerate(optical_profiles):
        if optical_profiles[type][0] != 'RADI' and optical_profiles[type][1] != 'KM/S':
            if components[x-1][0] in ['expdisk','edgedisk']:
                if not galfit_file:
                    print(f'We have detected the input to be a density profile hence we extropolate from that to get the rotation curve')
                    tmp = CubicSpline(optical_radii,np.array(optical_profiles[type][2:]),extrapolate=True)
                    found_RC= random_density_disk(kpc_radii,tmp(kpc_radii),h_z = opt_h_z,mode = vert_mode)
                else:
                    print(f'We have detected the input to be a galfit file hence we extropolate from that to get the rotation curve')
                    found_RC = exponential_parameter_RC(kpc_radii,components[x-1])

            elif components[x-1][0] in ['sersic']:
                if  galfit_file and (0.75 < components[x-1][4] < 1.25 or 3.75 < components[x-1][4] < 4.25)  :
                    found_RC = sersic_parameter_RC(kpc_radii,components[x-1])
                else:
                    found_RC = None
                    print(f'We have detected the input to be a density profile for a sersic profile that is too complicated for us')
                #found_RC = sercic_RC(optical_radii,np.array(optical_profiles[x][2:]), [components[x-1][3]])
            elif components[x-1][0] in ['bulge']:
                print(f'Assuming a classical bulge spherical profile in a Hernquist profile')
                found_RC = bulge_RC(kpc_radii,optical_radii,np.array(optical_profiles[type][2:]))
            else:
                found_RC = None
                print(f'We do not know how to convert the mass density of {components[x-1][0]}')
            if np.any(found_RC):
                RCs.append([optical_profiles[type][0], 'KM/S']+list(found_RC))
        else:
            if type != 'RADI':
                tmp = CubicSpline(optical_radii,np.array(optical_profiles[type][2:]),extrapolate=True)
                RCs.append(optical_profiles[type][:2]+tmp(kpc_radii))

    ########################### and last the gas which we do not interpolate  ###################
    #gas_scaleheight = 0.46
    #mode = 'sech-sq'
    #gas_scaleheight = 0.46
    #mode = 'sech-sq'
    #mode = None
    if gas_profile[1] != 'KM/S':
        found_RC = random_density_disk(kpc_radii,gas_profile[2:],h_z = gas_scaleheight,mode = vert_mode)
        RCs.append(['DISK_G', 'KM/S']+list(found_RC))
    else:
        RCs.append(gas_profile)

    return RCs


def sersic_RC(radii,density,axis_ratio = 0., eff_radius = 0.):
    # The problem here is that the scale in the bulge is not necessarily the one in the Miyago-Nagai potential hence we estimate it for each point from the density distribution
    if eff_radius ==0.:
        eff_radius = radii[-1]/2.

    scalelength,central,Mass_ind,Mass = get_individual_parameters(radii,density,initial_estimates =[density[0],eff_radius])

    RC = []
    for i in range(len(radii)):
        if scalelength[i] ==0. and central[i] == 0.:
            #print('When the sersic is a point mass')
            #print(c.Gsol,radii[i],c.pc, Mass)
            RC.append(np.sqrt(c.Gsol*Mass/(radii[i]*1000.*(c.pc/(100.*1000.)))))
        else:
            sersic_potential = MNP(amp=float(Mass_ind[i])*units.Msun,a=float(scalelength[i])*units.kpc,b=float(scalelength[i]*axis_ratio)*units.kpc)
            RC.append(sersic_potential.vcirc(radii[i]*units.kpc))
    #print(f'This is what we derive')
    #for i,rad in enumerate(radii):
    #    print(f'rad = {rad}, velocity = {RC[i]}')
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    return RC

def bulge_RC(radii,opt_rad,density,debug=False):
    density =np.array(density)
    density[np.where(density < 0.)] = 0.
    mass,effective_radius=get_effective_radius(opt_rad,density,debug=debug)
    hern_scale = float(effective_radius)/1.8153  #Equation 38 in Hernquist 1990
    bulge_potential = THP(amp=2.*float(mass)*units.Msun,a= hern_scale*units.kpc,b= 1.,c = 1.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = [bulge_potential.vcirc(x*units.kpc) for x in radii]

    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
        # take from the density profile
    return RC

def get_effective_radius(radii,density,debug=False):
    #The effective radius is where the integrated profile reaches half of the total
    ringarea= [0 if radii[0] == 0 else np.pi*((radii[0]+radii[1])/2.)**2]
    ringarea = np.hstack((ringarea,
                         [np.pi*(((y+z)/2.)**2-((y+x)/2.)**2) for x,y,z in zip(radii,radii[1:],radii[2:])],
                         [np.pi*((radii[-1]+0.5*(radii[-1]-radii[-2]))**2-((radii[-1]+radii[-2])/2.)**2)]
                         ))
    ringarea = ringarea*1000.**2
    #print(ringarea,density)
    mass = np.sum([x*y for x,y in zip(ringarea,density)])

    Cuma_prof = []
    for i,rad in enumerate(radii):
        if i == 0:
            Cuma_prof.append(ringarea[i]*density[i])
        else:
            new = Cuma_prof[-1]+ringarea[i]*density[i]
            Cuma_prof.append(new)
    #now to get the half of the total mass.
    return mass,radii[np.where(Cuma_prof<mass/2.)[-1][-1]]




def sersic_parameter_RC(radii,parameters,sech=True):
    # There is no parameterized version for a potetial of all sersic indices as there is no deprojected surface density profile
    # However for n = 4 we can use the hernquist profile, and n =1 the exponential disk, else we need to go through the parameterization of a deprojected rofile of Baes & Gentile 2010

    if 0.75 < float(parameters[4]) < 1.25:
        # assume this to be close enough to a infinitely thin exponential
        hr =  float(parameters[3])/1.678
        exp_disk_potential = EP(amp=float(parameters[2])*units.Msun/units.pc**2,hr=hr*units.kpc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RC = exp_disk_potential.vcirc(radii*units.kpc)
    elif 3.75 < float(parameters[4]) < 4.25:
        # take this to be a bulge and we use a hernequist potential
        hern_scale = float(parameters[3])/1.8153  #Equation 38 in Hernquist 1990
        bulge_potential = THP(amp=2.*float(parameters[1])*units.Msun,a= hern_scale*units.kpc,b= 1.,c = float(parameters[5]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RC = [bulge_potential.vcirc(x*units.kpc) for x in radii]
    else:
        print("This sersic  profile cannot be prameterized.")
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
        # take from the density profile
    return RC

def get_individual_parameters(radii,density,fit_bound=[[-np.inf,-np.inf],[np.inf,np.inf]],initial_estimates = [0.,0.]):
    extended_profile = CubicSpline(radii,density,extrapolate = True,bc_type ='natural')
    scalelength = []
    central = []
    Mass_ind = []
    curr_mass = 0.
    fix_radii = [radii[int(len(radii)/2.-2)],radii[int(len(radii)/2.-1)],radii[int(len(radii)/2.)],radii[int(len(radii)/2.+1)],radii[int(len(radii)/2.+2)]]
    Mass = integrate_surface_density(radii,density)
    s=0.
    sx =0.0
    sy =0.
    sxy = 0.
    sxx = 0.
    rcut =radii[-1]
    delt = rcut-radii[-2]

    for i in range(len(radii)):
        #This is the rotmod method
        if density[i] > 0.:
            s += 1.
            sx +=radii[i]
            sxx += radii[i]**2
            sy += np.log(density[i])
            sxy += radii[i]*np.log(density[i])
        det = s*sxx-sx*sx
        h = det/(sx*sy-s*sxy)
        #print(f"H before correcting = {h}")
        if h > 0.5*rcut:
            h = 0.5*rcut
        if h < 0.1*rcut:
            h = 0.1*rcut
        dens = np.exp((sy*sxx-sx*sxy)/det)
        #print(f"From rotmod we get h = {h} and dens0 = {dens}")
        if radii[i] == 0.:
            if np.isnan(dens):
                dens= density[i]
            if np.isnan(h):
                h = 0.5*rcut
        # The area in the ring
        if i == 0.:
            if radii[i] == 0.:
                pair0 = [-2*radii[i+1],extended_profile(-2*radii[i+1])]
                pair1 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]

            else:
                pair0 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]
                pair1 = [0,extended_profile(0.)]

            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        if i == 1.:
            if radii[i-1] == 0.:

                pair0 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]
            else:
                pair0 = [0,extended_profile(0.)]
            pair1 = [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        elif i == len(radii)-2:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [2.*radii[i+1]-radii[i],extended_profile(2.*radii[i+1]-radii[i])]
        elif i == len(radii)-1:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            outer_extra = 2.*radii[i]-radii[i-1]
            outer_extra2 = outer_extra+(radii[i]-radii[i-1])
            pair3 = [outer_extra,extended_profile(outer_extra)]
            pair4 = [outer_extra2,extended_profile(outer_extra2)]
        else:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        pair2 = [radii[i],density[i]]

        fit_radii = [pair0[0],pair1[0],pair2[0],pair3[0],pair4[0]]
        fit_density = [pair0[1],pair1[1],pair2[1],pair3[1],pair4[1]]
        fit_radii = [pair1[0],pair2[0],pair3[0]]
        fit_density = [pair1[1],pair2[1],pair3[1]]
        #fit_radii = [pair1[0],pair2[0],pair3[0]]
        #fit_density = [pair1[1],pair2[1],pair3[1]]
        #for i,par in enumerate(fit_density):
        #    if par < 1e-8:
        #        fit_density[i] = 0.
        #print('This is what well fit')
        #print(fit_radii,fit_density)

        if np.sum(fit_density) == 0.:
            exp_parameters =[0.,0.]
        else:
            try:
                exp_parameters, exp_covariance = curve_fit(exponential, fit_radii, fit_density,p0=initial_estimates,bounds=fit_bound)
            except RuntimeError:
                exp_parameters =[0.,0.]
            try:
                fit_bound_adj = [[0.,-np.inf],[dens*5.,np.inf]]
                use_rotmod = [dens,h]
                exp_parameters2, exp_covariance2 = curve_fit(exponential, fit_radii, fit_density,p0=use_rotmod,bounds=fit_bound_adj)
            except RuntimeError:
                exp_parameters2 =[0.,0.]

        #print(f"From fitting we get h = {exp_parameters[1]},{exp_parameters2[1]} and dens0 = {exp_parameters[0]},{exp_parameters2[0]}")
        #curr_mass = integrate_surface_density(radii[:i],density[:i])
        if exp_parameters2[1] > 0.5*rcut:
            exp_parameters2[1] = 0.5*rcut
        if exp_parameters2[1] < 0.1*rcut:
            exp_parameters2[1] = 0.1*rcut
        curr_mass = integrate_surface_density(radii,density)
        ok = True
        counter = 0
        while not ok:
            exp_profile =exponential(radii,*exp_parameters)
            #prof_mass = integrate_surface_density(radii[:i],exp_profile[:i])
            prof_mass = integrate_surface_density(radii,exp_profile)
            if curr_mass*0.9 <= prof_mass <= curr_mass*1.1:
                ok = True
            else:
                exp_parameters[0] = exp_parameters[0]* curr_mass/prof_mass
            counter += 1
            if counter > 1000:
                ok =True
        print(f"After correcting h = {exp_parameters[1]} and dens0 = {exp_parameters[0]}")
        scalelength.append(exp_parameters2[1])
        central.append(exp_parameters2[0])#
        #scalelength.append(h)
        #central.append(dens)
        exp_profile =exponential(radii,*exp_parameters)
        #print(exp_profile)
        Mass_ind.append(integrate_surface_density(radii,exp_profile))
    print(central,scalelength)
    print(f'This is the total mass in the exponential disk {Mass:.2e}')
    return scalelength,central,Mass_ind,Mass


# Obtain the velocities of a density profile where the vertical distribution is a exponential disk.
def exponential_RC(radii,density,vertical_distribution,sech= True, log = None):
    # First we need to get the total mass in the disk
    #print(f'We are using this vertical distributions {vertical_distribution}')
    #print(radii,density)
    tot_parameters, tot_covariance = curve_fit(exponential, radii, density,p0=[3.,5])

    fit_bound = [[0.,-np.inf],[tot_parameters[0]*5,np.inf]]
    #fit_bound = [[0,-np.inf],[np.inf,np.inf]]
    #print(f'for the total profile we find h = {tot_parameters[1]} and cent = {tot_parameters[0]}')
    scalelength,central,Mass,Mass_tot = get_individual_parameters(radii,density,fit_bound=fit_bound,initial_estimates = tot_parameters)


    #print('These are the scalelengths and and central densities')
    #print(scalelength,central)
    #print('These are the individual masses')
    #print([f'{x:.2e}' for x in Mass])

    # Now to caluclate the vcirc for each of these parameters
    RC = []
    print(vertical_distribution)
    for i in range(len(radii)):
        print(f"At radius {radii[i]:.2f}, h_R= {scalelength[i]:.2f}, central mass = {central[i]:.2e}")
        if (scalelength[i] ==0. and central[i] == 0.) :
            RC.append(np.sqrt(c.Gsol*Mass_tot/(radii[i]*1000.*c.pc/(100.*1000.))))
        #if we're hitting the edge of the disk things get funny
        #elif central[i] > tot_parameters[0]*3.:
        #    central_dens = (np.mean(central[:i])+central[i])*0.5
        #    h = (np.mean(scalelength[:i])+scalelength[i])*0.5
        #    normal = np.sqrt(c.Gsol*Mass_tot/(radii[i]*1000.*c.pc/(100.*1000.)))
        #    if vertical_distribution >0.:
        #        with warnings.catch_warnings():
        #            warnings.simplefilter("ignore")
        #            area = 2.*quad(sechsquare,0,np.inf,args=(float(vertical_distribution)))[0]
        #        central_now = central_dens/(1000.*area)
        #        exp_disk_potential = MNP(amp=central_now*units.Msun/units.pc**3,hr=float(h)*units.kpc,hz=float(vertical_distribution)*units.kpc,sech=True)
        #    else:
        #        exp_disk_potential = EP(amp=central_dens*units.Msun/units.pc**2,hr=float(h)*units.kpc)
        #    RC.append(np.mean([normal,exp_disk_potential.vcirc(radii[i]*units.kpc)]))
        else:
            if vertical_distribution >0.:
                #with warnings.catch_warnings():
                #    warnings.simplefilter("ignore")
                #    area = 2.*quad(sechsquare,0,np.inf,args=(float(vertical_distribution)))[0]
                #central_now = central[i]/(1000.*area)
                central_now = central[i]/(2.*vertical_distribution*1000.)
                exp_disk_potential = MNP(amp=central_now*units.Msun/units.pc**3,hr=float(scalelength[i])*units.kpc,hz=float(vertical_distribution)*units.kpc,sech=True)
            else:
                exp_disk_potential = EP(amp=central[i]*units.Msun/units.pc**2,hr=float(scalelength[i])*units.kpc)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print('This is the force')
                print(exp_disk_potential.Rforce(radii[i]*units.kpc,0.))
                if -radii[i]*exp_disk_potential.Rforce(radii[i]*units.kpc,0.) < 0.:
                    RC.append(-exp_disk_potential.vcirc(radii[i]*units.kpc))
                else:
                    RC.append(exp_disk_potential.vcirc(radii[i]*units.kpc))

        print(f"Gives this circular velocity {RC[-1]} km/s")
    #print(f'This is what we derive')
    #for i,rad in enumerate(radii):
    #    print(f'rad = {rad}, velocity = {RC[i]}')
    # Take  MiyamotoNagaiPotential from galpy and set scale length between neighbouring points
    # Interpolate with cubic (Akima) spline and fit exponential scale length at each point and calculate V_Circ with given scale height
    # If this works it will allow automatically allow for incorporrating flaring

    # Following Binney and Tremaine (eq 2.29)Centripal force is vc^2/r which means vc =sqrt(r*|F|)
    #   F = dPHI/dr with PHI the gravitational potential
    #vc^2 =

    #RC = np.sqrt(-r*Force)
    #print(radii,density,vertical_distribution)
    #RC = 1.
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    return RC

def sechsquare(x,b):
    #sech(x) = 1./cosh(x)
    return 1./np.cosh(x/b)**2
def exponential_parameter_RC(radii,parameters,sech=True,truncated =-1.,log= False):
    #print(f'This is the total mass in the parameterized exponential disk {float(parameters[1]):.2e}')
    if float(parameters[4]) > 0.:
        #This is not very exact for getting a 3D density
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area = 2.*quad(sechsquare,0,np.inf,args=(float(parameters[4])))[0]
        central = float(parameters[2])/(1000.*area)
        exp_disk_potential = MNP(amp=central*units.Msun/units.pc**3,hr=float(parameters[3])*units.kpc,hz=float(parameters[4])*units.kpc,sech=sech)
    else:
        exp_disk_potential = EP(amp=float(parameters[2])*units.Msun/units.pc**2,hr=float(parameters[3])*units.kpc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = exp_disk_potential.vcirc(radii*units.kpc)
    if truncated > 0.:
        ind =  np.where(radii > truncated)
        RC[ind] = np.sqrt(c.Gsol*float(parameters[1])/(radii[ind]*1000.*c.pc/(100.*1000.)))
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    return RC

def exponential(radii,central,h):
    return central*np.exp(-1.*radii/h)

def get_rotmod_scalelength(radii,density):
    s=0.
    sx =0.0
    sy =0.
    sxy = 0.
    sxx = 0.
    rcut =radii[-1]
    delt = rcut-radii[-2]
    for i in range(len(radii)):
        #This is the rotmod method
        if density[i] > 0.:
            s += 1.
            sx +=radii[i]
            sxx += radii[i]**2
            sy += np.log(density[i])
            sxy += radii[i]*np.log(density[i])
        det = s*sxx-sx*sx
    h = det/(sx*sy-s*sxy)
    if h > 0.5*rcut:
        h = 0.5*rcut
    if h < 0.1*rcut:
        h = 0.1*rcut
    dens = np.exp((sy*sxx-sx*sxy)/det)
    return h,dens


def random_density_disk(radii,density,h_z = 0.,mode = None):
    density = np.array(density) *1e6 # Apparently this is done in M-solar/kpc^2
    rcut = radii[-1]
    delta = 0.2
    ntimes =50.
    h_r,dens0 = get_rotmod_scalelength(radii,density)
    #print(h_r,dens0)
    RC = []
    h_z1 = set_limits(h_z,0.1*h_r,0.3*h_r)
    for i,r in enumerate(radii):
        #looping throught the radii
        vsquared =0.
        r1 = set_limits(r - 3.0 * h_z1, 0, np.inf)
        r2 = 0.
        if r1 < rcut +2.*delta:
            r2 = r+(r-r1)
            #This is very optimized
            ndens = int(6 * ntimes +1)
            step = (r2-r1)/(ndens -1)
            rstart = r1
            vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)
            if r1 > 0.:
                ndens = int(r1*ntimes/h_r)
                ndens =int(2 * ( ndens / 2 ) + 3)
                step = r1 / ( ndens - 1 )
                rstart = 0.
                vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)
        if r2 < ( rcut + 2.0 * delta ):
            ndens = ( rcut + 2.0 * delta - r2 ) * ntimes / h_r
            ndens = int(2 * ( ndens / 2. ) + 3.)
            step = ( rcut + 2.0 * delta - r2 ) / ( ndens - 1 )
            rstart = r2
            vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)

        if vsquared < 0.:
            RC.append(-np.sqrt(-vsquared))
        else:
            RC.append(np.sqrt(vsquared))
    return RC


def integrate_v(radii,density,R,rstart,h_z,step,iterations=200,mode = None ):
    # we cannot actually use quad or some such as the density profile is not a function but user supplied

    vsquared = 0.
    eval_radii = [rstart + step * i for i in range(iterations)]
    eval_density = np.interp(np.array(eval_radii),np.array(radii),np.array(density))
    weights = [4- 2*((i+1)%2) for i in range(iterations)]
    weights[0], weights[-1] = 1., 1.
    weights = np.array(weights,dtype=float)*step
    # ndens = the number of integrations
    for i in range(iterations):
        if 0 < eval_radii[i] < radii[len(density)-1] and eval_density[i] > 0 and R > 0.:
            zdz = integrate_z(R, eval_radii[i], h_z,mode)
            vsquared += 2.* np.pi*c.Grotmod/3. *zdz*eval_density[i]*weights[i]
    return vsquared

def integrate_z(radius,x,h_z,mode):
    if h_z != 0.:
        vertical_function = select_vertical(mode)
        zdz = quad(vertical_function,0,np.inf,args=(radius,x,h_z))[0]
    else:
        if not (radius == x) and not radius == 0. and not x == 0. and \
                    (radius**2 + x**2)/(2.*radius*x) > 1:
                    zdz = interg_function(h_z,radius, x)
        else:
            zdz = 0.
    return zdz

#this function is a carbon copy of the function in GIPSY's rotmod (Begeman 1989, vd Hulst 1992)
# Function to integrate from Cassertano 1983
def interg_function(z,x,y):
    xxx = (x**2 + y**2 + z**2)/(2.*x*y)
    rrr = (xxx**2-1.)
    ppp = 1.0/(xxx+np.sqrt(rrr))
    fm = 1.-ppp**2
    el1 = ( 1.3862944 + fm * ( 0.1119723 + fm * 0.0725296 ) ) - \
          ( 0.5 + fm * ( 0.1213478 + fm * 0.0288729 ) ) * np.log( fm )
    el2 = ( 1.0 + fm * ( 0.4630151 + fm * 0.1077812 ) ) - \
          ( fm * ( 0.2452727 + fm * 0.0412496 ) ) * np.log( fm )
    r = ( ( 1.0 - xxx * y / x ) * el2 / ( rrr ) + \
        ( y / x - ppp ) * el1 / np.sqrt( rrr ) ) /np.pi
    return r * np.sqrt( x / ( y * ppp ) )

def selected_vertical_dist(mode):
    if mode == 'sech-sq':
        return norm_sechsquare
    elif mode == 'exp':
        return norm_exponential
    elif mode == 'sech-simple':
        return simple_sech
    else:
        return None

def combined_rad_sech_square(z,x,y,h_z):
    return interg_function(z,x,y)*norm_sechsquare(z,h_z)
def combined_rad_exp(z,x,y,h_z):
    return interg_function(z,x,y)*norm_exponential(z,h_z)
def combined_rad_sech_simple(z,x,y,h_z):
    return interg_function(z,x,y)*simple_sech(z,h_z)

def select_vertical(mode):
    if mode == 'sech-sq':
        return combined_rad_sech_square
    elif mode == 'exp':
        return combined_rad_exp
    elif mode == 'sech-simple':
        return combined_rad_sech_simple
    else:
        return None

def norm_exponential(radii,h):
    return np.exp(-1.*radii/h)/h
def norm_sechsquare(radii,h):
    return 1./(np.cosh(radii/h)**2*h)
def simple_sech(radii,h):
    return 2./h/np.pi/np.cosh(radii/h)

def set_limits(value,minv,maxv,debug = False):
    if value < minv:
        return minv
    elif value > maxv:
        return maxv
    else:
        return value

set_limits.__doc__ =f'''
 NAME:
    set_limits
 PURPOSE:
    Make sure Value is between min and max else set to min when smaller or max when larger.
 CATEGORY:
    support_functions

 INPUTS:
    value = value to evaluate
    minv = minimum acceptable value
    maxv = maximum allowed value

 OPTIONAL INPUTS:
    debug = False

 OUTPUTS:
    the limited Value

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
