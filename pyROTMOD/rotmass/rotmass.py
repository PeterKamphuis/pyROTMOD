# -*- coding: future_fstrings -*-



import numpy as np
import pyROTMOD.rotmass.potentials as V
from sympy import symbols, sqrt,atan,pi,log,Abs

def the_action_is_go(radii, derived_RCs, total_RC,total_RC_err,debug=False,interactive = False,config = None):
    if not config:
            #let's make a dictionary with default settings
            # for masses of disk [Initial M/L, Fixed, Include]
            config = { 'MG': [1.4, True,True],
                       'MD': [1., False,True],
                       'MB': [1., False,False],
                       'HALO': 'NFW'
            }
    type =['MB','MD','MG']
    Baryonic_RC = get_three_RC(radii,derived_RCs,types = type)

    for component in type:
        if len(Baryonic_RC[component]) < 1:
            config[component][2] = False



    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print("Not functional yet")
    else:
        #First we get the DM function we want
        #DM_RC = get_DM_RC(config['HALO'])
        #Then we build the combined rotation curve
        fit_curve,fit_variable,fix_variables = build_curve(config,Baryonic_RC,types = type)


        print(fit_curve)
        #for key in dir(ML_ratios):
        #    print(f" for the {key} we start with M/L {getattr(ML_ratios,key)}")
        '''
        fit_curve = build_curve(config,DM_RC,Bulge_RC,Disk_RC,Gas_RC)
    return np.sqrt(NFW_h * NFW_h + Mg*V_gas*abs(V_gas)  + Md *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )
          return np.sqrt(NFW_h* NFW_h + Mg*V_gas*abs(V_gas)  + ML *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )

        print(DM_RC)
        '''
def build_curve(config,RC,types = ['MG']):

    curve = getattr(V, config['HALO'])()**2
    fit_variables = []
    fixed_variables = []
    Disk_Var = {'MD': 'Vdisk','MG': 'Vgas','MB': 'Vbulge' }
    for component in types:
        if config[component][2]:
            fit_variables.append(component,Disk_Var[component])
            fitM, fitV = symbols(f"{component} {Disk_Var[component]}")
            curve = curve +fitM*fitV*Abs(fitV)
            if config[component][1]:
                fixed_variables.append(component)
    print(curve)
    curve = sqrt(curve)


    return curve,fit_variables,fixed_variables

def get_three_RC(radii,derived_RCs,types =['MG']):
    radii=radii[2:]

    RC ={}
    for component in types:
        RC[component]=[]

    for x in range(len(derived_RCs)):
        if derived_RCs[x][0] == 'RADII':
            rad_in = derived_RCs[x][2:]
        elif derived_RCs[x][0][:3] == 'EXP':
            component = 'MD'
        elif derived_RCs[x][0][:3] == 'SER':
            component = 'MB'
        elif derived_RCs[x][0][:6] == 'DISK_G':
            component = 'MG'
        else:
            print("We do not recognize this type of RC and don't know what to do with it")
            exit()
        if len(RC[component]) < 1:
            RC[component] = derived_RCs[x][2:]
        else:
            RC[component] = [np.sqrt(x**2+y**2) for x,y in zip(RC[component],derived_RCs[x][2:])]

    # if our requested radii do not correspond to the wanted radii we interpolat


    for key in RC:
        if np.sum([float(x)-float(y) for x,y in zip(radii,rad_in)]) != 0.:
            RC[key] = np.array(np.interp(np.array(radii),np.array(rad_in),np.array(RC[key])),dtype=float)
        else:
            RC[key] = np.array(RC[key],dtype=float)

    return RC


def get_DM_RC(name):
    return getattr(V, name)






def composite_RC(MB,MD,MG,BRC,DRC,GRC,Dark):
    return np.sqrt(MB*BRC**2+MD*DRC**2+MG*GRC**2+Dark**2)

#written by Aditya K
def INITIAL_GUESS(f,R,V):
    func=f
    xData=R
    yData=V
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore")
        val = func(xData, *parameterTuple)
        return numpy.sum((yData - val) ** 2.0)
    def generate_Initial_Parameters():
        maxX = max(xData)
        minX = min(xData)
        maxY = max(yData)
        minY = min(yData)
        parameterBounds = []
        parameterBounds.append([0.1, 1000])
        parameterBounds.append([0.1, 1000])
        result = differential_evolution(sumOfSquaredError, parameterBounds)
        return result.x
    geneticParameters = generate_Initial_Parameters()
    fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
    print('Parameters', fittedParameters)
    modelPredictions = func(xData, *fittedParameters)
    absError = modelPredictions - yData
    A, B = fittedParameters
    dA, dB= \
          [np.sqrt(pcov[j,j]) for j in range(fittedParameters.size)]
    f_fit = radius
    s_fit = func(radius,A,B)
    resids = yData - func(radius,A,B)
    SE = numpy.square(absError) # squared errors
    MSE = numpy.mean(SE) # mean squared errors
    RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
    f_fit = r
    s_fit = func(r,A,B)
    resids = yData - func(r,A,B)
    redchisqr = ((resids/V_err)**2).sum()/float(r.size-2)
    if(func==pseudo_Isothermal):
        f = open("Inital_estimates_PIS.txt", "w+")
        print("Inital estimates for Pseudo-Isothermal-Halo: M/L Fixed",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"RC=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==pseudo_Isothermal_Minimum_Disk):
        f = open("Inital_estimates_PIS_MinimumDISK.txt", "w+")
        print("Inital estimates for Pseudo-Isothermal-Halo: Minimum Disk",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"RC=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==NFW):
        f = open("Inital_estimates_NFW.txt", "w+")
        print("Inital estimates for NFW-Halo: M/L Fixed",file=f)
        print("----------------------------------------------",file=f)
        print("C=",A,"R200=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==NFW_Minimum_Disk):
        f = open("Inital_estimates_NFW_Minimum Disk.txt", "w+")
        print("Inital estimates for NFW-Halo: Minimum Disk",file=f)
        print("----------------------------------------------",file=f)
        print("C=",A,"R200=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==BURKERT):
        f = open("Inital_estimates_BURKERT.txt", "w+")
        print("Inital estimates for BURKERT-Halo: M/L fixed",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"R0=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==BURKERT_Minimum_Disk):
        f = open("Inital_estimates_BURKERT.txt", "w+")
        print("Inital estimates for BURKERT-Halo: Minimum Disk",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"R0=",B,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    return geneticParameters

#written by Aditya K
def INITIAL_GUESS_ML_free(f,R,V):
    func=f
    xData=R
    yData=V
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore")
        val = func(xData, *parameterTuple)
        return numpy.sum((yData - val) ** 2.0)
    def generate_Initial_Parameters():
        maxX = max(xData)
        minX = min(xData)
        maxY = max(yData)
        minY = min(yData)
        parameterBounds = []
        parameterBounds.append([0.1, 100]) # seach bounds for RHO_0
        parameterBounds.append([0.1, 200]) # seach bounds for R_C
        parameterBounds.append([0.1, 200]) # seach bounds for M/L
        result = differential_evolution(sumOfSquaredError, parameterBounds)
        return result.x
    geneticParameters = generate_Initial_Parameters()
    fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
    print('Parameters', fittedParameters)
    modelPredictions = func(xData, *fittedParameters)
    absError = modelPredictions - yData
    A, B, ML = fittedParameters
    dA, dB, dML= \
          [np.sqrt(pcov[j,j]) for j in range(fittedParameters.size)]
    f_fit = radius
    s_fit = func(radius,A,B,ML)
    resids = yData - func(radius,A,B,ML)
    SE = numpy.square(absError) # squared errors
    MSE = numpy.mean(SE) # mean squared errors
    RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
    f_fit = r
    s_fit = func(r,A,B,ML)
    resids = yData - func(r,A,B,ML)
    redchisqr = ((resids/V_err)**2).sum()/float(r.size-2)
    if(func==pseudo_Isothermal):
        f = open("Inital_estimates_PIS_FreeML.txt","w+")
        print("Inital estimates for Pseudo-Isothermal-Halo: M/L Free",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"RC=",B,"M/L=",ML,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==NFW):
        f = open("Inital_estimates_NFW_FreeML.txt", "w+")
        print("Inital estimates for NFW-Halo: M/L Free",file=f)
        print("----------------------------------------------",file=f)
        print("C=",A,"R200=",B,"M/L=",ML,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    if(func==BURKERT):
        f = open("Inital_estimates_BURKERT_FreeML.txt", "w+")
        print("Inital estimates for BURKERT-Halo: Minimum Free",file=f)
        print("----------------------------------------------",file=f)
        print("RHO_0=",A,"R0=",B,"M/L=",ML,file=f)
        print("Mean Squared Error=",MSE,file=f)
        print("Root Mean Squared Error=",RMSE,file=f)
        print("Reduced Chi-Square=",redchisqr,file=f)
        f.close()
    return geneticParameters

#written by Aditya K
def MCMC_RUN(f,fit):
    func=f
    if(func==pseudo_Isothermal):
        p1_name='RHO_0'
        p2_name='R_C'
    if(func==NFW):
        p1_name='C'
        p2_name='R200'
    if(func==BURKERT):
        p1_name='RHO_0'
        p2_name='R_C'
    if(func==pseudo_Isothermal_Minimum_Disk):
        p1_name='RHO_0'
        p2_name='R_C'
    if(func==NFW_Minimum_Disk):
        p1_name='C'
        p2_name='R200'
    if(func==BURKERT_Minimum_Disk):
        p1_name='RHO_0'
        p2_name='R_C'
    model = lmfit.Model(func)
    model.set_param_hint(p1_name,value=fit[0], min=0.0,max=80,vary=True)
    model.set_param_hint(p2_name,value=fit[1], min=0.0,max=10,vary=True)
    p = model.make_params()
    result = model.fit(data=V_total, params=p, r=radius,method='NELDER', nan_policy='omit')
    emcee_kws = dict(steps=50, burn=5, thin=1, is_weighted=False)
    emcee_params = result.params.copy()
    emcee_params.add('__lnsigma',value=np.log(0.1*max(V_total)), min=-10, max=10)
    result_emcee = model.fit(data=V_total, r=radius, params=emcee_params, method='emcee',nan_policy='omit',
                         fit_kws=emcee_kws)
    if(func==pseudo_Isothermal):
        with open('pseudo_Isothrmal_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                            ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                             "$ \mathrm{R_{c}(kpc)}$","ln(f)"]);
        fig.savefig('PIS_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        V_th=pseudo_Isothermal(radius,RHO_0,R_C)
        V_DM=pseudo_Isothermal_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(Md)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('PIS_MM.pdf',dpi=300)
    if(func==pseudo_Isothermal_Minimum_Disk):
        with open('pseudo_Isothermal_MinimumDisc_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                    ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                    "$ \mathrm{R_{c}(kpc)}$","ln(f)"]);
        fig.savefig('PIS_MIN_DISC_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        V_th=pseudo_Isothermal_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('PIS_MinimumDisc_MM.pdf',dpi=300)
    if(func==NFW):
        with open('NFW_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},
            labels = ["$\mathrm{C}$","$ \mathrm{R_{200}(kpc)}$","ln(f)"]);
        fig.savefig('NFW_COV.pdf',dpi=300)
        C=result_emcee.best_values['C']
        R200=result_emcee.best_values['R200']
        V_DM=NFW_Minimum_Disk(radius,C,R200)
        V_th=NFW(radius,C,R200)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(Md)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('NFW_MM.pdf',dpi=300)
    if(func==NFW_Minimum_Disk):
        with open('NFW_MCMC_MinimumDisc_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},
        labels = ["$\mathrm{C}$","$ \mathrm{R_{200}(kpc)}$","ln(f)"]);
        fig.savefig('NFW_MIN_DISC_COV.pdf',dpi=300)
        C=result_emcee.best_values['C']
        R200=result_emcee.best_values['R200']
        V_th=NFW_Minimum_Disk(radius,C,R200)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('NFW_MinimumDiscMM.pdf',dpi=300)
    if(func==BURKERT):
        with open('Burkert_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                    ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                    "$ \mathrm{R_{c}(kpc)}$","ln(f)"]);
        fig.savefig('BURKERT_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        V_th=BURKERT(radius,RHO_0,R_C)
        V_DM=BURKERT_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(Md)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('BURKERT_MM.pdf',dpi=300)
    if(func==BURKERT_Minimum_Disk):
        with open('BURKERT_MCMC_MinimumDisc_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                    ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                    "$ \mathrm{R_{c}(kpc)}$","ln(f)"]);
        fig.savefig('BURKERT_MIN_DISC_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        V_th=BURKERT_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('BURKERT_MinimumDisc_MM.pdf',dpi=300)
    return lmfit.report_fit(result_emcee)

def MCMC_RUN_FREE_ML(f,fit):
    func=f
    if(func==pseudo_Isothermal_FREE_ML):
        p1_n='RHO_0'
        p2_n='R_C'
        p3_n='ML'
    if(func==NFW_FREE_ML):
        p1_n='C'
        p2_n='R200'
        p3_n='ML'
    if(func==BURKERT_FREE_ML):
        p1_n='RHO_0'
        p2_n='R_C'
        p3_n='ML'

    model = lmfit.Model(func)
    model.set_param_hint(p1_n,value=fit[0], min=0.0,max=1000,vary=True)
    model.set_param_hint(p2_n,value=fit[1], min=0.0,max=100,vary=True)
    model.set_param_hint(p3_n,value=fit[2], min=0.0,max=100,vary=True)
    p = model.make_params()
    result = model.fit(data=V_total, params=p, r=radius,method='NELDER', nan_policy='omit')
    emcee_kws = dict(steps=50, burn=5, thin=1, is_weighted=False)
    emcee_params = result.params.copy()
    emcee_params.add('__lnsigma',value=np.log(0.1*max(V_total)), min=-10, max=10)
    result_emcee = model.fit(data=V_total, r=radius, params=emcee_params, method='emcee',nan_policy='omit',
                         fit_kws=emcee_kws)
    if(func==pseudo_Isothermal_FREE_ML):
        with open('pseudo_Isothrmal_FREE_ML_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                            ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                             "$ \mathrm{R_{c}(kpc)}$"
                             ,"M/L","ln(f)"]);
        fig.savefig('PIS_FreeML_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        ML=result_emcee.best_values['ML']
        V_th=pseudo_Isothermal_FREE_ML(radius,RHO_0,R_C,ML)
        V_DM=pseudo_Isothermal_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(ML)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('PIS_Free_ML.pdf',dpi=300)
    if(func==NFW_FREE_ML):
        with open('NFW_MCMC_Free_ML_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},
                    labels = ["$\mathrm{c}$","$ \mathrm{R_{200}(kpc)}$","M/L","ln(f)"]);
        fig.savefig('NFW_FREE_ML_COV.pdf',dpi=300)
        C=result_emcee.best_values['C']
        R200=result_emcee.best_values['R200']
        ML=result_emcee.best_values['ML']
        V_DM=NFW_Minimum_Disk(radius,C,R200)
        V_th=NFW_FREE_ML(radius,C,R200,ML)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(ML)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('NFW_FreeML_MM.pdf',dpi=300)
    if(func==BURKERT_FREE_ML):
        with open('BURKERT_FREE_ML_MCMC_result.txt', 'w+') as fh:
            fh.write(result_emcee.fit_report())
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    title_kwargs={"fontsize": 15},labels =
                            ["$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$",
                             "$ \mathrm{R_{c}(kpc)}$"
                             ,"M/L","ln(f)"]);
        fig.savefig('BURKERT_FreeML_COV.pdf',dpi=300)
        R_C=result_emcee.best_values['R_C']
        RHO_0=result_emcee.best_values['RHO_0']
        ML=result_emcee.best_values['ML']
        V_th=BURKERT_FREE_ML(radius,RHO_0,R_C,ML)
        V_DM=BURKERT_Minimum_Disk(radius,RHO_0,R_C)
        figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.errorbar(radius, V_total, yerr=V_err, ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label='$V_{Obs}$')
        ax.plot(radius,V_gas,label='$V_{Gas}$',color='black',lw=5,ls='dotted')
        ax.plot(radius,np.sqrt(ML)*V_disc,label='$V_{stars}$',color='black',lw=5,ls='dashed')
        ax.plot(radius,V_DM,label='$V_{DM}$',lw=5,color='black',alpha=0.5,marker='s',ms=20)
        ax.plot(radius,V_th,label='$V_{Total}$',color='black',lw=5,ls='solid',alpha=0.5,marker='^',ms=20)
        plt.legend()
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig('BURKERT_Free_ML.pdf',dpi=300)

'''' Fitting part written by Aditya K
print('Fitting Pseudo-Isothermal Halo with user-defined M/L ratio ')
fit1=INITIAL_GUESS(pseudo_Isothermal,radius,V_total)
MCMC_RUN(pseudo_Isothermal,fit1)
print('Fit Pseudo-Isothermal Halo to the observed rotation curve case to find upper limit of the dark matter distribution (Minimum Disc case)  ')
fit2=INITIAL_GUESS(pseudo_Isothermal_Minimum_Disk,radius,V_total)
MCMC_RUN(pseudo_Isothermal,fit2)
print('Fit Pseudo-Isothermal dark matter profile keeping  the M/L ratio as a free parameter')
fit7=INITIAL_GUESS_ML_free(pseudo_Isothermal_FREE_ML,radius,V_total)
MCMC_RUN_FREE_ML(pseudo_Isothermal_FREE_ML,fit7)
print('Fitting NFW Halo with user-defined M/L ratio ')
fit3=INITIAL_GUESS(NFW,radius,V_total)
MCMC_RUN(NFW,fit3)
print('Fit NFW Halo to the observed rotation curve case to find upper limit of the dark matter distribution (Minimum Disc case)  ')
fit4=INITIAL_GUESS(NFW_Minimum_Disk,radius,V_total)
MCMC_RUN(NFW_Minimum_Disk,fit4)
print('Fit NFW dark matter profile keeping  the M/L ratio as a free parameter')
fit8=INITIAL_GUESS_ML_free(NFW_FREE_ML,radius,V_total)
MCMC_RUN_FREE_ML(NFW_FREE_ML,fit8)
print('Fitting Burkert Halo with user-defined M/L ratio ')
fit5=INITIAL_GUESS(BURKERT,radius,V_total)
MCMC_RUN(BURKERT,fit5)
print('Fit Burkert Halo to the observed rotation curve case to find upper limit of the dark matter distribution (Minimum Disc case)  ')
fit6=INITIAL_GUESS(BURKERT_Minimum_Disk,radius,V_total)
MCMC_RUN(BURKERT_Minimum_Disk,fit6)
print('Fit Burkert dark matter profile keeping  the M/L ratio as a free parameter')
fit9=INITIAL_GUESS_ML_free(BURKERT_FREE_ML,radius,V_total)
MCMC_RUN_FREE_ML(BURKERT_FREE_ML,fit9)
'''
