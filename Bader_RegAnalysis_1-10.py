# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:44:46 2021

@author: PerGeos
"""


import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt    
from scipy import stats
import pandas as pd
import seaborn as sns    
import math
from sklearn import linear_model
import statsmodels.api as sm

def plotsalot(FPATH, XLABEL, YLABEL, plotname, Datalabels, SIZES, colors='D', fit_type='linear'):
        
    # pip install uncertainties, if needed
    try:
        import uncertainties.unumpy as unp
        import uncertainties as unc
    except:
        try:
            from pip import main as pipmain
        except:
            from pip._internal import main as pipmain
        pipmain(['install','uncertainties'])
        import uncertainties.unumpy as unp
        import uncertainties as unc
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
    Qline='no'
    MillerK='yes'
    
    colorx = 'forestgreen'
    colory='blue'
    colorz='red'
    
    # import data
    cmap = plt.get_cmap('tab20b')
    fitstring=[]
    string=[]
    cstr=[]
    rstr=[]
    reslim_perm= 1E-18
    bottom_perm = 1
    top_perm=200
    alpha=.6
    LW = 2
    # SIZE=200
    x1= 0.01
    x2= 0.3
    y1 =0.0
    y2 = 0.30
    # Xlabelstr= 'Operative Melt Fraction (\u03C6' + '$_{o}$'+ '$_{p}$)'
    # Ylabelstr= 'Connected Melt Fraction (\u03C6' +'$_{c}$)'
    Ylabelstr='Anisotropy ([Kx+Ky]/2)/Kz'
    Xlabelstr = 'Total Melt Fraction (\u03C6'+'$_{t}$)'
    # Ylabelstr = 'Permeability k'
    # Ylabelstr = 'ED-Prism Melt Fraction Difference'
    fig15 = plt.figure()
    
    for i in range(len(FPATH)):
        SIZE=SIZES[i]
        DF = pd.read_csv(FPATH[i])
        Datalabel = Datalabels[i]
        if colors=='D':
            if 'x' in YLABEL[i] or 'X' in YLABEL[i] or 'x' in XLABEL[i] or 'X' in XLABEL[i]:
                color=colorx
                z_ord=2
                color2=color
            elif 'y' in YLABEL[i] or 'Y' in YLABEL[i] or 'y' in XLABEL[i] or 'Y' in XLABEL[i]:
                color=colory
                z_ord=1
                color2=color
            elif 'z' in YLABEL[i] or 'Z' in YLABEL[i] or 'z' in XLABEL[i] or 'Z' in XLABEL[i]: 
                color=colorz
                z_ord=3
                color2=color
            else:
                if i==0:
                    color=colorx
                    z_ord=1
                    color2=color
                elif i==1:
                    color=colory
                    z_ord=3
                    color2=color
                else:
                    color=colorz
                    z_ord=2
                    color2=color
        elif colors == 'Q':
            for n in range(len(DF)):
                z_ord=3
                if DF.loc[n, 'Quadrant'] == 1:
                    DF.loc[n, 'Color'] = 'firebrick'
                elif DF.loc[n, 'Quadrant'] == 2:
                    DF.loc[n, 'Color'] = 'red'
                elif DF.loc[n, 'Quadrant'] == 3:
                    DF.loc[n, 'Color'] = 'tomato'
                elif DF.loc[n, 'Quadrant'] == 4:
                    DF.loc[n, 'Color'] = 'steelblue'
                elif DF.loc[n, 'Quadrant'] == 5:
                    DF.loc[n, 'Color'] = 'cyan'
                elif DF.loc[n, 'Quadrant'] == 6:
                    DF.loc[n, 'Color'] = 'blue'
                elif DF.loc[n, 'Quadrant'] == 7:
                    DF.loc[n, 'Color'] = 'gray'
                elif DF.loc[n, 'Quadrant'] == 8:
                    DF.loc[n, 'Color'] = 'lime'
                elif DF.loc[n, 'Quadrant'] == 9:
                    DF.loc[n, 'Color'] = 'gold'
                else:
                    DF.loc[n, 'Color'] = 'pink'
            color=DF['Color'].tolist()
            if 'x' in YLABEL[i] or 'X' in YLABEL[i] or 'x' in XLABEL[i] or 'X' in XLABEL[i]:
                color2=colorx
                z_ord=2
            elif 'y' in YLABEL[i] or 'Y' in YLABEL[i] or 'y' in XLABEL[i] or 'Y' in XLABEL[i]:
                color2=colory
                z_ord=1
            elif 'z' in YLABEL[i] or 'Z' in YLABEL[i] or 'z' in XLABEL[i] or 'Z' in XLABEL[i]: 
                color2=colorz
                z_ord=3
            else: 
                color2=[0, 0, 0]
                z_ord=3
            
        else:    
            if i==1:
                color=colorx
                z_ord=1
            elif i==2:
                color=colory
                z_ord=3
            elif i==3:
                color=colorz
                z_ord=2
            else:
                color=np.array((cmap(1.*i/len(FPATH))))
                z_ord=3
            color2=color
                
                

        Bestfitlabel = Datalabel + ' Best Fit'

        
        if 'k' in YLABEL[i] or 'K' in YLABEL[i]:
            DF[YLABEL[i]] = DF[YLABEL[i]]*(.986923E-15)
            DF[YLABEL[i]] = DF[YLABEL[i]] + reslim_perm
            print(DF[YLABEL[i]])
            perm_for_regress = DF[DF[YLABEL[i]] != reslim_perm]
            perm_for_regress = perm_for_regress.dropna(subset=[YLABEL[i]])
            if 'Con' in XLABEL[i] or 'con' in XLABEL[i]:
                perm_for_regress = DF[DF[XLABEL[i]] != 0]
                perm_for_regress = perm_for_regress.dropna(subset=[XLABEL[i]])
            else:
                pass
            print(perm_for_regress)
            x = np.log10(perm_for_regress[XLABEL[i]])
            if 'ED' in YLABEL[i]:
                DF.loc[DF[YLABEL[i]] == bottom_perm, YLABEL[i]] = DF[YLABEL[i]]/4
                y = np.log10(perm_for_regress[YLABEL[i]]*4)   
                y_p = (DF[YLABEL[i]]*4)
                s='d'
            else:
                y = np.log10(perm_for_regress[YLABEL[i]])
                y_p =(DF[YLABEL[i]])
                s='o'
            
        else:
       
            y_p = DF[YLABEL[i]]
            s='s'
            perm_for_regress = DF[DF[YLABEL[i]] != 0]
            perm_for_regress = perm_for_regress.dropna(subset=[YLABEL[i]])            
            y = perm_for_regress[YLABEL[i]]
            x = perm_for_regress[XLABEL[i]]
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # print('Slope:' + str(slope) +'and intercept:' + str(intercept))
        
        
        x_p = DF[XLABEL[i]]

        n = len(y)
        
        def f(x, a, b):
            return (a * x) + b
        if fit_type=='first degree polynomial':
            def f(x, a, b, c):
                return (a * x) + (b * (x**2)) + c
        if fit_type=='second degree polynomial':
            def f(x, a, b, c, d):
                return (a*x) + (b * (x**2)) + (c* (x**3)) + d
        
        
        
        try:
            popt, pcov = curve_fit(f, x, y, absolute_sigma=False)
            popt_2, pcov_2 = curve_fit(f, x, y, absolute_sigma=False)
            popt_3, pcov_3 = curve_fit(f, perm_for_regress[XLABEL[i]], perm_for_regress[YLABEL[i]])
            slope, intercept, r, p, se = linregress(x, y)
            print(x, y)
            print((str(slope), str(intercept)))
            se=(np.sqrt(np.diag(pcov)))
            # retrieve parameter values
            a = popt[0]
            b = popt[1]
            if fit_type == 'first degree polynomial':
                c = popt[2]
            if fit_type == 'second degree polynomial':
                c=popt[2]
                d = popt[3]
            print('Optimal Values')
            print('a: ' + str(a))
            print('b: ' + str(b))
            if fit_type=='first degree polynomial':
                print('c: ' + str(c))
            if fit_type=='second degree polynomial':
                print('d: ' + str(d))
            
            # compute r^2
            if fit_type=='linear':
                r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
            if fit_type=='first degree polynomial':
                print('trying to find r2')
                r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y, ddof=1)))
            if fit_type=='second degree polynomial':
                print('trying to find r2')
                r2 = 1.0-(sum((y-f(x,a,b,c,d))**2)/((n-1.0)*np.var(y, ddof=1)))
            print('R^2: ' + str(r2))
            
            # calculate parameter confidence interval
            a,b = unc.correlated_values(popt, pcov)
            a_2, b_2 = unc.correlated_values(popt_2, pcov_2)
            print('Uncertainty')
            print('a: ' + str(a_2))
            print('b: ' + str(b_2))
            
            # plot data
            plt.scatter(x_p, y_p, s=SIZE, marker=s, c=color, alpha=alpha, label=Datalabel, zorder=z_ord)
        except:
            try:
                plt.scatter(x_p, y_p, s=SIZE, marker=s, c=color, alpha=alpha, label=Datalabel, zorder=z_ord)
            except:
                pass
        

        if 'k' in YLABEL[i] or 'K' in YLABEL[i]:
            plt.yscale('log')
            plt.xscale('log')
        if len(perm_for_regress) > 1:
            # calculate regression confidence interval
            if 'k' in YLABEL[i] or 'K' in YLABEL[i]:
                px = np.linspace(np.log10(x1), np.log10(x2), 100)
            else:
                px= np.linspace(x1, x2, 100)
            
            if fit_type=='linear':
                py = a*px+b
            if fit_type=='first degree polynomial':
                py=a*px+b*px**2+c
            if fit_type=='second degree polynomial':
                py=a*px+b*px**2+c*px**3+d
            nom = unp.nominal_values(py)
            std = unp.std_devs(py)
            # SE=a_2/(np.sqrt(len(perm_for_regress)))
            SE=se/(np.sqrt(len(perm_for_regress)))
            print(' standard deviation' + str(std))
            print('standard error: ' +str(SE))
            
            def predband(x, xd, yd, p, func, conf=0.95):
                # x = requested points
                # xd = x data
                # yd = y data
                # p = parameters
                # func = function name
                alpha = 1.0 - conf    # significance
                N = xd.size          # data sample size
                var_n = len(p)  # number of parameters
                # Quantile of Student's t distribution for p=(1-alpha/2)
                q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
                # Stdev of an individual measurement
                se = np.sqrt(1. / (N - var_n) * \
                             np.sum((yd - func(xd, *p)) ** 2))
                # Auxiliary definitions
                sx = (x - xd.mean()) ** 2
                sxd = np.sum((xd - xd.mean()) ** 2)
                # Predicted values (best-fit model)
                yp = (func(x, *p))
                # Prediction band
                dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
                # Upper & lower prediction bands.
                lpb, upb = (yp - dy), (yp + dy)
                return lpb, upb
            try:
                lpb, upb = predband(px, x, y, popt, f, conf=0.95)
            except:
                pass
            # plot the regression
            if 'k' in YLABEL[i] or 'K' in YLABEL[i]:    
                plt.plot(10**px, 10**nom, c=color2, label=Bestfitlabel, linewidth=LW)
                print('Best fit line should have plotted')
            # prediction band (95% confidence)
                # plt.plot(10**px, 10**lpb, c=color2, linestyle='dashed', label='95% Prediction Band', linewidth=LW)
                # plt.plot(10**px, 10**upb, linestyle='dashed',c=color2, linewidth=LW)
                plt.plot(10**px, 10**(nom - (1.96 * std)), c=color2, linestyle='dotted', label='95% Confidence Region', linewidth=LW)
                plt.plot(10**px, 10**(nom + (1.96 * std)), c=color2, linestyle='dotted', linewidth=LW)
                if fit_type=='linear':
                    fitstring.append(Datalabel + 'log(' + YLABEL[i] + ') = ' + str(a) + 'log(' + XLABEL[i] + ') + (' + str(b) + ')\n')
                elif fit_type=='first degree polynomial':
                    fitstring.append(Datalabel + 'log(' + YLABEL[i] + ') = ' + str(b) + 'log(' + XLABEL[i] + '$^{2}$'  + ') + (' + str(a) + 'log(' + XLABEL[i] + ') + (' + str(c) + ')\n')
                elif fit_type=='second degree polynomial': 
                    fitstring.append('log(' + YLABEL[i] + ') = ' + str(c) + 'log(' + XLABEL[i] + '$^{3}$' + ') + ('+ str(b) + 'log(' + XLABEL[i] + '$^{2}$' + ') + (' + str(a) + 'log(' + XLABEL[i] + ') + (' + str(d) + ')\n')
                # fitstring.append(Datalabel + ' Best Fit Line: log(' + YLABEL[i] + ') = ' + str(a) + 'log(' + XLABEL[i] + ') + ' +str(d) + '('+XLABEL[i]+')' + '$^2$(' +str(b)+')\n')
                rst=str(round(r2, 2))
                rstr.append(Datalabel + ' R$^2$'+ '('+ rst + ')\n')
            elif colors == 'Q':
                pass
            else:
                plt.plot(px, nom, c=color2, label=Bestfitlabel, linewidth=LW)
            # prediction band (95% confidence)
                # plt.plot(px, lpb, c=color2, linestyle='dashed', label='95% Prediction Band')
                # plt.plot(px, upb, linestyle='dashed', c=color2)
                
                
                #plt.plot(px, (nom - (1.96 * std)), c=color2, linestyle='dotted', label='95% Confidence Region', linewidth=LW)
                #plt.plot(px, (nom + (1.96 * std)), c=color2, linestyle='dotted', linewidth=LW)
                fitstring.append(Datalabel + ': Best Fit Line: (' + YLABEL[i] + ') = ' + (f'{a:.2f}') + '(' + XLABEL[i] + ') + (' + (f'{b:.3f}') + ')\n')
                rst=str(round(r2, 2))
                rstr.append(Datalabel + ' R$^2$'+ '('+ rst + ')\n')
        else:
            pass
        if x2==y2 and x1==y1:
            plt.plot([x1, x2], [y1, y2], linestyle='solid', color='black', linewidth=LW*2)
        else:
            pass
        ratio = round(len(perm_for_regress[YLABEL[i]])/len(DF[YLABEL[i]])*100,2)
        ratiostr = (Datalabel + ': % of Subvolumes where (\u03C6'+ '$_{c}$)' +'>0 = ' + str(ratio))
        cstr.append(ratiostr) 
        if Qline=='yes':
            if i==(len(FPATH)-2):
                print('True')
                min_yphi = min(DF['Total Melt'])
                max_yphi = max(DF['Total Melt'])
                min_ydiff = DF['EDPrismMeltRatio'].min()
                max_ydiff = DF['EDPrismMeltRatio'].max()
                print(min_ydiff)
                
                plt.plot(([min_yphi, min_yphi]),(y1, y2), color='black', linewidth=1.5)
                plt.plot(([max_yphi, max_yphi]),(y1, y2), color='black', linewidth=1.5)
                plt.plot(([x1, x2]), (min_ydiff, min_ydiff), color='black', linewidth=1.5)
                plt.plot(([x1, x2]), (max_ydiff, max_ydiff), color='black', linewidth=1.5)
                plt.plot(([x1, x2]), ([0, 0]), color='black', linewidth=1, linestyle='--')
            else:
                pass
        else:
            pass
        # tval=np.sqrt(r2)*(np.sqrt((len(perm_for_regress)-2)/(1-r2)))
        tval=(slope/(SE))
        print('T Value:' +  str(tval) + '...DF:'+ str(len(perm_for_regress)))
    cstr = '\n'.join(cstr)
    string = '\n'.join(fitstring)
    rstr='\n'.join(rstr)
    plt.xlim([x1, x2])
    plt.ylim([bottom_perm, top_perm])
    # plt.ylim([y1, y2])
    ax = plt.axes()
    ax.set_xlabel(Xlabelstr, fontsize=16)
    ax.set_ylabel(Ylabelstr, fontsize=16)
    if MillerK == 'yes':
        def k_eqn(melt):
            k = (melt**(2.6))*((3.1E-6)**2)/(38)
            return(k)
        [permk1, permk2] = k_eqn(x1), k_eqn(x2)
        ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-')
    if colors == 'Q':
        if color2 == [0, 0, 0]:
            pass
        else:
            ax.text(.392, .06, string, transform=ax.transAxes)
            ax.text(.20, .8, cstr, transform=ax.transAxes)
            plt.legend(loc='best')
    else:

        ax.text(.2, .06, cstr, transform=ax.transAxes)
        ax.text(.12, .8, string, transform=ax.transAxes)
        ax.text(.8, .2, rstr, transform=ax.transAxes)
        plt.legend(loc='upper left')
    # save and show figure

    SAVENAME = (plotname + '.png')
    plt.savefig(SAVENAME)
    plt.show()    
    plt.figure()
    # regtest =    sns.regplot(x, y, ci=95)
    # regtest.set(xlim=(0,.20), ylim=(0, 0.20))
    return


def plot_relcon(spreadsheetfpaths, xlabels, ylims, pointsize):
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(ylims)
    ax.set_xlim([0, len(spreadsheetfpaths)+1])
    
    
    
    i=0
    vals=[]
    for i in range(len(spreadsheetfpaths)):
        for j in range(len(spreadsheetfpaths)):
            if i==0:
                vals.append(j)
            df = pd.read_csv(spreadsheetfpaths[i, j])
            count_yescon_x = [0]
            count_yescon_y = [0]
            count_yescon_z = [0]
            count_total = len(df.index)
            df = df.set_index(['Subvolume Name'])
            for subvolume in df.index:
                if df[subvolume, 'X Connected Melt'] > 0:
                    count_yescon_x = count_yescon_x + 1
                if df[subvolume, 'Y Connected Melt'] > 0:
                    count_yescon_y = count_yescon_y + 1
                if df[subvolume, 'Z Connected Melt'] > 0:
                    count_yescon_z = count_yescon_z + 1
            if count_yescon_x == 0:
                relconx=0
            else:
                relconx=count_yescon_x/count_total
            if count_yescon_y == 0:
                relcony=0
            else:
                relcony=count_yescon_y/count_total
            if count_yescon_z == 0:
                relconz=0
            else:
                relconz=count_yescon_z/count_total
                
            plt.scatter(j+1, relconx, s=pointsize[i], color='forestgreen')
            plt.scatter(j+1, relcony, s=pointsize[i], color='blue')
            plt.scatter(j+1, relconz, s=pointsize[i], color='red')
    plt.xticks(vals, xlabels)
            
    return

def permnotlog_avg(spreadsheetfpathlist):
    meltfracstep=0.005
    range_fudge=[0,0]
    quartile_range=[10, 90]
    
    fig2=plt.figure()
    ax2=plt.axes()
    
    fig=plt.figure()
    ax=plt.axes()
    ax.set_xlim([0, 0.20])
    ax.set_ylim([0, 1E-15])
    
    plotstr=[]
    
    listshape=np.shape(spreadsheetfpathlist)
    for i in range(listshape[1]):    
        fit_type='first degree polynomial'
        combined_meltfrac=[]
        combined_kmean=[]
        rangelist1=[]
        rangelist2=[]
        kmeanlist1=[]
        kmeanlist2=[]
        meltfraclist1=[]
        meltfraclist2=[]
        kmean_uq=[]
        kmean_lq=[]
        kmean_uq_1=[]
        kmean_lq_1=[]
        kmean_uq_2=[]
        kmean_lq_2=[]
        datasheet=pd.read_csv(spreadsheetfpathlist[0][i])
        meltfracmin=min(datasheet['Dominant Melt'])
        meltfracmax=max(datasheet['Dominant Melt'])
        num_ranges = int(math.ceil((meltfracmax-meltfracmin)/meltfracstep))
        for j in range(num_ranges):
            rangelist1.append(meltfracmin+((j-1)*meltfracstep))
            rangelist2.append(meltfracmin+(meltfracstep/2)+((j-1)*meltfracstep))
        for j in range(len(rangelist1)-1):
            datasheetrangeapp1= datasheet[(datasheet['Dominant Melt'] < rangelist1[j+1]) & (datasheet['Dominant Melt'] > rangelist1[j])]
            datasheetrangeapp2= datasheet[(datasheet['Dominant Melt'] < rangelist2[j+1]) & (datasheet['Dominant Melt'] > rangelist2[j])]
            meltfraclist1.append((rangelist1[j]+rangelist1[j+1])/2)
            meltfraclist2.append((rangelist2[j]+rangelist2[j+1])/2)
            if datasheet['dX'][0] > datasheet['dY'][0]:
                kmeanlist1.append(np.mean(datasheetrangeapp1['EDKX']*(.986923E-15)))
                kmeanlist2.append(np.mean(datasheetrangeapp2['EDKX']*(.986923E-15)))
                try:
                    kmean_uq_1.append((np.percentile(datasheetrangeapp1['EDKX'], quartile_range[1]))*.986923E-15)
                    kmean_lq_1.append((np.percentile(datasheetrangeapp1['EDKX'], quartile_range[0]))*.986923E-15)
                except:
                    kmean_uq_1.append(0)
                    kmean_lq_1.append(0)
                try:
                    kmean_uq_2.append((np.percentile(datasheetrangeapp2['EDKX'], quartile_range[1]))*.986923E-15)
                    kmean_lq_2.append((np.percentile(datasheetrangeapp2['EDKX'], quartile_range[0]))*.986923E-15)
                except:
                    kmean_uq_2.append(0)
                    kmean_lq_2.append(0)
                colordesi='forestgreen'
            elif datasheet['dY'][0] > datasheet['dX'][0]:
                kmeanlist1.append(np.mean(datasheetrangeapp1['EDKY']*(.986923E-15)))
                kmeanlist2.append(np.mean(datasheetrangeapp2['EDKY']*(.986923E-15)))
                try:    
                    kmean_uq_1.append((np.percentile(datasheetrangeapp1['EDKY'], quartile_range[1]))*.986923E-15)
                    kmean_uq_2.append((np.percentile(datasheetrangeapp2['EDKY'], quartile_range[1]))*.986923E-15)
                    kmean_lq_1.append((np.percentile(datasheetrangeapp1['EDKY'], quartile_range[0]))*.986923E-15)
                    kmean_lq_2.append((np.percentile(datasheetrangeapp2['EDKY'], quartile_range[0]))*.986923E-15)
                except:
                    kmean_uq_1.append(0)
                    kmean_uq_2.append(0)
                    kmean_lq_1.append(0)
                    kmean_lq_2.append(0)
                colordesi='red'
            elif datasheet['dZ'][0] > datasheet['dY'][0]:
                kmeanlist1.append(np.mean(datasheetrangeapp1['EDKZ']*(.986923E-15)))
                kmeanlist2.append(np.mean(datasheetrangeapp2['EDKZ']*(.986923E-15)))
                try:
                    kmean_uq_1.append((np.percentile(datasheetrangeapp1['EDKZ'], quartile_range[1]))*.986923E-15)
                    kmean_uq_2.append((np.percentile(datasheetrangeapp2['EDKZ'], quartile_range[1]))*.986923E-15)
                    kmean_lq_1.append((np.percentile(datasheetrangeapp1['EDKZ'], quartile_range[0]))*.986923E-15)
                    kmean_lq_2.append((np.percentile(datasheetrangeapp2['EDKZ'], quartile_range[0]))*.986923E-15)
                except:
                    kmean_uq_1.append(0)
                    kmean_uq_2.append(0)
                    kmean_lq_1.append(0)
                    kmean_lq_2.append(0)
                colordesi='royalblue'
        kmeanlist1= [0 if pd.isna(x) else x for x in kmeanlist1]
        kmeanlist2= [0 if pd.isna(x) else x for x in kmeanlist2]
        kmean_uq_1= [0 if pd.isna(x) else x for x in kmean_uq_1]
        kmean_uq_2= [0 if pd.isna(x) else x for x in kmean_uq_2]
        kmean_lq_1= [0 if pd.isna(x) else x for x in kmean_lq_1]
        kmean_lq_2= [0 if pd.isna(x) else x for x in kmean_lq_2]
        
        
        
        ax.scatter(meltfraclist1, kmeanlist1, color=colordesi, marker='o')
        ax.scatter(meltfraclist2, kmeanlist2, color=colordesi, marker='o')
        ax.scatter(meltfraclist1, kmean_uq_1, color=colordesi, marker='s')
        ax.scatter(meltfraclist2, kmean_uq_2, color=colordesi, marker='s')
        ax.scatter(meltfraclist1, kmean_lq_1, color=colordesi, marker='s')
        ax.scatter(meltfraclist2, kmean_lq_2, color=colordesi, marker='s')
        ax.set_xlabel('Total Melt Fraction')
        ax.set_ylabel('Permeability (m$^2$)')
        
        
        def f(x, a, b):
            return (a * x) + b
        if fit_type=='first degree polynomial':
            def f(x, a, b, c):
                return (a * x) + (b * (x**2)) + c
        if fit_type=='second degree polynomial':
            def f(x, a, b, c, d):
                return (a*x) + (b * (x**2)) + (c* (x**3)) + d
        orig_range=range(len(meltfraclist1))
        ad_range_1=orig_range[0]+range_fudge[0]
        ad_range_2=orig_range[-1]-range_fudge[1]
        print(ad_range_1, ad_range_2)
        for j in range(ad_range_1, ad_range_2):
            if kmeanlist1[j]>0:    
                kmean_uq.append(kmean_uq_1[j])
                kmean_lq.append(kmean_lq_1[j])
                combined_kmean.append(kmeanlist1[j])
                combined_meltfrac.append(meltfraclist1[j])
            if kmeanlist2[j]>0:
                kmean_uq.append(kmean_uq_2[j])
                kmean_lq.append(kmean_lq_2[j])
                combined_kmean.append(kmeanlist2[j])
                combined_meltfrac.append(meltfraclist2[j])
        x=np.asarray(combined_meltfrac)
        y=np.asarray(combined_kmean)
        
        popt, pcov = curve_fit(f, x, y, absolute_sigma=False)
        print(x, y)
        # retrieve parameter values
        a = popt[0]
        b = popt[1]
        if fit_type == 'first degree polynomial':
            c = popt[2]
        if fit_type == 'second degree polynomial':
            c=popt[2]
            d = popt[3]
        print('Optimal Values')
        print('a: ' + str(a))
        print('b: ' + str(b))
        if fit_type=='first degree polynomial':
            print('c: ' + str(c))
        if fit_type=='second degree polynomial':
            print('d: ' + str(d))
        
        # compute r^2
        n = len(y)
        print(a, x, b, c)
        if fit_type=='linear':
            r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
        if fit_type=='first degree polynomial':
            print('trying to find r2')
            r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y, ddof=1)))
        if fit_type=='second degree polynomial':
            print('trying to find r2')
            r2 = 1.0-(sum((y-f(x,a,b,c,d))**2)/((n-1.0)*np.var(y, ddof=1)))
        print('R^2: ' + str(r2))
        px=np.linspace(meltfracmin, meltfracmax, 100)
        if fit_type=='linear':
            py = a*px+b
        if fit_type=='first degree polynomial':
            py=a*px+b*px**2+c
        if fit_type=='second degree polynomial':
            py=a*px+b*px**2+c*px**3+d
        
        ax.plot(px, py, linestyle='-', color=colordesi, linewidth=2)
        
        ##time to log-plot and do the linear-fit!
        fit_type='linear'
        
        x=np.log10(x)
        y=np.log10(y)
        
        def f(x, a, b):
            return (a * x) + b
        if fit_type == 'first degree polynomial':
            c = popt[2]
        if fit_type == 'second degree polynomial':
            c=popt[2]
            d = popt[3]
        
        popt, pcov = curve_fit(f, x, y, absolute_sigma=False)
        print(x, y)
        # retrieve parameter values
        a = popt[0]
        b = popt[1]

        print('Optimal Values')
        print('a: ' + str(a))
        print('b: ' + str(b))
        if fit_type=='first degree polynomial':
            print('c: ' + str(c))
        if fit_type=='second degree polynomial':
            print('d: ' + str(d))
        
        # compute r^2
        n = len(y)
        print(a, x, b, c)
        if fit_type=='linear':
            r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
        if fit_type=='first degree polynomial':
            print('trying to find r2')
            r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y, ddof=1)))
        if fit_type=='second degree polynomial':
            print('trying to find r2')
            r2 = 1.0-(sum((y-f(x,a,b,c,d))**2)/((n-1.0)*np.var(y, ddof=1)))
        print('R^2: ' + str(r2))
        px=np.linspace(np.log10(meltfracmin), np.log10(meltfracmax), 100)
        if fit_type=='linear':
            py = a*px+b
        if fit_type=='first degree polynomial':
            py=a*px+b*px**2+c
        if fit_type=='second degree polynomial':
            py=a*px+b*px**2+c*px**3+d
        
        ax2.plot(10**px, 10**py, color=colordesi, linewidth=2)
        ax2.scatter(10**x, 10**y, color=colordesi)
        ax2.scatter(combined_meltfrac, kmean_uq, color=colordesi, marker='s')
        ax2.scatter(combined_meltfrac, kmean_lq, color=colordesi, marker='s')
        round_n=str(round(a, 2))
        if colordesi=='forestgreen':
            plotstr.append('n' + '$_{x}$' + '~' + round_n +'\n')
        if colordesi=='royalblue':
            plotstr.append('n' + '$_{z}$' + '~' + round_n +'\n')
    plotstr='\n'.join(plotstr)
    ax2.text(.2, 0.06, plotstr, transform=ax2.transAxes)
    ax2.set_xlabel('Total Melt Fraction')
    ax2.set_ylabel('Permeability (m$^2$)')
    x1=0.01
    x2=0.20
    def k_eqn(melt):
        k = (melt**(2.6))*((3.1E-6)**2)/(38)
        return(k)
    [permk1, permk2] = k_eqn(x1), k_eqn(x2)
    ax2.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-')   
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim([0.01, 0.20])
    ax2.set_ylim([1E-19, 1E-13])
        
        
        
        # plt.yscale('log')
        # plt.xscale('log')

def z_test_regcoeff(coeff1, stan1, coeff2, stan2):
    Z = (coeff1-coeff2)/(np.sqrt((stan2*stan2)+(stan1*stan1)))
    print(Z)
    return(Z)
        
            
def reg_test_dummyset(spreadsheetinband, spreadsheetoutband, xlabel='Dominant Melt', ylabel='EDKX'):
    label2='100x800 mixed region'
    label1='100x800 inside-band'
    color1=(1,0,1,1)
    # color1=(0, 1, 0, 1)
    color2=(1,0,0, 1)
    size=120
    xmin=0.01
    xmax=0.20
    xplotter=np.linspace(np.log10(xmin), np.log10(xmax))
    meltfracstep=0.01
    rangelist1=[]
    rangelist2=[]
    meltfraclist1=[]
    meltfraclist2=[]
    kmeanlist1=[]
    kmeanlist2=[]
    
    inmeltfraclist1=[]
    inmeltfraclist2=[]
    inkmeanlist1=[]
    inkmeanlist2=[]
    
    
    spreadsheetinband=spreadsheetinband[spreadsheetinband[ylabel].notnull()]
    spreadsheetoutband=spreadsheetoutband[spreadsheetoutband[ylabel].notnull()]
    spreadsheetinband[ylabel] = (spreadsheetinband[ylabel]*0.986923E-15)+1.78E-18
    spreadsheetoutband[ylabel] = (spreadsheetoutband[ylabel]*0.986923E-15)+1.78E-18
    
    xdata_inband=np.array(spreadsheetinband[xlabel])
    ydata_inband=np.array(spreadsheetinband[ylabel])
    xdata_outband=np.array(spreadsheetoutband[xlabel])
    ydata_outband=np.array(spreadsheetoutband[ylabel])
    
    
    ###Find the separate regression estimates for both in and out band data, including and not including zeros, and 
    ###with the binned data
    xfit=np.array(np.log10([xdata_inband]))
    xfit=xfit.T
    yfit=np.array(np.log10([ydata_inband]))    
    yfit=yfit.T

    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result_inband = ols.fit()
    print(label1+ '; with zeros only')
    print(ols_result_inband.summary())
    
    xfit=np.array(np.log10([xdata_outband]))
    xfit=xfit.T
    yfit=np.array(np.log10([ydata_outband]))    
    yfit=yfit.T

    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result_outband = ols.fit()
    print(label2+ '; with zeros only')
    print(ols_result_inband.summary())
    
    ##
    spreadsheetinband_nozero=spreadsheetinband[spreadsheetinband[ylabel]!=1.78E-18]
    spreadsheetoutband_nozero=spreadsheetoutband[spreadsheetoutband[ylabel]!=1.78E-18]
    
    xdata_inband_nz=np.array(spreadsheetinband_nozero[xlabel])
    ydata_inband_nz=np.array(spreadsheetinband_nozero[ylabel])
    xdata_outband_nz=np.array(spreadsheetoutband_nozero[xlabel])
    ydata_outband_nz=np.array(spreadsheetoutband_nozero[ylabel])
    xfit_nz=np.array(np.log10([xdata_inband_nz]))
    xfit_nz=xfit_nz.T
    yfit_nz=np.array(np.log10([ydata_inband_nz]))    
    yfit_nz=yfit_nz.T

    xfitshape_nz=np.shape(xfit_nz)
    N_nz = (xfitshape_nz[0])
    p_nz = (xfitshape_nz[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept_nz = np.empty(shape=(N_nz, p_nz), dtype=np.float)
    X_with_intercept_nz[:, 0] = 1
    X_with_intercept_nz[:, 1:p_nz] = xfit_nz
    ols = sm.OLS(yfit_nz, X_with_intercept_nz)
    ols_result_inband_nozero = ols.fit()
    print(label1+ '; without zeros only')
    print(ols_result_inband_nozero.summary())
    
    ##
    xfit_nz=np.array(np.log10([xdata_outband_nz]))
    xfit_nz=xfit_nz.T
    yfit_nz=np.array(np.log10([ydata_outband_nz]))    
    yfit_nz=yfit_nz.T

    xfitshape_nz=np.shape(xfit_nz)
    N_nz = (xfitshape_nz[0])
    p_nz = (xfitshape_nz[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept_nz = np.empty(shape=(N_nz, p_nz), dtype=np.float)
    X_with_intercept_nz[:, 0] = 1
    X_with_intercept_nz[:, 1:p_nz] = xfit_nz
    
    ols = sm.OLS(yfit_nz, X_with_intercept_nz)
    ols_result_outband_nozero = ols.fit()
    print(label2 + 'only, without zeros')
    print(ols_result_outband_nozero.summary())
    
    
    
    
    meltfracmin=min(min(spreadsheetinband[xlabel]), min(spreadsheetoutband[xlabel]))
    meltfracmax=max(max(spreadsheetinband[xlabel]), max(spreadsheetoutband[xlabel]))
    num_ranges = int(math.ceil((meltfracmax-meltfracmin)/meltfracstep))
    
    fig=plt.figure()
    ax=plt.axes()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 0.20])
    ax.set_ylim([1E-19, 1E-14])
    x1=0.01
    x2=0.20
    def k_eqn(melt):
        k = (melt**(2.6))*((3.1E-6)**2)/(38)
        return(k)
    [permk1, permk2] = k_eqn(x1), k_eqn(x2)
    ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-')   

    
    height=1E-14-1E-19
    logmeltfracstep=np.log10(meltfracstep)
    for j in range(num_ranges):
        rangelist1.append(np.log10(meltfracmin+((j-1)*meltfracstep)))
        rangelist2.append(np.log10(meltfracmin+(meltfracstep/2)+((j-1)*meltfracstep)))
    for j in range(len(rangelist1)-1):
        datasheetrangeapp1= spreadsheetoutband[(np.log10(spreadsheetoutband['Dominant Melt']) < rangelist1[j+1]) & (np.log10(spreadsheetoutband['Dominant Melt']) > rangelist1[j])]
        datasheetrangeapp2= spreadsheetoutband[(np.log10(spreadsheetoutband['Dominant Melt']) < rangelist2[j+1]) & (np.log10(spreadsheetoutband['Dominant Melt']) > rangelist2[j])]
        meltfraclist1.append((10**rangelist1[j]+10**rangelist1[j+1])/2)
        meltfraclist2.append((10**rangelist2[j]+10**rangelist2[j+1])/2)
        kmeanlist1.append(np.mean(datasheetrangeapp1['EDKX']))
        kmeanlist2.append(np.mean(datasheetrangeapp2['EDKX']))
        print(meltfraclist1[-1])
        ax_histy=ax.inset_axes([meltfraclist1[-1], 1E-19, (meltfracstep), height], transform=ax.transData, zorder=1)
        ax_histy_two=ax.inset_axes([meltfraclist2[-1], 1E-19, (meltfracstep), height], transform=ax.transData, zorder=1)
        ax_histy.hist(datasheetrangeapp1['EDKX'], orientation='horizontal', zorder=1, color=color1)
        ax_histy_two.hist(datasheetrangeapp2['EDKX'], orientation='horizontal', zorder=1, color=color1)
        print(datasheetrangeapp1['EDKX'])
        datasheetrangeapp1= spreadsheetinband[(np.log10(spreadsheetinband['Dominant Melt']) < rangelist1[j+1]) & (np.log10(spreadsheetinband['Dominant Melt']) > rangelist1[j])]
        datasheetrangeapp2= spreadsheetinband[(np.log10(spreadsheetinband['Dominant Melt']) < rangelist2[j+1]) & (np.log10(spreadsheetinband['Dominant Melt']) > rangelist2[j])]
        inkmeanlist1.append(np.mean(datasheetrangeapp1['EDKX']))
        inkmeanlist2.append(np.mean(datasheetrangeapp2['EDKX']))
        ax_histy.hist(datasheetrangeapp1['EDKX'], orientation='horizontal', zorder=2, color=color2)
        ax_histy_two.hist(datasheetrangeapp2['EDKX'], orientation='horizontal', zorder=2, color=color2)

        # ax_histy.set_xticks([])
        # ax_histy.set_yticks([])
        # ax_histy_two.set_xticks([])
        # ax_histy_two.set_yticks([])

    
        ax_histy.grid(False)
        ax_histy.set_yscale('log')
        # ax_histy.set_xscale('log')
        ax_histy_two.set_yscale('log')             
        ax_histy.set_xlim([1, 100])
        ax_histy_two.set_xlim([1, 100])
        ax_histy.set_ylim([1E-19, 1E-14])
        ax_histy_two.set_ylim([1E-19, 1E-14])    
        ax_histy.set_xticks([])
        ax_histy.set_yticks([])
        ax_histy_two.set_xticks([])
        ax_histy_two.set_yticks([]) 
        ax_histy.minorticks_off()
        ax_histy_two.minorticks_off()
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        ax_histy_two.spines['top'].set_visible(False)
        ax_histy_two.spines['right'].set_visible(False)
        ax_histy_two.spines['bottom'].set_visible(False)
        ax_histy_two.spines['left'].set_visible(False)        
  
        # ax_histy_two.set_xscale('log')
        secax = ax_histy.secondary_xaxis('top')
        
    


    kmeanlist1= [1.78E-18 if pd.isna(x) else x for x in kmeanlist1]
    kmeanlist2= [1.78E-18 if pd.isna(x) else x for x in kmeanlist2]
    inkmeanlist1= [1.78E-18 if pd.isna(x) else x for x in inkmeanlist1]
    inkmeanlist2= [1.78E-18 if pd.isna(x) else x for x in inkmeanlist2]
    
    meltfraclist1=np.array([meltfraclist1])
    meltfraclist2=np.array([meltfraclist2])
    kmeanlist1=np.array([kmeanlist1])
    kmeanlist2=np.array([kmeanlist2])
    inkmeanlist1=np.array([inkmeanlist1])
    inkmeanlist2=np.array([inkmeanlist2])
    
    ax.scatter(meltfraclist1, inkmeanlist1, marker='s', s=size, facecolor='none', edgecolor=color1, zorder=3)
    ax.scatter(meltfraclist2, inkmeanlist2, marker='s', s=size, facecolor='none', edgecolor=color1, zorder=3)
    ax.scatter(meltfraclist1, kmeanlist1, marker='s', s=size, facecolor=color2, edgecolor=color2, zorder=4)
    ax.scatter(meltfraclist2, kmeanlist2, marker='s', s=size, facecolor=color2, edgecolor=color2, zorder=4)
    ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-')  
    
    ###Fit the data and do the regression comparison with the binned and averaged data
    kmeanlist_comb=np.concatenate([kmeanlist1, kmeanlist2], axis=None)
    inkmeanlist_comb=np.concatenate([inkmeanlist1, inkmeanlist2], axis=None)

    ##do the regression fit for the averaged data
    meltfrac=np.concatenate([meltfraclist1, meltfraclist2], axis=None)
    xfit=np.array(np.log10([meltfrac]))
    xfit=xfit.T
    yfit=np.array(np.log10([kmeanlist_comb]))    
    yfit=yfit.T
    width=meltfrac[1]-meltfrac[0]
    height=1E-15-1E-19      

    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print(label1 + '; binned, including zeros')
    print(ols_result.summary())
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result.params[0]+(ols_result.params[1] * xplotter[i]))))
    print(predvals)
    ax.plot(10**xplotter, predvals, label=(label1+'; binned, including zeros'), zorder=5, color=color2)
    
    ##do the regression fit for the averaged data
    yfit=np.array(np.log10([inkmeanlist_comb]))    
    yfit=yfit.T

    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print(label2 + '; binned, including zeros')
    print(ols_result.summary())
    predvals=[]
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result.params[0]+(ols_result.params[1] * xplotter[i]))))
    print(predvals)
    ax.plot(10**xplotter, predvals, label=(label2+'; binned, including zeros'), zorder=5, color=color1)
    ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-', zorder=5)  
    ax.legend()
    
    
        
    comb_xdata=np.concatenate([meltfraclist1, meltfraclist2, meltfraclist1, meltfraclist2], axis=None)
    xyes=np.full((len(inkmeanlist_comb), 1), 1)
    xno=np.full((len(kmeanlist_comb), 1), 0)
    comb_ydata=np.concatenate([np.log10(kmeanlist_comb), np.log10(inkmeanlist_comb)], axis=None)
    xyesno=np.concatenate((xno, xyes), axis=None)
    xyesno_mult=comb_xdata*xyesno
    
    xfit=np.stack((comb_xdata, xyesno, xyesno_mult), axis=-1)
    yfit=np.array([comb_ydata])    
    yfit=yfit.T


    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print('Combined regression for binned data, including zeros')
    print(ols_result.summary())
    
    
    
    
    
    ###Do the fit with the raw data from each spreadsheet
    ydata_inband=np.log10(ydata_inband)
    ydata_outband=np.log10(ydata_outband)
    xdata_inband=np.log10(xdata_inband)
    xdata_outband=np.log10(xdata_outband)
    
    
    xyes=np.full((len(xdata_inband), 1), 1)
    xno=np.full((len(xdata_outband), 1), 0)
    comb_xdata=np.concatenate([xdata_outband, xdata_inband], axis=None)
    comb_ydata=np.concatenate([ydata_outband, ydata_inband], axis=None)
    xyesno=np.concatenate((xno, xyes), axis=None)
    xyesno_mult=comb_xdata*xyesno
    
    xfit=np.stack((comb_xdata, xyesno, xyesno_mult), axis=-1)
    yfit=np.array([comb_ydata])    
    yfit=yfit.T


    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print(' Combined regression for datapoints, including zeros')
    print(ols_result.summary())
    
    
    fig=plt.figure()
    ax=plt.axes()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 0.20])
    ax.set_ylim([1E-19, 1E-14])
    comb_ydata=10**comb_ydata
    comb_xdata=10**comb_xdata
    ax.scatter(10**xdata_outband, 10**ydata_outband, marker='s', s=size, edgecolor=color2, facecolor=color2, zorder=3)
    ax.scatter(10**xdata_inband, 10**ydata_inband, marker='s', s=(size), facecolor='none',edgecolor=color1, zorder=4)
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result_outband.params[0]+(ols_result_outband.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label1+'; no zeros', color=color1, zorder=5)
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result_inband.params[0]+(ols_result_inband.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label2+'; no zeros', color=color2, zorder=5)
    ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-', zorder=5)  
    ax.legend()
    
    
    
    ##Do the fit with the raw data, zeros not included
    comb_ydata=np.log10(comb_ydata)
    comb_xdata=np.log10(comb_xdata)
    comb_xdata=comb_xdata[np.where(comb_ydata>(np.log10(1.78E-18)))]
    xyesno_mult=xyesno_mult[np.where(comb_ydata>(np.log10(1.78E-18)))]
    xyesno=xyesno[np.where(comb_ydata>(np.log10(1.78E-18)))]
    comb_ydata=comb_ydata[np.where(comb_ydata>(np.log10(1.78E-18)))]
    
    fig=plt.figure()
    ax=plt.axes()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 0.20])
    ax.set_ylim([1E-19, 1E-14])
    comb_ydata=10**comb_ydata
    comb_xdata=10**comb_xdata
    ax.scatter(comb_xdata, comb_ydata, marker='o', s=size, facecolor=color2, edgecolor=color2, zorder=4)
    ax.scatter(10**xdata_inband, 10**ydata_inband, marker='s', s=(size),facecolor='none', edgecolor=color1, zorder=5)
    ax.plot([x1, x2], [permk1, permk2], color=[.2, .2, .2], linewidth=1.5, linestyle='-', zorder=6)  
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result_outband_nozero.params[0]+(ols_result_outband_nozero.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label1+'; no zeros', color=color1, zorder=4)
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result_inband_nozero.params[0]+(ols_result_inband_nozero.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label2+'; no zeros', color=color2, zorder=5)
    ax.legend()
    
    xfit=np.stack((comb_xdata, xyesno, xyesno_mult), axis=-1)
    yfit=np.array([comb_ydata])    
    yfit=yfit.T
    

    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    p=ols.fit().params
    print('Combined regression for data points, no zeros')
    print(ols_result.summary())
    
    
    
    ##plot the binned data, no zeros and do separate fits. 
    
    meltfraclist1_o=meltfraclist1[np.where(inkmeanlist1>(1.78E-18))]
    meltfraclist2_o=meltfraclist2[np.where(inkmeanlist2>(1.78E-18))]
    meltfraclist1=meltfraclist1[np.where(kmeanlist1>(1.78E-18))]
    meltfraclist2=meltfraclist2[np.where(kmeanlist2>(1.78E-18))]
    
    
    kmeanlist1= kmeanlist1[np.where(kmeanlist1>1.78E-18)]
    kmeanlist2= kmeanlist2[np.where(kmeanlist2>1.78E-18)]
    inkmeanlist1= inkmeanlist1[np.where(inkmeanlist1>1.78E-18)]
    inkmeanlist2= inkmeanlist2[np.where(inkmeanlist2>1.78E-18)]
    
    
    
    
    
    fig=plt.figure()
    ax=plt.axes()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 0.20])
    ax.set_ylim([1E-19, 1E-14])
    ax.scatter(meltfraclist1, kmeanlist1, marker='s', s=size, facecolor=color2, edgecolor=color2, zorder=4)
    ax.scatter(meltfraclist2, kmeanlist2, marker='s', s=size, facecolor=color2, edgecolor=color2, zorder=4)
    ax.scatter(meltfraclist1_o, inkmeanlist1, marker='s', s=size, facecolor='none', edgecolor=color1, zorder=5)
    ax.scatter(meltfraclist2_o, inkmeanlist2, marker='s', s=size, facecolor='none', edgecolor=color1, zorder=5)
    ax.plot([x1, x2], [permk1, permk2], color='k', linewidth=1.5, linestyle='-')  
    
    
    ###Fit the data and do the regression comparison with the binned and averaged data
    kmeanlist_comb=np.concatenate([kmeanlist1, kmeanlist2], axis=None)
    inkmeanlist_comb=np.concatenate([inkmeanlist1, inkmeanlist2], axis=None)

    ##do the regression fit for the averaged data
    meltfrac=np.concatenate([meltfraclist1, meltfraclist2], axis=None)
    meltfrac_o=np.concatenate([meltfraclist1_o, meltfraclist2_o], axis=None)
    xfit=np.array(np.log10([meltfrac]))
    xfit=xfit.T
    yfit=np.array(np.log10([kmeanlist_comb]))    
    yfit=yfit.T

    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print(label1+'Binned Data, no zeros')
    print(ols_result.summary())
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result.params[0]+(ols_result.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label1+'; Binned data, no zeros', color=color2)
    
    ##do the regression fit for the averaged data
    xfit=np.array(np.log10([meltfrac_o]))
    xfit=xfit.T
    yfit=np.array(np.log10([inkmeanlist_comb]))    
    yfit=yfit.T
    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit


    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    predvals=[]
    for i in range(len(xplotter)):
        predvals.append(10**((ols_result.params[0]+(ols_result.params[1] * xplotter[i]))))
    
    ax.plot(10**xplotter, predvals, label=label2+'; Binned data, no zeros', color=color1)
    print(label2+'; Binned Data, no zeros')
    print(ols_result.summary())
    ax.legend()
    
    
    
    
    comb_xdata=np.concatenate([meltfraclist1, meltfraclist2, meltfraclist1_o, meltfraclist2_o], axis=None)
    xyes=np.full((len(inkmeanlist_comb), 1), 1)
    xno=np.full((len(kmeanlist_comb), 1), 0)
    comb_ydata=np.concatenate([np.log10(kmeanlist_comb), np.log10(inkmeanlist_comb)], axis=None)
    xyesno=np.concatenate((xno, xyes), axis=None)
    xyesno_mult=comb_xdata*xyesno
    
    xfit=np.stack((comb_xdata, xyesno, xyesno_mult), axis=-1)
    yfit=np.array([comb_ydata])    
    yfit=yfit.T


    xfitshape=np.shape(xfit)
    N = (xfitshape[0])
    p = (xfitshape[1]) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = xfit
    
    ols = sm.OLS(yfit, X_with_intercept)
    ols_result = ols.fit()
    print('Combined regression for binned data, not including zeros')
    print(ols_result.summary())
    # def linfit_dummy(x, xyesno, xyesnomult, intercept, slopeone, slopetwo, slopethree):
    #     return(intercept+(slopeone*x)+(slopetwo*xyesno)+(slopethree*xyesno_mult))
    # popt, pcov = curve_fit(linfit_dummy, comb_xdata, xyesno,xyesno_mult, comb_ydata)
    # # retrieve parameter values
    # a = popt[0]
    # b = popt[1]
    # c = popt[2]
    # d = popt[3]
    # print(a, b, c, d)
    
    
    return



def find_k_aniso(datasheet):
    num_subs=len(datasheet.index)
    array_of_half=np.zeros(num_subs)
    array_of_half.fill(0.5)
    array_of_reslim=np.zeros(num_subs)
    array_of_reslim.fill(1E-18)
    
    x_perm=datasheet['kX']*1E-16
    y_perm=datasheet['kY']*1E-16
    z_perm=datasheet['kZ']*1E-16
    print(x_perm, array_of_reslim)
    
    aniso_num=np.multiply((np.add(x_perm, y_perm)), array_of_half)
    permaniso=np.divide(aniso_num, z_perm)
    permaniso[permaniso== np.inf ]=np.nan
    
    datasheet['Permeability Anisotropy']=permaniso
    
    
    x_perm=np.add(datasheet['kX']*1E-16, array_of_reslim)
    y_perm=np.add(datasheet['kY']*1E-16, array_of_reslim)
    z_perm=np.add(datasheet['kZ']*1E-16, array_of_reslim)
    
    aniso_num=np.multiply((np.add(x_perm, y_perm)), array_of_half)
    permaniso=np.divide(aniso_num, z_perm)
    
    datasheet['Permeability Anisotropy (Upper Bound)']=permaniso












