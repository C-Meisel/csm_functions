''' This module contains functions to help format and plot Electrochemical Impedance Spectroscopy (EIS)
data. The data files are obtained by a Gamry potentiostat. The files are .DTA files
This module is to be using in conjunction with the hybrid_DRT package developed by Jake Huang
This is a second attempt at making plotting functions (more streamlined than plotting found in eis)
# C-Meisel
'''

'Imports'
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from hybdrt import fileload as fl
import os
from matplotlib.ticker import FormatStrFormatter
from io import StringIO

from .data_formatting import *

def plot_ocv(loc : str):
    '''
    Plots OCV vs time from a .DTA file generated from a Gamry Potentiostat

    Parameter:
    ----------
    loc,str: Location of the .DTA file that contains the OCV data.

    returns: the plot of the figure
    '''
    df = get_ocv_data(loc) # Reads the .DTA file and saves the OCV data as a dataframe

    df_useful = df[['s','V']]
    fig, ax = plt.subplots()
    ax.plot(df_useful['s'],df_useful['V'],'ko')

    # - Formatting
    ax.set_xlabel('Time (s)',fontsize='xx-large')
    ax.set_ylabel('Voltage (V)',fontsize='xx-large')
    ax.tick_params('both',labelsize = 'x-large')

    plt.tight_layout()
    plt.show()

def plot_peis(area:float, loc:str, ohmic_rtot:np.array = None, pro:bool = False, cut_inductance:bool = False,**plot_args):
    '''
    Plots Zreal and Zimag from a DTA file of potentiostatic electrochemical impedance spectroscopy (PEIS) 
    data taken from a Gamry potentiostat

    Parameters
    ----------
    area, float:
        The active cell area in cm^2
    loc, str:
        Location of the .DTA file that contains the EIS data.
    ohmic_rtot, np.array:
        The first number is the ohmic resistance and the second is the total resistance.
    pro, bool:
        If true, plots a dark mode EIS data.  If false, plots a light mode EIS data. The pro version
        has more experimental features.
    cut_inductance, bool: (default = False)
        If this is set to true, the negative inductance values at the beginning of the DataFrame
    plot_args, dict:
        Any arguments that are passed to the plot function.

    Return --> The figure. It also plots and shows one EIS spectra
    '''
    df = get_eis_data(loc) # Reads the .DTA file and saves the PEIS data as a dataframe

    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']]
    
    if cut_inductance == True:
        df_useful = cut_induct(df_useful)

    # Plotting
    if pro == False:
        fig, ax = plt.subplots()

        # -- Formatting
        ax.plot(df_useful['ohm'],df_useful['ohm.1'],'o',**plot_args,color = '#21314D',markersize = 10)
        ax.set_xlabel('Zreal (\u03A9 cm$^2$)', size = 'xx-large')
        ax.set_ylabel('-Zimag (\u03A9 cm$^2$)', size = 'xx-large')
        dashes = [6,2,2,2,6,2,2,4,6,2,6,4]
        ax.axhline(y=0, color='#c1741d',dashes=dashes)
        ax.margins(y=0.1)
        ax.axis('scaled') #Keeps X and Y axis scaled 1 to 1

        # - Excessive formatting
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize='x-large')
        plt.tight_layout()
        plt.show()

        return fig

    elif pro == True: # Plot like a Pro, breh
        fig,ax = plt.subplots()
        ax.plot(df_useful['ohm'],df_useful['ohm.1'],'D',**plot_args,color = '#336699',alpha=0.69,markeredgewidth=0.2,
            markeredgecolor='#cfcfcf',ms=8,antialiased=True) # #21314D is the color of Mines Navy blue ##09213c #0c2d52

        # --- Colors
        background = '#121212' # '#000023'
        fig.set_facecolor(background)
        ax.set_facecolor(background)
        frame_color = '#a0a0a0'
        ax.spines['bottom'].set_color(frame_color)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color(frame_color)
        ax.spines['right'].set_visible(False)
        ax.margins(y=0.1)
        ax.tick_params(colors=frame_color, which='both')

        # --- Horizontal dashed line
        dashes = [6,2,2,2,6,2,2,4,6,2,6,4]
        ax.axhline(y=0, color='#c1741d',dashes=dashes) # #D2492A is the color of Mines orange #7f4c13 #c1741d #D98B21

        # --- cutting down the X and Y axis and adding a grid
        if ohmic_rtot is not None:
            ax.set_xticks(ohmic_rtot)
            ax.spines['bottom'].set_bounds(ohmic_rtot[0], ohmic_rtot[1])
            y_bottom = df_useful['ohm.1'].iloc[0]
            if ohmic_rtot[1]<=1: # --- Cutting out inductance effects
                y_bottom = -0.1
            ax.spines['left'].set_bounds(y_bottom,df_useful['ohm.1'].max())
            ax.spines['left'].set_capstyle('butt')
            ax.tick_params(axis='x',length=10,direction='inout')
            ax.grid(axis='x',linestyle='--',color='#a0a0a0',alpha=0.35) # visible=None

        # --- Labels
        font = 'Helvetica'
        ax.set_xlabel('Zreal (\u03A9 cm$^2$)',color=frame_color,fontsize='x-large',family=font)
        ax.set_ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',color=frame_color,fontsize='x-large',family=font)
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.axis('scaled')
        if ohmic_rtot is not None and ohmic_rtot[1]<=1: # --- Cutting out inductance effects
            ax.set_ylim(y_bottom*1.01,df_useful['ohm.1'].max()*1.1)
        plt.tight_layout() 
        plt.show()

        return fig

def plot_peiss(area:float, condition:str, loc:str, ncol:int=1,
                legend_loc:str='best', cut_inductance:bool = False,**plot_args): #Enables multiple EIS spectra to be stacked on the same plot ,color=-1
    '''
    Enables multiple EIS spectra to be stacked on the same plot.
    
    Parameters:
    -----------
    area,float: 
        The active cell area in cm^2
    condition,str: 
        The condition of the EIS data. This is what will be in the legend.
    loc,str:
        Location of the .DTA file that contains the EIS data.
    ncol,int:
        The number of columns in the legend of the plot.
    legend_loc,str: 
        The location of the legend. Best is the best spot, Outside places the legend
        outside the plot.
    cut_inductance, bool: (default = False)
        If this is set to true, the negative inductance values at the beginning of the DataFrame
    plot_args, dict: 
        Any arguments that are passed to the plot function.

    Return --> none but it plots the figure
    '''
    df = get_eis_data(loc) # Reads the .DTA file and saves the PEIS data as a dataframe

    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']] #returns the useful information

    if cut_inductance == True:
        df_useful = cut_induct(df_useful)
        legend_loc = 'outside'

    # ----- Plotting
    # with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
    #     plt.plot(df_useful['ohm'],df_useful['ohm.1'],'o',**plot_args,label = condition) #plots data
    plt.plot(df_useful['ohm'],df_useful['ohm.1'],'o',**plot_args,label = condition) #plots data

    plt.xlabel('Zreal (\u03A9 cm$^2$)',size=18) #\u00D7
    plt.ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',size=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    dashes = [6,2,2,2,6,2,2,4,6,2,6,4]
    plt.axhline(y=0, color='k',dashes=dashes)
    plt.axis('scaled') #Keeps X and Y axis scaled 1 to 1
    plt.locator_params(axis='y', nbins=3) # Sets the number of ytick labels but still has matplotlib place them

    # If statement determines legend loc
    if legend_loc == 'best': 
        plt.legend(loc='best',fontsize='x-large')
    elif legend_loc == 'outside':
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),ncol=ncol)
    else: #this else statement is a default if one of the earlier statements causes an error
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),ncol=ncol)

    plt.tight_layout()

def plot_sc_peiss(area:float, condition:str, loc:str, ncol:int=1,
                legend_loc:str='best', cut_inductance:bool = False,**plot_args):
    '''
    Enables multiple EIS spectra to be stacked on the same plot.
    
    Parameters:
    -----------
    area,float: 
        The active cell area in cm^2
    condition,str: 
        The condition of the EIS data. This is what will be in the legend.
    loc,str:
        Location of the .DTA file that contains the EIS data.
    ncol,int:
        The number of columns in the legend of the plot.
    legend_loc,str: 
        The location of the legend. Best is the best spot, Outside places the legend
        outside the plot.
    cut_inductance, bool: (default = False)
        If this is set to true, the negative inductance values at the beginning of the DataFrame
    plot_args, dict: 
        Any arguments that are passed to the plot function.

    Return --> none but it plots the figure
    '''
    df = get_eis_data(loc) # Reads the .DTA file and saves the PEIS data as a dataframe

    area = area/2 # Cuts area in half because there are two electrodes in a symmetric cell

    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']] #returns the useful information

    df_useful = cut_ohmic(df_useful)

    if cut_inductance == True:
        df_useful = cut_induct(df_useful)
        legend_loc = 'outside'

    # ----- Plotting
    plt.plot(df_useful['ohm'],df_useful['ohm.1'],'o',**plot_args,label = condition) #plots data

    plt.xlabel('Zreal (\u03A9 cm$^2$)',size=18) #\u00D7
    plt.ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',size=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    dashes = [6,2,2,2,6,2,2,4,6,2,6,4]
    plt.axhline(y=0, color='k',dashes=dashes)
    plt.axis('scaled') #Keeps X and Y axis scaled 1 to 1
    plt.locator_params(axis='y', nbins=3) # Sets the number of ytick labels but still has matplotlib place them

    # If statement determines legend loc
    if legend_loc == 'best': 
        plt.legend(loc='best',fontsize='x-large')
    elif legend_loc == 'outside':
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),ncol=ncol)
    else: #this else statement is a default if one of the earlier statements causes an error
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),ncol=ncol)

    plt.tight_layout()


def plot_eis_ocvs(loc:str, label:str, ymin:float=1.00, ymax:float=1.10, ncol:int=1):
    '''
    Plots the ocv that is taken right before the EIS data. This function can stack to plot
    multiple ocv files on the same plot. Same as peiss.

    Parameters:
    -----------
    loc, str:
        Location of the .DTA file that contains the EIS data.
    label, str:
        The label of the ocv data. this will be in the plot legend
    ymin,float:
        The minimum y value of the plot. Defaults to 1.00.
    ymax, float: 
        The maximum y value of the plot. Defaults to 1.10.
    ncol, int:
        The number of columns in the legend of the plot.

    Return --> none but it plots the figure
    '''
    df = get_eis_ocv_data(loc) # Reads the .DTA file and saves the OCV data as a dataframe

    df_useful = df[['s','V vs. Ref.']] #returns the useful information
    
    plt.plot(df_useful['s'],df_useful['V vs. Ref.'],'o',label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.tight_layout()
    plt.legend(ncol=ncol)
    plt.ylim(ymin,ymax)

def plot_ivfc(area:float, loc:str):
    '''
    Extracts the polarization data from a fuel cell mode polarization curve taken with a Gamry potentiostat.
    Plots the IV curve and calculates and plots the corresponding power density curve
    Calculates the max power density and prints it on the plot

    Parameters:
    -----------
    area, float:
        The active cell area in cm^2
    loc, str:
        Location of the .DTA file that contains the IV data.

    Return --> none but it plots the figure and shows it    
    '''
    
    df = get_iv_data(loc) # Reads the .DTA file and saves the IVFC data as a dataframe

    # dt = df['s'].iloc[-1]-df['s'].iloc[0]
    # da = df['A'].iloc[-1]-df['A'].iloc[0]
    # print(sweep_rate)

    # -calculations and only keeping the useful data
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)
    df_useful = df[['V','A','W']]
    
    # --- Plotting
    fig, ax1 = plt.subplots()
   
    # - IV plotting
    color = '#21314D' #Navy color
    ax1.set_xlabel('Current Density ($A/cm^2$)',fontsize = 'xx-large')
    ax1.set_ylabel('Voltage (V)', color=color, fontsize = 'xx-large')
    ax1.plot(df_useful['A'], df_useful['V'],'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color,labelsize = 'x-large')
    ax1.tick_params(axis='x',labelsize = 'x-large')
    
    # - Power density plotting
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = '#D2492A' # Mines orange color
    ax2.set_ylabel('Power Density ($W/cm^2$)', color=color2, fontsize = 'xx-large')  # we already handled the x-label with ax1
    ax2.plot(df_useful['A'], df_useful['W'], 'o',color=color2) 
    ax2.tick_params(axis='y', labelcolor=color2, labelsize = 'x-large')
    ax2.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    # - Calculating and printing max values onto the graph
    max_w = df_useful['W'].max() #finds maximum power density
    max_v = df.loc[df.index[df['W'] == max_w],'V'].iloc[0] #finds voltage of max power density
    max_ws = f'{round(max_w,3)}' #sets float to a string
    max_vs = f'{round(max_v,3)}'

    plt.figtext(0.28,0.21,r'$P_{max} = $'+max_ws+r' $W/cm^2 at$ '+max_vs+r'$V$',size='x-large',weight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_ivfcs(curves_conditions:tuple, print_Wmax:bool=False, cmap:str=None,leg_cols:int=None):
    '''
    Plots multiple IV and power density curves, input is a tuple of (area,condition,location of IV curve)

    Parameters
    ----------
    curves_conditions,tuple: 
        A tuple containing data to plot and label the curve.
        The order is: (area,condition,location of IV curve)
    print_Wmax, bool: (default is True)
        Prints out Wmax in the terminal
    cmap,str: (default is None)
        If a colormap is defined here it will be used for the plots
    leg_cols,int: (default is None)
        how many columns the legend will have. If none, this will be based of the abount of curves being plot

    Return --> none but it plots the figure and shows it
    '''
    
    fig,ax1 = plt.subplots() # Initializing plot
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    w_max = [] #initializing a list to store the max power densities

    # --- Setting up colormap if a colormap is chosen
    if cmap is not None:
        # --- Setting up an array for the color map
        color_space = np.linspace(0,1,len(curves_conditions)) # array from 0-1 for the colormap for plotting
        c = 0 # index of the color array
        cmap = cmap
        # print(color_space)

    for iv in curves_conditions:
        loc = iv[2]
        df = get_iv_data(loc) # Reads the .DTA file and saves the IVFC data as a dataframe

        #calculations and only keeping the useful data
        area = iv[0]
        df['A'] = df['A'].div(-area)
        df['W'] = df['W'].div(-area)
        df_useful = df[['V','A','W']]

        if print_Wmax == True:
            w_max.append(df_useful['W'].max()) #finds maximum power density

        if cmap == None: # This if statement determines whether the default colors or a cmap is used
            # --- IV plotting:
            label = iv[1]
            ax1.plot(df_useful['A'], df_useful['V'],'o',fillstyle='none',label=label)

            # --- Power Density plotting
            ax2.plot(df_useful['A'], df_useful['W'],'o',label=label)
        else:
            # --- IV plotting:
            color = cmap(color_space[c])
            label = iv[1]
            ax1.plot(df_useful['A'], df_useful['V'],'o',fillstyle='none',label=label,color=color)

            # --- Power Density plotting
            ax2.plot(df_useful['A'], df_useful['W'],'o',label=label,color=color)
            c = c+1

    # --- Printing Max power density if desired:
    if print_Wmax == True:
        for i in range(len(w_max)):
            max_ws = f'{round(w_max[i],3)}'
            print('Max Power Density of the ' + curves_conditions[i][1] + ' condition' + ' is: '
                +'\033[1m'+ max_ws + '\033[0m'+' W/cm\u00b2')

    # ----- Plot Formatting:
    ax1.set_xlabel('Current Density (A/cm$^2$)',fontsize='xx-large')
    ax1.set_ylabel('Voltage (V)',fontsize='xx-large')
    ax1.tick_params(axis='both', which='major', labelsize='x-large')

    ax2.set_ylabel('Power Density (W/cm$^2$)',fontsize='xx-large')  # we already handled the x-label with ax1
    ax2.tick_params(axis='both', which='major', labelsize='x-large')

    # --- Legend Formatting
    if leg_cols == None:
        num_curves = len(curves_conditions)
        if num_curves <=4:
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=num_curves,handletextpad=0.02,columnspacing=1)
        elif num_curves <=8:
            ncol = int(round(num_curves/2))
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=ncol,handletextpad=0.02,columnspacing=1)
        elif num_curves <=12:
            ncol = int(round(num_curves/3))
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=num_curves,handletextpad=0.02,columnspacing=1)
    else:
        ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=leg_cols,handletextpad=0.02,columnspacing=1)

    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()

def plot_ivec(area:float, loc:str, CD_at_V:float = 1.3):
    '''
    Plots IV curve in EC mode and displays the current density at a desired voltage on the plot

    Parameters
    ----------
    area, float: 
        The active cell area in cm^2
    loc, string: 
        The location of the .DTA file that contains the IVEC curve
    CD_at_V, float: (default is 1.3)
        Current Density at a certain voltage

    Return --> none but it plots the figure and shows it
    '''
    
    df = get_iv_data(loc) # Reads the .DTA file and saves the IVEC data as a dataframe

    # ------- calculations and only keeping the useful data
    df['A'] = df['A'].div(area)
    df_useful = df[['V','A']]
    
    'Plotting'
    fig, ax1 = plt.subplots()

    'IV plotting'
    color = '#21314D'
    ax1.set_xlabel('Current Density ($A/cm^2$)',fontsize = 'xx-large')
    ax1.set_ylabel('Voltage (V)',fontsize = 'xx-large')
    
    if df_useful['V'].loc[0] <= 0: # This is put in because the Gamry 3000 and 5000 have different signs on the voltage of IVEC curve
        sign = -1
    else:
        sign = 1
    ax1.plot(df_useful['A'], sign*df_useful['V'],'o', color=color)
    ax1.tick_params(axis='both',labelsize = 'x-large')

    # ------- Calculating and printing current density at a given voltage
    CD_at_mod = CD_at_V-0.01
    current_density15 = -df_useful[abs(df_useful['V'])>=CD_at_mod].iloc[0,1] # Finds the current density of the first Voltage value above the desired Voltage
    V15 = df_useful[abs(df_useful['V'])>=CD_at_mod].iloc[0,0] # Same as before but returns the exact voltage value
    current_density15_string = f'{round(current_density15,3)}'
    V15_string = f'{round(V15,3)}'
    plt.figtext(0.28,0.21,current_density15_string+r' $A/cm^2\:at$ '+V15_string+r'$V$',size='x-large',weight='bold') #placing value on graph
    
    plt.tight_layout()
    plt.show()

def plot_ivecs(area:float,condition:str,loc:str, **plot_args:dict):
    '''
    Plots multiple electrolysis cell mode IV curves on same plot. This function can stack to plot
    multiple IVEC files on the same plot. Same as peiss.

    Parameters:
    -----------
    area, float: 
        The active cell area in cm^2
    condition, string: 
        The condition of the IV curve. This will be the label of the curve in the legend
    loc, string: 
        The location .DTA file that contains the IVEC curve
    plot_args, dict: (optional)
        passes arguments to the plot_peis function

    Return --> none but it plots the figure
    '''
    
    df = get_iv_data(loc) # Reads the .DTA file and saves the IVEC data as a dataframe

    # ------ calculations and only keeping the useful data
    df['A'] = df['A'].div(area)
    df_useful = df[['V','A']]
    if df_useful['V'].loc[0] <= 0: #this is put in because the Gamry 3000 and 5000 have different signs on the voltage of IVEC curve
        sign = -1
    else:
        sign = 1

    if df_useful['A'].loc[0] <= 0: #this is put in because the Gamry 3000 and 5000 have different signs on the voltage of IVEC curve
        a_sign = 1
    else:
        a_sign = -1

    # ------- Plotting
    with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
        plt.plot(a_sign*df_useful['A'], sign*df_useful['V'],'o', label=condition,markersize=9, **plot_args)
    plt.xlabel('Current Density ($A/cm^2$)', fontsize='xx-large')
    plt.ylabel('Voltage (V)', fontsize='xx-large')
    plt.legend(loc='best', fontsize = 'x-large')
    plt.xticks(fontsize = 'x-large')
    plt.yticks(fontsize = 'x-large')
    plt.tight_layout()

def plot_galvanoStb(folder_loc:str, fit:bool = True, fontsize:int = 20, smooth:bool=False,
                     first_file:str = 'default', clean_axis=True, quant_stb = 'mv', **kwargs):
    '''
    Looks through the specified folder and plots all the galvanostatic stability testing data in one plot, 
    and fits it. This function compliments the gamry sequence I use for fuel cell mode stability testing

    Parameters
    ----------
    folder_loc, string:
        The location of the folder that contains the .DTA files to be plotted
    fit, bool: (default = True)
        Whether or not to linearly fit the data and print on the plot
    fontsize,int: (default is 16)
        The fontsize of the words on the plot
    smooth, bool: (default = False)
        Whether or not to smooth the data
    first_file, string: (default = 'default')
        Identifies the first file in the stability test. This is used as a time reference
        If 'default' then the first file taken is used
        If you want to change the first file, put the loc in place of 'default'
    clean_axis, string: (default = True)
        When true, plots the Voltage axis with less points
        This is a much cleaner way to plot and looks better when presenting
        however it clears out the Y axis so if you want to zoom in on an image, set this to False
    quant_stb, str: (default = 'mv')
        This is how the stability data gets reported on the graph. There are many ways to quantify stability
        if = mv, the slope of the graph is multiplied by 1,000,000 to report the data in mV/khrs
        if = potential, then the slope is multiplied by 1,000,000 then divided by the starting potential to get %/khrs 
        (% of the starting potential lost during testing)
        if = overpotential, then the slope is multiplied by 1,000,000 then divided by the starting potential-OCV to get %/khrs
        (% of the overpotential lost during testing)
        if = all, then all three of the above options are printed at on the figure
    plot_args, dict:
        Any arguments that are passed to the plot function.
        generally for presentations I use markersize = 20

    Return --> none but it plots the figure and shows it
    '''

    files = os.listdir(folder_loc) # obtaining a list of the files in the desired folder
    useful_files = [] # initializing a list for the useful files

    # ------- Taking out all of the galvanostatic files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('GS')!=-1) and (file.find('.DTA')!=-1) and (file.find('Deg')!=-1):
            #Extracting the file number (used for sorting, yeah this is probably a roundabout way of doing it)
            start, end = file.rsplit('#',1)# cutting everything out before the # so only the number and file extension is left
            fnum, fileExt = end.rsplit('.',1) #cutting off the file extension leaving only the number as a string
            index = int(fnum) #Converts the file number to a string
            useful_file = (file,index)
            useful_files.append(useful_file)
    
    # ------- Sorting the files
    useful_files.sort(key=lambda x:x[1]) #Sorts by the second number in the tuple
    sorted_useful_files, numbers = zip(*useful_files) #splits the tuple
    sorted_useful_files = [folder_loc + '/' + f for f in sorted_useful_files] #Turning all files from their relative paths to the absolute path

    # ------- Getting the first time
    for file in files: #Finding the first file
        if file.find('Deg__#1.DTA')!=-1 and file.find('OCV')!=-1: # 'Deg__#1.DTA'
            file1 = os.path.join(folder_loc,file)

    if first_file == 'default': # if another file is specified as the first file, this file will be used to find T0
        T0_stamp = fl.get_timestamp(file1) # gets time stamp from first file
        t0 = T0_stamp.strftime("%s") # Converting Datetime to seconds from Epoch
    else:
        file1 = first_file
        T0_stamp = fl.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch

    
    # ------- Combining all DataFrames
    dfs = [] #Initializing list of dfs
    length = len(sorted_useful_files) #gets length of sorted useful files
    
    for i in range(0,length,1):
        loc = os.path.join(folder_loc,sorted_useful_files[i]) # Creates a file path to the file of choice
        df = get_iv_data(loc) # Reads the .DTA file and saves the galvanostatic data as a dataframe

        start_time = fl.get_timestamp(sorted_useful_files[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)

    cat_dfs = pd.concat(dfs,ignore_index=True)# (s) Combine all the DataFrames in the file folder
    cat_dfs.sort_values(by=['s'])
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    end_time = int(cat_dfs['s'].max())
    start_time = int(cat_dfs['s'].min()) #is nt necessarily 0. Maybe you are only looking at a small portion of the data

    # ------- plotting:
    fig, ax = plt.subplots()

    if len(kwargs)==0: # Basically if hte dictionary is empty
        kwargs['markersize'] = 20
    
    if smooth == False:
        ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k',**kwargs) # For presentations set markersize to 20

    if smooth == True: #Averages out data points to smooth the curve
        bin_size = 50
        bins = cat_dfs['V vs. Ref.'].rolling(bin_size)
        moving_avg_voltage = bins.mean()
        ax.plot(cat_dfs['s'],moving_avg_voltage,'k',**kwargs)

    # ------- Plot formatting
    ax.set_xlabel('Time (hrs)',fontsize = fontsize)
    ax.set_ylabel('Voltage (V)',fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2) #changing tick label size
    ocv = get_init_ocv(folder_loc) # folder_loc
    ax.set_xticks([start_time,end_time])
    
    if clean_axis == True:
        ax.set_yticks([0,0.8,ocv])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.yaxis.labelpad = -20
    
    ax.set_xticklabels([start_time,end_time],fontsize=18)
    ax.spines['bottom'].set_bounds(start_time, end_time)
    ax.xaxis.labelpad = -15
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,ocv)

    # ------- fitting and writing slope on graph:
    if fit == True:
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        
        if quant_stb == 'mv':
            mp = m * 1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' mV/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'potential':
            init_v = get_init_v(folder_loc,fc=True)
            mp = ((m * 1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            ms = f'{round(mp,2)}'
            plt.figtext(0.40,0.17, ms+' %/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'overpotential':
            init_v = get_init_v(folder_loc,fc=True)
            mp = ((m * 1000)/(ocv-init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' %η/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'all':
            init_v = get_init_v(folder_loc,fc=True) # folder_loc

            m_mv = m * 1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            m_mvs = f'{round(m_mv,2)}'
            plt.figtext(0.43,0.27, m_mvs + ' mV/khrs',weight='bold',size='xx-large')

            m_volt = ((m * 1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            m_volts = f'{round(m_volt,2)}'
            plt.figtext(0.43,0.20, m_volts + ' %/khrs',weight='bold',size='xx-large')

            m_over = ((m * 1000)/(ocv-init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            m_overs= f'{round(m_over,2)}'
            plt.figtext(0.43,0.13, m_overs + ' %η/khrs',weight='bold',size='xx-large')

    plt.tight_layout()

    plt.show()

def plot_ocvStb(folder_loc:str, fit:bool=True, first_file = 'default', fontsize = 20,
                clean_axis=True, quant_stb = 'mv'):
    '''
    Looks through the specified folder and plots all the ocv stability test data in one plot and fits it.
    This function compliments the gamry sequence I use for stability testing.
    
    Parameters:
    -----------
    folder_loc, string: (path to a directory)
        The location of the folder containing the files to be plotted
    fit, bool: (default = True)
        Whether or not to fit the data
    first_file, string: (default = 'default')
        Identifies the first file in the stability test. This is used as a time reference
        If 'default' then the first file taken is used
        If you want to change the first file, put the loc in place of 'default
    fontsize, int: (default = 16)
        The font size of the words on the plot
    quant_stb, str: (default = 'mv')
        This is how the stability data gets reported on the graph. There are many ways to quantify stability
        if = mv, the slope of the graph is multiplied by 1,000,000 to report the data in mV/khrs
        if = potential, then the slope is multiplied by 1,000,000 then divided by the starting potential to get %/khrs 
        (% of the starting potential lost during testing)
        if = overpotential, then the slope is multiplied by 1,000,000 then divided by the starting potential-OCV to get %/khrs
        (% of the overpotential lost during testing)
        if = all, then all three of the above options are printed at on the figure

    Return --> none, but it plots the data, fits it, and shows it
    '''

    files = os.listdir(folder_loc)
    useful_files = [] #initializing a list for the useful files

    # ======= Taking out all of the ocv files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('OCV')!=-1) and (file.find('.DTA')!=-1) and (file.find('Deg')!=-1):
            useful_files.append(os.path.join(folder_loc,file))

    # ======= Finding the first file
    for file in useful_files: #Finding the first file
        if file.find('Deg__#1.DTA')!=-1 and file.find('OCV')!=-1:
            file1 = os.path.join(folder_loc,file)
    if first_file == 'default': #if another file is specified as the first file, this file will be used to find T0
        T0_stamp = fl.get_timestamp(file1) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
    else:
        T0_stamp = fl.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch

    # ======= Concatenating all of the data frames
    dfs = [] #Initializing list of dfs
    length = len(useful_files) #gets length of sorted useful files
    
    for i in range(0,length,1):
        df = get_iv_data(useful_files[i]) # Reads the .DTA file and saves the galvanostatic data as a dataframe
        start_time = fl.get_timestamp(useful_files[i]).strftime("%s") # Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)

    # - Combining all of the DataFrames
    cat_dfs = pd.concat(dfs,ignore_index=True) # (s) Combine all the DataFrames in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and

    # - Finding the start and end time
    end_time = int(cat_dfs['s'].max())
    start_time = int(cat_dfs['s'].min()) #is not necessarily 0. Maybe you are only looking at a small portion of the data

    # ======= Plotting
    fig, ax = plt.subplots()
    ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'], '.k', markersize=20)

    # -- Plot formatting    
    ax.set_xlabel('Time (hrs)',fontsize = fontsize)
    ax.set_ylabel('Voltage (V)',fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2) #changing tick label size
    ocv = get_init_ocv(folder_loc)
    ax.set_xticks([start_time,end_time])
    
    if clean_axis == True:
        ax.set_yticks([0,0.8,ocv])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.yaxis.labelpad = -15
    
    ax.set_xticklabels([start_time,end_time],fontsize=18)
    ax.spines['bottom'].set_bounds(start_time, end_time)
    ax.xaxis.labelpad = -15
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,1.2)

    # ======= fitting and writing slope on graph: 
    if fit == True:
        # === Converting cat_dfs to floats so polyfit can work (sometimes the df columns are object types)
        cat_dfs['s'] = cat_dfs['s'].astype(float, errors = 'raise')
        cat_dfs['V vs. Ref.'] = cat_dfs['V vs. Ref.'].astype(float, errors = 'raise')

        # === Fitting and plotting
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        
        if quant_stb == 'mv':
            mp = m * 1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' mV/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'potential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * 1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            ms = f'{round(mp,2)}'
            plt.figtext(0.40,0.17, ms+' %/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'overpotential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * 1000)/(ocv-init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' %η/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'all':
            init_v = get_init_v(folder_loc,fc=True)

            m_mv = m * 1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            m_mvs = f'{round(m_mv,2)}'
            plt.figtext(0.39,0.27, m_mvs + ' mV/khrs',weight='bold',size='xx-large')

            m_volt = ((m * 1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            m_volts = f'{round(m_volt,2)}'
            plt.figtext(0.39,0.20, m_volts + ' %/khrs',weight='bold',size='xx-large')

            m_over = ((m * 1000)/(ocv-init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            m_overs= f'{round(m_over,2)}'
            plt.figtext(0.39,0.13, m_overs + ' %η/khrs',weight='bold',size='xx-large')


    plt.show()

def plot_EC_ocvStb(folder_loc:str, fit:bool=True, first_file = 'default', fontsize = 20,
                    clean_axis=True, quant_stb = 'mv'):
    '''
    Looks through the specified folder and plots all the Electrolysis cell mode ocv stability test data 
    in one plot and fits it. This function compliments the gamry sequence I use for EC stability testing.
    
    Parameters
    ----------
    folder_loc, string: 
        The location of the folder containing the files to be plotted
    fit, bool:
        Whether or not to fit the data
    first_file, string:
        Identifies the first file in the stability test. This is used as a time reference
        If 'default' then the first file taken is used though this currently doesn't work
        If you want to change the first file, put the loc in place of 'default
    fontsize, int:
        The font size of the words on the plot
    clean_axis, string: (default = True)
        When true, plots the Voltage axis with less points
        This is a much cleaner way to plot and looks better when presenting
        however it clears out the Y axis so if you want to zoom in on an image, set this to False
    quant_stb, str: (default = 'mv')
        This is how the stability data gets reported on the graph. There are many ways to quantify stability
        if = mv, the slope of the graph is multiplied by 1,000,000 to report the data in mV/khrs
        if = potential, then the slope is multiplied by 1,000,000 then divided by the starting potential to get %/khrs 
        (% of the starting potential lost during testing)
        if = overpotential, then the slope is multiplied by 1,000,000 then divided by the starting potential-OCV to get %/khrs
        (% of the overpotential lost during testing)
        if = all, then all three of the above options are printed at on the figure

    Return --> none, but it plots the data, fits it, and shows it
    '''
    
    files = os.listdir(folder_loc)
    useful_files = [] #initializing a list for the useful files

    # ======= Taking out all of the ocv files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('OCV')!=-1) and (file.find('.DTA')!=-1) and (file.find('ECstability')!=-1):
            useful_files.append(os.path.join(folder_loc,file))
    
    # ======= Finding or defining the first file
    for file in useful_files: #Finding the first file
        if file.find('ECstability__#1.DTA')!=-1:
            file1 = os.path.join(folder_loc,file)
    if first_file == 'default': #if another file is specified as the first file, this file will be used to find T0
        T0_stamp = fl.get_timestamp(file1) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
    else:
        T0_stamp = fl.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
    
    # ======= Concatenating the EC mode OCV stability data
    dfs = [] #Initializing list of dfs
    length = len(useful_files) #gets length of sorted useful files
    for i in range(0,length,1):
        df = get_iv_data(useful_files[i]) # Reads the .DTA file and saves the galvanostatic data as a dataframe

        start_time = fl.get_timestamp(useful_files[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)
    
    cat_dfs = pd.concat(dfs)# (s) Combine all the DataFrames in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    end_time = int(cat_dfs['s'].max())
    start_time = int(cat_dfs['s'].min()) #is not necessarily 0. Maybe you are only looking at a small portion of the data
    
    # ====== Plotting
    fig, ax = plt.subplots()
    ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k',markersize=20)

    # ===== Plot Formatting
    ax.set_xlabel('Time (hrs)',fontsize = fontsize)
    ax.set_ylabel('Voltage (V)',fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2) #changing tick label size
    ocv = get_init_ocv(folder_loc)
    ax.set_xticks([start_time,end_time])

    if clean_axis == True:
        ax.set_yticks([0,ocv,1.3])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.yaxis.labelpad = -20

    ax.set_xticklabels([start_time,end_time],fontsize=18)
    ax.spines['bottom'].set_bounds(start_time, end_time)
    ax.xaxis.labelpad = -15
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,1.5)

    # ======= fitting and writing slope on graph: 
    if fit == True:
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        
        if quant_stb == 'mv':
            mp = m * -1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' mV/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'potential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * -1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            ms = f'{round(mp,2)}'
            plt.figtext(0.40,0.17, ms+' %/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'overpotential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * -1000)/(init_v-ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' %η/khrs',weight='bold',size='xx-large')

        if quant_stb == 'all':
            init_v = get_init_v(folder_loc,fc=False)
            m_mv = m * -1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            m_p = ((m * -1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            m_o = ((m * -1000)/(init_v-ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            
            m_mvs = f'{round(m_mv,2)}'
            m_ps = f'{round(m_p,2)}'
            m_os = f'{round(m_o,2)}'

            plt.figtext(0.39,0.27, m_mvs + ' mV/khrs', weight='bold',size='xx-large')
            plt.figtext(0.39,0.20, m_ps + ' %/khrs', weight='bold',size='xx-large')
            plt.figtext(0.39,0.13, m_os + ' %η/khrs', weight='bold',size='xx-large')

    plt.tight_layout()
    plt.show()

def plot_EC_galvanoStb(folder_loc:str,fit:bool=True,first_file = 'default', fontsize = 20,
                         smooth:bool=False, plot_ocv:bool=False,clean_axis=True, quant_stb = 'mv'):
    '''
    Looks through the specified folder and plots all the Electrolysis cell mode galvanostatic stability test data 
    in one plot and fits it. This function compliments the gamry sequence I use for EC stability testing.
    
    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the folder containing the files to be plotted
    fit, bool: (default is True)
        Whether or not to fit the data
    first_file, str:  (default is 'default')
        Identifies the first file in the stability test. This is used as a time reference
        If 'default' then the first file taken is used
        If you want to change the first file, put the loc in place of 'default
    fontsize, int: (default is 16)
        The font size of the words on the plot
    smooth, bool: (default is False)
        Uses a moving average of 50 bins to average out data points and smooth out the line
    plot_ocv, bool: (default is False)
        Whether or not to plot the OCV as a dotted line on the plot
        The ocv value is the average OCV of the first OCV file in the stability test
    clean_axis, string: (default = True)
        When true, plots the Voltage axis with less points
        This is a much cleaner way to plot and looks better when presenting
        however it clears out the Y axis so if you want to zoom in on an image, set this to False
    quant_stb, str: (default = 'mv')
        This is how the stability data gets reported on the graph. There are many ways to quantify stability
        if = mv, the slope of the graph is multiplied by 1,000,000 to report the data in mV/khrs
        if = potential, then the slope is multiplied by 1,000,000 then divided by the starting potential to get %/khrs 
        (% of the starting potential lost during testing)
        if = overpotential, then the slope is multiplied by 1,000,000 then divided by the starting potential-OCV to get %/khrs
        (% of the overpotential lost during testing)
        if = all, then all three of the above options are printed at on the figure

    Return --> none, but it plots the data, fits it, and shows it
    '''

    files = os.listdir(folder_loc)
    useful_files = [] #initializing a list for the useful files

    # ------- Taking out all of the ocv files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('GS')!=-1) and (file.find('.DTA')!=-1) and (file.find('ECstability')!=-1):
            useful_files.append(os.path.join(folder_loc,file))

    # ------- Finding or defining the first file
    start = '' #Initializing for later
    for file in files: #Finding the first file
        if file.find('ECstability__#1.DTA')!=-1 and file.find('OCV')!=-1:
            file1 = os.path.join(folder_loc,file)
            start = file1
    if first_file == 'default': #if another file is specified as the first file, this file will be used to find T0
        T0_stamp = fl.get_timestamp(file1) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
    else:
        T0_stamp = fl.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
        start = first_file

    # ------- Concatenating the FC mode Galvanostatic DataFrames
    dfs = [] #Initializing list of dfs
    length = len(useful_files) #gets length of sorted useful files
    
    for i in range(0,length,1):
        df = get_iv_data(useful_files[i]) # Reads the .DTA file and saves the galvanostatic data as a dataframe

        # dta2csv(useful_files[i]) #convert DTA to a CSV
        # loc_csv = useful_files[i].replace('.DTA','')+'.csv' #access newly made file
        # data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        
        # for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        #     if row[0] == 'CURVE':
        #         skip = data.line_num+1
        #         break
        
        # df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') # create data frame for a file
        start_time = fl.get_timestamp(useful_files[i]).strftime("%s") # Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)

    cat_dfs = pd.concat(dfs,ignore_index=True)# (s) Combine all the DataFrames in the file folder
    cat_dfs.sort_values(by=['s'])
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    end_time = int(cat_dfs['s'].max())
    start_time = int(cat_dfs['s'].min()) #is not necessarily 0. Maybe you are only looking at a small portion of the data

    # ------- Plotting
    fig, ax = plt.subplots()

    if smooth == False:
        ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k',markersize=20)

    if smooth == True: #Averages out data points to smooth the curve
        bin_size = 50
        bins = cat_dfs['V vs. Ref.'].rolling(bin_size)
        moving_avg_voltage = bins.mean()
        ax.plot(cat_dfs['s'],moving_avg_voltage,'k',markersize=20)

    # ------- Plot formatting
    ax.set_xlabel('Time (hrs)',fontsize = fontsize)
    ax.set_ylabel('Voltage (V)',fontsize = fontsize)

    ocv = get_init_ocv(folder_loc)

    if clean_axis == True:
        ax.set_yticks([0,ocv,1.3])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.yaxis.labelpad = -20

    ax.tick_params(axis='both', which='major', labelsize=fontsize-2) #changing tick label size
    ax.set_xticks([start_time,end_time])
    ax.set_xticklabels([start_time,end_time],fontsize=fontsize)
    
    ax.spines['bottom'].set_bounds(start_time, end_time)
    ax.xaxis.labelpad = -15
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,1.5)

    # ------- Finding the OCV at the beginning this Stability test and plotting it
    if plot_ocv == True:
        loc_csv = start.replace('.DTA','')+'.csv' 
        initial_ocv_data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t")
        skip = 0
        for row in initial_ocv_data: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = initial_ocv_data.line_num+1
                break
            if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
                skip = initial_ocv_data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
        ocv = df['V'].mean()
        ax.axhline(y=ocv,color='k',alpha=0.5,linestyle='--') 

    # ------- fitting and writing slope on graph: 
    if fit == True:
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        
        if quant_stb == 'mv':
            mp = m * -1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' mV/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'potential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * -1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            ms = f'{round(mp,2)}'
            plt.figtext(0.40,0.17, ms+' %/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'overpotential':
            init_v = get_init_v(folder_loc,fc=False)
            mp = ((m * -1000)/(init_v-ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            ms = f'{round(mp,2)}'
            plt.figtext(0.39,0.17, ms+' %η/khrs',weight='bold',size='xx-large')
        
        if quant_stb == 'all':
            init_v = get_init_v(folder_loc,fc=False)
            m_mv = m * -1000000 #Converting the slope into a mV per khrs (*1000 to get from mV to V, *1000 to get to khrs,*-1 for degradation)
            m_p = ((m * -1000)/(init_v))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
            m_o = ((m * -1000)/(init_v-ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v-ocv to get η/khrs, and multiply by 100 to get %η/khrs))
            
            m_mvs = f'{round(m_mv,2)}'
            m_ps = f'{round(m_p,2)}'
            m_os = f'{round(m_o,2)}'

            plt.figtext(0.44,0.27, m_mvs + ' mV/khrs', weight='bold',size='xx-large')
            plt.figtext(0.44,0.20, m_ps + ' %/khrs', weight='bold',size='xx-large')
            plt.figtext(0.44,0.13, m_os + ' %η/khrs', weight='bold',size='xx-large')

    plt.tight_layout()
    plt.show()

def plot_bias_potentio_holds(area:float,folder_loc:str,voltage:bool=True):
    '''
    Plots the 30 minute potentiostatic holds in between the bias EIS spectra.
    This function complements the gamry sequence I use for bias testing.

    Parameters:
    -----------
    area, float:
        The active cell area in cm^2
    folder_loc, string:
        The location of the folder containing the files to be plotted
    voltage, bool: 
        Whether or not to plot the voltage with the current

    Return --> none, but it plots the data and shows it
    '''

    files = os.listdir(folder_loc)
    useful_files = []

    # >>>>>>>> Making a list of all the files of potentiostatic holds during a bias test
    for file in files:
        if (file.find('PSTAT')!=-1) and (file.find('.DTA')!=-1):
            useful_files.append(file)

    # >>>>>>>> Getting the first time for reference
    if len(useful_files) == 10:
        T0_stamp = fl.get_timestamp(os.path.join(folder_loc,'PSTAT_5bias.DTA')) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch

    elif len(useful_files) == 9:
        T0_stamp = fl.get_timestamp(os.path.join(folder_loc,'PSTAT_4bias.DTA')) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch

    # >>>>>>>> extracting the useful information from the files and placing it into a DataFrame
    dfs = [] #Initializing list of dfs
    size = len(useful_files) #gets length of the useful files list

    for i in range(0,size,1):
        loc = os.path.join(folder_loc,useful_files[i]) #Creates a file path to the file of choice
        df = get_iv_data(loc)

        start_time = fl.get_timestamp(loc).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','A','V vs. Ref.']]
        dfs.append(df_useful)

    cat_dfs = pd.concat(dfs)# (s) Combine all the DataFrames in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    cat_dfs['A'] = cat_dfs['A'].div(area)

    # >>>>>>>> plotting:
    if voltage == True:
        # Finding OCV:
        for file in files:
            if (file.find('0bias.DTA')!=-1) and (file.find('OCV')!=-1):
                ocv_path = os.path.join(folder_loc,file)

        df_ocv = get_ocv_data(ocv_path)
        avg_ocv = df_ocv['V'].mean()

        # --- Initializing FIgure
        fig,axs = plt.subplots(2)

        # --- Plotting Bias
        axs[0].set_xlabel('Time (hrs)',size='xx-large')
        axs[0].set_ylabel('Voltage (V)',size='xx-large')
        axs[0].plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k')
        axs[0].axhline(y=avg_ocv, color= 'r', linestyle='--')
        axs[0].tick_params(axis='both', which='major', labelsize='x-large')

        # --- Plotting Current Density
        axs[1].set_xlabel('Time (hrs)',size='xx-large')
        axs[1].set_ylabel('Current Density (A/cm$^2$)',size='xx-large')
        axs[1].plot(cat_dfs['s'],-cat_dfs['A'],'.k')
        axs[1].tick_params(axis='both', which='major', labelsize='x-large')


        # --- Extras
        axs[1].axhline(y=0, color= 'r', linestyle='--') #Plots 0 Bias on the Current density chart
        plt.figtext(0.16,0.85,'Electrolysis',weight='bold',size='large')
        plt.figtext(0.16,0.76,'Fuel Cell',weight='bold',size='large') # Voltage graph

        plt.figtext(0.16,0.39,'Fuel Cell',weight='bold',size='large') # Current graph
        plt.figtext(0.16,0.30,'Electrolysis',weight='bold',size='large')
        plt.tight_layout()
        plt.show()
        
    else:
        fig,ax = plt.subplots()
        ax.set_xlabel('Time (hrs)',size='xx-large')
        ax.set_ylabel('Current Density (A/cm$^2$)',size='xx-large')
        ax.plot(cat_dfs['s'],-cat_dfs['A'],'.k')
        # --- Plot extras
        plt.axhline(y=0, color= 'r', linestyle='--')
        plt.show()

def lnpo2(ohmic_asr:np.array,rp_asr:np.array,O2_conc:np.array): 
    '''
    Plots ln(1/ASRs) as a function of ln(PO2), inputs are arrays of floats

    Parameters:
    -----------
    ohmic_asr, array:
        The ohmic area specific resistance values of the eis spectra at different oxygen concentrations
    rp_asr, array: 
        The rp area specific resistance values of the eis spectra at different oxygen concentrations
    O2_conc, array: 
        The oxygen concentrations that the EIS spectra were taken at

    Returns --> None, but it plots the data and shows it
    '''
    # ----- Making ln arrays:
    ln_O2 = np.log(O2_conc)
    ln_ohmic_asr = np.log(1/ohmic_asr)
    ln_rp_asr = np.log(1/rp_asr)

    # ----- Plotting
    fig,ax = plt.subplots()
    ax.plot(ln_O2,ln_ohmic_asr,'o',color = '#21314D',label = r'ASR$_\mathrm{O}$')
    ax.plot(ln_O2,ln_rp_asr,'o',color = '#D2492A',label = r'ASR$_\mathrm{P}$')

    # ----- Fitting
    mo,bo = np.polyfit(ln_O2,ln_ohmic_asr,1)
    mr,br = np.polyfit(ln_O2,ln_rp_asr,1)
    fit_o = mo*ln_O2 + bo
    fit_r = mr*ln_O2 + br
    ax.plot(ln_O2,fit_o,color = '#21314D')
    ax.plot(ln_O2,fit_r,color = '#D2492A')

    # ----- Formatting
    ax.set_xlabel('ln(O$_2$) (%)')
    ax.set_ylabel('ln(1/ASR) (S/cm$^2$)') #(\u03A9*cm$^2$)
    ax.set_xlim(-1.7,0.1)
    ax.legend()

    # ----- Setting up second x axis
    axx2 = ax.twiny()
    axx2.set_xlabel('Oxygen Concentration (%)')
    axx2.set_xticks(ln_O2)
    axx2.set_xticklabels(O2_conc*100)
    axx2.set_xlim(-1.7,0.1)
    
    # Figtext - If statement is to compensate for the fact that if Rp>ohmic Rp line is lower and visa-versa
    if ohmic_asr[0]<rp_asr[0]: 
        mo_str = f'{round(mo,2)}'
        plt.figtext(0.5,0.84,r'ASR$_\mathrm{O}$ Slope = '+mo_str,weight='bold')
        mr_str = f'{round(mr,2)}'
        plt.figtext(0.5,0.15,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold')  

    elif ohmic_asr[0]>rp_asr[0]:
        mo_str = f'{round(mo,2)}'
        plt.figtext(0.5,0.15,r'ASR$_\mathrm{O}$ Slope = '+mo_str,weight='bold')
        mr_str = f'{round(mr,2)}'
        plt.figtext(0.5,0.84,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold')  

    plt.tight_layout()
    plt.show()

def plot_fc_ec_galvano(folder_loc:str, fit:bool = True, fc_ocv:bool = True):
    '''
    Plots the fuel cell and electrolysis cell galvanostatic holds on the same plot
    This function is for cells that were stability tested in fuel cell and electrolysis cell mode

    Parameters:
    -----------
    folder_loc, str:
        Location of the folder where the cell testing data is stored
    fit, bool: (Default = True)
        Whether or not to do a linear fit of the data and plot it
    fc_ocv, bool: (Default = True)
        This is needed to get the initial ocv value of the cell

    Return --> Nothing. This function will plot and show the data
    '''

    ' //////////// ------- Organize & Plot FC data  ------- //////////// '
    files = os.listdir(folder_loc) # obtaining a list of the files in the desired folder
    useful_files = [] # initializing a list for the useful files

    # ------- Taking out all of the galvanostatic files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('GS')!=-1) and (file.find('.DTA')!=-1) and (file.find('Deg')!=-1):
            #Extracting the file number (used for sorting, yeah this is probably a roundabout way of doing it)
            start, end = file.rsplit('#',1)# cutting everything out before the # so only the number and file extension is left
            fnum, fileExt = end.rsplit('.',1) #cutting off the file extension leaving only the number as a string
            index = int(fnum) #Converts the file number to a string
            useful_file = (file,index)
            useful_files.append(useful_file)
    
    # ------- Sorting the files
    useful_files.sort(key=lambda x:x[1]) #Sorts by the second number in the tuple
    sorted_useful_files_fc, numbers = zip(*useful_files) #splits the tuple
    sorted_useful_files_fc = [folder_loc + '/' + f for f in sorted_useful_files_fc] #Turning all files from their relative paths to the absolute path

    # ------- Getting the first time
    for file in files: #Finding the first file
        if file.find('Deg__#1.DTA')!=-1 and file.find('OCV')!=-1:
            file1_fc = os.path.join(folder_loc,file)

    T0_stamp = fl.get_timestamp(file1_fc) #gets time stamp from first file
    t0 = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch

    # ------- Combining all DataFrames
    dfs = [] #Initializing list of dfs
    length = len(sorted_useful_files_fc) #gets length of sorted useful files
    
    for i in range(0,length,1):
        loc = os.path.join(folder_loc,sorted_useful_files_fc[i]) # Creates a file path to the file of choice
        dta2csv(loc) #convert DTA to a CSV
        loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break

        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') # create data frame for a file
        start_time = fl.get_timestamp(sorted_useful_files_fc[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)

    cat_dfs_fc = pd.concat(dfs,ignore_index=True)# (s) Combine all the DataFrames in the file folder
    cat_dfs_fc.sort_values(by=['s'])
    cat_dfs_fc['s'] = (cat_dfs_fc['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    
    # ------- plotting:
    fig, ax = plt.subplots()
    ax.plot(cat_dfs_fc['s'],cat_dfs_fc['V vs. Ref.'],'.k')

    ' //////////// ------- Organize & Plot EC data  ------- //////////// '
    files = os.listdir(folder_loc)
    useful_files_ec = [] #initializing a list for the useful files

    # ------- Taking out all of the ocv files
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('GS')!=-1) and (file.find('.DTA')!=-1) and (file.find('ECstability')!=-1):
            useful_files_ec.append(os.path.join(folder_loc,file))

    # ------- Finding or defining the first file
    for file in files: #Finding the first file
        if file.find('ECstability__#1.DTA')!=-1 and file.find('OCV')!=-1:
            file1_ec = os.path.join(folder_loc,file)

    T0_stamp = fl.get_timestamp(file1_ec) #gets time stamp from first file
    t0_ec = T0_stamp.strftime("%s") #Converting Datetime to seconds from Epoch
 
    # ------- Concatenating the FC mode Galvanostatic DataFrames
    dfs = [] #Initializing list of dfs
    length = len(useful_files_ec) #gets length of sorted useful files
    
    for i in range(0,length,1):
        dta2csv(useful_files_ec[i]) #convert DTA to a CSV
        loc_csv = useful_files_ec[i].replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') # create data frame for a file
        start_time = fl.get_timestamp(useful_files_ec[i]).strftime("%s") # Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)
    
    cat_dfs_ec = pd.concat(dfs) # (s) Combine all the DataFrames in the file folder
    cat_dfs_ec['s'] = (cat_dfs_ec['s']-int(t0_ec))/3600 # (hrs) subtracting the start time to get Delta t and converting time from seconds to hours and

    # ------- Plotting
    ax.plot(cat_dfs_ec['s'],cat_dfs_ec['V vs. Ref.'],'.k')

    ' ------- Formatting, fitting, and writing the slope on the graph: '
    fontsize = 16
    ax.set_xlabel('Time (hrs)',fontsize = fontsize)
    ax.set_ylabel('Voltage (V)',fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=12) #changing tick label size
    ax.set_ylim(0,1.5)

    # ------- Finding the OCV at the beginning this Stability test and plotting it
    ocv = get_init_ocv(folder_loc,fc_ocv)
    ax.axhline(y=ocv,color='k',alpha=0.5,linestyle='--') 

    # ------- Fitting and writing the slope on the graph:
    if fit == True:
        # --- Fuel cell mode
        m_fc,b_fc = np.polyfit(cat_dfs_fc['s'],cat_dfs_fc['V vs. Ref.'],1)
        fit_fc = m_fc * cat_dfs_fc['s'] + b_fc
        ax.plot(cat_dfs_fc['s'],fit_fc,'--r')
        mp_fc = m_fc * 1000000 #Converting the slope into a % per khrs
        ms_fc = f'{round(mp_fc,3)}'
        
        # --- electrolysis cell mode
        m_ec,b_ec = np.polyfit(cat_dfs_ec['s'],cat_dfs_ec['V vs. Ref.'],1)
        fit_ec = m_ec * cat_dfs_ec['s'] + b_ec
        ax.plot(cat_dfs_ec['s'],fit_ec,'--r')
        mp_ec = m_ec * -1000000 #Converting the slope into a % per khrs
        ms_ec = f'{round(mp_ec,3)}'

        # --- Writing the slope on the graph
        plt.figtext(0.15,0.20, 'Electrolysis Cell Stability: ' + ms_ec + ' mV/khrs\n\n'
                            + 'Fuel Cell Stability: ' + ms_fc + ' mV/khrs',weight='bold',size='x-large')
    plt.tight_layout()
    plt.show()

def curtin_echem_plotting(loc:str, area:float = None, CD_at_V:float = 1.3, time_step:int = None):
    '''
    Plots data gathered using the labview testing software developed by Prof. Shao at Curtin university
    The data is gathered using a Kiethley

    Parameters
    ----------
    loc, string: 
        The location of the file that contains the electrochemical data
    area, float: 
        The active cell area in cm^2
    CD_at_V, float: (default is 1.3)
        This function prints the current density at a specific voltage onto the IVEC plot
        This parameter sets the specific voltage to print the current at (i.e the current at 1.3V)
    time_step, int: (default = None)
        Sets the time step in seconds between measurments for the OCV plot
        If this was variable throughout testing set to None

    Return --> none but it plots the figure and shows it
    '''
    
    df = pd.read_csv(loc,sep='\t',encoding='latin1',engine='python') # - Making a dataframe with the data

    # --- Figuring out which test was run
    file = os.path.basename(loc)

    if file.find('IV')!=-1 and file.find('FC')!=-1:
        test = 'ivfc'
    elif file.find('IV')!=-1 and file.find('EC')!=-1:
        test = 'ivec'
    elif file.find('OCV')!=-1:
        test = 'ocv'

    if test == 'ivfc': # >>> Plotting a fuel cell mode power curve
        # - Modifying the dataframe
        df.columns = ['A','V']

        # - Setting area specifit data
        if area == None:
            area = 0.5
        else:
            area = area

        df['A'] = df['A'].div(area * 1000) # A/cm^2
        df['W'] = (df['A'] * df['V']) # W/cm^2

        # ------ Plotting
        fig, ax1 = plt.subplots()
   
        # - IV plotting
        color = '#000000' #Black color
        ax1.set_xlabel('Current Density ($A/cm^2$)',fontsize = 'xx-large')
        ax1.set_ylabel('Voltage (V)', color=color, fontsize = 'xx-large')
        ax1.plot(df['A'], df['V'],'o', color=color)
        ax1.tick_params(axis='y', labelcolor=color,labelsize = 'x-large')
        ax1.tick_params(axis='x',labelsize = 'x-large')
        
        # - Power density plotting
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color2 = '#c79d22' # Curtin gold color #CC9900 Curtin web safe gold color
        ax2.set_ylabel('Power Density ($W/cm^2$)', color=color2, fontsize = 'xx-large')  # we already handled the x-label with ax1
        ax2.plot(df['A'], df['W'], 'o',color=color2) 
        ax2.tick_params(axis='y', color=color2, labelcolor=color2, labelsize = 'x-large')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color(color2)
        ax1.spines['top'].set_visible(False)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        # - Calculating and printing max values onto the graph
        max_w = df['W'].max() #finds maximum power density
        max_v = df.loc[df.index[df['W'] == max_w],'V'].item() #finds voltage of max power density
        max_ws = f'{round(max_w,3)}' #sets float to a string
        max_vs = f'{round(max_v,3)}'

        plt.figtext(0.28,0.21,r'$P_{max} = $'+max_ws+r' $W/cm^2 at$ '+max_vs+r'$V$',size='x-large',weight='bold')
        
        plt.tight_layout()
        plt.show()

    elif test == 'ivec': # >>> Plotting an electrolysis cell mode power curve
        # ---- Modifying the dataframe
        df.columns = ['A','V']

        # - Setting area specifit data
        if area == None:
            area = 0.5
        else:
            area = area

        # - calculating the area specific current
        df.columns = ['A','V']
        df['A'] = df['A'].div(area * 1000)

        # ----- Plotting
        fig, ax1 = plt.subplots()

        # - Formatting
        color = '#c79d22'
        ax1.set_xlabel('Current Density ($A/cm^2$)',fontsize = 'xx-large')
        ax1.set_ylabel('Voltage (V)',fontsize = 'xx-large')
        
        ax1.plot(-df['A'], df['V'],'o', color=color)
        ax1.tick_params(axis='both',labelsize = 'x-large')

        # ------- Calculating and printing current density at a given voltage
        CD_at_mod = CD_at_V-0.01
        current_density15 = df[abs(df['V'])>=CD_at_mod].iloc[0,0] # Finds the current density of the first Voltage value above the desired Voltage
        V15 = df[abs(df['V'])>=CD_at_mod].iloc[0,1] # Same as before but returns the exact voltage value
        current_density15_string = f'{round(current_density15,3)}'
        V15_string = f'{round(V15,3)}'
        plt.figtext(0.28,0.21,current_density15_string+r' $A/cm^2\:at$ '+V15_string+r'$V$',size='x-large',weight='bold') #placing value on graph
        
        plt.tight_layout()
        plt.show()

    elif test == 'ocv': # >>> Plotting an electrolysis cell mode power curve
        df.columns = ['A','V']

        # --- Plotting
        fig, ax = plt.subplots()
        if time_step == None:
            ax.plot(df.index.values,df['V'],'ko')
            ax.set_xlabel('Measurement (#)',fontsize='xx-large')

        else:
            x = df.index.values * time_step
            ax.plot(x, df['V'],'ko')
            ax.set_xlabel('Time (s)',fontsize='xx-large')

        # - Formatting
        ax.set_ylabel('Voltage (V)',fontsize='xx-large')
        ax.tick_params('both',labelsize = 'x-large')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    return()

def curtin_ivfcs(curves_conditions:tuple, print_Wmax:bool=False, cmap:str=None,leg_cols:int=None):
    '''
    Plots data gathered using the labview testing software developed by Prof. Shao at Curtin university
    The data is gathered using a Kiethley
    Plots multiple IV and power density curves, input is a tuple of (area,condition,location of IV curve)

    Parameters
    ----------
    curves_conditions,tuple: 
        A tuple containing data to plot and label the curve.
        The order is: (area,condition,location of IV curve)
    print_Wmax, bool: (default is True)
        Prints out Wmax in the terminal
    cmap,str: (default is None)
        If a colormap is defined here it will be used for the plots
    leg_cols,int: (default is None)
        how many columns the legend will have. If none, this will be based of the abount of curves being plot

    Return --> none but it plots the figure and shows it
    '''
    fig,ax1 = plt.subplots() # Initializing plot
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    w_max = [] #initializing a list to store the max power densities

    # --- Setting up colormap if a colormap is chosen
    if cmap is not None:
        # --- Setting up an array for the color map
        color_space = np.linspace(0,1,len(curves_conditions)) # array from 0-1 for the colormap for plotting
        c = 0 # index of the color array
        cmap = cmap

    for iv in curves_conditions:
        # - Gathering data
        try:
            loc = iv[2]
            df = pd.read_csv(loc,sep='\t',encoding='latin1',engine='python') # - Making a dataframe with the data
            df.columns = ['A','V']

            # - Calculations
            area = iv[0]
            df['A'] = df['A'].div(area * 1000) # A/cm^2
            df['W'] = (df['A'] * df['V']) # W/cm^2
        
        except:
            loc = iv[2]
            df = get_iv_data(loc) # Reads the .DTA file and saves the IVFC data as a dataframe

            #calculations and only keeping the useful data
            area = iv[0]
            df['A'] = df['A'].div(-area)
            df['W'] = df['W'].div(-area)
            df = df[['V','A','W']]

        if print_Wmax == True:
            w_max.append(df['W'].max()) #finds maximum power density

        if cmap == None: # This if statement determines whether the default colors or a cmap is used
            # --- IV plotting:
            label = iv[1]
            ax1.plot(df['A'], df['V'],'o',fillstyle='none',label=label)

            # --- Power Density plotting
            ax2.plot(df['A'], df['W'],'o',label=label)
        else:
            # --- IV plotting:
            color = cmap(color_space[c])
            label = iv[1]
            ax1.plot(df['A'], df['V'],'o',fillstyle='none',label=label,color=color)

            # --- Power Density plotting
            ax2.plot(df['A'], df['W'],'o',label=label,color=color)
            c = c+1

    # --- Printing Max power density if desired:
    if print_Wmax == True:
        for i in range(len(w_max)):
            max_ws = f'{round(w_max[i],3)}'
            print('Max Power Density of the ' + curves_conditions[i][1] + ' condition' + ' is: '
                +'\033[1m'+ max_ws + '\033[0m'+' W/cm\u00b2')

    # ----- Plot Formatting:
    ax1.set_xlabel('Current Density (A/cm$^2$)',fontsize='xx-large')
    ax1.set_ylabel('Voltage (V)',fontsize='xx-large')
    ax1.tick_params(axis='both', which='major', labelsize='x-large')

    ax2.set_ylabel('Power Density (W/cm$^2$)',fontsize='xx-large')  # we already handled the x-label with ax1
    ax2.tick_params(axis='both', which='major', labelsize='x-large')

    # --- Legend Formatting
    if leg_cols == None:
        num_curves = len(curves_conditions)
        if num_curves <=4:
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=num_curves,handletextpad=0.02,columnspacing=1)
        elif num_curves <=8:
            ncol = int(round(num_curves/2))
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=ncol,handletextpad=0.02,columnspacing=1)
        elif num_curves <=12:
            ncol = int(round(num_curves/3))
            ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=num_curves,handletextpad=0.02,columnspacing=1)
    else:
        ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize='large',ncol=leg_cols,handletextpad=0.02,columnspacing=1)


    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()

def curtin_ivecs(loc:str, area:float = None, label:str = None):

    # ------ calculations and only keeping the useful data
    try:
        df = get_iv_data(loc) # Reads the .DTA file and saves the IVEC data as a dataframe
        df['A'] = df['A'].div(area)
        df = df[['V','A']]
        if df['V'].loc[0] <= 0: #this is put in because the Gamry 3000 and 5000 have different signs on the voltage of IVEC curve
            sign = -1
        else:
            sign = 1

        asign = 1
    except:
        df = pd.read_csv(loc,sep='\t',encoding='latin1',engine='python') # - Making a dataframe with the data
        # ---- Modifying the dataframe
        df.columns = ['A','V']

        # - Setting area specifit data
        if area == None:
            area = 0.5
        else:
            area = area

        # - calculating the area specific current
        df.columns = ['A','V']
        df['A'] = df['A'].div(area * 1000)
        sign=1
        asign = -1


    # ------- Plotting
    with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
        plt.plot(df['A']*asign, sign*df['V'],'o', label=label,markersize=9)
    plt.xlabel('Current Density ($A/cm^2$)', fontsize='xx-large')
    plt.ylabel('Voltage (V)', fontsize='xx-large')
    plt.legend(loc='best', fontsize = 'x-large')
    plt.xticks(fontsize = 'x-large')
    plt.yticks(fontsize = 'x-large')
    plt.tight_layout()
