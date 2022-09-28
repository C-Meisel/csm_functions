''' This module contains functions to help format and plot XRD files
# C-Meisel
'''

'Imports'
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np

def xrd_format(location: str)->pd.DataFrame:
    '''
    Formats the data from a CSV file that is generated by the XRD mahine for plotting. It also converts intensity to relative intensity

    Parameters:
    -----------
    location, str:
        location of the CSV file
        
    :return -> dataframe of formatted data
    '''
    file = csv.reader(open(location, "r",encoding='latin1')) #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "Angle", then subtracts 1. This gives the right amount of rows to skip to get to the XRD data
        if row[0] == 'Angle':
            skip = file.line_num-1
            break
    df = pd.read_csv(location,skiprows=skip) #creates datafile from csv convert of XRDML file
    maximum = df['Intensity'].max() #calculates highes intensity value
    df['Intensity'] = df['Intensity']/maximum #
    df = df.rename({'Angle':'a','Intensity':'i'},axis=1) #renames columns to make further plotting quicker a = angle i = relative intensity
    return df

def xrd_format_icdd(sheet: str)->pd.DataFrame:
    '''
    Returns 2Theta and relative intensity data from my saved ICDD files.

    param sheet: str, the name of the excel sheet containing the desired material (material name)

    return -> dataframe of formatted data
    '''
    df = pd.read_excel('/Users/Charlie/Documents/CSM/XRD_Data/ICDD_XRD_Files.xlsx',sheet) #will need to change this line if the file location changes
    df = df[['2Theta','Relative Intensity']]
    df = df.rename({'2Theta':'a','Relative Intensity':'i'},axis=1)
    return df

def plot_xrd(loc: str,material: str):
    '''
    This function graphs an XRD spectra from a CSV containing XRD data.  material is what the line will be named
    
    Parameters:
    -----------
    loc, str: 
        the location of the file
    material, str: 
        material is what the line will be named in the legend

    return -> nothing
    '''
    df = xrd_format(loc)
    # plt.figure(dpi=250)  #change to change the quality of the return chart
    plt.plot(df['a'],df['i'], label = material)
    plt.xlim(20,80)
    plt.xlabel('2\u03B8')
    plt.ylabel('Relative Intensity')
    plt.tight_layout()
    plt.legend()

def plot_xrds(loc: str, material: str, y_offset: float = 0,linewidth: float = 2.5, **plt_args):
    '''
    This function enables multiple spectra to be on the same plot. This function graphs the XRD spectra from a CSV., while y_offset is the y offset and if left blank defaults to 0.

    Parameters:
    -----------
    loc, str:
        The location of the file
    material, str: 
        material is what the line will be named in the legend
    y_offset, float:
        is the y offset of the xrd spectra left blank it defaults to 0.
    linewidth, float: 
        width of the spectra line 

    return -> nothing
    '''
    try: #loc should be the sheet name in the ICDD file or the location of the csv file
        df = xrd_format_icdd(loc) 
    except:
        df = xrd_format(loc)
    
    with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
        plt.plot(df['a'],df['i']+y_offset, label = material,linewidth=linewidth,**plt_args) #offset is the y axis offset to stack graphs and is optional
    
    # Formatting
    plt.xlabel('2\u03B8',size='xx-large')
    plt.ylabel('Relative Intensity',size='xx-large')
    plt.xticks(fontsize = 'x-large')
    
    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])
    # plt.yticks([0],fontsize = 'x-large')

    plt.tight_layout()
    plt.legend(fontsize='large')
