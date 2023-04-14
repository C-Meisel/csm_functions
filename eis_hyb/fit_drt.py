''' This module contains functions to help it distribution of relaxation time (DRT) spectra to 
potentiostatic electrochemical impedance spectroscopy data taken with a Gamry potentiostat
All of the actual DRT fitting and analysis is done in the hybrid_drt module made by Jake Huang
# C-Meisel
'''

'Imports'
import os
from hybdrt import fileload as fl
from hybdrt.models import DRT
from hybdrt.models import elements
from hybdrt.fileload import read_eis, get_eis_tuple


def dual_drt_save(loc_eis:str, fit_path:str, fit_name:str,data=['config', 'fit_core','data'], indicator = False):
    '''
    Takes a potentiostatic EIS spectra, and fits it using the dual-regression
    classification function from hybrid-DRT by Jake Huang

    Parameters:
    -----------
    loc_eis, str: 
        location of the EIS spectra (path to file)
    fit_path, str: 
        path to save the fit. This will be the directory, or "Jar", that the fit will be saved in. (path to folder)
    fit_name, str:
        The name of the fit. This will be the file name
    data, str list: (Default=['config', 'fit_core','data'])
        Which data to store. the default is the most data, but ['config', 'fit_core'] can also be used
        rule of thumb is that the default wil lbe 340 KB while only the first two will be around 65 KB

    Return --> Nothing. The fit is saved to the jar    
    '''
    drt = DRT() # Create a DRT instance
    df = read_eis(loc_eis)
    freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe
    drt.dual_fit_eis(freq, z, discrete_kw=dict(prior=True, prior_strength=None)) # Fit the data

    if indicator == True:
        fdest = os.path.join(fit_path,fit_name) + '_dual.pkl'
    else:
        fdest = os.path.join(fit_path,fit_name) + '.pkl'


    drt.save_attributes(data, fdest)

def pfrt_drt_save(loc_eis:str, fit_path:str, fit_name:str,data=['config', 'fit_core','data'], indicator = False):
    '''
    Saves a probability function of relaxation times (PFRT) fit of an EIS spectra
    From hybrid-DRT from jake huang

    Parameters:
    -----------
    loc_eis, str: 
        location of the EIS spectra (path to file)
    fit_path, str: 
        path to save the fit. This will be the directory, or "Jar", that the fit will be saved in. (path to folder)
    fit_name, str:
        The name of the fit. This will be the file name
    data, str list: (Default=['config', 'fit_core','data'])
        Which data to store. the default is the most data, but ['config', 'fit_core'] can also be used
        rule of thumb is that the default wil lbe 340 KB while only the first two will be around 65 KB

    Return --> Nothing. The fit is saved to the jar    
    '''
    drt = DRT() # Create a DRT instance
    df = read_eis(loc_eis)
    freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe
    drt.pfrt_fit_eis(freq, z) # Fit the data

    fdest = os.path.join(fit_path,fit_name) + '_pfrt.pkl'
    drt.save_attributes(data, fdest)

def bias_dual_fit(folder_loc:str,fit_path:str):
    '''
    Dual fits DRT to all the changes in O2 concentration EIS files in a given folder and saves the 
    files to the specified jar (directory)

    param folder_loc, string: location of the folder containing the EIS spectra (path to folder)
    param fit_path, string: path to the jar to save the DRT fits to (path to folder)

    Return --> None
    '''

    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc
    bias_eis = [] #initializing bias files list
    
    for file in dta_files: #Finding all bias EIS files
        if (file.find('PEIS')!=-1) and (file.find('bias.DTA')!=-1):
            bias_eis.append(os.path.join(folder_loc,file))
    
    cell_name = os.path.basename(fit_path).split("_", 2)[1]

    for peis in bias_eis:
        bias = peis[peis.find('550C_')+len('PEIS_'):peis.rfind('bias')] #gets the bias from the file name
        fit_name = cell_name + '_dual_fit_' + bias + 'bias'
        
        dual_drt_save(peis,fit_path,fit_name)