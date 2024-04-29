''' This module contains functions to help it distribution of relaxation time (DRT) spectra to 
potentiostatic electrochemical impedance spectroscopy data taken with a Gamry potentiostat
All of the actual DRT fitting and analysis is done in the bayes_drt module made by Jake Huang
# C-Meisel
'''
# PRinting f is line 4229 in inversion
'Imports'
import os
from bayes_drt2.inversion import Inverter#inverter class in inversion module
from bayes_drt2 import file_load as fl

# ---------- Functions for saving a DRT fit for one spectra ----------
def bayes_drt_save(loc_eis:str, fit_path:str, fit_name:str, which:bool='core', 
                    init_from_ridge:bool=True,):
    '''
    Takes a potentiostatic EIS spectra, fits it using a Baysian model,
    and saves it to a certain fit path (directory) with a file name

    Parameters
    -----------
    loc_eis, string: (path to a directory)
        location of the EIS spectra 
    fit_path, string: (path to a directory)
        path to save the fit. This will be the directory, or "Jar", that the fit will be saved in. 
    fit_name, string:
        The name of the fit. This will be the file name
    which, string:
        which data to store. 'core' or 'sample'. Core file sizes are smaller
    init_from_ridge, boolean: (default: True)
        If True, use the hyperparametric ridge solution to initialize the Bayesian fit.
        Only valid for single-distribution fits
    Return --> None
    '''
    df = fl.read_eis(loc_eis)
    freq,z = fl.get_fZ(df)
    inv = Inverter() #new DRT package can figure out the right basis frequency to use
    inv.fit(freq,z,init_from_ridge=True,mode='sample',chains=2,samples=200) #Bayes fitting DRT
    #Chains=2 and samples=200
    inv.save_fit_data(os.path.join(fit_path,fit_name),which=which) #main thing that core doesnt save is the matricies (a lot of data)

def map_drt_save(loc_eis:str, fit_path:str, fit_name:str, which:bool='core', 
                init_from_ridge:bool=True, outliers:bool=False):
    '''
    Takes a potentiostatic EIS spectra, fits it using a mapping model,
    and saves it to a certain fit path (directory) with a file name.

    Made on 18May21, this is how I currently use Jake's Bayes fit function to fit eis spectra
    Changes will likely be made in the future as I learn more about DRT and Jakes Modules

    Parameters:
    loc_eis, string: 
        location of the EIS spectra (path to file)
    fit_path, string: 
        path to save the fit. This will be the directory, or "Jar", that the fit will be saved in. (path to folder)
    fit_name, string:
        The name of the fit. This will be the file name
    which, string:
        which data to store. 'core' or 'sample'. Core file sizes are smaller
    init_from_ridge, boolean: (default: True)
        If True, use the hyperparametric ridge solution to initialize the Bayesian fit.
        Only valid for single-distribution fits
    param outliers, boolean: (default: False)
        whether to use outliers to initialize the fit.
    '''
    
    df = fl.read_eis(loc_eis)
    freq,z = fl.get_fZ(df)
    inv = Inverter()
    inv.fit(freq,z,init_from_ridge=init_from_ridge,outliers=outliers) #If the data is not taken from 1e6hz init_from_ridge should be true
    inv.save_fit_data(os.path.join(fit_path,fit_name),which=which) #main thing that core doesnt save is the matricies (a lot of data)

# ---------- Functions for fitting and saving a DRT fit for multiple spectra ----------
def deg_eis_fitter(folder_loc:str, jar_bias:str, jar_ocv:str, bias_amount:str='n3', fit_ocv:bool=True, 
                    which:bool = 'core', init_from_ridge:bool=True):
    '''
    Map fit's DRT to all stabilty testing EIS files in a given folder and saves the files to the specified jar
    If the data has already been fit, it will not re-fit the data. This function compliments the Fuel Cell mode
    stabilty testing Gamry function.

    param folder_loc, string: (path to a directory)
        Location of the folder containing the stability testing EIS spectra 
    param jar_bias, string: (path to a directory)
        Path to the jar to save the bias DRT fits to 
    param jar_ocv, string: (path to a directory)
        Path to the jar to save the ocv DRT fits to 
    param bias_amount, string: 
        The amount of bias to use for the DRT fits. 
        Generally I do the bias EIS at -0.3V so bias_amount = n3 (n for negative, and 3 for 0.3V)
    param fit_ocv, boolean:
        Whether to map fit the ocv EIS spectra too.
    param which, string: (default = True)
        Which data to store. 'core' or 'sample'. Core file sizes are smaller
    param init_from_ridge, boolean: optional (default: False)
        If True, use the hyperparametric ridge solution to initialize the Bayesian fit.
        Only valid for single-distribution fits

    Return --> None
    '''
    files = os.listdir(folder_loc)
    files_eis_bias = []
    cell_name = os.path.basename(folder_loc).split("_", 3)[2]
    
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('PEIS')!=-1) and (file.find('_Deg')!=-1) and (file.find(bias_amount+'Bias')!=-1) and (file.find('.DTA')!=-1): #really specific to ensure no random EIS spectra get included
            number = file.split('_Deg')[1].split('.DTA')[0] #Splits the string of the file at _Deg and at .DTA and saves what is in the center
            name = cell_name+'_'+bias_amount+'_Bias'+number #adds more the name to make it more descriptive
            useful_file = (file,name) #creates a tuple of the file in question and the new name it has just been given
            files_eis_bias.append(useful_file) #adds tuple of file and name to a list

    for eis in files_eis_bias: #loops through list of degradation eis spectra at bias that was just created
        loc_eis = os.path.join(folder_loc,eis[0]) #creates the full location of desired the EIS spectra
        fit_path = jar_bias
        fit_name = eis[1]+'.pkl'
        
        pickle_jar = os.listdir(jar_bias)

        pickle_name = 0
        for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
            if fit_name == pickle:
                pickle_name = pickle_name + 1
                break
        
        if pickle_name == 0: 
            map_drt_save(loc_eis,fit_path,fit_name,which=which,init_from_ridge=init_from_ridge)

    print('Done Fitting EIS for Degradation at Bias')

    if fit_ocv == True:
        files_eis_ocv = []
        
        for file in files: #looping over all files in the folder folder_loc
            if (file.find('PEIS')!=-1) and (file.find('_Deg')!=-1) and (file.find(bias_amount+'Bias')==-1) and (file.find('.DTA')!=-1): #really specific to ensure no random EIS spectra get included
                #==-1 on the bias one because I do not want to re-fit the EIS at bias
                number = file.split('_Deg')[1].split('.DTA')[0] #Splits the string of the file at _Deg and at .DTA and saves what is in the center
                name = cell_name +'_OCV'+number #adds more the name to make it more descriptive
                useful_file = (file,name) #creates a tuple of the file in question and the new name it has just been given
                files_eis_ocv.append(useful_file) #adds tuple of file and name to a list

        for eis in files_eis_ocv: #loops through list of degradation eis spectra at bias that was just created
            loc_eis = os.path.join(folder_loc,eis[0]) #creates the full location of desired the EIS spectra
            fit_path = jar_ocv
            fit_name = eis[1]+'.pkl'

            pickle_jar = os.listdir(jar_ocv)
            name = 0
            for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
                if fit_name == pickle:
                    name = name + 1
                    break
            
            if name == 0: 
                map_drt_save(loc_eis,fit_path,fit_name,which=which,init_from_ridge=init_from_ridge)
        
        print('Done Fitting EIS for Degradation at OCV')

def ec_stb_eis_fitter(folder_loc:str, jar_bias_ec:str, jar_ocv_ec:str, fit_ocv:bool=True, which:bool = 'core', init_from_ridge:bool=True):
    '''
    Map fit's DRT to all stabilty testing EIS files in a given folder and saves the files to the specified jar
    If the data has already been fit, it will not re-fit the data. This function compliments the Electrolysis Cell mode
    stabilty testing Gamry function.

    param folder_loc, string: (path to a directory)
        Location of the folder containing the stability testing EIS spectra 
    param jar_bias, string: (path to a directory)
        Path to the jar to save the bias DRT fits to 
    param jar_ocv, string: (path to a directory)
        Path to the jar to save the ocv DRT fits to 
    param fit_ocv, boolean:
        Whether to map fit the ocv EIS spectra too.
    param which, string: (default = True)
        Which data to store. 'core' or 'sample'. Core file sizes are smaller
    param init_from_ridge, boolean: optional (default: False)
        If True, use the hyperparametric ridge solution to initialize the Bayesian fit.
        Only valid for single-distribution fits

    Return --> None
    '''
    files = os.listdir(folder_loc)
    files_eis_bias = []
    cell_name = os.path.basename(folder_loc).split("_", 3)[2]
    for file in files: #looping over all files in the folder folder_loc
        if (file.find('PEIS')!=-1) and (file.find('_ECstability')!=-1) and (file.find('TNV')!=-1) and (file.find('.DTA')!=-1): #really specific to ensure no random EIS spectra get included
            number = file.split('_ECstability')[1].split('.DTA')[0] #Splits the string of the file at _Deg and at .DTA and saves what is in the center
            name = cell_name+'_ECstb_TNV'+number #adds more the name to make it more descriptive
            useful_file = (file,name) #creates a tuple of the file in question and the new name it has just been given
            files_eis_bias.append(useful_file) #adds tuple of file and name to a list

    for eis in files_eis_bias: #loops through list of degradation eis spectra at bias that was just created
        loc_eis = os.path.join(folder_loc,eis[0]) #creates the full location of desired the EIS spectra
        fit_path = jar_bias_ec
        fit_name = eis[1]+'.pkl'
        
        pickle_jar = os.listdir(jar_bias_ec)

        pickle_name = 0
        for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
            if fit_name == pickle:
                pickle_name = pickle_name + 1
                break
        
        if pickle_name == 0: 
            map_drt_save(loc_eis,fit_path,fit_name,which=which,init_from_ridge=init_from_ridge)

    print('Done Fitting EIS for Degradation at Bias')

    if fit_ocv == True:
        files_eis_ocv = []
        for file in files: #looping over all files in the folder folder_loc
            if (file.find('PEIS')!=-1) and (file.find('_ECstability')!=-1) and (file.find('TNV')==-1) and (file.find('.DTA')!=-1): #really specific to ensure no random EIS spectra get included
                #==-1 on the bias one because I do not want to re-fit the EIS at bias
                number = file.split('_ECstability')[1].split('.DTA')[0] #Splits the string of the file at _Deg and at .DTA and saves what is in the center
                name = cell_name +'_ECstb_OCV'+number #adds more the name to make it more descriptive
                useful_file = (file,name) #creates a tuple of the file in question and the new name it has just been given
                files_eis_ocv.append(useful_file) #adds tuple of file and name to a list

        for eis in files_eis_ocv: #loops through list of degradation eis spectra at bias that was just created
            loc_eis = os.path.join(folder_loc,eis[0]) #creates the full location of desired the EIS spectra
            fit_path = jar_ocv_ec
            fit_name = eis[1]+'.pkl'

            pickle_jar = os.listdir(jar_ocv_ec)
            name = 0
            for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
                if fit_name == pickle:
                    name = name + 1
                    break
            
            if name == 0: 
                map_drt_save(loc_eis,fit_path,fit_name,which=which,init_from_ridge=init_from_ridge)
        
        print('Done Fitting EIS for Degradation at OCV')

def po2_mapfit(folder_loc:str,fit_path:str):
    '''
    Map fits DRT to all the changes in O2 concentration EIS files in a given folder and saves the 
    files to the specified jar (directory)

    Parameters
    ----------
    folder_loc, string: (path to folder)
        location of the folder containing the EIS spectra
    fit_path, string: (path to folder)
        path to the jar to save the DRT fits to

    Return --> None
    '''

    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc
    substring_list = ['PEIS_20O2.80Ar','PEIS_40O2.60Ar','PEIS_60O2.40Ar','PEIS_80O2.20Ar','PEIS_100O2.0Ar'] # setting variables to look for
    O2_conc_list = [dta_files for dta_files in dta_files if any(sub in dta_files for sub in substring_list)] # placing all the changes in O2 eis files into a list
    cell_name = os.path.basename(fit_path).split("_", 2)[1]
    
    for peis in O2_conc_list:
        loc = os.path.join(folder_loc, peis) # creates the full path to the file
        po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name
        fit_name = cell_name + '_map_fit_' + po2 + 'pO2.pkl'

        map_drt_save(loc,fit_path,fit_name)

def bias_mapfit(folder_loc:str,fit_path:str):
    '''
    Map fits DRT to all the changes in O2 concentration EIS files in a given folder and saves the 
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
        fit_name = cell_name + '_map_fit_' + bias + 'bias.pkl'
        
        map_drt_save(peis,fit_path,fit_name)



