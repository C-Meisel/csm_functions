''' This module contains functions to help format Electrohchemical Impedance Spectroscopy (EIS)
data. The data files are obtained by a Gamry potentiostat. The files are .DTA files
This is a second attempt at making data processing functions
# C-Meisel
'''

'Imports'
import pandas as pd
import csv
from shutil import copyfile
import os
import re

def read_dta(loc:str):
    '''
    Reads a gamry .DTA file and saves the data as an string
    These 6 lines of code are copied from fileload.py in Hybrid_drt (by Jake Huang)

    Parameters:
    ------------
    loc, str: location of a gamry .DTA file

    return --> string containg the data stored in the .DTA file
    '''
    try:
        with open(loc, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(loc, 'r', encoding='latin1') as f:
            txt = f.read()

    return(txt)

def get_ocv_data(loc:str):
    '''
    Reads a gamry .DTA file for an OCV scan and returns the OCV data
    This will work for potentiostatic holds too.
     
    Parameters:
    -----------
    loc, str: location of gamry .dta file to get the eis data from

    return --> dataframe with the ocv data 
    '''
    txt = read_dta(loc) # Reading the Gamry .DTA file and storing as a string

    # --- Finding the start of the data (Also informed by Hybrid_drt)
    data_flag = txt.find('CURVE')
    metadata = txt[:data_flag] # figure out where the metadata ends
    skiprows = len(metadata.split('\n')) + 1
    df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',engine='python')

    return(df)

def get_eis_data(loc:str):
    '''
    Reads a gamry .DTA file for a potentiostatic EIS scan 
    and returns the EIS data as a dataframe.
     
    Parameters:
    ------------
    loc, str: location of gamry .dta file to get the eis data from

    return --> dataframe with the eis data 
    '''
    txt = read_dta(loc) # Reading the Gamry .DTA file and storing as a string

    # --- Finding the start of the data (Also informed by Hybrid_drt)
    data_flag = txt.find('ZCURVE')
    metadata = txt[:data_flag] # figure out where the metadata ends
    skiprows = len(metadata.split('\n')) + 1
    df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',engine='python')
    
    return(df)

def get_eis_ocv_data(loc:str):
    '''
    Reads a gamry .DTA file for a potentiostatic EIS scan 
    and returns the OCV data as a dataframe.
     
    Parameters:
    ------------
    loc, str: location of gamry .dta file to get the eis data from

    return --> dataframe with the eis data 
    '''    
    txt = read_dta(loc) # Reading the Gamry .DTA file and storing as a string

    # --- Finding the start of the data (Also informed by Hybrid_drt)
    data_flag = txt.find('OCVCURVE')
    metadata = txt[:data_flag] # figure out where the metadata ends
    skiprows = len(metadata.split('\n')) + 1
    footer_flag = txt.find('EOC') # Figure out where OCV data ends
    skipfooter = len(txt[footer_flag:].split('\n')) - 1
    df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',
    skipfooter=skipfooter,engine='python')

    return(df)

def get_iv_data(loc:str):
    '''
    Reads a gamry .DTA file for a polarization (IV) curve
    and returns the data table as a dataframe.

    This function also works for galvanostatic holds
     
    Parameters:
    ------------
    loc, str: location of gamry .dta file to get the IV data

    return --> dataframe with the IV data 
    '''
    txt = read_dta(loc) # Reading the Gamry .DTA file and storing as a string

    # --- Finding the start of the data (Also informed by Hybrid_drt)
    try:
        data_flag = txt.find('CURVE')
        metadata = txt[:data_flag] # figure out where the metadata ends
        skiprows = len(metadata.split('\n')) + 1 # Finding how many rows to skip before the data starts
        df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',engine='python')

    except pd.errors.EmptyDataError: # If the test stand breaks during the test and it is aborted...
        data_flag = txt.find('EXPERIMENTABORTED')
        metadata = txt[:data_flag] # figure out where the metadata ends
        skiprows = len(metadata.split('\n')) + 2 # Finding how many rows to skip before the data starts
        df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',engine='python')
        print(df)

    return(df)
 
def get_gs_data(loc:str):
    '''
    Reads a gamry .DTA file for a polarization (IV) curve
    and returns the data table as a dataframe.

    This function also works for galvanostatic holds
     
    Parameters:
    ------------
    loc, str: location of gamry .dta file to get the IV data

    return --> dataframe with the IV data 
    '''
    txt = read_dta(loc) # Reading the Gamry .DTA file and storing as a string
    
    # with txt as file:
    #     lines = file.readlines()

    # for line in lines:
    #     if line.startswith('CURVE') and not 'OCV CURVE' in line:
    #         print(line)
    #         break
    # --- Finding the start of the data (Also informed by Hybrid_drt)
    data_flag = txt.find('COMPLIANCEVOLTAGE')
    metadata = txt[:data_flag] # figure out where the metadata ends
    skiprows = len(metadata.split('\n')) + 2 # Finding how many rows to skip before the data starts

    df = pd.read_csv(loc,sep='\t',skiprows=skiprows,encoding='latin1',engine='python')
    return(df)

def get_iv_ocv(loc:str):
    '''
    Reads a gamry .DTA file for a polarization (IV) curve
    and returns the ocv reading for the iv curve
     
    Parameters:
    ------------
    loc, str: location of gamry .dta file to get the IV data

    return --> float: OCV
    '''
    # Read the text file
    with open(loc, 'r') as file:
        text = file.read()

    # Define a regular expression pattern to match the line containing OCV
    pattern = r'EOC\s+QUANT\s+(\d+\.\d+)\s+Open Circuit\(V\)'

    # Search for the OCV value using the pattern
    match = re.search(pattern, text)

    # Check if a match was found
    if match:
        ocv_value = float(match.group(1))
    else:
        print("OCV Value not found in the text.")
    
    return(ocv_value)


' --- Older Data Functions '
def dta2csv(loc:str):
    '''
    Duplicates the .DTA files and converts it to a .CSV file
     
    param loc: str, location of dta file to convert to a csv

    return --> None 
    '''
    file = loc
    copyfile(file, file.replace('.DTA','')+'.csv')

def iv_data(area:float, loc:str) -> pd.DataFrame:
    '''
    Takes a .DTA file from an polarization curve (IV curve), extracts the voltage and amperage,
    and calculates the power density of each point. Then converts this data to a dataframe

    param area: float, active cell area for the cell that the data is from
    param loc: str, location of the .DTA file

    return --> Dataframe
    '''
    "Converts and finds CSV then turns it into a dataframe"
    dta2csv(loc)
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    "calculations and only keeping the useful data"
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)
    df_useful = df[['V','A','W']]

    return df_useful   
    
def ocv_data(loc:str):
    '''
    Takes a .DTA file that read the cell voltage, extracts the voltage and time,
    then converts this data to a dataframe.

    param loc: str, location of the .DTA file

    return --> none
    '''

    dta2csv(loc)
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    skip = 0
    for row in file: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            print(skip)
            break
        if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
            skip = file.line_num+1
            print(skip)
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    df_useful = df[['s','V']]
    df_useful.to_csv(loc_csv)

def peis_data(area:float,loc:str):
    '''
    Extracts area zreal and zimag .DTA file for potentiostatic eis. Converts z data to be area specific
    then places data in a pandas dataframe

    Parameters:
    -----------
    area, float:
        active cell area in cm^2
    loc, str: 
        location of the .DTA file

    return --> df_useful
    '''
    
    #Returns Zreal and Zimag from a DTA file of PotentiostaticEis in a CSV - not tested
    dta2csv(loc) #convert DTA to a CSV
    loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
    #find right amount of rows to skip
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'ZCURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1')
    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']]
    df_useful.to_csv(loc_csv)

    return df_useful

def get_init_ocv(folder_loc:str, fc:bool = True) -> float:
    '''
    This function finds the first OCV data file for a stability test, finds the average OCV, and returns it

    Parameters
    ----------
    folder_loc, str: (path to directory)
        location of the folder where the stability test data is stored
    fc, bool: Fuel Cell (default = True)
        determines whether to find the first ocv file for a fuel cell mode stability test,
        or an electrolysis cell stability test.
        set to False to find the first value of an electrolysis cell mode stability test.

    Returns --> The average OCV of the first OCV data file as a float.
    '''
    
    # ----- Finding the first OCV file
    files = os.listdir(folder_loc)

    if fc == True:
        for file in files:
            # if file.find('Deg__#1.DTA')!=-1 and file.find('OCV')!=-1 or (file.find('__#1.DTA')!=-1 and file.find('OCV_50')!=-1 and file.find('Bias')==-1 and file.find('Ahp')==-1):
            if file.find('Deg10__#4.DTA')!=-1 and file.find('OCV')!=-1 or (file.find('__#1.DTA')!=-1 and file.find('OCV_50')!=-1 and file.find('Bias')==-1 and file.find('Ahp')==-1): # Use for Cell 55
            # if file.find('Deg10_2__#1.DTA')!=-1 and file.find('OCV')!=-1 or (file.find('__#1.DTA')!=-1 and file.find('OCV_50')!=-1 and file.find('Bias')==-1 and file.find('Ahp')==-1): # Use for Cell 44 Continuation (Switch Deg10_3__1 to Deg10_2__1 for the inbetween sect)
                file1 = os.path.join(folder_loc,file)

    if fc == False:
        for file in files: #Finding the first file
            if file.find('ECstability__#1.DTA')!=-1 and file.find('OCV')!=-1:
                file1 = os.path.join(folder_loc,file)

    # ----- Finding the average OCV and returning it
    dta2csv(file1)
    loc_csv = file1.replace('.DTA','') + '.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t")
    
    skip = 0
    for row in file: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
        if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
            skip = file.line_num+1
            break
    
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    
    return df['V'].mean()

def get_init_v(folder_loc:str, fc:bool = True) -> float:
    '''
    This function finds the first galvanostatic hold data file for a stability test, finds the average Voltage, and returns it

    Parameters
    ----------
    folder_loc, str: (path to directory)
        location of the folder where the stability test data is stored
    fc, bool: Fuel Cell (default = True)
        determines whether to find the first ocv file for a fuel cell mode stability test,
        or an electrolysis cell stability test.
        set to False to find the first value of an electrolysis cell mode stability test.

    Returns --> The average voltage of the first galvanostatic hold data file as a float.
    '''
    
    # ----- Finding the first galvanostatic file
    files = os.listdir(folder_loc)

    if fc == True:
        for file in files:
            # if file.find('Deg__#1.DTA')!=-1 and file.find('GS')!=-1:
            if file.find('Deg10__#4.DTA')!=-1 and file.find('GS')!=-1: # Use for Cell 55
            # if file.find('Deg10_4__#1.DTA')!=-1 and file.find('GS')!=-1: # Use for Cell 44 Continuation (Switch Deg10_3 to Deg10_4 for the inbetween sect)

                file1 = os.path.join(folder_loc,file)

    if fc == False:
        for file in files: #Finding the first file
            if file.find('ECstability__#1.DTA')!=-1 and file.find('GS')!=-1:
                file1 = os.path.join(folder_loc,file)

    # ----- Finding the average galvanostatic and returning it
    dta2csv(file1)
    loc_csv = file1.replace('.DTA','') + '.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t")
    
    skip = 0
    for row in file: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
        if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
            skip = file.line_num+1
            break
    
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    
    return df['V vs. Ref.'].mean()

def cut_induct(df):
    '''
    Cut Inductance
    This function is used to cut off the inductance values off of an EIS spectra
    It finds the negative values in the EIS spectra, and removed that number of values from the
    beginning of the data. for example if there are 11 negative values, the first 11 values will be omitted

    Parameters
    ----------
    df, pandas.dataframe:
        dataframe where the inductance points will be cutoff
        This dataframe should be one from an EIS spectra taken from a Gamry
    
    Return --> the new dataframe without the negative values in the beginning
    '''
    cut_inductance_points = 0 # 0
    for i,row in df.iterrows():
        if df.iloc[i]['ohm.1'] < 0:
            cut_inductance_points +=1
        else:
            break
    
    df = df.iloc[cut_inductance_points:,:]

    return df

def cut_ohmic(df):
    '''
    Cut Ohmic
    This function is used to subtract off the ohmic resistance off of an EIS spectra
    All spectra will be plotted to start roughly at 0
    It finds the last negative value in the EIS data and subtracts the zreal from all values in the df

    If the spectra does not cross the x-axis, the point with the lowest zimag is used as a proxie for the point
    that crosses the x axis.

    Parameters
    ----------
    df, pandas.dataframe:
        dataframe where the ohmic resistance will be subtracted
        This dataframe should be one from an EIS spectra taken from a Gamry
    
    Return --> the new dataframe where the ohmic resistance is subtracted off the data
    '''
    no_neg = False

    # Seeing if the Zreal crosses the x-axis
    for i,row in df.iterrows():
        if df.iloc[i]['ohm.1'] < 0:
            no_neg = True
            break
            
    # - Finding the ohmic resistance
    if no_neg == True:
        for i,row in df.iterrows():
            if df.iloc[i]['ohm.1'] < 0:
                ohmic = df.iloc[i]['ohm']
            else:
                break
    else:
        ohmic = df['ohm'].min()
        # min_idx = df['ohm.1'].idxmin()
        # ohmic = df['ohm'].loc[min_idx]
        # df = df.truncate(before = min_idx)
        df = df.reset_index(drop=True)

    for i,row in df.iterrows():
        df.iloc[i]['ohm'] = df.iloc[i]['ohm']-ohmic

    return df
