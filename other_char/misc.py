''' 
This module contains all other functions that I made for analyzing data.
They mostly pertain to random characterization techniques or software.

# C-Meisel
'''

'Imports'
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import cmasher as cmr
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.table import Table
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib as mpl
import hashlib
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.sans-serif'] = 'Arial'


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Mass Spec functions"
'Functions to help me format data from the Prima DB mass spec in GRL241'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def ms_df_tc(loc:str)->pd.DataFrame: 
    '''
    Takes a CSV file from the prima DB and turns it into a df with the useful materials and
    converts time into relative time from the first measurement taken
    ms = mass spec, df = DataFrame, tc = time converter
    
    Param loc, str: (path to a file)
        Location of the CSV file containing the mass spec data

    Return --> a DataFrame containing the desired time values
    '''
    # Time conversion
    df = pd.read_csv(loc)
    t_init = int(pd.to_datetime(df.at[0,'Time&Date']).strftime("%s")) #converts excel time into something more useable and sets this as the initial time
    df['Time&Date'] = pd.to_datetime(df['Time&Date']).apply(lambda x: x.strftime('%s')) #-t_init#.dt.strftime("%s")-t_init #converts absolute time to relative time
    t_values = [int(t) for t in df['Time&Date'].to_numpy()]
    df['Time&Date'] = [t-t_init for t in t_values]
    return df

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Drycal Functions"
'Functions to help me format data from the Drycal used to calculate gas flow'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def plot_drycal_flow(loc_excel:str, sheet:str): 
    '''
    Plots drycal flow from a specific sheet in a specific file in excel

    Param loc_excel, str: (path to a file)
        Location of the excel file containing the drycal data
    Param sheet, str:
        Name of the sheet in the excel file to be plotted
    
    Return --> none but a plot is generated and shown
    '''
    
    df = pd.read_excel(loc_excel,sheet,skiprows=3)
    flow = df['DryCal scc/min ']
    time= [int(t.strftime('%s')) for t in df['Time'].to_numpy()] #Converts the date time into an integer array of seconds from epoch
    #Each time in df_dc_c1['Time'] is converted into seconds from epoch and then converted into an integer. 
    # df_dc_c1['Time'] is first converted into an array in numpy
    
    t0 = time[0] #Gets first time
    test_time = [t-t0 for t in time] #Getting delta t from start of test
    
    # - Plotting
    fig, ax = plt.subplots()
    ax.plot(test_time,flow,'b',linewidth=2.5)
    ax.set_xlabel('Time (s)',weight='bold')
    ax.set_ylabel('Flow (SCCM)',weight='bold')
    plt.tight_layout()
    plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Gas Chromatography Functions"
'Functions to help me format data from the gas chromatograph (GC) and plot it'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def plot_gc(loc_excel:str, sheet:str):
    '''
    Plots GC data from a specific sheet in a specific file in excel
    It is currently set to plot the Hydrogen, Nitrogen, and Methane Data

    Param loc_excel, str: (path to a file)
        Location of the excel file containing the GC data
    Param sheet, str:
        Name of the sheet in the excel file to be plotted

    Return --> none but a plot is generated and shown
    '''
    df= pd.read_excel(loc_excel,sheet,skiprows=3,usecols=[1,2,3,4])
    fig,ax = plt.subplots()

    # - Plotting
    ax.plot(df['Run'],df['Hydrogen (%)'],label='H$_2$')
    ax.plot(df['Run'],df['Nitrogen (%)'],label='N$_2$')
    ax.plot(df['Run'],df['Methane (%)'],label='CH$_4$')
    ax.set_xlabel('GC run',weight='bold')
    ax.set_ylabel('Gas Concentration (%)',weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"SEM and TEM Functions"
'Functions to help me format and plot Scanning Electron Microscope (SEM) data'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def element_to_color(element):
    '''
    For EDS linescan mapping. Made by ChatGPT, colors asjusted by me
    '''
    # Dictionary mapping elements to hex colors
    hex_color_map = {
        'Ba': '#1f77b4',  # Example hex color for Barium
        'Ce': '#ff7f0e',  
        'Zr': '#2ca02c',  
        'Y': '#d62728',
        'Yb': '#9467bd',
        'Ni': '#8c564b',  
        'O': '#e377c2',
        'S': '#bcbd22',
        'Si': '#7f7f7f',
        'Gd': '#17becf', # 10   
        'C': '#c64886', 
        'Nb': '#59276f',
        'Mg':'#32a88d',
        'Fe':'#a87d32',
        'Al':'#8f7ca3',
        'Co': '#0047AB',
        'Ag': '#a3c9c7',
        'Au': '#FFD700',
    }

    # Return the corresponding hex color, or a default hex color if the element is not in the dictionary
    return hex_color_map.get(element, '#7f7f7f')  # Grey as the default hex color

def string_to_hex_color(string):
    '''
    For linescan mapping. Made by ChatGPT
    '''
    # Hash the string to a hexadecimal number
    hash_object = hashlib.md5(string.encode())
    hex_hash = hash_object.hexdigest()

    # Take the first 6 characters for the color code
    hex_color = '#' + hex_hash[:6]
    return hex_color

def eds_line_plot(loc_csv:str, only:list = [''], reverse:bool = False,
                    sort_bczyyb:bool = False, dp:bool = False, net:bool = False,
                    nm:bool = False):
    '''
    Plots EDS line scan data from the Apex EDAX software from the Tescan SEM

    Parameters:
    -----------
    loc_csv, str: (path to a file)
        Path to the CSV file containing the EDS line scan data
    only, list: (Default = [''])
        List of elements that you want to plot. If empty, all elements are plotted
    reverse, bool: (Default = False)
        If True, the data is plotted with the distance reversed
    sort_bczyyb, bool: (Default = False)
        If True, the data is sorted by the order of the elements in BCZYYb for the plot legend
    dp, bool: (Default = False)
        If True, this lets the function know it is plotting a depth profile which will just
        change the plot formatting
    net, bool: (Default = False)
        If true, this means that this is plotting a ZAF fitting, so the y axis will print intensity
        instead of atomic %
    nm, bool: (Default = False)
        If the measurement is in nm then this should be set to true

    Return --> None but a plot is generated and shown
    '''
    
    file = csv.reader(open(loc_csv, "r",encoding='latin1'),delimiter=',') #enables the file to be read by python
    for row in file: #searches for start of data
        if row[0] == 'Point': #string in first column in the row where the data starts
            skip = file.line_num-1
            break
    df = pd.read_csv(loc_csv,sep= ',',skiprows=skip,encoding='latin1',index_col=False) #creates DataFrame, in this case sep needs to be ,
    df.drop(columns=['Point',' Image',' Frame'],inplace=True) #Drops values that are not needed 

    if sort_bczyyb == True:
        df, cols = sort_bczyyb_eds(df)
    else:
        cols = list(df.columns.values) #Makes list of cols
        for i,col in enumerate(cols): #Ensures that the index for distance is right
            if col == ' Distance (um)':
                index = i
                break

        cols.pop(index) #Drops Distance from cols list
        
    if reverse == True:
        df[" Distance (um)"] = df[" Distance (um)"].values[::-1]

    # Plotting
    fig, ax = plt.subplots() #Starts Plot
    if nm == True:
        df[' Distance (um)'] = df[' Distance (um)'].values * 1000 #Converts to nm
    if len(only[0])==0: #If there are no specified elements in only, then all elements will be plotted
        for col in cols: #Plots all of the columns
            ax.plot(df[' Distance (um)'],df[col],label = col,linewidth=2)

    elif len(only[0])>=1: #if certain elements are specified then those elements will be
        col_new = []
        for atom in cols: #Searches through all the elements
            for element in only: #searches through all the selected elements
                if atom.find(element+' ')!=-1: #if the selected element matches with one of the elements in the dataset
                    col_new.append(atom) # appending the matching element to a new element list
                    break

        for col in col_new: #Plots the columns for the selected elements only 
            ax.plot(df[' Distance (um)'],df[col],label = col,linewidth=2)

    # --- Plot formatting
    plt.legend()
    if dp == True:
        ax.annotate("Positrode", xy=(0,0), xycoords="axes fraction",
                        xytext=(-11,-21), textcoords="offset points",
                        ha="left", va="top",size='large') # Places "Positrode" in a certain spot
    
    ax.spines['left'].set_bounds(0,df.to_numpy().max()*1.07)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if reverse == False:
        ax.spines['bottom'].set_bounds(0, round(df[' Distance (um)'].iloc[-1],0))
    elif reverse == True:
        ax.spines['bottom'].set_bounds(0, round(df[' Distance (um)'].iloc[0],0))
    plt.legend(loc='upper left',bbox_to_anchor=(0.97,1),ncol=1,fontsize='large')
    ax.set_xlabel('Distance (μm)',size='x-large')
    ax.set_ylabel('Atomic %',size='x-large')
    if net == True:
        ax.set_ylabel('Intensity',size='x-large')
    ax.tick_params(axis='both', which='major', labelsize='large')
    plt.tight_layout()
    plt.show()

def tem_eds_line_plot(loc:str, only:list = [''], sort_bczyyb:bool = True, 
                    nm:bool = True):
    '''
    Plots EDS line scan data from the Bruker software from the Talos SEM

    Parameters:
    -----------
    loc, str: (path to a file)
        Path to the .txt file containing the EDS line scan data
    only, list: (Default = [''])
        List of elements that you want to plot. If empty, all elements are plotted
    sort_bczyyb, bool: (Default = True)
        If True, the data is sorted by the order of the elements in BCZYYb for the plot legend
    nm, bool: (Default = True)
        If the measurement is in nm then this should be set to true

    Return --> None but a plot is generated and shown
    '''
    
    df = pd.read_csv(loc,encoding='latin1',index_col=False,delim_whitespace=True) #creates DataFrame, in this case sep needs to be ,
    df.drop(columns=['Index'],inplace=True)

    group_cols = df.columns[1:].values.tolist() # if there are multiple data points from one spot, it averages them
    df = df.groupby('µm',as_index=False)[group_cols].mean()

    if sort_bczyyb == True: # Sorts the elements into the proper BCZYYb order
        print(df.columns)
        df, cols = sort_bczyyb_eds(df, SEM=False)
    else:
        cols = list(df.columns.values) #Makes list of cols
        for i,col in enumerate(cols): #Ensures that the index for distance is right
            if col == 'µm':
                index = i
                break
        cols.pop(index) #Drops Distance from cols list

    if nm == True: # Converts distance to nm from microns
        df['µm'] = df['µm'].values * 1000 #Converts to nm

    # ----- Plotting
    fig, ax = plt.subplots() #Starts Plot

    for col in cols: #Plots all of the columns
        ax.plot(df['µm'],df[col],label = col,linewidth=2)

    # --- Plot formatting
    plt.legend(fontsize='large')

    if nm == False:
        ax.set_xlabel('Distance (μm)',size='xx-large')
    else:
        ax.set_xlabel('Distance (nm)',size='xx-large')

    ax.set_ylabel('Intensity (a.u.)',size='xx-large')

    # - Excess formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize='x-large')

    plt.tight_layout()
    plt.show()
           
def eds_mapping(folder_loc:str):
    '''
    When given a folder containing EDS mapping data, this will extract the data for each element
    and plot it in a separate subplot in one figure. Each subplot will be in the same color scheme

    Parameters
    ----------
    folder_loc, str: (path to a folder)
        Path to the folder containing the EDS mapping data
    
    Return --> None but a plot is generated and shown
    '''
    data = [file for file in os.listdir(folder_loc) if file.endswith('.csv')]
    data.remove("ZafWt 1_Field of View.csv")

    # ----- Sorting the data files:
    i = 0

    for file in data:
        if file.find('Ba') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ce') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Zr') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Y ') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Yb') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ni') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    # ---- Setting up subplots
    if len(data) > 4 and len(data) <= 6:
        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(11, 7)
    if len(data) >6 and len(data) <=8:
        fig, axs = plt.subplots(2,4)
        fig.set_size_inches(10, 5)
    if len(data) == 9:
        fig,axs = plt.subplots(3,3)

    # --- Plotting
    for idx, ax in enumerate(axs.reshape(-1)):
        if idx <= len(data)-1:
            element_data = os.path.join(folder_loc,data[idx])

            # --- Extracting useful data from the file
            df_header = pd.read_csv(element_data,nrows=2,skiprows=1,header=None,sep= ',',lineterminator='\n',encoding='latin1')
            scale = df_header.iloc[0,10]*1000 # in microns


            # --- Converting the data to a DataFrame and obtaining the element name
            df = pd.read_csv(element_data,skiprows=5,header=None,encoding='latin1').iloc[:, 1:]
            df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
            df = df.transpose()

            # --- Extracting hte element from the file title
            element = os.path.basename(element_data).split("_", 2)[1]
            element = element.split(".")[0]

            # --- Initializing the plot
            cmap = cmr.get_sub_cmap('cmr.chroma', 0.10, 0.95) #cmr.chroma cmr.rainforest
            im = ax.imshow(df.values,cmap=cmap,vmin=0,vmax=100)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)

            # --- Formatting the plot
            font = 'Helvetica'
            df_len = len(df.index)

            len_col = len(df.columns)
            xticks = [0,len_col-1]
            labels = [0,round(scale,2)]

            ax.xaxis.set_ticks(xticks)
            ax.set_xticklabels(labels)

            ax.tick_params(left=False, which='major', labelsize='large')
            ax.yaxis.set_visible(False)

            ax.set_xlabel('\u03BCm',size='x-large',family=font)
            ax.xaxis.labelpad = -15

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            cbar.ax.yaxis.set_ticks([])

            ax.set_title(element,family=font,size='xx-large')

            fig.tight_layout()

        elif idx > len(data)-1:
            ax.set_visible(False)

    plt.show()

def sort_bczyyb_eds(df:pd.DataFrame, SEM=True, semtek = False):
    '''
    Takes a DataFrame of EDS line data, sorts it such that the first elements are 
    Ba, Ce, Zr, Y, Yb, Ni, O, and then the rest of the elements are in alphabetical order
    If one of the BCZYYb Ni, O elements are not in the data frame, it will not be included and there
    will be no error

    Parameters
    ----------
    df, pd.DataFrame:
        DataFrame of EDS line data
    SEM, bool: (Default = True)
        Sorts an SEM EDS line scan from the TESCAN SEM in the Coorstek building
        if false, sorts a line scan from the TALOS TEM also in the Coorstek building
    semtek, bool: (Default = False)
        Set to True if the EDS data is from the SEMTEK sem in Hill Hall
        
    Return:
    -------
    df --> The new DataFrame with the columns sorted the desired way
    cols --> The list of the columns in the new DataFrame without distance
    '''
    if SEM == True:
        df = df.reindex(sorted(df.columns), axis=1) #sorts elements alphabetically

        Ba_column = df.pop(' Ba L') # Barium to first spot
        df.insert(0, ' Ba L', Ba_column)
        Ce_column = df.pop(' Ce L') # Cerium to second spot
        df.insert(1, ' Ce L', Ce_column)
        if ' Zr L' in df.columns: # Zirconium to third spot
            Zr_column = df.pop(' Zr L') 
            df.insert(2, ' Zr L', Zr_column)
        else:
            Zr_column = df.pop(' Zr K') 
            df.insert(2, ' Zr K', Zr_column)  
        if ' Y L' in df.columns:
            Y_column = df.pop(' Y L') 
            df.insert(3, ' Y L', Zr_column)
        else:
            Y_column = df.pop(' Y K') # Yttrium
            df.insert(3, ' Y K', Y_column)
        if ' Yb M' in df.columns: #ytterbium, checking to see which band is being plotted
            Yb_column = df.pop(' Yb M')
            df.insert(4, ' Yb M', Yb_column)
        else:
            Yb_column = df.pop(' Yb L')
            df.insert(4, ' Yb L', Yb_column)
        if ' Ni K' in df.columns:
            Ni_column = df.pop(' Ni K')
            df.insert(5, ' Ni K', Ni_column)
        if ' O K' in df.columns:
            O_column = df.pop(' O K')
            df.insert(6, ' O K', O_column)

        cols = list(df.columns.values) #Makes list of cols
        index = 0
        for i,col in enumerate(cols): #Ensures that the index for distance is right
            if col == ' Distance (um)':
                index = i
                break

        cols.pop(index) #Drops Distance from cols list
    
    else:
        df = df.reindex(sorted(df.columns), axis=1) #sorts elements alphabetically

        Ba_column = df.pop('Ba') # Barium to first spot
        df.insert(0, 'Ba', Ba_column)

        Ce_column = df.pop('Ce') # Cerium to second spot
        df.insert(1, 'Ce', Ce_column)

        Zr_column = df.pop('Zr') # Zirconium to the third spot
        df.insert(2, 'Zr', Zr_column)

        Y_column = df.pop('Y') # Yttrium to the fourth spot
        df.insert(3, 'Y', Y_column)
        
        Yb_column = df.pop('Yb') # Ytterbium to the fifth spot
        df.insert(4, 'Yb', Yb_column)

        Ni_column = df.pop('Ni') # Oxygen to the sixth spot
        df.insert(5, 'Ni', Ni_column)

        O_column = df.pop('O') # Oxygen to the sixth spot
        df.insert(6, 'O', O_column)
        
        cols = list(df.columns.values) #Makes list of cols
        index = 0
        for i,col in enumerate(cols): #Ensures that the index for distance is right
            if col == 'µm':
                index = i
                break

        if semtek == False:
            cols.pop(index) #Drops Distance from cols list

    return df, cols

def haadf_eds_map(folder_loc:str, font:str = 'Arial',element_list:list = None,
                  save_fig:str = None, pxl_nm:float = None, sb_length:int = None):
    '''
    Function for plotting EDS STEM HAADF maps. This function plots the image
    and the chemical maps, using the turbo cmap, for all the chemical eds map files
    in the selected folder. This function also sorts and plots the BCZYYbNiO atoms first.
    The element files do need to be text files (the talos defaults to BMP)

    This function is mostly done and is functional but future additions include:
    - The choice of whether or not to sort by BCZYYb
    - Taking away the black bars from the tops of the image
    - Figuring out a way to include the scale bar on the original HAADF image

    Parameters
    ----------
    Folder_loc, str: (path to a directory)
        location of the folder where the .txt data files are located
    font, str: (default = Ariel)
        Font used on the plot
    element_list, list (str): (Default = None)
        List of elements to plot.
        If none, then all elements/files in the folder are plot
    save_fig, str: (Default = None)
        If this is not none, the fig will be saved
        save_fig is the file name and path of the saved file.
    img_width, float: (Default = None)
        width of the image in nm, used to make the scalebar
    sb_length, int: (Default = None)
        Scalebar length in nm.
    
    Return --> Nothing, but the image and the chemical maps are plotted
    '''
    
    folder_loc = folder_loc
    # ---
    data = [file for file in os.listdir(folder_loc) if file.endswith('.txt')]

    if element_list is not None:
        def extract_element(filename):
            return filename.split('_')[-1].split('.')[0]
        data = [file for file in data if extract_element(file) in element_list or 'HAADF' in file]
        # data = filtered_files

    # ----- Sorting the data files:
    i = 0

    for file in data:
        if file.find('HAADF') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ba') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ce') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Zr') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Y.') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Yb') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ni') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('O ') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    # ---- Setting up subplots
    if len(data) <= 4:
        fig, axs = plt.subplots(2,2)
        # fig.set_size_inches(11, 7)

    if len(data) > 4 and len(data) <= 6:
        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(11, 7)
    
    if len(data) > 6 and len(data) <=8:
        fig, axs = plt.subplots(2,4)
        fig.set_size_inches(10, 5)
    
    if len(data) == 9:
        fig,axs = plt.subplots(3,3)
    
    if len(data) > 9 and len(data) <= 12:
        fig,axs = plt.subplots(2,6)
        fig.set_size_inches(14, 5)

    # --- Plotting
    title_size = 28 #'xx-large'
    for idx, ax in enumerate(axs.reshape(-1)):
        if idx <= len(data)-1 and data[idx].find('HAADF') != -1:
            image_loc = os.path.join(folder_loc, data[idx])

            df = pd.read_csv(image_loc,header=None,encoding='latin1',sep=';')

            max = df.max().max()
            img = ax.imshow(df.values,cmap='gray', vmin=0, vmax=max)
            cbar = plt.colorbar(img,ax=ax)
            cbar.remove()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('HAADF',family=font,size=title_size)

            if pxl_nm is not None: # Setting the scalebar
                # - Calculations
                image_width_pixels = df.shape[1]
                image_height_pixels = df.shape[0]
                sb_length_pxls = pxl_nm * sb_length


                # - backround box
                x_anchor_bg = image_width_pixels * 0.7
                y_anchor_bg = image_height_pixels * 0.75
                bg_length_offset = 30 # % of the scalebar length
                bg_width_offset = 50 # % of the background length
                bg_length = sb_length_pxls * (1 + bg_length_offset / 100)
                bg_height = bg_length * (bg_width_offset / 100)

                sb_background = patches.Rectangle((x_anchor_bg, y_anchor_bg), bg_length, bg_height, linewidth=1, edgecolor='none', facecolor='k')
                
                # - Scalebar
                x_anchor = ((bg_length - sb_length_pxls) / 2) + x_anchor_bg
                y_anchor = y_anchor_bg + sb_length * 10

                sb = patches.Rectangle((x_anchor, y_anchor), sb_length_pxls, sb_length_pxls*0.1, linewidth=1, edgecolor='none', facecolor='w')

                # - Scalebar text:
                if pxl_nm < 100:
                    scale = ' nm'
                else:
                    scale = ' μm'

                txt = str(sb_length) + scale
                
                bg_center_x = x_anchor_bg + bg_length / 2
                bg_center_y = y_anchor_bg + bg_height / 3

                ax.text(bg_center_x, bg_center_y, txt, ha='center', va='top', fontsize=17, color='w')

                ax.add_patch(sb_background)
                ax.add_patch(sb)

            plt.tight_layout()
        
        elif idx <= len(data)-1:
            # --- Converting the data to a DataFrame and obtaining the element name
            image_loc = os.path.join(folder_loc, data[idx])
            df = pd.read_csv(image_loc,header=None,encoding='latin1',sep=';')
            element = os.path.basename(image_loc).split("_", 4)[3].split('.',2)[0]

            # --- Initializing the plot
            cmap = plt.cm.get_cmap('turbo') #plt.cm.get_cmap('gnuplot2'),  plt.cm.get_cmap('turbo'), #'cmr.chroma'
            max = df.max().max()
            if max < 10:
                im = ax.imshow(df.values,cmap=cmap,vmin=0,vmax=max/5) # max/5
            else:
                im = ax.imshow(df.values,cmap=cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)

            # --- Formatting the plot
            ax.xaxis.set_visible(False)
            ax.axis('off')

            cbar.ax.yaxis.set_ticks([])

            ax.set_title(element,family=font,size=title_size)

            fig.tight_layout()

        elif idx > len(data)-1:
            ax.set_visible(False)

    if save_fig is not None:
        fmat = save_fig.split('.', 1)[-1]
        plt.savefig(save_fig, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

def st_eds_line_overlay(loc_image:str, loc_data:str, scale:float = 0.5,
                        mag:int = None, title:str = None, save_img:str = None,
                        sort_bczyyb:bool = True, publication=False,legend_outside=False,
                        element_list=None):
    '''
    This function plots an SEM image taken during EDS and has
    the EDS linescan overlayed over the image
    This function accepts data taken from the SEMTEK SEM

    Parameters:
    -----------
    loc_image, str:
        Path to the EDS-SEM image
    loc_data, str:
        Path to the EDS data
    scale, float: (Default = 0.5)
        Scales the height of the EDS lines over the SEM image
        0.5 means the max EDS line height reaches half way up the image
    mag, int: (Default = None)
        magnification of the image
        if None, then the distance is normalized to one
    title, str: (default = None)
        title of the plot if this is set to a string
        The title resides inside the plot
    save_img, str: (default = None)
        If this is not none, the figure will be plot and saved
        Save_img is the file name and path of the saved file.
    sort_bczyyb, bool: (default = True)
        If true sort the elements in the legend so that they are ordered BCZYYb
    publication, bool: (default = False)
        If false the figure is formatted for a presentation
        If true the figure is formatted to be a subfigure in a journal paper.
        Setting publication to true increases all feature sizes
    legend_outside, bool: (default = False)
        Whether or not to place the legend outside the figure
    element_list, list (str): (Default = None)
        List of elements to plot.
        If none, then all elements/files in the folder are plot
    
    Return --> None, but plots and shows an image
    '''

    ' ----- Plotting the SEM image ----- '
    # -- Appending data to a dataframe
    df = pd.read_csv(loc_data,encoding='latin1',engine='python')

    if element_list is not None:
        df = df[element_list]

        print(df)

        # def extract_element(filename):
        #     return filename.split('_')[-1].split('.')[0]
        # data = [file for file in data if extract_element(file) in element_list or 'HAADF' in file]
        # # data = filtered_files

    if sort_bczyyb == True and element_list is None:
        df, cols = sort_bczyyb_eds(df, SEM=False, semtek=True)
    else:
        cols = list(df.columns.values)
    
    len_df = len(df)
    
    # - Calculating distance
    if mag == 1000:
        img_width = 112 # um
    if mag == 1500:
        img_width = 75 # μm
    if mag == 2000:
        img_width = 57 # μm
    if mag == 2500:
        img_width = 45 # μm
    elif mag == 3500:
        img_width = 38 # μm
    elif mag == 5000:
        img_width = 23 # μm
    elif mag is None:
        img_width = 1 # Normalizing the distance to 1

    ' ----- Plotting the SEM image ----- '
    fig, ax = plt.subplots()

    # - Load the image using matplotlib.image.imread
    img = mpimg.imread(loc_image)

    # - Display the image using matplotlib.pyplot.imshow
    ax.imshow(img)
    ax.margins(x=0,y=0)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])

    # - Excessive formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ' ----- Plotting the EDS linescan data ----- '
    ax_p = ax.twinx()
    dist_list = np.linspace(0, img.shape[1], len_df)
    df['Distance'] = dist_list

    if publication == False:
        label_size = 'xx-large'
        tick_size = 'large'
    else:
        label_size = 26
        tick_size = label_size * 0.9


    # --- Plotting    
    max_val = 1
    for col in cols:
        max_intensity = df[col].max()
        color = element_to_color(col)
        ax_p.plot(df['Distance'],df[col],label = col,linewidth=2, color=color)

        if max_intensity > max_val:
            max_val = max_intensity

    # - Formatting
    ax_p.yaxis.set_tick_params(labelleft=False)
    ax_p.set_yticks([])
    ax_p.yaxis.tick_left()
    ax_p.yaxis.set_label_position('left')
    ax_p.set_ylabel('Intensity (a.u.)',fontsize=label_size)
    ax_p.margins(x=0,y=0)
    
    recip_scale = 1/scale
    ax_p.set_ylim(0,max_val*recip_scale)

    # - Excessive formatting
    ax_p.spines['top'].set_visible(False)
    ax_p.spines['right'].set_visible(False)

    # - Adjusting the xticks
    x_tick_labels = np.linspace(0,img_width,4)
    x_tick_locs = x_tick_labels * img.shape[1]/img_width
    ax.set_xticks(x_tick_locs)
    x_tick_labels = ["%.1f" % number for number in x_tick_labels]
    x_tick_labels[0] = int(float(x_tick_labels[0]))
    x_tick_labels[-1] = int(float(x_tick_labels[-1]))
    ax.set_xticklabels(x_tick_labels, fontsize=tick_size)

    if mag is None:
        ax.set_xlabel('Distance (a.u.)',fontsize=label_size)
    else:
        ax.set_xlabel('Distance (μm)',fontsize=label_size)

    # - Adding a legend
    if legend_outside == False:
        leg = plt.legend(loc='upper left', handlelength = 0.5, handletextpad=0.35,
                   fontsize= 'xx-large', framealpha=0.9,labelspacing=0.35)
    if legend_outside == True:
        leg = plt.legend(loc='upper left', handlelength = 0.5, handletextpad=0.35,
                fontsize= 'xx-large', frameon=False,labelspacing=0.4,bbox_to_anchor=(0.97,1))
    
    for line in leg.get_lines():
        line.set_linewidth(4.0)
        
    # - Adding a title:
    if title is not None:
        ax.text(0.5, 0.98, title, 
                fontsize=34, 
                color='black', 
                ha='center', 
                va='top', 
                transform=ax.transAxes,  # Coordinate system relative to the axes
                bbox=dict(facecolor='white', alpha=0.8,edgecolor='none'))

    # - Saving the image
    if save_img is not None:
        fmat = save_img.split('.', 1)[-1]
        fig.savefig(save_img, dpi=300, format=fmat, bbox_inches='tight')
    
    fig.tight_layout()
    plt.show()

def st_eds_image_line(loc_image:str, loc_data, mag = None):
    '''
    THis function plots
    
    '''
    fig, (ax_i, ax_p) = plt.subplots(nrows=2, ncols=1, figsize=(5, 6.4),
                                      gridspec_kw={'height_ratios': [2, 1]})

    ' ----- Plotting the SEM image ----- '
    # Load the image using matplotlib.image.imread
    img = mpimg.imread(loc_image)

    # Display the image using matplotlib.pyplot.imshow
    ax_i.imshow(img)
    ax_i.axis('off')

    ' ----- Plotting the EDS linescan data ----- '
    # -- Appending data to a dataframe
    df = pd.read_csv(loc_data,encoding='latin1',engine='python')
    df, cols = sort_bczyyb_eds(df, SEM=False, semtek=True)
    len_df = len(df)
    
    # -- Calculating distance
    if mag == 1500:
        img_width = 75 # μm
    elif mag == 3500:
        img_width = 38 # μm
    elif mag is None:
        img_width = 1 # Normalizing the distance to 1
    
    dist_list = np.linspace(0, img_width, len_df)
    df['Distance'] = dist_list

    # --- Plotting
    for col in cols:
        ax_p.plot(df['Distance'],df[col],label = col,linewidth=2)

    # - Formatting
    ax_p.set_ylabel('Intensity (a.u.)',fontsize='x-large')
    if mag is None:
        ax_p.set_xlabel('Distance (a.u.)',fontsize='x-large')
    else:
        ax_p.set_xlabel('Distance (μm)',fontsize='x-large')

    # - Excessive formatting
    ax_p.yaxis.set_tick_params(labelleft=False)
    ax_p.set_yticks([])
    ax_p.margins(0)
    ax_p.spines['top'].set_visible(False)
    ax_p.tick_params(axis='x', which='major', labelsize='large')
    ax_p.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    # - Verticle lines
    # x_ticks = ax_p.get_xticks()
    # for tick in x_ticks:
    #     if tick < max(x_ticks) and tick > min(x_ticks):
    #         ax_p.axvline(x=tick, color= '#C3C411', linestyle = '--', linewidth = 1, alpha = 0.8)

    #         img_loc = img.shape[1] * (tick/img_width)
    #         ax_i.axvline(x=img_loc, color= '#C3C411',  linestyle = '--',linewidth = 1, alpha = 0.8)
    
    # - Final formatting and showing
    plt.legend(loc='upper left',bbox_to_anchor=(1,1), handlelength = 1)
    fig.subplots_adjust(hspace=-0.52)
    plt.tight_layout()
    plt.show()


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Particle size analyzer (PSA) functions"
'Functions to help me format and plot data from the PSA in Hill Hall 375'
' Also Functions to format and plot data from the Microtrac Flowsync PSA in CK250'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def psa_d50(loc:str)->float: 
    '''
    Takes in a PSA excel sheet and returns the D50 of the sample (float)
    D50 is the diameter of a particle in the 50th percentile of size

    Param loc, str: (path to a file)
        Location of the excel file containing the PSA data
    
    Return --> D50, float
    '''
    
    # --- Making a DataFrame for the useful data
    df = pd.read_excel(loc,'Sheet1',skiprows=25,nrows=10) #The percentiles always start at row 26 and will only need 10 rows
    
    # --- returning the D50
    d50 = df[df['%Tile']==50].iloc[0,2] #This returns a float. Looks through df %tile column for 50 then returns the value in column 3 (2+1)
    
    return d50

def psa_plot(loc:str, plot_d50:bool=True):
    '''
    Plots Particle size analyzer data and shows the D50 on the plot if plot_d50 is True

    Param loc, str: (path to a file)
        Location of the excel file containing the PSA data
    Param plot_d50, bool: (Default = True)
        If True, the D50 will be plotted on the plot as a vertical line
    
    Return --> None but a plot is created and shown
    '''
    
    df = pd.read_excel(loc,'Sheet1',skiprows=68) #the sheet will always be called sheet 1 and the data will always start at row 69 (nice)
    d50 = psa_d50(loc) # retrieves d50

    # --- Plotting
    fig,ax = plt.subplots()
    ax.plot(df['Size(um)'],df['%Chan'],'k',linewidth=4)

    # --- plotting D50 and printing value
    if plot_d50==True:
        ax.axvline(d50,color='#D2492A')
        d50s = f'{d50}' #converting the int to a string so I can print it on the figure
        ax.text(d50,0.1,r'$D50 =$'+d50s+r' $\mu m$',ha='left',size='large',weight='bold',color='#D2492A')

    # - Misc plot formatting
    ax.set_xscale('log')
    ax.set_ylabel('% Chance (%)',size=16)
    ax.set_xlabel('Particle Diameter (μm)',size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

def psa_plots(list:list):
    '''
    Plots multiple PSA data sets on the same plot

    Param list, list of strs:
        List of paths to the PSA files to be plotted

    Return --> None, but a plot is created and shown
    '''
    
    # --- Plotting
    fig,ax = plt.subplots()
    for psa in list:
        df = pd.read_excel(psa[0],'Sheet1',skiprows=68) #the sheet will always be called sheet 1 and the data will always start at row 69 (nice)
        ax.plot(df['Size(um)'],df['%Chan'],label=psa[1],linewidth=4)
    # - Mist plotting stuff
    ax.set_xscale('log')
    ax.set_ylabel('% Chance (%)',size=16)
    ax.set_xlabel('Particle Diameter (μm)',size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

def sync_psa_plot(data):
    '''
    Plots PSA data. Can plot one or multiple files

    Parameters:
    -----------
    data, str or str list:
        Path or list of paths to the PSA files to be plotted.
        Accepts .csv files. 
        data should be a string or a list of strings

    Return --> None, but a plot is created and shown
    '''

    if isinstance(data, str): # If input_data is a single string, plot data from a single file
        # - Looking for the blank line at the beginning of the file
        with open(data, 'r', newline='') as file:
            reader = csv.reader(file)
            first_row = next(reader) # Read the first row (header or data)
            first_cell = first_row[0]  # Extract the first cell

            if first_cell == "":
                adjust = 1
            else:
                adjust = 0

        # - Extracting  Plotting Data
        df = pd.read_csv(data, skiprows=69 + adjust, usecols = ['Size(um)', '%Chan']) # 
        df = df.iloc[0:69]
        x = df['Size(um)']
        y = df['%Chan']

        # - Extracting the D-50 value:
        df_pcent = pd.read_csv(data, skiprows=24 + adjust, usecols = ['%Tile', 'Size(um)'],encoding='utf-8-sig', engine='python' )
        d50 = float(df_pcent.iloc[4,1])

        # - Extracting the peaks
        df_peaks = pd.read_csv(data, skiprows=48 + adjust, usecols = ['Dia(um)', ' Volume  % ', 'Width'])
        new_names = {'Dia(um)':'D50 (μm)', ' Volume  % ':'Volume (%)', 'Width':'Range (μm)'}
        df_peaks = df_peaks.rename(columns=new_names)

        index_of_first_nan = df_peaks.index[df_peaks.isna().any(axis=1)].tolist()[0]
        sliced_df_peaks = df_peaks.loc[:index_of_first_nan - 1]

        # -  Calculate the differences between adjacent x-values to set proper bin widths
        diff_x = np.diff(x)
        diff_x = np.append(diff_x,0) # 0 is added to keep the list the same size as the data list.

        # - Figuring out the color of the chart:
        fname = os.path.basename(data)
        if "Elyte" in fname:
            color = '#48443c'
        elif "Anode" in fname:
            color = '#6b7c6f'
        else:
            color = 'darkmagenta'

        # - Create a bar plot using the calculated widths
        fig, ax = plt.subplots()
        ax.bar(x, y, width = diff_x, ec="k", align="edge",color=color)

        # - Formatting
        ax.set_xscale('log')
        ax.set_xlim(0.1,1000)
        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.set_xlabel('Log Particle size (μm)', size= 'xx-large')
        ax.set_ylabel('Channel (%)', size= 'xx-large')

        # - Printing a table for the peaks
        num_rows, num_cols = sliced_df_peaks.shape

        bbox_height = 0.12 + num_rows * 0.06
        
        table = Table(ax, bbox=[0.56, 0.6, 0.44, bbox_height])
        # Define the table properties
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1, 1.6)  # Adjust the table scaling as needed

        # Add header rows with labels
        for i in range(num_cols):
            text = ''
            if i == 1:
                text = 'Peak Properties'
            table.add_cell(0, i, 1, 1, text= text, loc='center', edgecolor='white')
  
        for i in range(num_cols):
            table.add_cell(1, i, 1, 1, text=sliced_df_peaks.columns[i], loc='center', facecolor=color)
            cell = table[1,i]
            cell.get_text().set_color('white')

        # Add data to the table iteratively
        for i in range(num_rows):
            for j in range(num_cols):
                table.add_cell(i + 2, j, 1, 1, text=sliced_df_peaks.iloc[i, j], loc='center')

        # Add the table to the plot
        table.auto_set_column_width([i for i in range(num_cols)])  # Adjust column widths
        ax.add_table(table)

        # - Plotting the D50 line
        ax.axvline(d50,color='#D2492A')
        d50s = f'{d50}' #converting the int to a string so I can print it on the figure
        ax.text(d50,y.max()*1.07,r'$D50 =$'+d50s+r' $\mu m$',ha='center',va='center',size='large',weight='bold',color='#D2492A')

        # - Excessive formatting:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize='x-large')

        plt.tight_layout()
        plt.show()

    elif isinstance(data, list): # If input_data is a list of strings, plot data from multiple files
        fig, ax = plt.subplots()

        for file_path in data:
            # --- Adjusting for the random blank row at the start of some files
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                first_row = next(reader) # Read the first row (header or data)
                first_cell = first_row[0]  # Extract the first cell

                if first_cell == "":
                    adjust = 1
                else:
                    adjust = 0
            
            # --- Extracting data
            df = pd.read_csv(file_path, skiprows=69 + adjust, usecols = ['Size(um)', '%Chan']) # 
            df = df.iloc[0:69]
            x = df['Size(um)']
            y = df['%Chan']

            fname = os.path.basename(file_path)
            parts = fname.split("_Data")
            label = parts[0]

            ax.semilogx(x, y, label=label, linewidth=2)

               
        # - Formatting
        ax.set_xscale('log')
        ax.set_xlim(0.1,1000)
        ax.legend(fontsize='large')

        ax.set_xlabel('Log Particle size (μm)', size= 'xx-large')
        ax.set_ylabel('Channel (%)', size= 'xx-large')

        # - Excessive formatting:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize='x-large')

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Input must be a string or a list of strings")


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"LEIS"
'Functions to help me format and plot Low energy ion scattering (LEIS) data from the IONTOF in the SEEL building at CU Boulder'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def leis_plot(data_loc:str,label:str, normalize:bool=False, **plot_args):
    '''
    Plots LEIS data from a text file gathered from an IONTOF LEIS machine
    To plot multiple plots on the same figure, just stack these functions on top of each other

    Parameters:
    -----------
    data_loc, str:
        Location of the text file containing the LEIS data
    label, str:
        The label for the given data in the plot legend
    normalize, bool: (Default = False)
        Whether or not to normalize the data to the highest value in the data
    plot_args,dict: 
        Any arguments that are passed to the plot function.
    '''

    # ~~~~~ Creating a DataFrame from the text file
    df = pd.read_csv(data_loc, header=None, encoding='latin1', sep = '\t',
                        names = ['Energy (eV)','Intensity'])
    
    # ~~ Formatting data if desired:
    if normalize ==True:
        maximum = df['Intensity'].max()
        df['Intensity'] = df['Intensity']/maximum


    # ~~~~~ Plotting
    with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
        plt.plot(df['Energy (eV)'],df['Intensity'],'-',**plot_args, label = label) #plots data

    # ~ Axis labels
    plt.xlabel('Energy (eV)',size=18) #\u00D7
    if normalize == True:
        plt.ylabel('Relative Intensity',size=18)
    else:
        plt.ylabel('Intensity',size=18)
        
    plt.legend(loc='best',fontsize='x-large')


    # ~~~~~ Excessive Plot formatting
    plt.tick_params(left = False)
    plt.yticks([])
    plt.xticks(fontsize=14)

    plt.tight_layout()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"XPS"
'Functions to help me format and plot X-ray photoelectron spectroscopy raw data taken from the Environmental XPS in the Coorstek Building'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def plot_xps(loc:str, normalize:bool=False, label:str = None, flip_x:bool = False):
    '''
    Takes a .txt file from the XPS machine and plots the data
    Can plot multiple plots on one figure
    Parameters
    ----------
    loc, str:
        Location of the .txt XPS file
    normalize, boolean: (default = False)
        Whether or not to normalize the data to the highest point
    label, str: (default = None)
        Label of the spectra in the plot legend. If the default is selected no legend will be added
        if a label is added then a legend will pe placed on the plot and the label will coincide with the spectra
    flip_x, bool: (default = False)
        The x-data for each plot needs to be flipped. Each plot has the data for all plots flipped.
        Therefore when an even number of plots are plot, the x-axes needs to be flipped one more time
    
    Return --> None, but a graph is plotted though it is not shown
    '''

    # ----- Extracting the X axis for the XPS plot
    datafile = open(loc, 'r')
    data = [line.split(',') for line in datafile.readlines()]

    x_string = str(data[8])
    x_string = x_string[20:-4]
    x_list = x_string.split(' ')
    x_int_list = [float(str) for str in x_list]
    # print(len(x_int_list))

    ' ----- Extracting the Y data '
    # -- Figuring out how many rows to skip
    for i,line in enumerate(data):
        line = str(line)
        if line[2:10] == '[Data 1]':
            data_start = i+1

    raw_data = data[data_start:-1] # For whatever reason this doesnt work if the last datapoint is selected (likely it is just a line break)
    
    # -- Extracting each point and finding the average of each point (each point is a binding energy (x value))
    y_data = []
    for point in raw_data:
        string = str(point) # Converting measurement array into a string
        y_string = string[3:-4] # Removing garbage at the beginning and end of the number
        y_list = y_string.split('  ') # Splitting up the string by a few spaces, this isolates each number
        y_float_list = [float(str) for str in y_list] # Converting each number to a float
        y_float_list = y_float_list[1:] # The first value in the dataset is the binding energy, this is not needed and will throw off the data
        y_array = np.array(y_float_list) # converitin the list to an numpy array
        avg = np.mean(y_array) # finding the average of the array at each datapoint. This is just my guess at what Casa XPS graphs
        y_data.append(avg) # Appending the average value to the Y-data list

    # -- Whether or not to normalize the data
    if normalize == True:
        maximum = np.max(y_data)
        y_data = y_data/maximum

    # ----- Plotting
    plt.plot(x_int_list, y_data, label=label)

    # -- Basic formatting
    plt.ylabel('Intensity', fontsize='xx-large')
    plt.xlabel('Binding Energy (Ev)', fontsize='xx-large')
    plt.tick_params(axis='both', which='major', labelsize = 'x-large')

    # Inverting the x axis
    ax = plt.gca()
    ax.set_xlim(ax.get_xlim()[::-1])

    if flip_x == True:
        ax.set_xlim(ax.get_xlim()[::-1])

    # -- Excessive formatting (less than usual)
    if normalize == True:
        plt.ylabel('Relative Intensity (a.u)', fontsize='xx-large')
        plt.yticks([])

    # Getting rid on axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if label is not None:
        plt.legend(loc='best',fontsize='x-large')


    plt.tight_layout()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"EPMA"
'Functions to help me format and plot electron probe micro analyzer (EPMA) data taken at KICET in South Korea'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def epma_plots(folder:str, sem:bool = True):
    '''
    Plots EPMA elemental maps from a folder of EPMA files. Each EPMA scan data is in its own distinct folder.
    If sem = True then the SEM images of the cells are plotted as well.

    Parameters:
    -----------
    folder, str: (path to directory)
        Location of the file containing the EPMA data for a sample
    sem, bool: (Default = True)
        Plots the Secondary Electron (SEI) and Back Scattered Electron (COMPO) images of the cell
        where the elemental data was taken.

    Return --> None, but a plot of plots is created and shown
    '''

    # --- Extracting all data files from the folder:
    if sem == False:
        data = [file for file in os.listdir(folder) if file.endswith('La.csv')]
    elif sem == True:
        data = [file for file in os.listdir(folder) if file.endswith('.csv')]

    # ----- Sorting the data files:
    i = 0
    for file in data:
        if file.find('Ba') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ce') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Zr') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Y') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Yb') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    for file in data:
        if file.find('Ni') != -1:
            data.remove(file)
            data.insert(i,file)
            i = i + 1

    if sem == True:
        for file in data:
            if file.find('SEI') != -1:
                data.remove(file)
                data.insert(i,file)
                i = i + 1
        for file in data:
            if file.find('COMPO') != -1:
                data.remove(file)
                data.insert(i,file)
                i = i + 1

    # ---- Plotting data
    if len(data) > 4 and len(data) <= 6:
        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(11, 7)
    if len(data) >6 and len(data) <=8:
        fig, axs = plt.subplots(2,4)
        fig.set_size_inches(10, 5)
    if len(data) == 9:
        fig,axs = plt.subplots(3,3)

    for idx, ax in enumerate(axs.reshape(-1)):
        if idx <= len(data)-1 and data[idx].find('SEI')==-1 and data[idx].find('COMPO')==-1:
            element_data = os.path.join(folder,data[idx])

            # --- Converting the data to a DataFrame and obtaining the element name
            df = pd.read_csv(element_data,header=None,encoding='latin1')
            element = os.path.basename(element_data).split("_", 2)[1]

            # --- Initializing the plot
            cmap = cmr.get_sub_cmap('cmr.chroma', 0.10, 0.95) #cmr.chroma cmr.rainforest
            im = ax.imshow(df.values,cmap=cmap,vmin=0,vmax=100)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)

            # --- Formatting the plot
            font = 'Helvetica'
            df_len = len(df.index)
            xticks = [0,df_len-1]
            yticks = [0,df_len-1]
            labels = [0,25]
            ax.xaxis.set_ticks(xticks)
            ax.set_xticklabels(labels)
            ax.yaxis.set_ticks(yticks)
            ax.set_yticklabels(np.flip(labels))
            ax.tick_params(axis='both', which='major', labelsize='large')
            ax.set_ylabel('\u03BCm',size='x-large',family=font)
            ax.yaxis.labelpad = -15
            ax.set_xlabel('\u03BCm',size='x-large',family=font)
            ax.xaxis.labelpad = -15

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar.ax.tick_params(labelsize=12)
            cbar.ax.yaxis.set_ticks([0,100])

            ax.set_title(element,family=font,size='xx-large')

            fig.tight_layout()

        elif idx <= len(data)-1 and sem == True and (data[idx].find('SEI')!=-1 or data[idx].find('COMPO')!=-1):
            image = os.path.join(folder,data[idx])

            # --- Converting the data to a DataFrame and obtaining the element name
            df = pd.read_csv(image,header=None,encoding='latin1')
            if data[idx].find('SEI')!=-1:
                image_type = os.path.basename(image).split("_", 1)[1].replace('.csv','')
            elif data[idx].find('COMPO')!=-1:
                image_type = 'BSE'
            max = df.max().max()
            im = ax.imshow(df.values,cmap='gray', vmin=0, vmax=max)
            cbar = fig.colorbar(im,ax=ax)
            cbar.remove()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(image_type,family=font,size='xx-large')
            plt.tight_layout()

        elif idx > len(data)-1:
            ax.set_visible(False)

    plt.show()