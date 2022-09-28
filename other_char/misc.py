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
        df[' Distance (um)'] = df[' Distance (um)'].values/1000 #Converts to nm
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
    ax.set_xlabel('Distance ($\mu$m)',size='x-large')
    ax.set_ylabel('Atomic %',size='x-large')
    if net == True:
        ax.set_ylabel('Intensity',size='x-large')
    ax.tick_params(axis='both', which='major', labelsize='large')
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

def sort_bczyyb_eds(df:pd.DataFrame):
    '''
    Takes a DataFrame of EDS line data, sorts it such that the first elements are 
    Ba, Ce, Zr, Y, Yb, Ni, O, and then the rest of the elements are in alphabetical order
    If one of the BCZYYb Ni, O elements are not in the data frame, it will not be included and there
    will be no error

    Parameters
    ----------
    df, pd.DataFrame:
        DataFrame of EDS line data
    
    Return:
    -------
    df --> The new DataFrame with the columns sorted the desired way
    cols --> The list of the columns in the new DataFrame without distance
    '''
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
    
    return df, cols

def haadf_eds_map(folder_loc:str):
    '''
    Function for plotting EDS STEM HAADF maps. This function plots the image
    and the chemical maps, using the turbo cmap, for all the chemical eds map files
    in the selected folder. This function also sorts and plots the BCZYYbNiO atoms first.
    The element files do need to be text files (the talos defaults to BMP)

    This function is mostly done and is functional but future additions include:
    - being able to select specific atoms to plot in the function call
    - The choice of whether or not to sort by BCZYYb
    - Taking away the black bars from the tops of the image
    - Figuring out a way to include the scale bar on the original HAADF image

    Parameters
    ----------
    Folder_loc, str: (path to a directory)
        location of the folder where the .txt data files are located
    
    Return --> Nothing, but the image and the chemical maps are plotted
    '''
    
    folder_loc = folder_loc
    font = 'Helvetica'
    # ---
    data = [file for file in os.listdir(folder_loc) if file.endswith('.txt')]

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
            ax.set_title('HAADF',family=font,size='xx-large')
            plt.tight_layout()
        
        elif idx <= len(data)-1:
            # --- Converting the data to a DataFrame and obtaining the element name
            image_loc = os.path.join(folder_loc, data[idx])
            df = pd.read_csv(image_loc,header=None,encoding='latin1',sep=';')
            element = os.path.basename(image_loc).split("_", 4)[3].split('.',2)[0]

            # --- Initializing the plot
            cmap = 'cmr.chroma' #plt.cm.get_cmap('turbo')
            max = df.max().max()
            if max < 10:
                im = ax.imshow(df.values,cmap=cmap,vmin=0,vmax=max/5)
            else:
                im = ax.imshow(df.values,cmap=cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)

            # --- Formatting the plot
            font = 'Helvetica'

            ax.xaxis.set_visible(False)
            ax.axis('off')

            cbar.ax.yaxis.set_ticks([])

            ax.set_title(element,family=font,size='xx-large')

            fig.tight_layout()

        elif idx > len(data)-1:
            ax.set_visible(False)

    plt.show()


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Particle size analyzer (PSA) functions"
'Functions to help me format and plot data from the PSA in Hill Hall 375'
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
    ax.set_xlabel('Particle Diameter ($\mu$m)',size=16)
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
    ax.set_xlabel('Particle Diameter ($\mu$m)',size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

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