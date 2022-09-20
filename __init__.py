''' Here is where I will set global parameters for my plotting modules'''


import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Changing background color of the plots
# plt.rcParams['axes.facecolor'] = '#f2f2f2' # The background color for inside the figure area
# plt.rcParams['figure.facecolor'] = '#f2f2f2' # The background color for the area around the figure (outside the axes)

# --- Changing the font of the plots
from matplotlib import font_manager
lato = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Lato'
roboto = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Roboto,Roboto_Condensed'
ssp = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Source_Sans_Pro'
font_dir = [ssp]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Source Sans Pro']