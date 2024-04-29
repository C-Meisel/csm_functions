
'Imports'
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import findfont, FontProperties

# Global Parameters
plt.rcParams['axes.facecolor'] = '#f2f2f2' # The backround color for inside the figure area
plt.rcParams['figure.facecolor'] = '#f2f2f2' # The backround color for the area areound the figure (outside the axes)


## Playing around with fonts
# --> Use this line to clear the cache: rm -fr ~/.cache/matplotlib
# DejaVu Sans
# print(font_manager.findSystemFonts(fontpaths=None, fontext="ttf"))
font_dir = ['/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Lato']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# mpl.rcParams['font.sans-serif'] = 'Lato-Regular'
# print(font_manager.findfont("Lato"))
# mpl.rcParams['font.family'] = 'sans-serif'

# mpl.rcParams['font.sans-serif'] = ['Lato']
# font = findfont(FontProperties(family=['sans-serif']))
# print(font)

ssp = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Source_Sans_Pro/SourceSansPro-Regular.ttf'
ariel = '/System/Library/Fonts/Supplemental/Arial.ttf'
path = ssp
path_ariel = ariel
prop = font_manager.FontProperties(fname=path)
prop_ariel = font_manager.FontProperties(fname=path_ariel)
# mpl.rcParams['font.family'] = prop_ariel.get_name()

# --- Changing fonts (this took soooo many hours to figure out)
# Only to realize that Lato uses different unicode symbols and doesnt have tau (fuck)
from matplotlib import font_manager
lato = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Lato'
roboto = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Roboto,Roboto_Condensed'
ssp = '/Users/Charlie/Documents/Misc/Random Art stuff/DownloadedFonts/Source_Sans_Pro/SourceSansPro-Regular.ttf'
font_dir = [lato]
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(font)
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['lato']
# print(mpl.rcParams['font.sans-serif'])


