''' 
This module contains functions to help format and plot Arrhenius data.
The data is electrohchemical impedance spectroscopy (EIS) data obtained by a Gamry potentiostat. 
The files are .DTA files and this module plots EIS and IV curves as well as fits and plots DRT using
the Hybrid_DRT package developed by Dr. Jake Huang
# C-Meisel
'''

'Imports'
import os #operating system useful for navigating directories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import scipy as scipy
import seaborn as sns
import sys
import traceback

# from bayes_drt2.inversion import Inverter
# from bayes_drt2 import plotting as bp 

# from .fit_drt import map_drt_save
from .plotting import plot_peiss, plot_ivfcs
from .convenience import excel_datasheet_exists