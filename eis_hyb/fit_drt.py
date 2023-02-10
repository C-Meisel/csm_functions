''' This module contains functions to help it distribution of relaxation time (DRT) spectra to 
potentiostatic electrochemical impedance spectroscopy data taken with a Gamry potentiostat
All of the actual DRT fitting and analysis is done in the hybrid_drt module made by Jake Huang
# C-Meisel
'''

'Imports'
import os
from hybdrt import file_load as fl
from hybdrt.models import DRT
from hybdrt.models import elements
