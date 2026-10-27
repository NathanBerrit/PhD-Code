import os
import numpy as np
#import pickle
import scipy as sc
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib
from matplotlib.transforms import Bbox

import matplotlib.colors
import plotly.offline
import plotly.graph_objs as go
import pandas as pd


os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\Permittivity_study\fit_interband_permittivity')
from fit_interband_func import *
import fit_interband_func
importlib.reload(fit_interband_func)
from fit_interband_func  import *



