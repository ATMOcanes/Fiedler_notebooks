
# coding: utf-8

# # METR 4323: Weather Simulation with Computers
# 
# A course in the School of Meteorology at the University of Oklahoma
# 
# Prof. Brian Fiedler
# metrprof@gmail.com
# 
# v1.1 10 July 2018
# 
# Although this is an ipython notebook, the intent is to use it solely for documentation for the course. There is no Python to execute. No need to download it.  Just view it right here.
# 
# The important ipython notebooks to download are in the last section.
# 
# The official class website (for due dates, the posting of grades, official syllabus) is at http://canvas.ou.edu

# ## Installing Anaconda Python for Jupyter Notebooks
# 
# You will need your own installation of [Anaconda Python 3](https://www.continuum.io/downloads).  Most likely you want 64-bit, unless your computer is ancient.
# 
# In METR 4323, we will use python via a [jupyter notebook](http://jupyter.org). (You are looking at such a notebook now)
# 
# At that site, you are welcome to click "Try it in your browser", and then "Welcome_to_Python.ipynb".
# 
# Here is a  [video introduction to jupyter notebooks](https://youtu.be/e9cSF3eVQv0).  You just need
# to type `jupyter notebook` on the command line of your Windows|OSX|Linux computer, and the jupyter notebook opens in *your browser*.  That's right: the app that you use for web surfing will be your development environment for python.
# 
# Note the terminology [jupyter versus ipython](https://ipython.org/notebook.html).  Here we learn that "jupyter" is the [language-agnostic part](https://ipython.org/) of the notebook, which can be used for several languages.
# So, more accurately, we may say METR 4323 uses [ipython](http://ipython.org/) as the kernel in jupyter.

# ## Jupyter/Ipython warm up
# 
# Here is a notebook on the level of my "freshman" Python course: 
# [ScrabbleWords.ipynb](https://anaconda.org/bfiedler/scrabblewords).  
# **Click on "Download" and save it to your computer.**
# You may want to save it in a directory/folder named `notebooks`, or something like that. 
# 
# 
# Then, from your [Jupyter dashboard](https://youtu.be/e9cSF3eVQv0?t=5m0), open `ScrabbleWords.ipynb`.
# 
# I make no claim that `ScrabbleWords.ipynb` is the ideal place to start for you.  
# You may want to investigate [A gallery of interesting IPython Notebooks](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks).
# 
# You may also find my [matplotlib1.ipynb](https://anaconda.org/bfiedler/matplotlib1/notebook) useful.
# 
# Students in a previous semester enjoyed my [N005_CollatzConjecture](https://anaconda.org/bfiedler/n005_collatzconjecture)
# 

# ## The course notebooks

# No summaries are provided here (yet). Just click on a link to get the html preview at anaconda.org. The html rendering may take a few seconds, perhaps because of the LaTeX equations.
# 
# **If you want to download and run the notebook, be sure to click the Download tab.  The Download tab is just above and to the right of the html preview.**
# 
# 

# 
#  * [N005_CollatzConjecture](https://anaconda.org/bfiedler/n005_collatzconjecture/notebook)
#  * [N010_JuliaSetFractals.ipynb](https://anaconda.org/bfiedler/n010_juliasetfractals)                           
#  * [N020_ConwayGameOfLife.ipynb](https://anaconda.org/bfiedler/n020_conwaygameoflife)                 
#  * [N030_CoupledODE.ipynb](https://anaconda.org/bfiedler/n030_coupledode)                                
#  * [N040_DiffusionPDE1D.ipynb](https://anaconda.org/bfiedler/n040_diffusionpde1d)                             
#  * [N050_AdvectionPDE1D.ipynb](https://anaconda.org/bfiedler/n050_advectionpde1d)                             
#  * [N060_SympySchemes.ipynb](https://anaconda.org/bfiedler/n060_sympyschemes)                              
#  * [N070_ShallowWater1D.ipynb](https://anaconda.org/bfiedler/n070_shallowwater1d)
#  * [N080_ShallowWater2D.ipynb](https://anaconda.org/bfiedler/n080_shallowwater2d)
#  * [N090_StreamfunctionVorticity2D.ipynb](https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d)
#  * [N100_PressureSolver2D.ipynb](https://anaconda.org/bfiedler/n100_pressuresolver2d)
#  * [N110_Hydrostatic_vs_Nonhydrostatic2D.ipynb](https://anaconda.org/bfiedler/n110_hydrostatic_vs_nonhydrostatic2d)
#  * [N120_BaroclinicInstability3D.ipynb](https://anaconda.org/bfiedler/n120_baroclinicinstability3d)
#  * [N130_PetTornado3D.ipynb](https://anaconda.org/bfiedler/n130_pettornado3d)
#  * [N140_Convection3D](https://anaconda.org/bfiedler/n140_convection3d) Though not formally a part of METR 4323, this notebook shows how to do publishable fluid-dynamics research with an ipython notebook.
# 

# In[2]:


import urllib.request
from IPython.core.display import HTML
HTML(urllib.request.urlopen('http://metrprof.xyz/metr4323.css').read().decode())
#HTML( open('metr4323.css').read() ) #or use this, if you have downloaded metr4233.css to your computer

