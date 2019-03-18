
# coding: utf-8

# In[1]:


import glob
import netCDF4
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap 
from IPython.core.display import HTML
HTML( open('my_css.css').read() ) # if you don't have my_css.css, comment this line out


# In[2]:


# http://matplotlib.org/users/customizing.html
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


# # Elementary time-series  analysis of the PDSI
# 
# v1.1, 27 August 2016, by Brian Fiedler
# 
# We will explore some time-series data of the Palmer Drought Severity Index (PDSI).  Our analysis is not great:
# we will just use numpy to examine how drought is correlated around the globe (correlation) and how it is correlated with itself in time (autocorrelation). 
# 
# Learn about the PDSI:
# 
#  * The PDSI is also referred to as just the [PDI, or Palmer drought index](https://en.wikipedia.org/wiki/Palmer_drought_index)
# 
#  * [North American Drought: a Paleo Perspective](http://www.ncdc.noaa.gov/paleo/drought/drght_history.html)
# 
#  * [Paths to more data](https://climatedataguide.ucar.edu/climate-data/palmer-drought-severity-index-pdsi)
# 
# You will need this file: [pdsi.mon.mean.nc](https://anaconda.org/bfiedler/data/1/download/pdsi.mon.mean.nc). It is also available [from ESRL](http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/dai_pdsi/catalog.html?dataset=Datasets/dai_pdsi/pdsi.mon.mean.nc).  I cannot vouch whether this dataset is the best available. It does provide some exercise with time-series analysis.
# 
# We also enjoy some plotting and analysis of time-series of a single variables, in very simple short netCDF files. 
# 
# For that, you will need these files too: [knmi.zip](https://anaconda.org/bfiedler/data/1/download/knmi.zip)
# 
# **Your student [Student Tasks](#Student-Tasks) are below.**

# # Some helpful functions 

# In[3]:


def ncsummary(ncf): # prints a summary of the contents of netCDF file
    for item in ncf.ncattrs():
        print(item,': ',ncf.getncattr(item))

    print() #blank line
    for  key in list(ncf.dimensions.keys()):
        print(ncf.dimensions[key])
    
    for  key in list(ncf.variables.keys()):
        print(ncf.variables[key])


# In[4]:


def time2years(tvals,nc_time_units):
    # nc_time_units could be like so: "months since 1958-01-01"
    ts=nc_time_units.split() # splits at white space, into a list
    startdate=ts[2] # ts[-1] could be 1958-01-01
    year,month,day=startdate.split('-') # could be strings: 1958, 01, 01
    if ts[0]=='months':
        fs=12.
    elif ts[0]=='hours':
        fs=8766.
    elif ts[0]=='days':
        fs=365.25
    elif ts[0]=='years':
        fs=1.
    else:
        sys.exit(ts[0]+' not allowed')
    return tvals/fs + float(year) # always plot time axis in years


# In[5]:


def time_series_plot(times,vals,title="",xlab="",ylab="",comment="",negc='blue',posc='red'):
    fig = plt.figure(figsize=(16,4),facecolor='white')  
    ax1 = fig.add_axes([0.05,0.2,0.9,.7])  
    ax1.fill_between(times,vals,y2=0,where=vals>0,color=posc) # filled plot
    ax1.fill_between(times,vals,y2=0,where=vals<0,color=negc) # filled plot
    ax1.grid() # show grid lines
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.set_xlim((times[0],times[-1]))
    dmin=vals.min()
    dmax=vals.max()
    if dmin>=0.:
        if dmin>.5*dmax:
            ax1.set_ylim(ymin=.98*dmin)
        else:
            ax1.set_ylim(ymin=0.0)
    ax1.set_title(title)
    ax1.text(0.01,-0.2,comment,fontsize=10,transform=ax1.transAxes)  
    plt.close(fig)
    return fig


# In[6]:


def spectral_density_plot(times, vals, nfft=2048, title="",
                          unit="year", ylab="", comment=""):
    fig = plt.figure(figsize=(16,4),facecolor='white')  
    ax1 = fig.add_axes([0.05,0.2,0.9,.7]) 
    timesx = times[vals.mask == False] # take only unmasked values, may be unnecessary
    valsx = vals[vals.mask == False] # take only unmasked values, may be unnecessary
    dta = timesx[1:]-timesx[:-1]
    dtamin = dta.min()
    dtamax = dta.max()
    if not dtamin > .99*dtamax:
        print("trouble ahead, dt not uniform:", dtamin, dtamax)
    pf = np.polyfit(timesx,valsx,1) # best linear fit: data = pf[0]*tval+pf[1]
    valsd = valsx - pf[0]*timesx - pf[1] # detrended values
    ax1.psd(valsd, NFFT=nfft, Fs=1/dtamax) # dtamin should also work here
    ax1.set_ylabel(ylab)
    ax1.set_xlabel("frequency (cycles/"+unit+")")
    ax1.set_title( title+', Power Spectral Density from detrended data')
    ax1.text(0.01,-0.2,comment,fontsize=10,transform=ax1.transAxes)


# # Choosing and plotting a KNMI file
# 
# You will need to unzip [knmi.zip](https://anaconda.org/bfiedler/data/1/download/knmi.zip), which contains small files of a time series of a single quantity.  There is a task for `knmi/ilabrijn.nc` , defined below with the other tasks.

# In[7]:


files = glob.glob("knmi/*.nc")
for filename in files:
    ncf = Dataset(filename,'r')   
    print(filename,ncf.comment,"\n")


# In[8]:


#Select one:
inputfile = "knmi/imaunaloa_f.nc"
#inputfile = "knmi/isunspots.nc"
#inputfile = "knmi/ihadisst1_nino3.4a.nc"
#inputfile = "knmi/ilabrijn.nc"
ncf = Dataset(inputfile,'r') 
print(ncf)


# In[9]:


ncsummary(ncf) # my own summarizer function


# In[10]:


#### get time array and data array:
if inputfile == "knmi/imaunaloa_f.nc":
    varname = 'co2'
if inputfile == "knmi/isunspots.nc":
    varname = 'sunspot'
if inputfile == "knmi/ihadisst1_nino3.4a.nc":
    varname = 'SST'
if inputfile == "knmi/ilabrijn.nc":
    varname = "tair"
tvals = ncf.variables['time'][:]
tunits = ncf.variables['time'].units
data = ncf.variables[varname][:] # data, to plot vs. time
title = ""
if 'title' in ncf.ncattrs(): title += ncf.getncattr('title')
try:
    long_name=ncf.variables[varname].long_name
except:
    long_name=varname
title += ", "+long_name
print(title)
units=ncf.variables[varname].units
print(units)


# In[11]:


# Students: task modifications here
if inputfile == "knmi/ilabrijn.nc":
    pass


# In[12]:


plt.plot(tvals,data); # quick look


# In[13]:


print(tunits)


# In[14]:


tvalys = time2years(tvals,tunits)


# In[15]:


plt.plot(tvalys,data);


# In[16]:


tsplot=time_series_plot(tvalys,data,title=title,ylab=units,xlab='year')
tsplot


# In[17]:


fname = title.replace(' ','')
fname = fname.replace(',','')
outname = fname+'.png'
print("will save to:",outname)
tsplot.savefig(outname)


# In[18]:


sum(data.mask) # spectral_density_plot will deal with this


# In[19]:


spectral_density_plot(tvalys,data,title=title)


# # Plot the PDSI data
# 
# 

# In[20]:


proj = 'cyl' # choose a map projection, lcc or cyl

## Pick a point for particular study
# Norman, Oklahoma:
pointlat = 35.329
pointlon = -97.274
where = "Norman OK"
# Toowoomba, Queensland, Australia:


# In[21]:


ncf = Dataset('pdsi.mon.mean.nc')
print(ncf)


# In[22]:


ncsummary(ncf) # use my own summarizer function


# In[23]:


pdsi = ncf.variables['pdsi'][:]
print( pdsi.min(), pdsi.max() )


# In[24]:


lons = ncf.variables['lon'][:]
print(lons)


# In[25]:


lats = ncf.variables['lat'][:]
print(lats)


# In[26]:


tvals = ncf.variables['time'][:]
tunits = ncf.variables['time'].units
print(tunits,tvals)


# In[27]:


#http://stackoverflow.com/questions/37895666/how-to-calculate-hour-to-day-in-netcdf-file-using-scala
dates = netCDF4.num2date(tvals,tunits)
dates


# For plotting, I make `tvalys` as time in years, as a floating point number

# In[28]:


tvalys = time2years(tvals,tunits) # I needs years as floating number for plotting


# In[29]:


tvalys[0],tvalys[-1]


# For plotting with either `pcolor` or `pcolormesh`, we need the coordinates of the 4 points, in longitude and latitude, that surround a data value.
# So we **extend** `lons` and `lats` to `lonsx` and `latsx`.

# In[30]:


lonsx = np.zeros( lons.shape[0]+1 )
lonsx[1:-1] = (lons[1:]+lons[:-1])*0.5
lonsx[0] = lonsx[1] -(lonsx[2]-lonsx[1] )
lonsx[-1] = lonsx[-2] + (lonsx[-2]-lonsx[-3])
lonsx


# In[31]:


latsx = np.zeros(lats.shape[0]+1)
latsx[1:-1] = (lats[1:]+lats[:-1])*0.5
latsx[0] = latsx[1] -(latsx[2]-latsx[1] )
latsx[-1] = latsx[-2] + (latsx[-2]-latsx[-3])
latsx


# In[32]:


lonsxa,latsxa = np.meshgrid(lonsx,latsx) 
print(lonsxa.shape, latsxa.shape, pdsi.shape)


# We also need "conventional" arrays of latitude and longitude at the center of the pixels.

# In[33]:


lonsa,latsa = np.meshgrid(lons,lats) # makes 2-D arrays of same shape as pdsi 
print(lonsa.shape, latsa.shape, pdsi.shape)


# In[34]:


# the d2 distance^2 metric is only accurate for small distances, 
# which is okay because we seeking where the minimum im d2 is
d2 = (latsa - pointlat)**2 + np.cos(latsa*np.pi/180.)**2*(lonsa - pointlon)**2
print("where the min of d2 is in the flattened array:",np.argmin(d2))
pointlati, pointloni = np.unravel_index( np.argmin(d2), d2.shape ) # need to convert the flattened index

print("we see to study longitude", pointlon,", which is closest to longitude index", pointloni,
      ", which is at ",lonsa[pointlati,pointloni])
print("we see to study latitude", pointlat,", which is closest to latitude index", pointlati,
      ", which is at ",latsa[pointlati,pointloni])


# Let's inspect one value of `pdsi`. If you have set `pointlat` and `pointlat` for Norman OK, we find that August 1956 was a time of severe drought.

# In[35]:


tindex = 1039
tvalys[tindex]


# In[36]:


dates[tindex]


# In[37]:


dates[tindex].strftime('%Y-%m-%d')


# In[38]:


pdsi[tindex,pointlati,pointloni]


# In[39]:


startlon=-180. # for this dataset, longitudes start at -180

if proj=='lcc': #Lambert Conformal
    m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.,urcrnrlon=-2.566,urcrnrlat=46.352,            resolution='l',area_thresh=1000.,projection='lcc',            lat_1=50.,lon_0=-107.)
    parallels = np.arange(0.,80,20.)
    meridians = np.arange(10.,360.,30.)
else: #cylindrical is default
    m = Basemap(llcrnrlon=startlon,llcrnrlat=-90,urcrnrlon=startlon+360.,urcrnrlat=90.,            resolution='c',area_thresh=10000.,projection='cyl')
    parallels = np.arange(-90.,90.,30.)
    if startlon==-180:
        meridians = np.arange(-180.,180.,60.)
    else:
        meridians = np.arange(0.,360.,60.)
        
X,Y = m(lonsxa,latsxa) # Very important! For pcolormesh 


# In[40]:


colormap = plt.cm.jet_r # _r means reverse, red will be negative, for drought
colorbounds=[-10,10] # numbers mapped to extremes of color map
fig = plt.figure(figsize=(15,10))
ax = fig.add_axes( [0.08,0.1,0.7,0.7], axisbg='white' )

p = m.pcolormesh(X, Y, pdsi[tindex], cmap=colormap) 
when = dates[tindex].strftime('%Y-%m-%d')
ax.set_title("PDSI  "+when)

m.drawcoastlines()
m.drawstates()

im = plt.gci()
im.set_clim( vmin=colorbounds[0], vmax=colorbounds[1] )
cax = fig.add_axes( [0.85, 0.1, 0.05, 0.7] ) # setup colorbar axes
plt.colorbar(format='%d', cax=cax) # draw colorbar

plt.axes(ax)  # make the original axes current again

xpt,ypt=m(pointlon,pointlat) # make a pink dot here
m.plot([xpt],[ypt],'o',color='pink',markersize=5,zorder=2) 

m.drawparallels(parallels,labels=[1,1,0,1])
m.drawmeridians(meridians,labels=[1,1,0,1]);


# In[41]:


pointpdsi = pdsi[:,pointlati,pointloni].copy()


# In[42]:


pointpdsi.max()


# In[43]:


tvalys.shape


# In[44]:


title="PDSI for "+where
units='pdsi'
tsp = time_series_plot(tvalys, pointpdsi, title=title, ylab=units,xlab='year',negc='red',posc='blue')
tsp


# In[45]:


whr = where.replace(' ','')
whr = whr.replace(',','')
outname = "PDSI_"+whr+'.png'
print("will save to:",outname)


# In[46]:


tsp.savefig(outname)


# # Correlation of PDSI with the chosen point
# 
# We make a time series with a time-step of one year: all the months of March.
# There are certainly other possibilities, such as examining other months. But there are no tasks assigned here for changing the month.
# 

# In[47]:


dates[2] # 2 is the 3rd month, March


# In[48]:


pdsi_March = pdsi[2::12] # 2 is the first March, then skip every 12 months


# In[49]:


# Make a short variable name
v = pdsi_March.copy()
vmean = v.mean(0) #mean along first axis, time
v = v - vmean 


# In[50]:


vp = v[:,pointlati,pointloni] # time series at the chosen point


# In[51]:


vp.shape # 136 values for March


# In[52]:


v.shape


# `.T` means *transpose*, which puts the time index last:

# In[53]:


v.T.shape


# `v.T*vp` is succesful below. Every March pdsi at a point is multiplied by the value at Norman (or other chosen point) at that time.

# In[54]:


vvp = (v.T*vp).T # Tranpose back, to put time index first again


# In[55]:


vcorr = vvp.mean(0)/np.sqrt( (v*v).mean(0) * (vp*vp).mean(0) ) # correlation coefficient


# In[56]:


#Plot the correlation coefficient
colormap = plt.cm.jet
colorbounds=[-1,1]
fig = plt.figure(figsize=(15,10))

ax = fig.add_axes([0.08,0.1,0.7,0.7],axisbg='white')

p = m.pcolormesh(X, Y, vcorr, cmap=colormap) # colorize area around data point
m.drawcoastlines()
m.drawstates()

im = plt.gci() # these two statement map extreme colors out chosen numbers
im.set_clim(vmin=colorbounds[0],vmax=colorbounds[1]) 

cax = fig.add_axes([0.85, 0.1, 0.05, 0.7]) # setup colorbar axes

plt.colorbar(format='%4.1f', cax=cax) # draw colorbar

plt.axes(ax)  # make the original axes current again

xpt,ypt=m(pointlon,pointlat)
m.plot([xpt],[ypt],'o',color='pink',markersize=5,zorder=2) 

ax.set_title("March PDSI correlation with "+where)
m.drawparallels(parallels,labels=[1,1,0,1])
m.drawmeridians(meridians,labels=[1,1,0,1]);


# In[57]:


outname = "corr"+whr+".png"
print(outname)
fig.savefig(outname)


# <hr style="height:6px;border:none;background-color:#f00;" />
# 
# # Student Tasks
#  * The plots for `knmi/ilabrijn.nc` are a bit crowded until we filter the data a bit. Plot only the January temperatures.
#  
#  <img src="http://metr4330.net/img/TdebiltJanuaryX.png">
#  
#  * Plot and post the "March PDSI correlation with Toowoomba"
#  * Complete the Autocorrelation task below:

# ## Autocorrelation
# 
# [Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).
# 
# This is what you are shooting for:
# 
# <img src="http://metr4330.net/img/corrauto12monthX.png">
# 
# Produce and post a 3 month, 12 month and 36 month autocorrelation plot.

# In[58]:


lag = 12 # number of months to lag by


# In[59]:


# I will give you a start:
v1 = pdsi[lag:]
v2 = pdsi[:-lag]
v1 = v1 - v1.mean(0)
v2 = v2 - v2.mean(0)


# In[60]:


outname = "corrauto%dmonth.png" % lag
print(outname)
figa.savefig(outname)

