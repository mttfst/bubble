# %% codecell
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")
# %% markdown
# ## Definitions
#
# |var name|value<br>exp1|value<br>exp2|unit|definition|
# |---|---|---|---|---|
# |posrlon|2.54517|same|[°E]||
# |posrlat|2.54879|same|[°N]|[coordinats](https://xkcd.com/2170/) of the emission point<br>(here in rotated coordinats of the forecast model)|
# |dt|30|same|[s]|model time step|
# |scale|2.e-7|same|-|scaling between measured particles and model particles<br>scale=2.e-7 1 model particle = 5 million measured particles|
# |emission time|10:50-11:45|13:40-14:30|-|time with particle emission|
# |ntmsp|110|100|-| number of time steps with particle emission|
# |itime|20.85|23.65|[h]| start time of the emission in hours after model start|
# |velox|0.|same|[ms-1]|the velocity of moving source in x direction|
# |veloy|0.|same|[ms-1]|the velocity of moving source in y direction|
# |expname|exp1|exp2|-|name of experiment|
# |plumeheight|5.|same|[m]|height of the dust plume|
# |plumeradius|5.|same|[m]|radius of the dust plume|
# |ngridpoints|100|same|-|number of horizontal gridpoints of the plume|
#
#
#
# %% codecell
posrlon=2.54517
posrlat=2.54879

dt=30

scale=2.e-7
border=100000


velox=0.
veloy=0.

expname = 'exp2'
ofilename=expname+'_'+str(scale)+'.out'

# ntmsp
if expname == 'exp1':
    ntmsp = 110
if expname == 'exp2':
    ntmsp = 100

# itime
if expname == 'exp1':
    itime=20.85
if expname == 'exp2':
    itime=23.65


plumeheight=5.
plumeradius=5.

ngridpoints=100

# filenames
fname1 = 'GRIMM_1.5m_diff_'+expname+'.data'
fname2 = 'GRIMM_3.8m_diff_'+expname+'.data'
# %% markdown
# ## Pre-calculations
# %% codecell
def meter2degree(meter):
    degree=np.zeros(len(meter))
    for i in range(len(meter)):
        degree[i] = meter[i]/6.3710088e6 * 180./np.pi
    return degree


dx=2.*plumeradius/ngridpoints # spacing of the gridboxes
dz=dx

# I divide the area form -plumeradius to +plumeradius in x and y direction
# and from 0 to +plumeheight in z direction in grid boxes with the size dx**3
ngridbox=int((ngridpoints+1)**2*(plumeheight/dz+1))

# list of the bubble midpoint position
rlons=[posrlon+meter2degree([i*velox*dt])[0] for i in range(ntmsp)]
rlats=[posrlat+meter2degree([i*veloy*dt])[0] for i in range(ntmsp)]
# %% markdown
# ## Input
# ### Produce random data for testing
# The EDM data is not included in this repository but you can test the software with a randomized input.
# %% codecell
# 2 data arrays with a len of 31

dat1 = np.zeros(31)
dat2 = np.zeros(31)

npmax = 25000
for i in range(len(dat1)):
    dat1[i]=np.random.random_integers(0,npmax)
    dat2[i]=np.random.random_integers(0,npmax)
    npmax -= npmax/5.


# %% markdown
# ### Read in data from Environmental Dust Monitor (EDM)
# %% codecell
# open input files
f1 = open(fname1,encoding = 'unicode_escape')
f2 = open(fname2,encoding = 'unicode_escape')

# read header
head=f1.readline()
head=f1.readline()
head=f2.readline()
head=f2.readline()

# read bin names form the header
binnames = []
for i in head.split():
    if(int(i[0].isdigit())):
        binnames.append(i)
binnames.append('>32.0')

# read the data to np.array
data1 = np.loadtxt(f1,usecols=range(2,33))
data2 = np.loadtxt(f2,usecols=range(2,33))

#dimension of the data set
dim_data=data1.shape


# definition of the peak in the data & removal of the background noise

# exp1
if expname == 'exp1':
    peak1 = data1[454:464,:]-np.median(data1[454-50:464+50,:],axis=0)
    peak2 = data2[458:465,:]-np.median(data2[458-50:465+50,:],axis=0)

# exp2
if expname == 'exp2':
    peak1 = data1[179:183,:]-np.median(data1[179-50:183+50,:],axis=0)
    peak2 = data2[177:181,:]-np.median(data2[177-50:181+50,:],axis=0)


peak1=np.where(peak1 > 0.,peak1,0.)
peak2=np.where(peak2 > 0.,peak2,0.)

dat1 = [np.sum(peak1[:,i])/(len(peak1)*6.) * dt for i in range(peak1.shape[1])]
dat2 = [np.sum(peak2[:,i])/(len(peak2)*6.) * dt for i in range(peak2.shape[1])]
# %% markdown
# ## Create the Bubble
#
# In this section, I use the EDM data to create a vertical profile of the particle concentration.
# From there I project the profile to a half-sphere to get a concentration field.
# Then I calculate how many particles there should be in every grid box of the sphere.
# Afterward, the particles are distributed randomly in the bubble. The probability that a particle appears at a certain position depends on the concentration field.
#
#
# %% codecell
# this lists store the data for the plotting later
xp_plt=[]
yp_plt=[]
contour_plt=[]
points_plt=[]
nrmax_sv=[]
ppgb_sv=[]
nrsum=0 # sum of all particles in all bins


# loop over all bins
for ibin in range(len(dat1)):   # range(11,23) -> particles between 1 and 10 µm
#    print(ibin)
#    print(binnames[ibin])

    # negative particle concentrations may occur in some bins because of the background noise removal
    # filter them out
    if dat1[ibin] <= 0:
        dat1[ibin] = 0.

    if dat2[ibin] <= 0:
        dat2[ibin] = 0.


    # define the data points we are using: surface:dat1, 1.5m:dat1, 3,8m:dat2, top of plume:0
    # it is defined as x points because we need a funktion y(x) for the fitting
    # it is flipped later
    xpoints=[dat1[ibin],dat1[ibin],dat2[ibin],0]
    ypoints=[0.,1.5,3.8,plumeheight]



    # fitting with 5th order Poly (5th order seems to work well)
    x_p = np.linspace(0,plumeheight,int(plumeheight/dz)+1)
    fit = np.polyfit(ypoints,xpoints,5)
    y_p = np.polyval(fit,x_p)

    # filter out values <1 below the altitude with the peak value
    maxpos=np.where(y_p == np.max(y_p))[0][0]
    for i in range(maxpos):
        if y_p[i] < 1.:
            y_p[i] = 1.



    # save data for plotting
    xp_plt.append(x_p)
    yp_plt.append(y_p)

    # the profile is now defined as funktion y_p(x_p).
    # to make it a vertikal profile we flip it x_p(y_p)

    # not every profile reaches the possible plumeheight
    # the next few lines search for the height where the profile becomes < 1
    # this height is used later at the ellipsoid
    zid=0
    while y_p[zid] < 1.:
        zid+=1
    hmin=zid
    while y_p[zid] >= 1.:
        zid+=1
        if zid == int(plumeheight/dz):
            break
    height=x_p[zid]

    # the vertikal profile defines the particle conzentration
    # in the center of the dust plume as a function of higth

    # now we use the profile to create a 3D conzentration field in the shape of a half sphere
    # in the standart setting plumeradius=5. and plumehight=5.
    # the conzentration field is defined in a grid from -5 to 5 meter horizontal and 0 to 5 meter vertical.
    # with ngridpoints=100 each gridbox has a size of 0.1**3 meter
    xg = np.linspace(-plumeradius, plumeradius,ngridpoints+1)
    yg = np.linspace(-plumeradius, plumeradius,ngridpoints+1)
    zg = np.linspace( 0.0, plumeheight,int(plumeheight/dz)+1)


    # get coordinates of the gridboxes with np.meshgrid
    X, Y, Z = np.meshgrid(xg, yg,zg)

    # I define the concentration field as an ellipsoid
    # (x**2/a**2 + y**2/b**2 + z**2/c**2) = 1
    # in this case, a and b is the plumeradius and c is the height
    # This produces a normalized spherical concentration field with 1 in the centre and 0 at the edge

    contour = 1 - (X**2/plumeradius**2  + Y**2/plumeradius**2  + Z**2/height**2 )

    # now we need to multiply the normalized concentration field with a vector
    # that returns the vertical profile at the midpoint of the concentration field
    # we search for the multiply value on every altitude level
    # multiply(level) = profiel value (level) / conture value (level)

    # multiply vector
    multi=np.zeros(y_p.shape)

    # the midpoint of the bubble
    mid=int(ngridpoints/2)

    # loop through level
    for k in range(multi.shape[0]):
        if contour[mid,mid,k] > 0.:
            multi[k]=y_p[k]/contour[mid,mid,k]
        else:
            multi[k] = 0.

    # multiply the concentration field with the vector
    contour *= multi


    # if contour > 0. then contour == contour eles contour == 0.
    contour = np.where(contour > 0., contour,0.)

    # remove possible artificials below the lower edge of the bubble
    contour[:,:,:hmin]=0.0

    # save this contour field for plotting
    contour_plt.append(contour)

    # ppgb - parts per grid box
    # The conzentration [parts per liter] in transformed into parts per gridbox.
    # size gridbox m**3 * part/liter * 1000 liter/m**3

    ppgb = dx**3 * contour*1000
    ppgb = ma.masked_where(ppgb < 0.,ppgb)

    # maximum number of particles in this bin. here the scaling took place
    # (see table at the begin of this notebook)
    # summ of particles * scale * number of emission time steps
    nrmax=int(round(np.sum(ppgb)*scale)*ntmsp )

    # save nrmax of this bin, this is needed later for the output
    nrmax_sv.append(nrmax)
    ppgb_sv.append(ppgb)

    # total sum of particles
    nrsum+=nrmax

    # print particle number per bin
    # print(nrmax)

    nrmax=nrmax_sv[ibin]
    ppgb=ppgb_sv[ibin]
    # 3d array that holds the coordinates of the paticles
    points=np.zeros((nrmax,3))


    # flatten the ppgb array and normalize it with the sum of particles
    props=np.reshape(ppgb,ngridbox)/np.sum(ppgb)
    npp=0
    for pp in props.mask:
        if pp:
            props[npp]=0.0
        npp+=1


    # create nrmax random numbers in the range of ngridbox with the probabilty props
    # So the single randmo number (ran in randoms) points to a specific cell in the flatten array
    randoms = np.random.choice(ngridbox, nrmax,p=props)

    nr=0
    for ran in randoms:
        # get the indices of the 3d array from the cell number of the flattened array
        idx = np.unravel_index(np.ravel_multi_index((ran,), props.shape), ppgb.shape)

        # get the position of the point in cartesian coordinates
        # then shift the particles randomly away from the middle point of the grid box
        points[nr,0]=X[idx] + np.random.random()*dx-dx/2.
        points[nr,1]=Y[idx] + np.random.random()*dx-dx/2.
        points[nr,2]=Z[idx] + np.random.random()*dx-dx/2.

        # if the point are shifted underneath the surface, than get them back.
        if points[nr,2] < 0.:
            #print('warnig',points[nr,2])
            points[nr,2]*=-1.


        if points[nr,2] > 5.:
            print('warnig',points[nr,2])
            print(Z[idx],dx-dx/2.)


        nr+=1

    points_plt.append(points)


print('total particles: ' + str(nrsum))
# %% markdown
# #### particle numbers per bin
#
# |   bin   |exp1  |exp2  |
# |---------|------|------|
# |0.25-0.28|22440 |59500 |
# |0.28-0.30|24310 |34000 |
# |0.30-0.35|16390 |30000 |
# |0.35-0.40|7040  |29400 |
# |0.40-0.45|4730  |19500 |
# |0.45-0.50|3520  |7500  |
# |0.50-0.58|3410  |8800  |
# |0.58-0.65|2310  |7500  |
# |0.65-0.70|1100  |6600  |
# |0.70-0.80|3520  |12200 |
# |0.80-1.00|1320  |6600  |
# |1.00-1.30|1320  |6700  |
# |1.30-1.60|880   |3300  |
# |1.60-2.00|1100  |6500  |
# |2.00-2.50|1650  |13400 |
# |2.50-3.00|1430  |9700  |
# |3.00-3.50|440   |3000  |
# |3.50-4.00|440   |2100  |
# |4.00-5.00|880   |4200  |
# |5.00-6.50|660   |2500  |
# |6.50-7.50|440   |1100  |
# |7.50-8.50|220   |400   |
# |8.50-10.0|220   |700   |
# |10.0-12.5|220   |400   |
# |12.5-15.0|110   |300   |
# |15.0-17.5|110   |300   |
# |17.5-20.0|110   |200   |
# |20.0-25.0|110   |200   |
# |25.0-30.0|110   |200   |
# |30.0-32.0|110   |100   |
# |>32.0    |550   |300   |
# |total    |101200|277200|
#
# %% markdown
# ## Output
#
# The output file of this notebook is the start file of the Lagrangian particle dispersion model Itpas.
# The above-defined points are transformed into the rotated coordinates of the model.
#
# %% codecell
def numspace(num):
    nspace=0      # number of spaces between colums,
                  # 3 for numbers < -10,
                  # 4 for numbers < 0,
                  # 5 for numbers >0,
                  # 4 for numbers >10
    if num <= -10.:
        nspace=3
        ndigs=8+4
    elif num < 0.:
        nspace=4
        ndigs=8+3
    elif num >= 0.:
        nspace=5
        ndigs=8+2
        if num >= 10.:
            nspace=4
            ndigs=8+3
    numlen=len(str(num))
    dot=str(num).find('.')
    afterdot=numlen-dot
    nfill=0
    if afterdot < 9:
        nfill=9-afterdot
    return nspace,ndigs,nfill



# open new file
f=open(ofilename,'w')

# write header
f.write(ofilename+' \n')
hline='time'
hline+=5*' '+'lon'
hline+=12*' '+'lat'
hline+=12*' '+'height'
hline+=7*' '+'npart'
hline+=3*' '+'diam'
hline+=14*' '+'dens'
hline+=7*' '+'emission'
hline+='\n'
f.write(hline)
f.write(50*'-'+'\n')


# loop over all bins
for ibin in range(len(dat1)):

    # get points of the bin
    points=points_plt[ibin]

    # get nrmax of the bin
    nrmax=nrmax_sv[ibin]

    # transform relative particle possitions to rotated coordinates
    nrmax=int(nrmax/ntmsp)
    for rlon,rlat,nr in zip(rlons,rlats,range(len(rlons))):
        points[nr*nrmax:(nr+1)*nrmax,0]=rlon + meter2degree(points[nr*nrmax:(nr+1)*nrmax,0])
        points[nr*nrmax:(nr+1)*nrmax,1]=rlat + meter2degree(points[nr*nrmax:(nr+1)*nrmax,1])


    # define min and max particle diameter
    pos=binnames[ibin].find('-')
    if pos > 0:
        Dmin=float(binnames[ibin][:pos])
        Dmax=float(binnames[ibin][pos+1:])
    else:
        Dmin=float(binnames[ibin][1:])
        Dmax=50.0


    # write points to ofile
    npoint=0
    time=0.0
    initime=itime
    for point in points:
        npoint+=1
        # create the output row
        # time
        if npoint> nrmax:
            time+=0.0083
            npoint=1

        stime='{:8.4f}'.format(time+initime)
        row=stime
        # lon
        nspace,ndigs,nfill=numspace(point[0])
        row+=nspace*' '+str(point[0])[:ndigs]+nfill*'0'
        #lat
        nspace,ndigs,nfill=numspace(point[1])
        row+=nspace*' '+str(point[1])[:ndigs]+nfill*'0'
        # height
        nspace,ndigs,nfill=numspace(point[2])
        row+=nspace*' '+str(point[2])[:ndigs]+nfill*'0'
        # npart
        npart=1
        nspace,ndigs,nfill=numspace(npart)
        row+=nspace*' '+str(npart)
        # Diam, chosen randomly in the size range of bin
        diam=(np.random.random()*(Dmax-Dmin)+Dmin)*1e-6
        nspace,ndigs,nfill=numspace(diam)
        row+=nspace*' '+str(diam)[:ndigs]+str(diam)[-4:]
        # Dens
        dens=2650.0 #[kg/m**3]
        nspace,ndigs,nfill=numspace(dens)
        row+=nspace*' '+str(dens)
        # emission
        emission='0.00'
        nspace,ndigs,nfill=numspace(float(emission))
        row+=nspace*' '+emission
        row+='\n'
        f.write(row)

# close file
f.close()

# %% markdown
# ## Plotting
#
# ### Vertical profile
# %% codecell
# Open figure
fig = plt.figure(figsize=(22,12))

# define grid of plots
gs = gridspec.GridSpec(nrows=4, ncols=8)

# x,y specify the subplot
x=0
y=0

#loop over bins
for ibin in range(len(dat1)):

    #open subplots
    ax = fig.add_subplot(gs[y, x])

    # set bin name as title
    ax.set_title(binnames[ibin]+' $\mu m$')

    # draw grid
    ax.grid()

    # get profile data
    xpoints=[dat1[ibin],dat1[ibin],dat2[ibin],0]
    ypoints=[0.,1.5,3.8,plumeheight]

    x_p=xp_plt[ibin]
    y_p=yp_plt[ibin]

    # plot profile
    ax.plot(xpoints,ypoints)
    ax.plot(y_p,x_p)
    ax.set_xlim(0)
    ax.set_ylim(0)


    # increas x,y
    x+=1
    if x > 7:
        x = 0
        y+=1

fig.tight_layout()
fig.savefig(ofilename[:-4]+'_profiles.png')
# %% markdown
#  ### Cross Section through the concentration field
# %% codecell
# Open figure
fig = plt.figure(figsize=(22,12))

# define grid of plots
gs = gridspec.GridSpec(nrows=4, ncols=8)

# x,y specify the subplot
x=0
y=0

# loop over bins
for ibin in range(len(dat1)):

    #open subplots
    ax = fig.add_subplot(gs[y, x])

    # set bin name as title
    ax.set_title(binnames[ibin]+' $\mu m$')

    # draw grid
    ax.grid()

    # get stored data
    contour=contour_plt[ibin]

    # masked the areas where contour <= 0. so that they appear white in the plot
    contour = ma.masked_where(contour <= 0.,contour)

    # the midpoint of the bubble
    mid=int(ngridpoints/2-1)
    # plot cross-section through the bubble
    con = ax.contourf(X[0,:,:], Z[:,0,:], contour[:,mid,:],cmap='jet') #ppgb
    fig.colorbar(con)


    # increas x,y
    x+=1
    if x > 7:
        x = 0
        y+=1

fig.tight_layout()
fig.savefig(ofilename[:-4]+'_contour.png')
# %% markdown
# ### 3d scatter plot of the particles
# %% codecell
# Open figure
fig = plt.figure(figsize=(22,12))

# define grid of plots
gs = gridspec.GridSpec(nrows=4, ncols=8)

# x,y specify the subplot
x=0
y=0

#loop over bins
for ibin in range(len(dat1)):

    #open subplots
    ax = fig.add_subplot(gs[y, x],projection='3d')

    # set bin name as title
    ax.set_title(binnames[ibin]+' $\mu m$')

    # draw grid
    ax.grid()

    # get points
    points = points_plt[ibin]

    # 3d scatter plot
    ax.scatter(points[:,0],points[:,1],points[:,2],'b.',s=0.25)
    #ax.set_xticks([])
    #ax.set_yticks([])


    # increas x,y
    x+=1
    if x > 7:
        x = 0
        y+=1

fig.tight_layout()
fig.savefig(ofilename[:-4]+'_points.png')
# %% markdown
# ### panel plot: Profile, cross-section, scatter
# %% codecell
# Open figure
fig = plt.figure(figsize=(12,4))

# define grid of plots
gs = gridspec.GridSpec(nrows=1, ncols=3)

# plot only bin 13 (1.6 - 2. µm)
ibin = 10


# open sub plots
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2],projection='3d')

# ax0 profile
x_p=xp_plt[ibin]
y_p=yp_plt[ibin]

xpoints=[dat1[ibin],dat1[ibin],dat2[ibin],0]
ypoints=[0.,1.5,3.8,plumeheight]

# plot profile

ax0.plot(y_p,x_p)

ax0.scatter(xpoints,ypoints,s=100)


ax0.set_xlim(0)
ax0.set_ylim(0,plumeheight+0.1)
ax0.set_xlabel('particle per liter',size='x-large')
ax0.set_ylabel('altitude [m]',size='x-large')

ax0.annotate('Assumption #1',
            xy=(xpoints[0], ypoints[0]), xycoords='data',
            xytext=(-100, 25), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-.2"))

ax0.annotate('Assumption #2',
            xy=(xpoints[3], ypoints[3]), xycoords='data',
            xytext=(15, -35), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-.2"))

ax0.annotate('Measurements',
            xy=(xpoints[1], ypoints[1]), xycoords='data',
            xytext=(-80, 45), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))

ax0.annotate('',
            xy=(xpoints[2], ypoints[2]), xycoords='data',
            xytext=(-73, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-.3"))

# ax1
# get stored data
contour=contour_plt[ibin]



# masked the areas where contour <= 0. so that they appear white in the plot
contour = ma.masked_where(contour <= 0.,contour)

# the midpoint of the bubble
mid=int(ngridpoints/2)
# plot cross-section through the bubble
con = ax1.contourf(X[0,:,:], Z[:,0,:], contour[:,mid,:],cmap='jet') #ppgb
cb = fig.colorbar(con,ax=ax1)
cb.ax.tick_params(labelsize='x-large')
cb.set_label('particle per liter',size='x-large')

ax1.set_ylim(0,plumeheight+0.1)
ax1.set_ylabel('altitude [m]',size='x-large')
ax1.set_xlabel('width [m]',size='x-large')

# ax2
# get points
points = points_plt[ibin]

# 3d scatter plot
ax2.scatter(points[:,0],points[:,1],points[:,2],'b.',s=0.25)
#ax2.set_ylabel('width [m]',size='x-large')
#ax2.set_zlabel('altitude [m]',size='x-large')
#ax2.view_init(30, -65)

ax0.tick_params(labelsize='x-large')
ax1.tick_params(labelsize='x-large')
ax2.tick_params(labelsize='x-large')

ax0.set_title('a) Vertical profile',size='x-large')
ax1.set_title('b) Concentration field',size='x-large')
ax2.set_title('c) Particle positions',size='x-large')

fig.tight_layout()
plt.savefig('out.png',dpi=500)
# %% codecell
