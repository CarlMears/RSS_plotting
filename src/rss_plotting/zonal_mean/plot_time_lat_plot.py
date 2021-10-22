def plt_time_lat(*,z,year_range,xlabel='Year',ylabel='Latitude',title='',cmap = 'BrBG',vmin = -1.0,vmax = 1.0,fig_in = None,ax_in = None):

    if fig_in is None:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = fig_in 

    if ax_in = None:
        ax = fig.add_subplot(111)
    else:
        ax = ax_in

    num_months = z.shape[0]
    num_lats = z.shape[1]
 
    yrs = year_range[0] + np.arange(0,num_months)/12.0 + 1.0/24.0
    dlat = 90.0/num_lats
    lats = -90.0 + dlat/2.0 + dlat*np.arange(0,num_lats)

    X,Y = np.meshgrid(yrs,lats)
    im = ax.pcolormesh(X,Y, np.transpose(z), cmap=cmap, vmin  = vmin,vmax=vmax)

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel(title, fontsize=16)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig,ax
