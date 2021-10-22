# RSS_Plotting

This module a number of useful plotting routines, mostly for internal RSS use.  Package maturity varies ;-).
The intent is to have each main plotting routine be multipanel friendly -- see note after the list of routines

## binned_stats.py 
Contains BinnedStat, a class for accumulating data to calculate binned means and stddevs for "cross" talk plots

## global_map.py
Makes a global map of the values in "a".

Example Call:
	fig,ax = global_map(global_wind_speed,plt_colorbar=True,cmap='viridis',title='Wind Speed',units='Wind Speed (m/s)',panel_label='A')
    
*A similar, but less mature routine, __global_map_w_zonal_mean__, is also included*

*A routine, __write_global_map_netcdf__, to write a global map to a netcdf file is also included*
	
## plot_2d_array.py
Plots a 2d array of values with labels and colorbar

Example Call:
	fig,ax = plot_2d_array(data, xvals, yvals,zrange=(0.0,12.0), title='Example', xtitle='X', ytitle='Y',cmap='plasma',plt_colorbar = True,zlabel='data values'):

## plot_2d_hist.py
Plots a color-coded representation of a 2-d histogram

Example Call:
	fig.ax = plot_2d_hist(hist, xedges, yedges,
                 title='Histogram Example', xtitle='X', ytitle='Y',  
                 x_range=(0.0, 1.2), y_range=(0.0, 1.2), 
                 plot_diagonal=False, plot_horiz_means=False,
                 norm='Linear',cmap = 'viridis',plt_colorbar=True,fontsize=16,panel_label='A'):

         
## plot_scat.py
Plots a color-coded representation of a 2-d histogram

Example Call:
	plot_scat(x,y, title='Scatter Plot', xtitle='X', ytitle='Y', 
				x_range=None, y_range=None, aspect='equal', 
				plot_diagonal=True, marker = '.'):
				

## sea_ice_plotting
This package contains specialized set ice plotting routines.  Not documented yet
            
## plt_time_lat
Plots a time-lat color coded plot - i.e. a Hovmeuller plot.
Example Call
	plt_time_lat(z=zonal_data,year_range=[2003,2012],xlabel='Year',ylabel='Latitude',title='Example Time Lat plot',units='data units',cmap = 'BrBG',vmin = -1.0,vmax = 1.0,plot_colorbar=True):


    





















