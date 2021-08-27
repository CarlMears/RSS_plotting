import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from netCDF4 import Dataset as netcdf_dataset

def global_map(a, vmin=0.0, vmax=30.0, cmap=None, plt_colorbar=False,title='',extent=None,central_longitude = 0.0,units = ''):


    img_extent = [-180.0, 180.0, -90.0, 90.0]
    fig = plt.figure(figsize=(10, 5))  # type: Figure
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = central_longitude),title=title)
    for item in ([ax.title]):
        item.set_fontsize(16)
    sz = a.shape
    num_lons = sz[1]

    map = ax.imshow(np.flipud(np.roll(a, int(num_lons/2), axis=1)), cmap=cmap, origin='upper', transform=ccrs.PlateCarree(),
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)
    if plt_colorbar:
        cbar = fig.colorbar(map, shrink=0.7, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
    ax.coastlines()
    ax.set_global()
    ax.text(0.5, -0.2, units, va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
    if extent is not None:
        ax.set_extent(extent,crs = ccrs.PlateCarree())
    return fig, ax

def write_global_map_netcdf(data_in,file_nc='',standard_name='',long_name = '',valid_range = [],nc_data_type='f4'):

    
    shp = data_in.shape
    lats = np.arange(0,720)*0.25 - 90.0 + 0.125
    lons = np.arange(0,1440)*0.25 + 0.125

    if len(shp) != 2:
        raise ValueError('data should be a 2 dimensional array')

    root_grp = netcdf_dataset(file_nc, 'w', format='NETCDF4')

    root_grp.createDimension('latitude', 720)
    root_grp.createDimension('longitude', 1440)

    latitude = root_grp.createVariable('latitude', 'f4', ('latitude',))
    longitude = root_grp.createVariable('longitude', 'f4', ('longitude',))
    data = root_grp.createVariable(standard_name, nc_data_type, ('latitude', 'longitude',))
   


    latitude.standard_name = "latitude"
    latitude.long_name = "latitude"
    latitude.units = "degrees_north"
    latitude.valid_range = (-90.0, 90.0)
    latitude.FillValue = -999.0
    latitude.renameAttribute('FillValue', '_FillValue')

    longitude.standard_name = "longitude"
    longitude.long_name = "longitude"
    longitude.units = "degrees_east"
    longitude.valid_range = (0.0, 360.0)
    longitude.FillValue = -999.0
    longitude.renameAttribute('FillValue', '_FillValue')

    data.standard_name = standard_name
    data.long_name = long_name

    latitude[:] = lats
    longitude[:] = lons
    data[:, :] = data_in
    root_grp.close()

    print('Wrote: ',file_nc)

    return