from plotting import plot_polar_stereographic
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from polar_projections import polarstereo_inv

class map_accumulator:
    def __init__(self,pole = 'north',type_map = False):

        if pole == 'north':
            num_x = 304
            num_y = 448
            x_range = [-154.0, 149.0]
            y_range = [-214.0, 233.0]
            std_lat = 70.0
            std_lon = -45.0
        elif pole == 'south':
            num_x = 316
            num_y = 332
            x_range = [-158.0, 157.0]
            y_range =[-158.0, 173.0]
            std_lat = -70.0
            std_lon = 180.0
        else:
            raise(ValueError,f"pole: {pole} not defined")


        self.num_x = num_x
        self.num_y = num_y
        self.x_range = x_range
        self.y_range = y_range
        self.type_map = type_map

        if type_map:
            self.num_stats = 4
            self.stat_type = ['ocean','new','fy','my']
        else:
            self.num_stats = 3
            self.stat_type = [0,1,2]

        iy_list = np.arange(self.y_range[0], self.y_range[1]+1, dtype=np.float32)
        ix_list = np.arange(self.x_range[0], self.x_range[1]+1, dtype=np.float32)
        xdist_list = (ix_list+0.5)*25000.0
        ydist_list = (iy_list+0.5)*25000.0
        self.data =  xr.Dataset(
            data_vars={},
            coords={'ygrid': ydist_list,
                    'xgrid': xdist_list,
                    'stat_type': self.stat_type})

        if pole == 'north':
            self.data.xgrid.attrs = {'valid_range':[-3850000.0,3750000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_x_centers',
                                 'standard_name' : 'projection_x_coordinate',
                                 'axis' : 'X'}
        
            self.data.ygrid.attrs = {'valid_range':[-5350000.0,5850000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_y_centers',
                                 'standard_name' : 'projection_y_coordinate',
                                 'axis' : 'Y'}

        else:
            self.data.xgrid.attrs = {'valid_range':[-3950000.0,3950000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_x_centers',
                                 'standard_name' : 'projection_x_coordinate',
                                 'axis' : 'X'}
        
            self.data.ygrid.attrs = {'valid_range':[-3950000.0,4350000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_y_centers',
                                 'standard_name' : 'projection_y_coordinate',
                                 'axis' : 'Y'}

        self.data['Latitude'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])
        self.data['Longitude'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])

        self.data['X'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])
        
        self.data['Y'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])

        lat_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        lon_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        x_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        y_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)

        for j in np.arange(0, ydist_list.shape[0]):
            x_temp[j,:] = xdist_list
        for i in np.arange(0, xdist_list.shape[0]):
            y_temp[:,i] = ydist_list

        if pole == 'north':
            lon_temp, lat_temp = polarstereo_inv(x_temp/1000.0,y_temp/1000.0, std_parallel = std_lat, lon_y = std_lon)
        else:
            lon_temp, lat_temp = polarstereo_inv(-x_temp/1000.0, y_temp/1000.0, std_parallel = -std_lat, lon_y = std_lon)
            lat = -lat

        self.data['Longitude'].values = lon_temp
        self.data['Latitude'].values  = lat_temp
        self.data['X'].values = x_temp
        self.data['Y'].values = y_temp

    def add_map(self,map_name = 'none',dtype = np.float32):
        data = np.zeros((self.num_y,self.num_x,self.num_stats),dtype=dtype)
        self.data[map_name] = xr.DataArray(data, coords=[self.data.ygrid,self.data.xgrid,self.data.stat_type], dims=['ygrid', 'xgrid','stat'])
    
    def add_data_to_map_lat_lon(self,map_name='none',values = None,lat = None, lon = None, percent_land = None,land_thres = 1.0,pole = 'north'):

        from polar_grids import polarstereo_fwd,polarstereo_fwd_SP
        if pole == 'north':
            x, y = polarstereo_fwd(lat, lon)
        else:
            x, y = polarstereo_fwd_SP(lat, lon)

        ix = np.floor(x * 0.04).astype(np.int32)
        iy = np.floor(y * 0.04).astype(np.int32)

        map = np.zeros((self.num_y, self.num_x, self.num_stats))

        if percent_land is None:
            #ignore land  mask because not present
            ok = np.all([(np.isfinite(values)),
                         (ix >= self.x_range[0]),
                         (ix <= self.x_range[1]),
                         (iy >= self.y_range[0]),
                         (iy <= self.y_range[1])], axis=0)
        else:
            ok  = np.all([(np.isfinite(values)),
                          (ix >= self.x_range[0]),
                          (ix <= self.x_range[1]),
                          (iy >= self.y_range[0]),
                          (iy <= self.y_range[1]),
                          (percent_land <= land_thres)],axis=0)

        rng = np.array([[self.y_range[0] - 0.5,self.y_range[1] + 0.5],
               [self.x_range[0] - 0.5,self.x_range[1] + 0.5]])

        
        map[:, :, 0], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='count',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 1], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 2], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], np.square(values[ok]),
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)

        self.data[map_name] = self.data[map_name] + map
    def add_data_to_map(self,map_name='none',values = None,ix = None, iy = None, percent_land = None,land_thres = 1.0):

        map = np.zeros((self.num_y, self.num_x, self.num_stats))
        if percent_land is None:
            #ignore land  mask because not present
            ok = np.all([(np.isfinite(values)),
                         (ix >= self.x_range[0]),
                         (ix <= self.x_range[1]),
                         (iy >= self.y_range[0]),
                         (iy <= self.y_range[1])], axis=0)
        else:
            ok  = np.all([(np.isfinite(values)),
                          (ix >= self.x_range[0]),
                          (ix <= self.x_range[1]),
                          (iy >= self.y_range[0]),
                          (iy <= self.y_range[1]),
                          (percent_land <= land_thres)],axis=0)

        rng = np.array([[self.y_range[0] - 0.5,self.y_range[1] + 0.5],
               [self.x_range[0] - 0.5,self.x_range[1] + 0.5]])

        
        map[:, :, 0], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='count',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 1], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 2], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], np.square(values[ok]),
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)

        self.data[map_name] = self.data[map_name] + map

    def add_type_data_to_map(self,map_name='none',values = None,ix = None, iy = None, percent_land = None,land_thres = 1.0):

        if not self.type_map:
            raise(ValueError,'Map object is not an ice type object')

        map = np.zeros((self.num_y, self.num_x, self.num_stats))

   #def add_map(self,map_name = 'none',dtype = np.float32):
   #     data = np.zeros((self.num_y,self.num_x,self.num_stats),dtype=dtype)
   #     self.data[map_name] = xr.DataArray(data, coords=[self.data.ygrid,self.data.xgrid,self.data.stat_type], dims=['ygrid', 'xgrid','stat'])

        for type_index in range(0,self.num_stats):
            if percent_land is None:
                #ignore land  mask because not present
                ok = np.all([(np.isfinite(values)),
                             (np.abs(values - type_index) < 0.1),
                             (ix >= self.x_range[0]),
                             (ix <= self.x_range[1]),
                             (iy >= self.y_range[0]),
                             (iy <= self.y_range[1])], axis=0)
            else:
                ok  = np.all([(np.isfinite(values)),
                              (np.abs(values - type_index) < 0.1),
                              (ix >= self.x_range[0]),
                              (ix <= self.x_range[1]),
                              (iy >= self.y_range[0]),
                              (iy <= self.y_range[1]),
                              (percent_land <= land_thres)],axis=0)

            rng = np.array([[self.y_range[0] - 0.5,self.y_range[1] + 0.5],
                [self.x_range[0] - 0.5,self.x_range[1] + 0.5]])
            z, xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='count',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
            map[:, :, type_index] = z

        self.data[map_name] = self.data[map_name] + map


    def combine_maps_xr(map1,map2):
        map_out = map1.copy(deep = True)
        for var in map1.data_vars:
            try:
                map_out[var] = map1[var]+map2[var]
            except:
                print('Can not find '+var+' in second map')
                map_out[var] = map1[var]
        return map_out

    def merge_maps(self,map2):
        for var in self.data.data_vars:
            if var in ['Latitude','Longitude','X','Y']:
                continue
            else:
                try:
                    self.data[var] = self.data[var] + map2.data[var]
                except:
                    print('Can not find '+var+' in second map')


    def mean(self,var = 'ice_map'):

        mean_map = (self.data[var].values)[:,:,1]/(self.data[var].values)[:,:,0]
        return mean_map

    def rms(self,var = 'ice_map'):
        rms_map = np.sqrt((self.data[var].values)[:,:,2]/(self.data[var].values)[:,:,0])
        return rms_map

    def most_common(self,var = 'ice_type'):

        if self.type_map:
            data = np.argmax(self.data[var].values,axis=2)
            most_common_type = xr.DataArray(data, coords=[self.data['ygrid'], self.data['xgrid']],
                                               dims=['ygrid', 'xgrid'])

            tot = np.sum(self.data[var],axis=2)
            most_common_type.values[tot < 1] = -1
        else:
            raise(ValueError,'Object is not a type map')

        return most_common_type

    def to_netcdf(self,nc_file,include_x_y=False):
        
        temp = self.data['Latitude']
        temp = temp.reindex(ygrid = temp.ygrid[::-1])
        temp.to_netcdf(nc_file)
        temp = self.data['Longitude']
        temp = temp.reindex(ygrid = temp.ygrid[::-1])
        temp.to_netcdf(nc_file,mode = 'a')

        if include_x_y:
            temp = self.data['X']
            temp = temp.reindex(ygrid = temp.ygrid[::-1])
            temp.to_netcdf(nc_file,mode='a')
            temp = self.data['Y']
            temp = temp.reindex(ygrid = temp.ygrid[::-1])
            temp.to_netcdf(nc_file,mode = 'a')

        temp_DS =  xr.Dataset(
                        data_vars={})
        for var in self.data.data_vars:
            if var in ['Latitude','Longitude','X','Y']:
                continue
            else:
                temp = self.mean(var=var)
                temp2 = xr.DataArray(temp, 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])
                temp2 = temp2.reindex(ygrid = temp2.ygrid[::-1])
                temp_DS[var] = temp2
        temp_DS.to_netcdf(nc_file,mode='a')


