"""
Title/Version
-------------
Marshall MRMS Mosaic Python Toolkit (MMM-Py)
mmmpy v1.3.2
Developed & tested with Python 2.7.6-2.7.8
Last changed 8/27/2014
    
    
Author
------
Timothy Lang
NASA MSFC
timothy.j.lang@nasa.gov
(256) 961-7861


Overview
--------
This python script defines a class, MosaicTile, which can be
populated with data from a NOAA MRMS mosaic tile file containing
mosaic reflectivities on a national 3D grid. Simple diagnostics and plotting,
as well as computation of composite reflectivity, are available. A
child class, MosaicStitch, is also defined. This can be populated
with stitched-together MosaicTiles. To access these classes, add the
following to your program and then make sure the path to this script
is in your PYTHONPATH:
import mmmpy


Notes
-----
Dependencies: numpy (1.8.1+), scipy (0.13.0+), time, os,
matplotlib (1.3.1+), Basemap, struct, calendar, gzip


Change Log
----------
v1.3.2 major changes:
1. Converted epochtime_to_string() to a public independent function.
2. Added compute_grid_attributes as a public independent function. This will
   return volumes and areas of 3-D and 2-D grid cells, respectively.

v1.3.1 major changes:
1. Added output_composite() method to MosaicTile to compute a 2D composite,
   set it as the only available '3D' field, and write to an MRMS binary file.
2. Added resolution and area_thresh keywords to plot_horiz() and 
   three_panel_plot() methods, to allow adjustment of these Basemap 
   characteristics.

v1.3 major changes
1. Added write_mosaic_binary() method to MosaicTile. Writes out MRMS-format
   binary file.
2. Added capability to read gzipped NetCDF files w/out first decompressing.
   Also now can read uncompressed binary files.
3. Added MosaicTile.Tile attribute, which stores the tile number as a string.
   Number determined based on MRMS Version and StartLat/Lon values.
   MosaicStitch. Tile attribute becomes a concatenated string of tile numbers
   upon stitching one or more MosaicTiles together.
4. Added linewidth keyword to plot_horiz() and three_panel_plot() methods.
   Can now add/thicken gridlines by setting to something other than default 0.
5. Added subsection() method to MosaicTile. It subsections a tile/stitch to a
   smaller grid.

v1.2.1 major changes
1. Added cross-section information to default plot titles.
2. Improved the performance of read_mosaic_binary(), ~3x speedup!

v1.2 major changes
1. Added read_mosaic_binary() method to MosaicTile class. This method reads in
   gzipped binary MRMS mosaics. It runs in pure Python so it takes a short while
   (~25 seconds on my Mac laptop) to read in a single v2 tile.
2. Removed the gridlines (but kept the numbers) from the Basemap plots, 
   if show_grid=True.

v1.1 major changes
1. Signficant refactoring of code to offload many tasks to internal methods of 
   MMM-Py, MosaicTile, and MosaicStitch. This simplified most public
   methods, reduced the number of local variables appearing in any one method,
   and reduced duplication in the code.
2. Added MosaicTile.three_panel_plot() method, which creates a multi-panel 
   plot that shows one horizontal and two vertical plot cross-sections.
3. Plotting methods now can return figure, axis, and Basemap objects if you set
   return_flag=True. This enables the creation of highly customized plots, 
   including overlays, but still using MMM-Py's plotting methods as a baseline.
4. Fixed a nasty bug where apparently v2 mosaics were not fully converting from
   scipy.io.netcdf objects into true ndarrays. This was killing the kernel when
   attempting to do something quantitative or visual with mrefl3d.

v1.0.2 major changes:
1. Changed names of Mosaic_Tile to MosaicTile and Mosaic_Stitch to
   MosaicStitch to conform with PEP8 class naming standards.

v1.0.1 major changes:
1. Added fix for v1/v2 mosaic changeover on 7/30/2013, so that 
   v1 mosaics read from NetCDF produced by MRMS_to_CFncdf will correctly 
   be identified as v1 mosaics.


Planned updates
---------------
1. Support two constant longitude or two constant latitude vertical
   cross-sections in MosaicTile.three_panel_plot(), instead of only separate
   lat and lon cross-sections.
2. Add MosaicTile methods that write out and read in subsets of mosaics. For 
   example, read one or more tiles (stitching them together as necessary),
   then subset over a smaller domain and write out the result as a separate file,
   which is also readable by MMM-Py.

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from scipy.io import netcdf
from struct import unpack
import os
import time
import calendar
import gzip

VERSION = '1.3.2'

#Hard coding of constants
DEFAULT_CLEVS = np.arange(15)*5.0
DEFAULT_VAR = 'mrefl3d'
DEFAULT_VAR_LABEL = 'Reflectivity (dBZ)'
V1_DURATION = 300.0 #seconds
V2_DURATION = 120.0 #seconds
ALTITUDE_SCALE_FACTOR = 1000.0 #Divide meters by this to get something else
DEFAULT_CMAP = cm.GMT_wysiwyg
DEFAULT_PARALLELS = [20, 37.5, 40, 55]
DEFAULT_MERIDIANS = [230, 250, 265, 270, 280, 300]
HORIZONTAL_PLOT = [0.1, 0.1, 0.8, 0.8]
VERTICAL_PLOT = [0.1, 0.2, 0.8, 0.8]
THREE_PANEL_SUBPLOT_A = [0.05, 0.10, 0.52, 0.80]
THREE_PANEL_SUBPLOT_B = [0.64, 0.55, 0.33, 0.32]
THREE_PANEL_SUBPLOT_C = [0.64, 0.14, 0.33, 0.32]
DEFAULT_LONLABEL = 'Longitude (deg)'
DEFAULT_LATLABEL = 'Latitude (deg)'
DEFAULT_ZLABEL = 'Height (km MSL)'
DEFAULT_LATRANGE = [20, 55]
DEFAULT_LONRANGE = [-130, -60]
DEFAULT_LINEWIDTH = 0

#Following is relevant to MRMS binary format read/write methods
ENDIAN = '' #Endian currently set automatically by machine type
INTEGER = 'i'
DEFAULT_VALUE_SCALE = 10
DEFAULT_DXY_SCALE = 100000
DEFAULT_Z_SCALE = 1
DEFAULT_MAP_SCALE = 1000
DEFAULT_MISSING_VALUE = -99
DEFAULT_MRMS_VARNAME = 'mosaicked_refl1     ' #20 characters
DEFAULT_MRMS_VARUNIT = 'dbz   ' #6 characters
DEFAULT_FILENAME =  './mrms_binary_file.dat.gz'

#v1/v2 changeover occurred on 07/30/2013 around 1600 UTC (epoch = 1375200000)
#See https://docs.google.com/document/d/1Op3uETOtd28YqZffgvEGoIj0qU6VU966iT_QNUOmqn4/edit
#for details (doc claims 14 UTC, but CSU has v1 data thru 1550 UTC)
V1_TO_V2_CHANGEOVER_EPOCH_TIME = 1375200000


class MosaicTile(object):

    """
    
    Overview
    --------
    To create a new MosaicTile instance:
    new_instance = MosaicTile() or new_instance = MosaicTile(filename)
    
    Notable attributes
    ------------------
    mrefl3d - Three-dimensional reflectivity on the tile (dBZ).
              Only produced after reading an MRMS file.
              Array = (Height, Latitude, Longitude).
    mrefl3d_comp - Two-dimensional composite reflectivity on the tile (dBZ).
                   Only produced after reading an MRMS file and if required
                   by a plotting or get_comp() call. Once produced, it remains 
                   in memory. Array = (Latitude, Longitude).
    Latitude -  Latitude on the 2-D grid (deg).
    Longitude - Longitude on the 2-D grid (deg).
    Height - One-dimensional array of heights (km MSL).
    Lat/LonGridSpacing - Scalar spacing between gridpoints (deg).
    StartLat - Starting Latitude of the grid (Northernmost border).
    StartLon - Starting Longitude of the grid (Westernmost border).
    Time - Epoch time (seconds since 1/1/1970).
    Duration - Time duration that grid is valid (seconds).
               Total time of grid: Time => Time + Duration.
               Version 1 mosaics = 300 s, Version 2 = 120 s.
    nz, nlat, nlon - Number of gridpoints in each direction.
    Version - Mosaic version (1: <= 7/30/2013, 2: >= 7/30/2013).
    Filename - String containing filename used to populate class (sans path).
    Variables - List of string variable names. Placeholder for when 
                dual-pol mosaics are available.
    
    """

###################################################

    def __init__(self, filename=None, verbose=False, binary=True):

        """
        If initialized with a filename (incl. path), will call
        read_mosaic_netcdf() to populate the class instance.
        If not, it simply instances the class but does not populate
        its attributes.
        filename: Full path and filename of file.
        verbose: Set to True for text output. Useful for debugging.
        binary: Set to True to read an MRMS binary file. False = NetCDF
        
        """

        if isinstance(filename, str) != False:
            if binary:
                self.read_mosaic_binary(filename, verbose=verbose)
            else:
                self.read_mosaic_netcdf(filename, verbose=verbose)
        else:
            #Initializes class instance but leaves it to other methods to
            #populate the class attributes.
            return

###################################################

    def help(self):

        _method_header_printout('help')
        print 'To define a new MosaicTile(), use instance = MosaicTile().'
        print 'then read in a file to populate attributes.'
        print 'Available read methods:'
        print '    read_mosaic_netcdf(<FILE>):'
        print '    read_mosaic_binary(<FILE>):'
        print '    Read v1 MRMS mosaics (7/30/2013 & earlier)'
        print '    Also v2 MRMS mosaics (7/30/2013 & after)'
        print 'Can also use instance = MosaicTile(filepath+name).'
        print 'Set binary=True as keyword above if reading an MRMS binary file.'
        print 'Other available methods:'
        print 'diag(), get_comp(), plot_vert(), plot_horiz(), three_panel_plot()'
        print 'subsection(), write_mosaic_binary(), output_composite()'
        _method_footer_printout()

###################################################

    def read_mosaic_netcdf(self, full_path_and_filename, verbose=False):

        """
        Reads MRMS NetCDF mosaic tiles.
        Attempts to distinguish between v1 (<= 7/30/2013) 
        and v2 (>= 7/30/2013) mosaics.
        v2 are produced from original binary tiles by MRMS_to_CFncdf.
        Reads the file and populates class attributes.
        
        """
        method_name = 'read_mosaic_netcdf'
        if verbose:
            _method_header_printout(method_name)
            print method_name + '(): Reading', full_path_and_filename
        if full_path_and_filename[-3:] == '.gz':
            fopen = gzip.GzipFile(full_path_and_filename, 'rb')
            fileobj = netcdf.netcdf_file(fopen, 'r')
            fopen.close()
        else:
            fileobj = netcdf.netcdf_file(full_path_and_filename, 'r')
        
        #Get data and metadata
        keys = fileobj.variables.keys()
        self.Version = None
        for element in keys:
            if element == 'mrefl_mosaic':
                self.Version = 1
                label = element
                break
            if element == 'MREFL':
                self.Version = 2
                label = element
                break
        if self.Version == None:
            del self.Version
            print 'read_mosaic_netcdf(): Unknown MRMS version, cannot read'
            if verbose:
                _method_footer_printout()
            return
           
        self.Filename = os.path.basename(full_path_and_filename)
        self.nz = fileobj.variables[label].shape[0]
        self.nlat = fileobj.variables[label].shape[1]
        self.nlon = fileobj.variables[label].shape[2]
        self.LatGridSpacing = fileobj.LatGridSpacing
        self.LonGridSpacing = fileobj.LonGridSpacing

        if self.Version == 1:
            lat, lon = self._populate_v1_specific_data(fileobj, label)
        if self.Version == 2:
            lat, lon = self._populate_v2_specific_data(fileobj, label)
        self.Longitude, self.Latitude = np.meshgrid(lon, lat)

        #Fix for v1 MRMS NetCDFs produced by mrms_to_CFncdf from v1 binaries
        #These look like v2 to mmmpy, and thus could impact stitching
        #as v1 tiles overlapped slightly and v2 tiles don't
        if self.Version == 2 and self.Time < V1_TO_V2_CHANGEOVER_EPOCH_TIME:
            self.Version = 1
            self.Duration = V1_DURATION
        self._get_tile_number()

        if verbose:
            _print_method_done()
            _method_footer_printout()
        
###################################################

    def read_mosaic_binary(self, full_path_and_filename, verbose=False):
    
        """
Reads gzipped MRMS binary files and populates MosaicTile fields.
Attempts to distinguish between v1 (<= 7/30/2013) and v2 (>= 7/30/2013) mosaics.
Major reference:
ftp://ftp.nssl.noaa.gov/users/langston/MRMS_REFERENCE/MRMS_BinaryFormat.pdf
        
        """
        
        if verbose:
            begin_time = time.time()
            _method_header_printout('read_mosaic_binary')
            print 'Reading', full_path_and_filename
        
        #Check to see if a real MRMS binary file
        if full_path_and_filename[-3:] == '.gz':
            f = gzip.open(full_path_and_filename, 'rb')
        else:
            f = open(full_path_and_filename, 'rb')
        try:
            self.Time = calendar.timegm(1*np.array(_fill_list(f, 6, 0)))
        except:
            print 'Not an MRMS binary file, nothing read ...'
            if verbose:
                _method_footer_printout()
            return

        if self.Time >= V1_TO_V2_CHANGEOVER_EPOCH_TIME:
            self.Version = 2
            self.Duration = V2_DURATION
        else:
            self.Version = 1
            self.Duration = V1_DURATION
        self.Variables = [DEFAULT_VAR]
        self.Filename = os.path.basename(full_path_and_filename)

        #Get dimensionality from header, use to define datatype
        f.seek(24)
        self.nlon, self.nlat, self.nz = unpack(ENDIAN+3*INTEGER, f.read(12))
        f.seek(80 + self.nz*4 + 78)
        NR, = unpack(ENDIAN+INTEGER, f.read(4))
        dt = self._construct_dtype(NR)
       
        #Rewind and then read everything into the pre-defined datatype.
        #np.fromstring() nearly 3x faster performance than struct.unpack()!
        f.seek(0)
        fileobj = np.fromstring(f.read(80 + 4*self.nz + 82 + 4*NR +
                                2*self.nlon*self.nlat*self.nz), dtype=dt)
        f.close()

        #Populate Latitude, Longitude, and Height
        self.StartLon = 1.0 * fileobj['StartLon'][0] / fileobj['map_scale'][0]
        self.StartLat = 1.0 * fileobj['StartLat'][0] / fileobj['map_scale'][0]
        self.LonGridSpacing = 1.0 * fileobj['dlon'][0] / fileobj['dxy_scale'][0]
        self.LatGridSpacing = 1.0 * fileobj['dlat'][0] / fileobj['dxy_scale'][0]
        #Note the subtraction in lat!
        lat = self.StartLat - self.LatGridSpacing * np.arange(self.nlat)
        lon = self.StartLon + self.LonGridSpacing * np.arange(self.nlon)
        self.Longitude, self.Latitude = np.meshgrid(lon, lat)
        self._get_tile_number()
        self.Height = 1.0 * fileobj['Height'][0] / fileobj['z_scale'][0] /\
                      ALTITUDE_SCALE_FACTOR
        if self.nz == 1:
            self.Height = [self.Height] #Convert to array for compatibility
    
        #Actually populate the mrefl3d data, need to reverse Latitude axis
        data3d = 1.0 * fileobj['data3d'][0] / fileobj['var_scale'][0]
        data3d[:,:,:] = data3d[:,::-1,:]
        setattr(self, DEFAULT_VAR, data3d)

        #Done!
        if verbose:
            print time.time()-begin_time, 'seconds to complete'
            _method_footer_printout()
    
###################################################

    def get_comp(self, var=DEFAULT_VAR, verbose=False):

        """
        Compute maximum reflectivity in column and returns as a new 2-D field.
        Uses numpy.amax() function, which provides great performance.
        
        """
        method_name = 'get_comp'
        if verbose:
            _method_header_printout(method_name)
        
        if not hasattr(self, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        else:
            if verbose:
                print 'Computing composite field'

        temp_3d = getattr(self, var)
        temp_comp = np.amax(temp_3d, axis=0)
        setattr(self, var+'_comp', temp_comp)

        if verbose:
            _method_footer_printout()

###################################################

    def diag(self, verbose=False):

        """
        Prints out diagnostic information and produces 
        a basic plot of tile/stitch composite reflectivity.

        """

        _method_header_printout('diag')
        if not hasattr(self, DEFAULT_VAR):
            print DEFAULT_VAR, 'does not exist, try reading in a file'
            _method_footer_printout()
            return
        print 'Printing basic metadata and making a simple plot' 
        print 'Data are from', self.Filename
        print 'Min, Max Latitude =',  np.min(self.Latitude),\
              np.max(self.Latitude)
        print 'Min, Max Longitude =', np.min(self.Longitude),\
              np.max(self.Longitude)
        print 'Heights (km) =', self.Height
        print 'Grid shape =', np.shape(self.mrefl3d)
        print 'Now plotting ...'
        self.plot_horiz(verbose=verbose)
        print 'Done!'
        _method_footer_printout()

###################################################

    def plot_horiz(self, var=DEFAULT_VAR, latrange=DEFAULT_LATRANGE,
                   lonrange=DEFAULT_LONRANGE, resolution='l',
                   level=None, parallels=None, area_thresh=10000,
                   meridians=None, title=None, clevs=DEFAULT_CLEVS,
                   cmap=DEFAULT_CMAP, save=None, show_grid=True,
                   linewidth=DEFAULT_LINEWIDTH,
                   verbose=False, return_flag=False):
        
        """
        Plots a basemap projection with a plan view of the mosaic radar data.
        The projection can be used to incorporate other data into figure 
        (e.g., lightning).
        var = Variable to be plotted.
        latrange = Desired latitude range of plot (2-element list).
        lonrange = Desired longitude range of plot (2-element list).
        level = If set, performs horizontal cross-section thru that altitude,
                or as close as possible to it. If not set, will plot composite.
        meridians, parallels = Scalars to denote desired gridline spacing.
        linewidth = Width of gridlines (default=0).
        show_grid = Set to False to suppress gridlines and lat/lon tick labels.
        title = String for plot title, None = Basic time & date string as title.
                So if you want a blank title use title='' as keyword.
        clevs = Desired contour levels.
        cmap = Desired color map.
        save = File to save image to. Careful, PS/EPS/PDF can get large!
        verbose = Set to True if you want a lot of text for debugging.
        resolution = Resolution of Basemap instance (e.g., 'c', 'l', 'i', 'h')
        area_thresh = Area threshold to show lakes, etc. (km^2)
        return_flag = Set to True to return plot info.
                      Order is Figure, Axis, Basemap
        
        """

        method_name = 'plot_horiz'
        plt.close() #mpl seems buggy if you don't clean up old windows
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        if self.nlon <= 1 or self.nlat <= 1:
            print 'Latitude or Longitude too small to plot'
            if verbose:
                _method_footer_printout()
            return
        if verbose:
            print 'Executing plot'

        zdata, slevel = self._get_horizontal_cross_section(var, level, verbose)
        #Note the need to transpose for plotting purposes
        plon = np.transpose(self.Longitude)
        plat = np.transpose(self.Latitude)

        fig = plt.figure(figsize=(8, 8))  
        ax = fig.add_axes(HORIZONTAL_PLOT)
        m = self._create_basemap_instance(latrange, lonrange, resolution,
                                          area_thresh)
        m = self._add_gridlines_if_desired(m, parallels, meridians, linewidth,
                                           latrange, lonrange, show_grid)
        x, y = m(plon, plat) # compute map proj coordinates.

        #Draw filled contours
        cs = m.contourf(x, y, zdata, clevs, cmap=cmap)
        #cs = m.pcolormesh(x, y, zdata, vmin=np.min(clevs),
        #                  vmax=np.max(clevs), cmap=cmap)
        
        #Add colorbar, title, and save
        cbar = m.colorbar(cs, location='bottom', pad="7%")
        if var == DEFAULT_VAR:
            cbar.set_label(DEFAULT_VAR_LABEL)
        else:
            #Placeholder for future dual-pol functionality
            cbar.set_label(var)
        if title == None:
            title = epochtime_to_string(self.Time) + slevel
        plt.title(title)
        if save != None:
            plt.savefig(save)

        #Clean up
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax, m

###################################################

    def plot_vert(self, var=DEFAULT_VAR, lat=None, lon=None,
                  xrange=None, xlabel=None,
                  zrange=None, zlabel=DEFAULT_ZLABEL,
                  clevs=DEFAULT_CLEVS, cmap=DEFAULT_CMAP,
                  title=None, save=None, verbose=False,
                  return_flag=False):

        """
        Plots a vertical cross-section through mosaic radar data.
        var = Variable to be plotted.
        lat/lon = If set, performs vertical cross-section thru that lat/lon,
                  or as close as possible to it. Only one or the other 
                  can be set.
        xrange = Desired latitude or longitude range of plot (2-element list).
        zrange = Desired height range of plot (2-element list).
        xlabel, zlabel = Axes labels.
        clevs = Desired contour levels.
        cmap = Desired color map.
        title = String for plot title, None = Basic time & date string as title.
                So if you want a blank title use title='' as keyword.
        save = File to save image to. Careful, PS/EPS/PDF can get large!
        verbose = Set to True if you want a lot of text for debugging.
        return_flag = Set to True to return Figure, Axis objects (in that order)

        """
        
        method_name = 'plot_vert'
        plt.close() #mpl seems buggy if you don't clean up old windows
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return

        #Get the cross-section
        vcut, xvar, xrange, xlabel, tlabel = self._get_vertical_slice(var, lat,
                                                   lon, xrange, xlabel, verbose)
        if vcut == None:
            return
        
        #Plot details
        if not title:
            title = epochtime_to_string(self.Time) + ' ' + tlabel
        if not zrange:
            zrange = [0, np.max(self.Height)]
            
        #Plot execution
        fig = plt.figure()
        ax = fig.add_axes(VERTICAL_PLOT)
        ax, cs = self._plot_vertical_cross_section(ax, vcut, xvar, xrange,
                                                   xlabel, zrange, zlabel, clevs,
                                                   cmap, title, mappable=True)
        cbar = fig.colorbar(cs)
        if var == DEFAULT_VAR:
            cbar.set_label(DEFAULT_VAR_LABEL, rotation=90)
        else:
            #Placeholder for future dual-pol functionality
            cbar.set_label(var, rotation=90)
    
        #Finish up
        if save != None:
            plt.savefig(save)
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax

###################################################

    def three_panel_plot(self, var=DEFAULT_VAR, lat=None, lon=None,
                         latrange=DEFAULT_LATRANGE, lonrange=DEFAULT_LONRANGE,
                         meridians=None, parallels=None,
                         linewidth=DEFAULT_LINEWIDTH, resolution='l',
                         show_grid=True, level=None, area_thresh=10000,
                         lonlabel=DEFAULT_LONLABEL,
                         latlabel=DEFAULT_LATLABEL,
                         zrange=None, zlabel=DEFAULT_ZLABEL,
                         clevs=DEFAULT_CLEVS, cmap=DEFAULT_CMAP,
                         title_a=None, title_b=None, title_c=None,
                         xrange_b=None, xrange_c=None,
                         save=None, verbose=False,
                         return_flag=False, show_crosshairs=True):

        """
        Plots horizontal and vertical cross-sections through mosaic radar data.
        Subplot (a) is the horizontal view, (b) is the
        var = Variable to be plotted.
        latrange = Desired latitude range of plot (2-element list).
        lonrange = Desired longitude range of plot (2-element list).
        level = If set, performs horizontal cross-section thru that altitude,
                or as close as possible to it. If not set, will plot composite.
        meridians, parallels = Scalars to denote desired gridline spacing.
        linewidth = Width of gridlines (default=0).
        show_grid = Set to False to suppress gridlines and lat/lon tick labels.
        title_a, _b, _c = Strings for subplot titles, None = Basic time & date
                          string as title for subplot (a), and constant lat/lon
                          for (b) and (c). So if you want blank titles use 
                          title_?='' as keywords.
        clevs = Desired contour levels.
        cmap = Desired color map.
        save = File to save image to. Careful, PS/EPS/PDF can get large!
        verbose = Set to True if you want a lot of text for debugging.
        zrange = Desired height range of plot (2-element list).
        resolution = Resolution of Basemap instance (e.g., 'c', 'l', 'i', 'h')
        area_thresh = Area threshold to show lakes, etc. (km^2)
        lonlabel, latlabel, zlabel = Axes labels.
        lat/lon = Performs vertical cross-sections thru those lat/lons,
                  or as close as possible to them. Both are required to be set!
        return_flag = Set to True to return plot info. 
                      Order is Figure, Axes (3 of them), Basemap.
        show_crosshairs = Set to False to suppress the vertical cross-section
                          crosshairs on the horizontal cross-section.
        xrange_b, _c = Subplot (b) is constant latitude, so xrange_b is is a 
                       2-element list that allows the user to adjust the 
                       longitude domain of (b). Defaults to lonrange if not set. 
                       Similar setup for xrange_c - subplot (c) - except for
                       latitude (i.e., defaults to latrange if not set). The
                       xrange_? variables determine the length of the crosshairs.
                       
        """
 
        method_name = 'three_panel_plot'
        plt.close() #mpl seems buggy if you don't clean up old windows
        if verbose:
            _method_header_printout(method_name)
        
        if not hasattr(self, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        if self.nlon <= 1 or self.nlat <= 1:
            print 'Latitude or Longitude too small to plot'
            if verbose:
                _method_footer_printout()
            return
        if lat == None or lon == None:
            print method_name + '(): Need both constant latitude and',\
                  'constant longitude for slices'
            if verbose:
                _method_footer_printout()
            return

        fig = plt.figure()
        fig.set_size_inches(11, 8.5)

        #Horizontal Cross-Section + Color Bar (subplot a)
        zdata, slevel = self._get_horizontal_cross_section(var, level, verbose)
        #Note the need to transpose for horizontal plotting purposes
        plon = np.transpose(self.Longitude)
        plat = np.transpose(self.Latitude)
        ax1 = fig.add_axes(THREE_PANEL_SUBPLOT_A)
        m = self._create_basemap_instance(latrange, lonrange, resolution,
                                          area_thresh)
        m = self._add_gridlines_if_desired(m, parallels, meridians, linewidth,
                                           latrange, lonrange, show_grid)
        x, y = m(plon, plat)
        cs = m.contourf(x, y, zdata, clevs, cmap=cmap)
        if xrange_b == None:
            xrange_b = lonrange
        if not xrange_c:
            xrange_c = latrange
        if show_crosshairs:
            m = self._add_crosshairs(m, lat, lon, xrange_b, xrange_c)
        cbar = m.colorbar(cs, location='bottom', pad="7%")
        if var == DEFAULT_VAR:
            cbar.set_label(DEFAULT_VAR_LABEL)
        else:
            #Placeholder for future dual-pol functionality
            cbar.set_label(var)
        if not title_a:
            title_a = '(a) ' + epochtime_to_string(self.Time) + slevel
        ax1.set_title(title_a)

        #Vertical Cross-Section (subplot b)
        vcut2, xvar2, tlabel2 =\
               self._get_constant_latitude_cross_section(var, lat)
        if zrange == None:
            zrange = [0, np.max(self.Height)]
        if not title_b:
            title_b = '(b) ' + tlabel2
        ax2 = fig.add_axes(THREE_PANEL_SUBPLOT_B)
        ax2 = self._plot_vertical_cross_section(ax2, vcut2, xvar2,
                                                xrange_b, lonlabel,
                                                zrange, zlabel, clevs, cmap,
                                                title_b)

        #Vertical Cross-Section (subplot c)
        vcut3, xvar3, tlabel3 =\
               self._get_constant_longitude_cross_section(var, lon)
        if not title_c:
            title_c = '(c) ' + tlabel3
        ax3 = fig.add_axes(THREE_PANEL_SUBPLOT_C)
        ax3 = self._plot_vertical_cross_section(ax3, vcut3, xvar3,
                                                xrange_c, latlabel,
                                                zrange, zlabel, clevs, cmap,
                                                title_c)
                                            
        #Finish up
        if save != None:
            plt.savefig(save)
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax1, ax2, ax3, m

###################################################

    def write_mosaic_binary(self, full_path_and_filename=None, verbose=False):

        """
        Major reference:
ftp://ftp.nssl.noaa.gov/users/langston/MRMS_REFERENCE/MRMS_BinaryFormat.pdf
        Note that user will need to keep track of Endian for machine used to
        write the file. MMM-Py's ENDIAN global variable may need to be adjusted
        if reading on a different Endian machine than files were produced.
        You can write out a subsectioned or a stitched mosaic and it will
        be readable by read_mosaic_binary().
        full_path_and_filename = Filename (including path). Include the .gz suffix.
        verbose = Set to True to get some text response.

        """
        if verbose:
            _method_header_printout('write_mosaic_binary')
            begin_time = time.time()
      
        if full_path_and_filename == None:
            full_path_and_filename = DEFAULT_FILENAME
        elif full_path_and_filename[-3:] != '.gz':
            full_path_and_filename += '.gz'
        if verbose:
            print 'Writing MRMS binary format to', full_path_and_filename

        header = self._construct_header()
        data1d = self._construct_1d_data()
        output = gzip.open(full_path_and_filename, 'wb')
        output.write(header+data1d.tostring())
        output.close()

        if verbose:
            print time.time() - begin_time, 'seconds to complete'
            _method_footer_printout()

###################################################

    def subsection(self, latrange=None, lonrange=None, zrange=None,
                   verbose=False):
        
        """
        Subsections a tile (or stitch) by keeping data only within the given
        2-element lists: latrange (deg), lonrange (deg), zrange (km).
        Lists that are not defined will lead to no subsectioning along those
        axes.
        verbose = Set to True to get some text response.
        
        """
        if verbose:
            _method_header_printout('subsection')
            if latrange and np.size(latrange) == 2:
                print 'Latitude Range to Keep =', latrange
            else:
                print 'No subsectioning in Latitude'
            if lonrange and np.size(lonrange) == 2:
                print 'Longitude Range to Keep =', lonrange
            else:
                print 'No subsectioning in Longitude'
            if zrange and np.size(zrange) == 2:
                print 'Height Range to Keep =', zrange
            else:
                print 'No subsectioning in Height'
    
        self._subsection_in_latitude(latrange)
        self._subsection_in_longitude(lonrange)
        self._subsection_in_height(zrange)
        
        if verbose:
            _method_footer_printout()

###################################################

    def output_composite(self, full_path_and_filename=DEFAULT_FILENAME,
                         var=DEFAULT_VAR, verbose=False):
        """
        Produces a gzipped binary file containing only a composite of
        the chosen variable. The existing tile now will only consist of a single
        vertical level (e.g., composite reflectivity)
        
        """
        method_name = 'output_composite'
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        if not hasattr(self, var+'_comp'):
            if verbose:
                print var+'_comp does not exist,',\
                          'computing it with get_comp()'
            self.get_comp(var=var, verbose=verbose)
        self.subsection(zrange=[self.Height[0], self.Height[0]], verbose=verbose)
        temp2d = getattr(self, var+'_comp')
        temp3d = getattr(self, var)
        temp3d[0,:,:] = temp2d[:,:]
        setattr(self, var, temp3d)
        self.write_mosaic_binary(full_path_and_filename, verbose)
        if verbose:
            _method_footer_printout()

###################################################
#MosaicTile internal methods below
###################################################

    def _populate_v1_specific_data(self, fileobj=None, label='mrefl_mosaic'):
        """v1 MRMS netcdf data file"""
        self.StartLat = fileobj.Latitude
        self.StartLon = fileobj.Longitude
        self.Height = fileobj.variables['Height'][:] / ALTITUDE_SCALE_FACTOR
        self.Time = np.float64(fileobj.Time)
        self.Duration = V1_DURATION
        ScaleFactor = fileobj.variables[label].Scale
        self.mrefl3d = fileobj.variables[label][:, :, :] / ScaleFactor
        #Note the subtraction in lat!
        lat = self.StartLat - self.LatGridSpacing * np.arange(self.nlat)
        lon = self.StartLon + self.LonGridSpacing * np.arange(self.nlon)
        self.Variables = [DEFAULT_VAR]
        return lat, lon

###################################################

    def _populate_v2_specific_data(self, fileobj=None, label='MREFL'):
        """v2 MRMS netcdf data file"""
        self.Height = fileobj.variables['Ht'][:] / ALTITUDE_SCALE_FACTOR
        #Getting weird errors with scipy 0.14 when np.array() not invoked below.
        #Think it was not properly converting from scipy netcdf object.
        #v1 worked OK because of the ScaleFactor division in
        #_populate_v1_specific_data().
        self.mrefl3d = np.array(fileobj.variables[label][:, :, :])
        lat = fileobj.variables['Lat'][:]
        lon = fileobj.variables['Lon'][:]
        self.StartLat = lat[0]
        self.StartLon = lon[0]
        self.Time = fileobj.variables['time'][0]
        self.Duration = V2_DURATION
        self.Variables = [DEFAULT_VAR]
        return lat, lon

###################################################

    def _construct_dtype(self, NR=1):
        """This is the structure of a complete binary MRMS file"""
        dt = np.dtype([('year', 'i4'), ('month', 'i4'), ('day', 'i4'),
               ('hour', 'i4'), ('minute', 'i4'), ('second', 'i4'),
               ('nlon', 'i4'), ('nlat', 'i4'), ('nz', 'i4'), ('deprec1', 'i4'),
               ('map_scale', 'i4'), ('deprec2', 'i4'), ('deprec3', 'i4'),
               ('deprec4', 'i4'), ('StartLon', 'i4'), ('StartLat', 'i4'),
               ('deprec5', 'i4'), ('dlon', 'i4'), ('dlat', 'i4'),
               ('dxy_scale', 'i4'), ('Height', ('i4', self.nz)),
               ('z_scale', 'i4'), ('placeholder1', ('i4', 10)),
               ('VarName', 'a20'), ('VarUnit', 'a6'),
               ('var_scale', 'i4'), ('missing_value', 'i4'), ('NR', 'i4'),
               ('Radars', ('a4', NR)),
               ('data3d', ('i2', (self.nz, self.nlat, self.nlon))) ])
        return dt
    
###################################################

    def _get_tile_number(self):
        self.Tile = '?'
        if self.Version == 1:
            if _are_equal(self.StartLat, 55.0) and\
               _are_equal(self.StartLon, -130.0):
                self.Tile = '1'
            elif _are_equal(self.StartLat, 55.0) and\
                 _are_equal(self.StartLon, -110.0):
                self.Tile = '2'
            elif _are_equal(self.StartLat, 55.0) and\
                 _are_equal(self.StartLon, -90.0):
                self.Tile = '3'
            elif _are_equal(self.StartLat, 55.0) and\
                 _are_equal(self.StartLon, -80.0):
                self.Tile = '4'
            elif _are_equal(self.StartLat, 40.0) and\
                 _are_equal(self.StartLon, -130.0):
                self.Tile = '5'
            elif _are_equal(self.StartLat, 40.0) and\
                 _are_equal(self.StartLon, -110.0):
                self.Tile = '6'
            elif _are_equal(self.StartLat, 40.0) and\
                 _are_equal(self.StartLon, -90.0):
                self.Tile = '7'
            elif _are_equal(self.StartLat, 40.0) and\
                 _are_equal(self.StartLon, -80.0):
                self.Tile = '8'
        elif self.Version == 2:
            if _are_equal(self.StartLat, 54.995) and\
               _are_equal(self.StartLon, -129.995):
                self.Tile = '1'
            elif _are_equal(self.StartLat, 54.995) and\
                 _are_equal(self.StartLon, -94.995):
                self.Tile = '2'
            elif _are_equal(self.StartLat, 37.495) and\
                 _are_equal(self.StartLon, -129.995):
                self.Tile = '3'
            elif _are_equal(self.StartLat, 37.495) and\
                 _are_equal(self.StartLon, -94.995):
                self.Tile = '4'

###################################################

    def _construct_header(self):
        """This is the structure of the header of a binary MRMS file"""
        nr = np.int32(1).tostring()
        rad_name = 'none'
        nz = np.int32(self.nz).tostring()
        nlat = np.int32(self.nlat).tostring()
        nlon = np.int32(self.nlon).tostring()
        year = np.int32(time.gmtime(self.Time)[0]).tostring()
        month = np.int32(time.gmtime(self.Time)[1]).tostring()
        day = np.int32(time.gmtime(self.Time)[2]).tostring()
        hour = np.int32(time.gmtime(self.Time)[3]).tostring()
        minute = np.int32(time.gmtime(self.Time)[4]).tostring()
        second = np.int32(time.gmtime(self.Time)[5]).tostring()
        map_scale = np.int32(DEFAULT_MAP_SCALE).tostring()
        dxy_scale = np.int32(DEFAULT_DXY_SCALE).tostring()
        z_scale = np.int32(DEFAULT_Z_SCALE).tostring()
        var_scale = np.int32(DEFAULT_VALUE_SCALE).tostring()
        missing = np.int32(DEFAULT_MISSING_VALUE).tostring()
        VarName = DEFAULT_MRMS_VARNAME
        VarUnit = DEFAULT_MRMS_VARUNIT
        StartLat = np.int32(self.StartLat * DEFAULT_MAP_SCALE).tostring()
        StartLon = np.int32(self.StartLon * DEFAULT_MAP_SCALE).tostring()
        Height = np.int32(self.Height * DEFAULT_Z_SCALE *\
                          ALTITUDE_SCALE_FACTOR).tostring() #km to m
        dlon = np.int32(self.LonGridSpacing * DEFAULT_DXY_SCALE).tostring()
        dlat = np.int32(self.LatGridSpacing * DEFAULT_DXY_SCALE).tostring()
        #Set depreciated and placeholder values.
        #Don't think exact number matters, but for now set as same values
        #obtained from MREF3D33L_tile2.20140619.010000.gz
        deprec1 = np.int32(538987596).tostring()
        deprec2 = np.int32(30000).tostring()
        deprec3 = np.int32(60000).tostring() 
        deprec4 = np.int32(-60005).tostring() 
        deprec5 = np.int32(1000).tostring()
        ph = 0 * np.arange(10) + 19000
        placeholder = np.int32(ph).tostring() #10 of these placeholder values
        header = year+month+day+hour+minute+second+nlon+nlat+nz+deprec1+map_scale+\
                 deprec2+deprec3+deprec4+StartLon+StartLat+deprec5+dlon+dlat+\
                 dxy_scale+Height+z_scale+placeholder+VarName+VarUnit+var_scale+\
                 missing+nr+rad_name
        return header

###################################################

    def _construct_1d_data(self):
    
        """
        Turns a 3D float mosaic into a 1-D short array suitable for writing
        to a binary file
        
        """
        data1d = DEFAULT_VALUE_SCALE * getattr(self, DEFAULT_VAR)
        #MRMS binaries have the Latitude axis flipped
        data1d[:,:,:] = data1d[:,::-1,:]
        data1d = data1d.astype(np.int16)
        data1d = data1d.ravel()
        return data1d

###################################################

    def _get_horizontal_cross_section(self, var=DEFAULT_VAR, level=None,
                                      verbose=False):
        if level == None:
            slevel=' Composite '
            if verbose:
                print 'No vertical level specified,',\
                      'plotting composite reflectivity'
            if not hasattr(self, var+'_comp'):
                if verbose:
                    print var+'_comp does not exist,',\
                          'computing it with get_comp()'
                self.get_comp(var=var, verbose=verbose)
            zdata = 1.0 * getattr(self, var+'_comp')
            zdata = np.transpose(zdata)
        else:
            if verbose:
                print 'Attempting to plot cross-section thru', level, 'km MSL'
            if level < np.min(self.Height):
                level = np.min(self.Height)
            elif level > np.max(self.Height):
                level = np.max(self.Height)
            index = np.argmin(np.abs(level - self.Height))
            level = self.Height[index]
            slevel = ' %.1f' % level + ' km MSL'
            temp_3d = 1.0 * getattr(self, var)
            zdata = temp_3d[index, :, :]
            zdata = np.transpose(zdata)
            if verbose:
                print 'Actually taking cross-section thru', level, 'km MSL'
        return zdata, slevel

###################################################

    def _create_basemap_instance(self, latrange=None, lonrange=None,
                                 resolution='l', area_thresh=10000):
        #create Basemap instance
        lon_0 = np.mean(lonrange)
        lat_0 = np.mean(latrange)
        m = Basemap(projection='merc', lon_0=lon_0, lat_0=lat_0, lat_ts=lat_0,
                llcrnrlat=np.min(latrange), urcrnrlat=np.max(latrange),
                llcrnrlon=np.min(lonrange), urcrnrlon=np.max(lonrange),
                rsphere=6371200., resolution=resolution, area_thresh=area_thresh)
        #Draw coastlines, state and country boundaries, edge of map
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
        return m

###################################################

    def _add_gridlines_if_desired(self, m=None, parallels=None, meridians=None,
                                  linewidth=DEFAULT_LINEWIDTH,
                                  latrange=None, lonrange=None, show_grid=True):
        if show_grid:
            #Draw parallels
            if parallels == None:
                vparallels = DEFAULT_PARALLELS
            else:
                vparallels = np.arange(np.floor(np.min(latrange)),
                                       np.ceil(np.max(latrange)), parallels)
            m.drawparallels(vparallels, labels=[1,0,0,0], fontsize=10,
                            linewidth=linewidth)
            #Draw meridians
            if meridians == None:
                vmeridians = DEFAULT_MERIDIANS
            else:
                vmeridians = np.arange(np.floor(np.min(lonrange)),
                                       np.ceil(np.max(lonrange)), meridians)
            m.drawmeridians(vmeridians, labels=[0,0,0,1], fontsize=10,
                            linewidth=linewidth)
        return m

###################################################

    def _add_crosshairs(self, m=None, lat=None, lon=None,
                        xrange_b=None, xrange_c=None):
        cross_xrange = [np.min(xrange_b), np.max(xrange_b)]
        constant_lat = [lat, lat]
        cross_yrange = [np.min(xrange_c), np.max(xrange_c)]
        constant_lon = [lon, lon]
        xc1, yc1 = m(cross_xrange, constant_lat)
        xc2, yc2 = m(constant_lon, cross_yrange)
        m.plot(xc1, yc1, 'k--', linewidth=2)
        m.plot(xc1, yc1, 'r--', linewidth=1)
        m.plot(xc2, yc2, 'k--', linewidth=2)
        m.plot(xc2, yc2, 'r--', linewidth=1)
        return m

###################################################

    def _get_vertical_slice(self, var=DEFAULT_VAR, lat=None, lon=None,
                            xrange=None, xlabel=None, verbose=False):
        """Execute slicing, get xvar, vcut"""
        fail = [None, None, None, None, None]
        if lat == None and lon == None:
            print 'plot_vert(): Need a constant latitude or longitude for slice'
            if verbose:
                _method_footer_printout()
            return fail
        elif lat != None and lon != None:
            print 'plot_vert(): Need either a lat or a lon for slice, not both!'
            if verbose:
                _method_footer_printout()
            return fail
        else:
            if lon == None:
                if self.nlon <= 1:
                    print 'Available Longitude range too small to plot'
                    if verbose:
                        _method_footer_printout()
                    return fail
                if verbose:
                    print 'Plotting vertical cross-section thru', lat,\
                          'deg Latitude'
                if not xrange:
                    xrange = [np.min(self.Longitude), np.max(self.Longitude)]
                if not xlabel:
                    xlabel = 'Longitude (deg)'
                vcut, xvar, tlabel =\
                      self._get_constant_latitude_cross_section(var, lat)
            if lat == None:
                if self.nlat <= 1:
                    print 'Available Latitude range too small to plot'
                    if verbose:
                        _method_footer_printout()
                    return fail
                if verbose:
                    print 'Plotting vertical cross-section thru', lon,\
                          'deg Longitude'
                if not xrange:
                    xrange = [np.min(self.Latitude),  np.max(self.Latitude)]
                if not xlabel:
                    xlabel = 'Latitude (deg)'
                vcut, xvar, tlabel =\
                      self._get_constant_longitude_cross_section(var, lon)
        return vcut, xvar, xrange, xlabel, tlabel

###################################################

    def _get_constant_latitude_cross_section(self, var=DEFAULT_VAR, lat=None):
        if lat > np.max(self.Latitude):
            lat = np.max(self.Latitude)
            print 'Outside domain, plotting instead thru',\
                  lat,' deg Latitude'
        if lat < np.min(self.Latitude):
            lat = np.min(self.Latitude)
            print 'Outside domain, plotting instead thru',\
                  lat, 'deg Latitude'
        index = np.round(np.abs(lat-self.StartLat) / self.LatGridSpacing)
        xvar = self.Longitude[index, :]
        temp_3d = getattr(self, var)
        vcut = temp_3d[:, index, :]
        return vcut, xvar, 'Latitude = ' + '%.2f' % lat + ' deg'

###################################################

    def _get_constant_longitude_cross_section(self, var=DEFAULT_VAR, lon=None):
        if lon > np.max(self.Longitude):
            print 'max',lon, np.max(self.Longitude)
            lon = np.max(self.Longitude)
            print 'Outside domain, plotting instead thru',\
                  lon,' deg Longitude'
        if lon < np.min(self.Longitude):
            print 'min',lon, np.max(self.Longitude)
            lon = np.min(self.Longitude)
            print 'Outside domain, plotting instead thru',\
                  lon, 'deg Longitude'
        index = np.round(np.abs(lon-self.StartLon)/self.LonGridSpacing)
        xvar = self.Latitude[:, index]
        temp_3d = getattr(self, var)
        vcut = temp_3d[:, :, index]
        return vcut, xvar, 'Longitude = ' + '%.2f' % lon + ' deg'

###################################################

    def _plot_vertical_cross_section(self, ax=None, vcut=None, xvar=None,
                                     xrange=None, xlabel=None,
                                     zrange=None, zlabel=None,
                                     clevs=DEFAULT_CLEVS, cmap=DEFAULT_CMAP,
                                     title=None, mappable=False):
        cs = ax.contourf(xvar, self.Height, vcut, clevs, cmap=cmap)
        if title:
            ax.set_title(title)
        ax.set_xlim(np.min(xrange), np.max(xrange))
        ax.set_ylim(np.min(zrange), np.max(zrange))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        if mappable:
            return ax, cs
        else:
            return ax

###################################################

    def _subsection_in_latitude(self, latrange=None):
        if latrange and np.size(latrange) == 2:
            temp = self.Latitude[:, 0]
            condition = np.logical_or(temp < np.min(latrange),
                                      temp > np.max(latrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nlat:
                print 'Refusing to delete all data in Latitude'
            else:
                self.Latitude = np.delete(self.Latitude, indices[0], axis=0)
                self.Longitude = np.delete(self.Longitude, indices[0], axis=0)
                self.nlat = self.nlat - np.size(indices[0])
                self.StartLat = np.max(self.Latitude)
                self._subsection_data3d(indices, 1)

###################################################

    def _subsection_in_longitude(self, lonrange=None):
        if lonrange and np.size(lonrange) == 2:
            temp = self.Longitude[0, :]
            condition = np.logical_or(temp < np.min(lonrange),
                                      temp > np.max(lonrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nlon:
                print 'Refusing to delete all data in Longitude'
            else:
                self.Latitude = np.delete(self.Latitude, indices[0], axis=1)
                self.Longitude = np.delete(self.Longitude, indices[0], axis=1)
                self.nlon = self.nlon - np.size(indices[0])
                self.StartLon = np.min(self.Longitude)
                self._subsection_data3d(indices, 2)

###################################################

    def _subsection_in_height(self, zrange=None):
        if zrange and np.size(zrange) == 2:
            temp = self.Height[:]
            condition = np.logical_or(temp < np.min(zrange),
                                      temp > np.max(zrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nz:
                print 'Refusing to delete all data in Height'
            else:
                self.Height = np.delete(self.Height, indices[0], axis=0)
                self.nz = self.nz - np.size(indices[0])
                self._subsection_data3d(indices, 0)

###################################################

    def _subsection_data3d(self, indices, axis):
        for var in self.Variables:
            temp3d = getattr(self, var)
            temp3d = np.delete(temp3d, indices[0], axis=axis)
            setattr(self, var, temp3d)

###################################################

###################################################

###################################################

###################################################
###################################################

class MosaicStitch(MosaicTile):

    """
    Child class of MosaicTile().
    To create a new MosaicStitch instance:
    new_instance = MosaicStitch() or
    new_instance = stitch_mosaic_tiles(map_array=map_array, direction=direction)
    
    """

###################################################

    def __init__(self, verbose=False):

        """
        Initializes class instance but leaves it to other methods to
        populate the class attributes.
        
        """
        MosaicTile.__init__(self, verbose=False)

###################################################

    def help(self):

        _method_header_printout('help')
        print 'Call stitch_mosaic_tiles() function to populate variables'
        print 'This standalone function makes use of this class\'s'
        print 'stitch_ns() and stitch_we() methods'
        print 'Instance = stitch_mosaic_tiles(map_array=map_array,',\
              'direction=direction)'
        print 'All MosaicTile() methods are available as well,'
        print 'except read_mosaic_*() methods are disabled to avoid problems'
        _method_footer_printout()

###################################################

    def read_mosaic_netcdf(self, filename, verbose=False):

        """Disabled in a MosaicStitch to avoid problems."""
        
        _method_header_printout('read_mosaic_netcdf')
        print 'To avoid problems with reading individual MosaicTile classes'
        print 'into a MosaicStitch, this method has been disabled'
        _method_footer_printout()

###################################################

    def read_mosaic_binary(self, filename, verbose=False):

        """Disabled in a MosaicStitch to avoid problems."""
        
        _method_header_printout('read_mosaic_binary')
        print 'To avoid problems with reading individual MosaicTile classes'
        print 'into a MosaicStitch, this method has been disabled'
        _method_footer_printout()

###################################################

    def stitch_ns(self, n_tile=None, s_tile=None, verbose=False):
        
        """
        Stitches MosaicTile pair or MosaicStitch pair (or mixed pair) 
        in N-S direction.
        
        """
        
        method_name = 'stitch_ns'
        if verbose:
            _method_header_printout(method_name)
        
        #Check to make sure method was called correctly
        if n_tile == None or s_tile == None:
            _print_method_called_incorrectly('stitch_ns')
            return
        
        #Check to make sure np.append() will not fail due to different grids
        if n_tile.nlon != s_tile.nlon:
            print method_name + '(): Grid size in Longitude does not match,',\
                  'fix this before proceeding'
            return
        
        inlat, index = self._stitch_radar_variables(n_tile, s_tile, verbose,
                                                    ns_flag=True)
                                                    
        if index == 0:
            print method_name + '(): No radar variables to stitch! Returning ...'
            return
            
        self._stitch_metadata(n_tile, s_tile, inlat, ns_flag=True)
        
        if verbose:
            print 'Completed normally'
            _method_footer_printout()

###################################################

    def stitch_we(self, w_tile=None, e_tile=None, verbose=False):
        
        """
        Stitches MosaicTile pair or MosaicStitch pair (or mixed pair) 
        in W-E direction
        
        """
        
        method_name = 'stitch_we'
        if verbose:
            _method_header_printout(method_name)

        #Check to make sure method was called correctly
        if w_tile == None or e_tile == None:
            _print_method_called_incorrectly('stitch_we')
            return

        #Check to make sure np.append() will not fail due to different grids
        if w_tile.nlat != e_tile.nlat:
            print method_name + '(): Grid size in Latitude does not match,',\
                  'fix this before proceeding'
            return

        inlon, index = self._stitch_radar_variables(w_tile, e_tile,
                                                    ns_flag=False)
        if index == 0:
            print method_name + '(): No radar variables to stitch! Returning ...'
            return
        
        self._stitch_metadata(w_tile, e_tile, inlon, ns_flag=False)

        if verbose:
            print 'Completed normally'
            _method_footer_printout()

###################################################
#MosaicStitch internal methods follow
###################################################

    def _stitch_metadata(self, a_tile=None, b_tile=None, index=None,
                         ns_flag=True):
        """
        ns_flag=True means N-S stitch, False means W-E stitch.
        Uses np.append() to stitch together.
           
        """
        if ns_flag:
            self.Longitude = np.append(a_tile.Longitude,
                                       b_tile.Longitude[:index,:], axis=0)
            self.Latitude = np.append(a_tile.Latitude,
                                      b_tile.Latitude[:index,:], axis=0)
            if a_tile.Version == 1:
                self.nlat = a_tile.nlat + b_tile.nlat - 1
            if a_tile.Version == 2:
                self.nlat = a_tile.nlat + b_tile.nlat
            self.nlon = a_tile.nlon
        else:
            self.Longitude = np.append(a_tile.Longitude[:,:index],
                                       b_tile.Longitude, axis=1)
            self.Latitude = np.append(a_tile.Latitude [:,:index],
                                      b_tile.Latitude, axis=1)
            if a_tile.Version == 1:
                self.nlon = a_tile.nlon + b_tile.nlon - 1
            if a_tile.Version == 2:
                self.nlon = a_tile.nlon + b_tile.nlon
            self.nlat = a_tile.nlat

        #Populate the other metadata attributes
        self.Height = a_tile.Height
        self.StartLat = a_tile.StartLat
        self.StartLon = a_tile.StartLon
        self.Filename = a_tile.Filename + '+' + b_tile.Filename
        self.nz = a_tile.nz
        self.LatGridSpacing = a_tile.LatGridSpacing
        self.LonGridSpacing = a_tile.LonGridSpacing
        self.Version = a_tile.Version
        self.Variables = a_tile.Variables
        self.Time = a_tile.Time
        self.Duration = a_tile.Duration
        self.Tile = a_tile.Tile + b_tile.Tile

###################################################

    def _stitch_radar_variables(self, a_tile=None, b_tile=None,
                                verbose=False, ns_flag=True):
        """
        ns_flag=True means N-S stitch, False means W-E stitch.
        Uses np.append() to stitch together.
        Ignore composite - can recalculate quickly rather than stitch.
        
        """
        index = 0
        for var in a_tile.Variables:
            if hasattr(a_tile, var) and hasattr(b_tile, var):
                if verbose:
                    print 'Stitching', var
                a_temp = 1.0 * getattr(a_tile, var)
                b_temp = 1.0 * getattr(b_tile, var)
                if ns_flag:
                    if a_tile.Version == 1:
                        inl = np.shape(b_temp)[1] - 1
                    if a_tile.Version == 2:
                        inl = np.shape(b_temp)[1]
                    temp_3d = np.append(a_temp, b_temp[:,:inl,:], axis=1)
                else:
                    if a_tile.Version == 1:
                        inl = np.shape(a_temp)[2] - 1
                    if a_tile.Version == 2:
                        inl = np.shape(a_temp)[2]
                    temp_3d = np.append(a_temp[:,:,:inl], b_temp, axis=2)
                setattr(self, var, temp_3d)
                a_temp = None
                b_temp = None
                temp_3d = None
                index += 1
        return inl, index

###################################################
#Independent MMM-Py functions follow
###################################################

def stitch_mosaic_tiles(map_array=None, direction=None, verbose=False):

    """
    Standalone function for stitching.
    Interprets map_array, a 1- or 2-rank list containing MosaicTile class
    instances arranged in proper locations. If there is not a tile 
    (determined by checking for the mrefl3d attribute), then no attempt will be 
    made to stitch. This method does some simple error checks to avoid problems
    but the user is primarily responsible for not inputing garbage.
    After that, it will iteratively call MosaicStitch.stitch_ns() and
    MosaicStitch.stitch_we() to do the heavy lifting.
    direction = Expected to be a string showing which direction to stitch
                (we or ns), and is only used if this function is sent a
                1-D map_array.
    Returns a MosaicStitch() class containing every tile provided in map_array,
    with each tile located where map_array told it to be.
    This function is standalone due to weird errors occurring if it was
    part of the MosaicStitch() class.
    
    """

    method_name = 'stitch_mosaic_tiles'
    if verbose:
        _method_header_printout(method_name)
    
    #1-D stitching, either N-S or W-E
    if np.rank(map_array) == 1: 

        #direction unset or not a string = direction fail
        if direction == None or isinstance(direction, str) == False:
            _print_direction_fail(method_name)
            return
        #E-W Stitching only
        if direction.upper() == 'EW' or direction.upper() == 'E' or\
                    direction.upper() == 'W' or direction.upper() == 'WE':
            result = _stitch_1d_array_we(map_array, verbose)
            if result == None:
                return
        #N-S Stitching only
        elif direction.upper() == 'NS' or direction.upper() == 'N' or\
                    direction.upper() == 'S' or direction.upper() == 'SN':
            result = _stitch_1d_array_ns(map_array, verbose)
            if result == None:
                return
        #everything else = direction fail
        else:
            _print_direction_fail(method_name)
            return

    #map_array fail
    elif np.rank(map_array) < 1 or np.rank(map_array) > 2:
        print method_name+'(): map_array is not right,',\
              'use 1- or 2-rank array'
        return

    #2-D stitching in N-S and W-E
    else:
        if verbose:
            print 'Sent a 2-D matrix, attempting to stitch in both',\
                  'N-S and E-W directions'

        #Tile number fail
        if not _right_number_of_tiles(map_array):
            return

        #Actual stitching done here
        result = _stitch_2d_array(map_array, verbose)
        if not result:
            return

        if verbose:
            _print_method_done()
            _method_footer_printout()

    return result

def compute_grid_attributes(dz3d, lat, lon, height):
    
    """
    Sent 3-D reflectivity array, 2-D lat/lon arrays and 1-D height arrays,
    compute the volume of each grid cell. Assumes km. Set as independent
    function so that it is easier for other programs to use it for their
    specific grids. For MosaicTiles and Stiches, send it the mrefl3d, Latitude,
    Longitude, and Height attributes. Assumes constant grid spacing in horiz.
    Returns volumes of 3-D grid cells (km**3) and areas of 2-D grid cells (km**2)
    
    """
    re = 6371.1 #km
    vol = 0.0 * dz3d
    latdel = np.abs(lat[0,0]-lat[1,0])
    londelr = np.abs(lon[0,1]-lon[0,0]) * np.pi/180.0
    sa = 0.0 * lat
    for j in xrange(len(lat[:,0])):
        th1 = np.deg2rad(90.0 + lat[j,0] - latdel/2.0)
        th2 = np.deg2rad(90.0 + lat[j,0] + latdel/2.0)
        sa[j,:] = re**2 * londelr * (np.cos(th1)-np.cos(th2))
        for k in xrange(len(height)):
            if k == 0:
                hdel = height[k]
            else:
                hdel = height[k] - height[k-1]
            vol[k,j,:] = hdel * sa[j,0]
    return vol, sa

def epochtime_to_string(epochtime=None, use_second=False):

    """
    Given an epoch time (seconds since 1/1/1970), return a string useful
    as a plot title. Set the use_second flag to also include seconds in
    the string.
    
    """
    if epochtime:
        if use_second:
            time_string = time.strftime('%m/%d/%Y %H:%M:%S UTC',
                                        time.gmtime(epochtime))
        else:
            time_string = time.strftime('%m/%d/%Y %H:%M UTC',
                                        time.gmtime(epochtime))
    else:
        time_string=''
    return time_string

###################################################
#MMM-Py internal functions below
###################################################

def _right_number_of_tiles(map_array=None):
    num_tiles = np.shape(map_array)[0] * np.shape(map_array)[1]
    if num_tiles % 2 != 0 or num_tiles > 8 or num_tiles <= 0:
        _print_wrong_number_of_tiles('stitch_mosaic_tiles')
        return False
    else:
        return True

def _stitch_1d_array_we(map_array=None, verbose=False,
                        method_name='stitch_mosaic_tiles'):
    result = MosaicStitch()
    if verbose:
        print 'Sent a 1-D matrix, attempting to stitch in E-W directions'
    for i in xrange(np.shape(map_array)[0]):
        if not hasattr(map_array[i], DEFAULT_VAR):
            _print_missing_a_tile(method_name)
            return False
        if i == 0:
            result.stitch_we(w_tile=map_array[i], e_tile=map_array[i+1],
                             verbose=verbose)
        if 1 <= i < np.shape(map_array)[0]-1:
            result.stitch_we(w_tile=result, e_tile=map_array[i+1],
                             verbose=verbose)
    if verbose:
        _print_method_done()
        _method_footer_printout()
    return result

def _stitch_1d_array_ns(map_array=None, verbose=False,
                        method_name='stitch_mosaic_tiles'):
    result = MosaicStitch()
    if verbose:
        print 'Sent a 1-D matrix, attempting to stitch in N-S directions'
    for i in xrange(np.shape(map_array)[0]):
        if not hasattr(map_array[i], DEFAULT_VAR):
            _print_missing_a_tile(method_name)
            return False
        if i == 0:
            result.stitch_ns(n_tile=map_array[i], s_tile=map_array[i+1],
                             verbose=verbose)
        if 1 <= i < np.shape(map_array)[0]-1:
            result.stitch_ns(n_tile=result, s_tile=map_array[i+1],
                             verbose=verbose)
    if verbose:
        _print_method_done()
        _method_footer_printout()
    return result

def _stitch_2d_array(map_array=None, verbose=False,
                     method_name='stitch_mosaic_tiles'):
    """Pass thru first doing N-S stitches, then E-W after"""
    ns_stitches=[]
    for i in xrange(np.shape(map_array)[1]):
        if not hasattr(map_array[0][i],DEFAULT_VAR) or \
           not hasattr(map_array[1][i],DEFAULT_VAR):
            _print_missing_a_tile(method_name)
            return False
        tmp = MosaicStitch()
        ns_stitches.append(tmp)
        ns_stitches[i].stitch_ns(n_tile=map_array[0][i],
                                 s_tile=map_array[1][i], verbose=verbose)
        if i == 0:
            result = ns_stitches[i]
        if i >= 1:
            result.stitch_we(w_tile=result, e_tile=ns_stitches[i],
                             verbose=verbose)
    return result

###################################################

def _method_header_printout(method_name=None):
    print
    print '********************'
    print method_name + '():'

def _method_footer_printout():
    print '********************'
    print

def _print_direction_fail(method_name=None):
    print method_name + '(): Sent a 1-D array but no direction to stitch!'
    print 'Use direction=\'we\' or direction=\'ns\' in argument to fix'

def _print_missing_a_tile(method_name=None):
    print method_name + '(): Missing a tile, fix and try again'

def _print_wrong_number_of_tiles(method_name=None):
    print method_name + '(): Wrong number of tiles, fix and try again'

def _print_method_done():
    print 'Task complete, data contained as class attributes'
    print 'Use dir(), help(), or __dict__ to find out what is available'

def _print_method_called_incorrectly(method_name=None):
    print method_name + '(): Method called incorrectly, check syntax'

def _print_variable_does_not_exist(method_name=None, var=DEFAULT_VAR):
    print method_name+'():', var, 'does not exist, try reading in a file'

def _fill_list(f, size, offset):
    _list = []
    for i in xrange(size):
        f.seek(i*4+offset)
        _list.append(unpack(ENDIAN+INTEGER, f.read(4))[0])
    return _list

def _are_equal(num1, num2):
    if np.abs(num1-num2) < 0.001:
        return True
    else:
        return False

###################################################

###################################################

###################################################


