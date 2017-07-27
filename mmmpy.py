"""
Title/Version
-------------
Marshall MRMS Mosaic Python Toolkit (MMM-Py)
mmmpy v1.6
Developed & tested with Python 2.7 & 3.4
Last changed 05/23/2017


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
mosaic reflectivities on a national 3D grid. Simple diagnostics and plotting
(via the MosaicDisplay class), as well as computation of composite
reflectivity, are available. A child class, MosaicStitch, is also defined.
This can be populated with stitched-together MosaicTiles. To access these
classes, add the following to your program and then make sure the path to
this script is in your PYTHONPATH:
import mmmpy


Notes
-----
Dependencies: numpy, time, os, matplotlib, Basemap, struct,
calendar, gzip, netCDF4, six, __future__, datetime
Optional: pygrib
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import datetime
from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset
from struct import unpack
import os
import time
import calendar
import gzip
import six
try:
    import pygrib
    IMPORT_FLAG = True
except ImportError:
    IMPORT_FLAG = False

VERSION = '1.6'

# Hard coding of constants
DEFAULT_CLEVS = np.arange(15) * 5.0
DEFAULT_VAR = 'mrefl3d'
DEFAULT_VAR_LABEL = 'Reflectivity (dBZ)'
V1_DURATION = 300.0  # seconds
V2_DURATION = 120.0  # seconds
ALTITUDE_SCALE_FACTOR = 1000.0  # Divide meters by this to get something else
DEFAULT_CMAP = cm.GMT_wysiwyg
DEFAULT_PARALLELS = 10  # [20, 37.5, 40, 55]
DEFAULT_MERIDIANS = 10  # [230, 250, 265, 270, 280, 300]
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
DEFAULT_LINEWIDTH = 0.1

# Following is relevant to MRMS binary format read/write methods
ENDIAN = ''  # Endian currently set automatically by machine type
INTEGER = 'i'
DEFAULT_VALUE_SCALE = 10
DEFAULT_DXY_SCALE = 100000
DEFAULT_Z_SCALE = 1
DEFAULT_MAP_SCALE = 1000
DEFAULT_MISSING_VALUE = -99
DEFAULT_MRMS_VARNAME = b'mosaicked_refl1     '  # 20 characters
DEFAULT_MRMS_VARUNIT = b'dbz   '  # 6 characters
DEFAULT_FILENAME = './mrms_binary_file.dat.gz'

# Following is relevant to MRMS grib2 format read/write
BASE_PATH = '/Users/tjlang/Downloads'
TMPDIR = BASE_PATH + '/tmpdir/'
WGRIB2_PATH = BASE_PATH + '/grib2/wgrib2/'
WGRIB2_NAME = 'wgrib2'
MRMS_V3_LATRANGE = [20.0, 55.0]
MRMS_V3_LONRANGE = [-130.0, -60.0]

# v1/v2 changeover occurred on 07/30/2013 around 1600 UTC (epoch = 1375200000)
# See 'https://docs.google.com/document/d/' +
# '1Op3uETOtd28YqZffgvEGoIj0qU6VU966iT_QNUOmqn4/edit'
# for details (doc claims 14 UTC, but CSU has v1 data thru 1550 UTC)
V1_TO_V2_CHANGEOVER_EPOCH_TIME = 1375200000

###################################################
# MosaicTile class
###################################################


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

    def __init__(self, filename=None, verbose=False, wgrib2_path=WGRIB2_PATH,
                 keep_nc=True, wgrib2_name=WGRIB2_NAME, nc_path=TMPDIR,
                 latrange=None, lonrange=None):
        """
        If initialized with a filename (incl. path), will call
        read_mosaic_netcdf() to populate the class instance.
        If not, it simply instances the class but does not populate
        its attributes.
        filename: Full path and filename of file.
        verbose: Set to True for text output. Useful for debugging.
        Other keywords are described in read_mosaic_grib() method.
        """
        if filename is None:
            return

        if not isinstance(filename, six.string_types):
            self.read_mosaic_grib(filename, verbose=verbose,
                                  wgrib2_path=wgrib2_path, keep_nc=keep_nc,
                                  wgrib2_name=wgrib2_name, nc_path=nc_path,
                                  latrange=latrange, lonrange=lonrange)
        else:
            try:
                flag = self.read_mosaic_binary(filename, verbose=verbose)
                if not flag:
                    flag = self.read_mosaic_netcdf(filename, verbose=verbose)
                    if not flag:
                        try:
                            self.read_mosaic_grib(
                                [filename], verbose=verbose,
                                wgrib2_path=wgrib2_path, keep_nc=keep_nc,
                                wgrib2_name=wgrib2_name, nc_path=nc_path,
                                latrange=latrange, lonrange=lonrange)
                        except:
                            print('Unknown file format, nothing read')
            except:
                print('No valid filename provided')

    def help(self):
        """Basic printout of module capabilities"""
        _method_header_printout('help')
        print('To use: instance = MosaicTile(filepath+name).')
        print('Available read methods:')
        print('    read_mosaic_netcdf(<FILE>):')
        print('    read_mosaic_binary(<FILE>):')
        print('    read_mosaic_grib(<FILE(S)>):')
        print('Other available methods:')
        print('diag(), get_comp(),')
        print('subsection(), write_mosaic_binary(), output_composite()')
        print('To plot: display = MosaicDisplay(tile_instance)')
        print('Available plotting methods: plot_horiz(), plot_vert(),')
        print('                            three_panel_plot()')
        _method_footer_printout()

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
            print(method_name+'(): Reading', full_path_and_filename)
        try:
            fileobj = Dataset(full_path_and_filename, 'r')
        except:
            if verbose:
                print('Not an MRMS netcdf file')
                _method_footer_printout()
            return False
        # Get data and metadata
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
        if self.Version is None:
            del self.Version
            if verbose:
                print('read_mosaic_netcdf(): Unknown MRMS version, not read')
                _method_footer_printout()
            return False
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
        # Fix for v1 MRMS NetCDFs produced by mrms_to_CFncdf from v1 binaries
        # These look like v2 to mmmpy, and thus could impact stitching
        # as v1 tiles overlapped slightly and v2 tiles don't
        if self.Version == 2 and self.Time < V1_TO_V2_CHANGEOVER_EPOCH_TIME:
            self.Version = 1
            self.Duration = V1_DURATION
        self._get_tile_number()
        if verbose:
            _print_method_done()
            _method_footer_printout()
        return True

    def read_mosaic_binary(self, full_path_and_filename, verbose=False):
        """
Reads gzipped MRMS binary files and populates MosaicTile fields.
Attempts to distinguish between v1 (<= 7/30/2013) and v2 (>= 7/30/2013)
mosaics.
Major reference:
ftp://ftp.nssl.noaa.gov/users/langston/MRMS_REFERENCE/MRMS_BinaryFormat.pdf
        """
        if verbose:
            begin_time = time.time()
            _method_header_printout('read_mosaic_binary')
            print('Reading', full_path_and_filename)
        # Check to see if a real MRMS binary file
        if full_path_and_filename[-3:] == '.gz':
            f = gzip.open(full_path_and_filename, 'rb')
        else:
            f = open(full_path_and_filename, 'rb')
        try:
            self.Time = calendar.timegm(1*np.array(_fill_list(f, 6, 0)))
        except:
            if verbose:
                print('Not an MRMS binary file')
                _method_footer_printout()
            return False
        if self.Time >= V1_TO_V2_CHANGEOVER_EPOCH_TIME:
            self.Version = 2
            self.Duration = V2_DURATION
        else:
            self.Version = 1
            self.Duration = V1_DURATION
        self.Variables = [DEFAULT_VAR]
        self.Filename = os.path.basename(full_path_and_filename)
        # Get dimensionality from header, use to define datatype
        f.seek(24)
        self.nlon, self.nlat, self.nz = unpack(ENDIAN+3*INTEGER, f.read(12))
        f.seek(80 + self.nz*4 + 78)
        NR, = unpack(ENDIAN+INTEGER, f.read(4))
        dt = self._construct_dtype(NR)
        # Rewind and then read everything into the pre-defined datatype.
        # np.fromstring() nearly 3x faster performance than struct.unpack()!
        f.seek(0)
        fileobj = np.fromstring(f.read(80 + 4*self.nz + 82 + 4*NR +
                                2*self.nlon*self.nlat*self.nz), dtype=dt)
        f.close()
        # Populate Latitude, Longitude, and Height
        self.StartLon = 1.0 * fileobj['StartLon'][0] / fileobj['map_scale'][0]
        self.StartLat = 1.0 * fileobj['StartLat'][0] / fileobj['map_scale'][0]
        self.LonGridSpacing = 1.0 * fileobj['dlon'][0] /\
            fileobj['dxy_scale'][0]
        self.LatGridSpacing = 1.0 * fileobj['dlat'][0] /\
            fileobj['dxy_scale'][0]
        # Note the subtraction in lat!
        lat = self.StartLat - self.LatGridSpacing * np.arange(self.nlat)
        lon = self.StartLon + self.LonGridSpacing * np.arange(self.nlon)
        self.Longitude, self.Latitude = np.meshgrid(lon, lat)
        self._get_tile_number()
        self.Height = 1.0 * fileobj['Height'][0] / fileobj['z_scale'][0] /\
            ALTITUDE_SCALE_FACTOR
        if self.nz == 1:
            self.Height = [self.Height]  # Convert to array for compatibility
        # Actually populate the mrefl3d data, need to reverse Latitude axis
        data3d = 1.0 * fileobj['data3d'][0] / fileobj['var_scale'][0]
        data3d[:, :, :] = data3d[:, ::-1, :]
        setattr(self, DEFAULT_VAR, data3d)
        # Done!
        if verbose:
            print(time.time()-begin_time, 'seconds to complete')
            _method_footer_printout()
        return True

    def read_mosaic_grib(self, filename, wgrib2_path=WGRIB2_PATH, keep_nc=True,
                         wgrib2_name=WGRIB2_NAME, verbose=False,
                         nc_path=TMPDIR, latrange=None, lonrange=None):
        """
        Method that is capable of reading grib2-format MRMS mosaics.
        Relies on MosaicGrib and NetcdfFile classes to do the heavy lifting.
        This method is mainly concerned with taking their output and creating
        a properly formatted MosaicTile object.

        Arguments/Keywords (passed to MosaicGrib)
        -----------------------------------------
        filename = Single string or list of strings, can be for grib2 or
                   netCDFs created by wgrib2
        wgrib2_path = Path to wgrib2 executable
        wgrib2_name = Name of wgrib2 executable
        keep_nc = Set to False to erase netCDFs created by wgrib2
        verbose = Set to True to get text updates on progress
        nc_path = Path to directory where netCDFs will be created
        lat/lonrange = 2-element lists used to subsection grib data
                       before ingest
        """
        if verbose:
            begin_time = time.time()
            _method_header_printout('read_mosaic_grib')
        self.Tile = '?'  # MRMS grib2 covers entire contiguous US
        self.Version = 3
        self.Duration = V2_DURATION  # MRMS grib2 timing still every 2 min
        self.Filename = os.path.basename(filename[0]) if not \
            isinstance(filename, six.string_types) else \
            os.path.basename(filename)
        self.Variables = [DEFAULT_VAR]
        gribfile = MosaicGrib(filename, wgrib2_path=wgrib2_path,
                              keep_nc=keep_nc,
                              wgrib2_name=wgrib2_name, verbose=verbose,
                              nc_path=nc_path, latrange=latrange,
                              lonrange=lonrange)
        # MosaicGrib objects have very similar attributes to MosaicTiles
        varlist = [DEFAULT_VAR, 'Latitude', 'Longitude', 'StartLat',
                   'StartLon', 'LatGridSpacing', 'LonGridSpacing', 'Time',
                   'nlon', 'nlat', 'nz', 'Height']
        for var in varlist:
            setattr(self, var, getattr(gribfile, var))
        if verbose:
            _method_footer_printout()

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
                print('Computing composite field')
        temp_3d = getattr(self, var)
        temp_comp = np.amax(temp_3d, axis=0)
        setattr(self, var+'_comp', temp_comp)
        if verbose:
            _method_footer_printout()

    def diag(self, verbose=False):
        """
        Prints out diagnostic information and produces
        a basic plot of tile/stitch composite reflectivity.
        """
        _method_header_printout('diag')
        if not hasattr(self, DEFAULT_VAR):
            print(DEFAULT_VAR, 'does not exist, try reading in a file')
            _method_footer_printout()
            return
        print('Printing basic metadata and making a simple plot')
        print('Data are from', self.Filename)
        print('Min, Max Latitude =',  np.min(self.Latitude),
              np.max(self.Latitude))
        print('Min, Max Longitude =', np.min(self.Longitude),
              np.max(self.Longitude))
        print('Heights (km) =', self.Height)
        print('Grid shape =', np.shape(self.mrefl3d))
        print('Now plotting ...')
        display = MosaicDisplay(self)
        display.plot_horiz(verbose=verbose)
        print('Done!')
        _method_footer_printout()

    def write_mosaic_binary(self, full_path_and_filename=None, verbose=False):
        """
        Major reference:
ftp://ftp.nssl.noaa.gov/users/langston/MRMS_REFERENCE/MRMS_BinaryFormat.pdf
        Note that user will need to keep track of Endian for machine used to
        write the file. MMM-Py's ENDIAN global variable may need to be adjusted
        if reading on a different Endian machine than files were produced.
        You can write out a subsectioned or a stitched mosaic and it will
        be readable by read_mosaic_binary().
        full_path_and_filename = Filename (including path).
                                 Include the .gz suffix.
        verbose = Set to True to get some text response.
        """
        if verbose:
            _method_header_printout('write_mosaic_binary')
            begin_time = time.time()
        if full_path_and_filename is None:
            full_path_and_filename = DEFAULT_FILENAME
        elif full_path_and_filename[-3:] != '.gz':
            full_path_and_filename += '.gz'
        if verbose:
            print('Writing MRMS binary format to', full_path_and_filename)
        header = self._construct_header()
        data1d = self._construct_1d_data()
        output = gzip.open(full_path_and_filename, 'wb')
        output.write(header+data1d.tostring())
        output.close()
        if verbose:
            print(time.time() - begin_time, 'seconds to complete')
            _method_footer_printout()

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
                print('Latitude Range to Keep =', latrange)
            else:
                print('No subsectioning in Latitude')
            if lonrange and np.size(lonrange) == 2:
                print('Longitude Range to Keep =', lonrange)
            else:
                print('No subsectioning in Longitude')
            if zrange and np.size(zrange) == 2:
                print('Height Range to Keep =', zrange)
            else:
                print('No subsectioning in Height')
        self._subsection_in_latitude(latrange)
        self._subsection_in_longitude(lonrange)
        self._subsection_in_height(zrange)
        if verbose:
            _method_footer_printout()

    def output_composite(self, full_path_and_filename=DEFAULT_FILENAME,
                         var=DEFAULT_VAR, verbose=False):
        """
        Produces a gzipped binary file containing only a composite of
        the chosen variable. The existing tile now will only consist
        of a single vertical level (e.g., composite reflectivity)
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
                print(var+'_comp does not exist,',
                      'computing it with get_comp()')
            self.get_comp(var=var, verbose=verbose)
        self.subsection(zrange=[self.Height[0], self.Height[0]],
                        verbose=verbose)
        temp2d = getattr(self, var+'_comp')
        temp3d = getattr(self, var)
        temp3d[0, :, :] = temp2d[:, :]
        setattr(self, var, temp3d)
        self.write_mosaic_binary(full_path_and_filename, verbose)
        if verbose:
            _method_footer_printout()

    def _populate_v1_specific_data(self, fileobj=None, label='mrefl_mosaic'):
        """v1 MRMS netcdf data file"""
        self.StartLat = fileobj.Latitude
        self.StartLon = fileobj.Longitude
        self.Height = fileobj.variables['Height'][:] / ALTITUDE_SCALE_FACTOR
        self.Time = np.float64(fileobj.Time)
        self.Duration = V1_DURATION
        ScaleFactor = fileobj.variables[label].Scale
        self.mrefl3d = fileobj.variables[label][:, :, :] / ScaleFactor
        # Note the subtraction in lat!
        lat = self.StartLat - self.LatGridSpacing * np.arange(self.nlat)
        lon = self.StartLon + self.LonGridSpacing * np.arange(self.nlon)
        self.Variables = [DEFAULT_VAR]
        return lat, lon

    def _populate_v2_specific_data(self, fileobj=None, label='MREFL'):
        """v2 MRMS netcdf data file"""
        self.Height = fileobj.variables['Ht'][:] / ALTITUDE_SCALE_FACTOR
        # Getting errors w/ scipy 0.14 when np.array() not invoked below.
        # Think it was not properly converting from scipy netcdf object.
        # v1 worked OK because of the ScaleFactor division in
        # _populate_v1_specific_data().
        self.mrefl3d = np.array(fileobj.variables[label][:, :, :])
        lat = fileobj.variables['Lat'][:]
        lon = fileobj.variables['Lon'][:]
        self.StartLat = lat[0]
        self.StartLon = lon[0]
        self.Time = fileobj.variables['time'][0]
        self.Duration = V2_DURATION
        self.Variables = [DEFAULT_VAR]
        return lat, lon

    def _construct_dtype(self, NR=1):
        """
        This is the structure of a complete binary MRMS file.
        This method breaks in Python 2.7 if you import the
        unicode_literals module from __future__. However,
        the method works fine under Python 3.4 as written.
        """
        dt = np.dtype(
              [('year', 'i4'), ('month', 'i4'), ('day', 'i4'),
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
               ('data3d', ('i2', (self.nz, self.nlat, self.nlon)))])
        return dt

    def _get_tile_number(self):
        """Returns tile number as a string based on starting lat/lon"""
        self.Tile = '?'
        if self.Version == 1:
            if _are_equal(self.StartLat, 55.0) and \
               _are_equal(self.StartLon, -130.0):
                self.Tile = '1'
            elif (_are_equal(self.StartLat, 55.0) and
                  _are_equal(self.StartLon, -110.0)):
                self.Tile = '2'
            elif (_are_equal(self.StartLat, 55.0) and
                  _are_equal(self.StartLon, -90.0)):
                self.Tile = '3'
            elif (_are_equal(self.StartLat, 55.0) and
                  _are_equal(self.StartLon, -80.0)):
                self.Tile = '4'
            elif (_are_equal(self.StartLat, 40.0) and
                  _are_equal(self.StartLon, -130.0)):
                self.Tile = '5'
            elif (_are_equal(self.StartLat, 40.0) and
                  _are_equal(self.StartLon, -110.0)):
                self.Tile = '6'
            elif (_are_equal(self.StartLat, 40.0) and
                  _are_equal(self.StartLon, -90.0)):
                self.Tile = '7'
            elif (_are_equal(self.StartLat, 40.0) and
                  _are_equal(self.StartLon, -80.0)):
                self.Tile = '8'
        elif self.Version == 2:
            if _are_equal(self.StartLat, 54.995) and \
               _are_equal(self.StartLon, -129.995):
                self.Tile = '1'
            elif (_are_equal(self.StartLat, 54.995) and
                  _are_equal(self.StartLon, -94.995)):
                self.Tile = '2'
            elif (_are_equal(self.StartLat, 37.495) and
                  _are_equal(self.StartLon, -129.995)):
                self.Tile = '3'
            elif (_are_equal(self.StartLat, 37.495) and
                  _are_equal(self.StartLon, -94.995)):
                self.Tile = '4'

    def _construct_header(self):
        """This is the structure of the header of a binary MRMS file"""
        nr = np.int32(1).tostring()
        rad_name = b'none'
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
        Height = np.int32(self.Height * DEFAULT_Z_SCALE *
                          ALTITUDE_SCALE_FACTOR).tostring()  # km to m
        dlon = np.int32(self.LonGridSpacing * DEFAULT_DXY_SCALE).tostring()
        dlat = np.int32(self.LatGridSpacing * DEFAULT_DXY_SCALE).tostring()
        # Set depreciated and placeholder values.
        # Don't think exact number matters, but for now set as same values
        # obtained from MREF3D33L_tile2.20140619.010000.gz
        deprec1 = np.int32(538987596).tostring()
        deprec2 = np.int32(30000).tostring()
        deprec3 = np.int32(60000).tostring()
        deprec4 = np.int32(-60005).tostring()
        deprec5 = np.int32(1000).tostring()
        ph = 0 * np.arange(10) + 19000
        placeholder = np.int32(ph).tostring()  # 10 placeholder values
        header = b''.join([year, month, day, hour, minute, second, nlon,
                           nlat, nz, deprec1, map_scale,
                           deprec2, deprec3, deprec4, StartLon, StartLat,
                           deprec5, dlon, dlat, dxy_scale, Height, z_scale,
                           placeholder, VarName, VarUnit, var_scale,
                           missing, nr, rad_name])
        return header

    def _construct_1d_data(self):
        """
        Turns a 3D float mosaic into a 1-D short array suitable for writing
        to a binary file
        """
        data1d = DEFAULT_VALUE_SCALE * getattr(self, DEFAULT_VAR)
        # MRMS binaries have the Latitude axis flipped
        data1d[:, :, :] = data1d[:, ::-1, :]
        data1d = data1d.astype(np.int16)
        data1d = data1d.ravel()
        return data1d

    def _subsection_in_latitude(self, latrange=None):
        if latrange and np.size(latrange) == 2:
            temp = self.Latitude[:, 0]
            condition = np.logical_or(temp < np.min(latrange),
                                      temp > np.max(latrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nlat:
                print('Refusing to delete all data in Latitude')
            else:
                self.Latitude = np.delete(self.Latitude, indices[0], axis=0)
                self.Longitude = np.delete(self.Longitude, indices[0], axis=0)
                self.nlat = self.nlat - np.size(indices[0])
                self.StartLat = np.max(self.Latitude)
                self._subsection_data3d(indices, 1)

    def _subsection_in_longitude(self, lonrange=None):
        if lonrange and np.size(lonrange) == 2:
            temp = self.Longitude[0, :]
            condition = np.logical_or(temp < np.min(lonrange),
                                      temp > np.max(lonrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nlon:
                print('Refusing to delete all data in Longitude')
            else:
                self.Latitude = np.delete(self.Latitude, indices[0], axis=1)
                self.Longitude = np.delete(self.Longitude, indices[0], axis=1)
                self.nlon = self.nlon - np.size(indices[0])
                self.StartLon = np.min(self.Longitude)
                self._subsection_data3d(indices, 2)

    def _subsection_in_height(self, zrange=None):
        if zrange and np.size(zrange) == 2:
            temp = self.Height[:]
            condition = np.logical_or(temp < np.min(zrange),
                                      temp > np.max(zrange))
            indices = np.where(condition)
            if np.size(indices[0]) >= self.nz:
                print('Refusing to delete all data in Height')
            else:
                self.Height = np.delete(self.Height, indices[0], axis=0)
                self.nz = self.nz - np.size(indices[0])
                self._subsection_data3d(indices, 0)

    def _subsection_data3d(self, indices, axis):
        for var in self.Variables:
            temp3d = getattr(self, var)
            temp3d = np.delete(temp3d, indices[0], axis=axis)
            setattr(self, var, temp3d)

###################################################
# NetcdfFile class
###################################################


class NetcdfFile(object):

    """
    Reads a given netCDF file and populates its attributes with the file's
    variables. Also adds a variable_list attribute which lists all the
    file-specific attributes contained by the object. Uses netCDF4 module's
    Dataset object.
    """

    def __init__(self, filename=None):
        self.read_netcdf(filename)

    def read_netcdf(self, filename):
        """variable_list = holds all the variable key strings """
        volume = Dataset(filename, 'r')
        self.filename = os.path.basename(filename)
        self.fill_variables(volume)

    def fill_variables(self, volume):
        """Loop thru all variables and store them as attributes"""
        self.variable_list = []
        for key in volume.variables.keys():
            new_var = np.array(volume.variables[key][:])
            setattr(self, key, new_var)
            self.variable_list.append(key)

###################################################
# MosaicGrib class
###################################################


class MosaicGrib(object):

    """
    This is an intermediary class that assists with reading MRMS grib2 files.
    It utilizes wgrib2 to create netCDFs from MRMS grib2 files. Then it
    reads the netCDFs using NetcdfFile class and consolidates all the levels
    into a single object that is similar to MosaicTile in terms of attributes.
    """

    def __init__(self, file_list, wgrib2_path=WGRIB2_PATH, keep_nc=True,
                 wgrib2_name=WGRIB2_NAME, verbose=False, nc_path=TMPDIR,
                 latrange=None, lonrange=None):
        """
        file_list = Single string or list of strings, can be for grib2
                    or netCDFs created by wgrib2
        wgrib2_path = Path to wgrib2 executable
        wgrib2_name = Name of wgrib2 executable
        keep_nc = Set to False to erase netCDFs created by wgrib2
        verbose = Set to True to get text updates on progress
        nc_path = Path to directory where netCDFs will be created
        lat/lonrange = 2-element lists used to subsection grib data
                       before ingest
        """
        if not isinstance(file_list, six.string_types):
            self.read_grib_list(file_list, wgrib2_path=wgrib2_path,
                                keep_nc=keep_nc, wgrib2_name=wgrib2_name,
                                verbose=verbose, nc_path=nc_path,
                                latrange=latrange, lonrange=lonrange)
        else:
            self.read_grib_list([file_list], wgrib2_path=wgrib2_path,
                                keep_nc=keep_nc, wgrib2_name=wgrib2_name,
                                verbose=verbose, nc_path=nc_path,
                                latrange=latrange, lonrange=lonrange)

    def read_grib_list(self, file_list, wgrib2_path=WGRIB2_PATH, keep_nc=True,
                       wgrib2_name=WGRIB2_NAME, verbose=False, nc_path=TMPDIR,
                       latrange=None, lonrange=None):
        """
        Actual reading of grib2 and netCDF files occurs here.
        Input arguments and keywords same as __init__() method.
        Now capable of ingesting grib2 directly via pygrib.
        """
        if verbose:
            begin_time = time.time()
        # Make the directory where netCDFs will be stored
        os.system('mkdir ' + TMPDIR)
        tmpf = nc_path + 'default.grib2'
        nclist = []
        gblist = []
        for grib in file_list if not isinstance(file_list, six.string_types)\
                else [file_list]:
            try:
                # See if passed netCDFs already created by wgrib2
                nc = NetcdfFile(grib)
                nclist.append(nc)
            except:
                # Attempt to read grib2
                # Can try to decompress if gzipped
                gzip_flag = False
                if grib[-3:] == '.gz':
                    os.system('gzip -d ' + grib)
                    grib = grib[0:-3]
                    gzip_flag = True
                # wgrib2 call is made via os.system()
                gribf = os.path.basename(grib)
                if IMPORT_FLAG:
                    if verbose:
                        print('Reading', gribf)
                    gr = pygrib.open(grib)
                    gblist.append(gr)
                else:
                    if latrange is None and lonrange is None:
                        command = wgrib2_path + wgrib2_name + ' ' + grib + \
                            ' -netcdf ' + nc_path+gribf + '.nc'
                    # Subsectioning before reading
                    else:
                        if latrange is None and lonrange is not None:
                            latrange = MRMS_V3_LATRANGE
                        elif latrange is not None and lonrange is None:
                            lonrange = MRMS_V3_LONRANGE
                        slat = self.convert_array_to_string(latrange)
                        slon = self.convert_array_to_string(
                            np.array(lonrange) + 360.0)
                        command = wgrib2_path + wgrib2_name + ' ' + grib + \
                            ' -small_grib ' + slon + ' ' + slat + ' ' + \
                            tmpf + '; ' + wgrib2_path + wgrib2_name + ' ' + \
                            tmpf + ' -netcdf ' + nc_path+gribf + '.nc; ' + \
                            'rm -f ' + tmpf
                    if verbose:
                        print('>>>> ', command)
                    os.system(command)
                    # Here the output netCDF is actually read
                    nclist.append(NetcdfFile(nc_path + gribf + '.nc'))
                    if not keep_nc:
                        os.system('rm -f ' + nc_path + gribf + '.nc')
                if gzip_flag:
                    os.system('gzip ' + grib)
        if IMPORT_FLAG:
            self.gblist = gblist
            self.format_grib_data()
        else:
            self.nclist = nclist
            self.format_netcdf_data()
        if verbose:
            print('MosaicGrib:', time.time() - begin_time, 'seconds to run')

    def convert_array_to_string(self, array):
        return str(np.min(array)) + ':' + str(np.max(array))

    def get_height_from_name(self, name):
        """
        Given a reflectivity variable name, get height information.
        Works on CONUS and CONUSPlus MRMS mosaics.
        """
        if name[0:9] == 'CONUSPlus':
            index = 30
        else:
            index = 26
        str_height = name[index:index+5]
        if str_height[4] != '0':
            str_height = str_height[0:4]
        if str_height[3] != '0':
            str_height = str_height[0:3]
        return float(str_height) / ALTITUDE_SCALE_FACTOR

    def get_reflectivity_data(self, ncfile):
        """Grab 2D reflectivity data from CONUS* variable"""
        for var in ncfile.variable_list:
            if var[0:5] == 'CONUS':
                return getattr(ncfile, var)

    def format_grib_data(self):
        """
        This method takes a list of ingested grib files and formats the data
        to match the MMM-Py model.
        """
        height = []
        refstore = []
        for i, gr in enumerate(self.gblist):
            grb = gr[1]
            if i == 0:
                lat, lon = grb.latlons()
                self.Longitude = lon - 360.0
                self.Latitude = lat
                self.LatGridSpacing = np.round(np.abs(
                    self.Latitude[0, 0] - self.Latitude[1, 0]), decimals=2)
                self.LonGridSpacing = np.round(np.abs(
                    self.Longitude[0, 1] - self.Longitude[0, 0]), decimals=2)
                self.StartLat = np.max(self.Latitude)
                self.StartLon = np.min(self.Longitude)
                dtgrb = datetime.datetime.strptime(
                    str(grb['dataDate']) + str(grb['dataTime']), '%Y%m%d%H%M')
                self.Time = (dtgrb -
                             datetime.datetime(1970, 1, 1)).total_seconds()
                mrefl3d = np.zeros(
                    (np.size(self.gblist), np.shape(self.Latitude)[0],
                     np.shape(self.Longitude)[1]), dtype='float')
            refstore.append(1.0 * grb['values'])
            height.append(grb['level'] / 1000.0)
            gr.close()
        self.Height = np.array(height)[np.argsort(height)]
        for index in np.argsort(height):
            mrefl3d[index, :, :] = refstore[index][:, :]
        self.nz, self.nlat, self.nlon = np.shape(mrefl3d)
        setattr(self, DEFAULT_VAR, mrefl3d)
        del self.gblist

    def format_netcdf_data(self):
        """
        Method to group all the reflectivity 2D planes into a 3D array.
        Also populates attributes that will be necessary for MosaicTile.
        """
        height = []
        for nc in self.nclist:
            for var in nc.variable_list:
                if var[0:5] == 'CONUS':
                    height.append(self.get_height_from_name(var))
        height = np.array(height)
        mrefl3d = np.zeros((np.size(height),
                           np.size(self.nclist[0].latitude),
                           np.size(self.nclist[0].longitude)),
                           dtype='float')
        # Following should work even if files are randomly sorted in list
        for index in np.argsort(height):
            tmpdata = self.get_reflectivity_data(self.nclist[index])
            mrefl3d[index, :, :] = tmpdata[0, ::-1, :]  # Swap Latitude axis
        setattr(self, DEFAULT_VAR, mrefl3d)
        self.Height = height[np.argsort(height)]
        # For the following, assuming first file just like rest (e.g., time)
        self.Longitude, self.Latitude = np.meshgrid(
            self.nclist[0].longitude, self.nclist[0].latitude[::-1])
        self.StartLat = np.max(self.nclist[0].latitude)
        self.StartLon = np.min(self.nclist[0].longitude)
        self.nz, self.nlat, self.nlon = np.shape(self.mrefl3d)
        self.LatGridSpacing = np.abs(self.nclist[0].latitude[0] -
                                     self.nclist[0].latitude[1])
        self.LonGridSpacing = np.abs(self.nclist[0].longitude[0] -
                                     self.nclist[0].longitude[1])
        self.Time = self.nclist[0].time[0]

###################################################
# MosaicStitch class
###################################################


class MosaicStitch(MosaicTile):

    """
    Child class of MosaicTile().
    To create a new MosaicStitch instance:
    new_instance = MosaicStitch() or
    new_instance = stitch_mosaic_tiles(map_array=map_array,
                   direction=direction)
    """

    def __init__(self, verbose=False):
        """
        Initializes class instance but leaves it to other methods to
        populate the class attributes.
        """
        MosaicTile.__init__(self, verbose=verbose)

    def help(self):
        _method_header_printout('help')
        print('Call stitch_mosaic_tiles() function to populate variables')
        print('This standalone function makes use of this class\'s')
        print('stitch_ns() and stitch_we() methods')
        print('Instance = stitch_mosaic_tiles(map_array=map_array,',
              'direction=direction)')
        print('All MosaicTile() methods are available as well,')
        print('except read_mosaic_*() methods are disabled to avoid problems')
        _method_footer_printout()

    def read_mosaic_grib(self, filename, verbose=False):
        """Disabled in a MosaicStitch to avoid problems."""
        _method_header_printout('read_mosaic_grib')
        print('To avoid problems with reading individual MosaicTile classes')
        print('into a MosaicStitch, this method has been disabled')
        _method_footer_printout()

    def read_mosaic_netcdf(self, filename, verbose=False):
        """Disabled in a MosaicStitch to avoid problems."""
        _method_header_printout('read_mosaic_netcdf')
        print('To avoid problems with reading individual MosaicTile classes')
        print('into a MosaicStitch, this method has been disabled')
        _method_footer_printout()

    def read_mosaic_binary(self, filename, verbose=False):
        """Disabled in a MosaicStitch to avoid problems."""
        _method_header_printout('read_mosaic_binary')
        print('To avoid problems with reading individual MosaicTile classes')
        print('into a MosaicStitch, this method has been disabled')
        _method_footer_printout()

    def stitch_ns(self, n_tile=None, s_tile=None, verbose=False):
        """
        Stitches MosaicTile pair or MosaicStitch pair (or mixed pair)
        in N-S direction.
        """
        method_name = 'stitch_ns'
        if verbose:
            _method_header_printout(method_name)
        # Check to make sure method was called correctly
        if n_tile is None or s_tile is None:
            _print_method_called_incorrectly('stitch_ns')
            return
        # Check to make sure np.append() will not fail due to different grids
        if n_tile.nlon != s_tile.nlon:
            print(method_name + '(): Grid size in Longitude does not match,',
                  'fix this before proceeding')
            return
        inlat, index = self._stitch_radar_variables(n_tile, s_tile, verbose,
                                                    ns_flag=True)
        if index == 0:
            print(method_name + '(): No radar vars to stitch! Returning ...')
            return
        self._stitch_metadata(n_tile, s_tile, inlat, ns_flag=True)
        if verbose:
            print('Completed normally')
            _method_footer_printout()

    def stitch_we(self, w_tile=None, e_tile=None, verbose=False):
        """
        Stitches MosaicTile pair or MosaicStitch pair (or mixed pair)
        in W-E direction
        """
        method_name = 'stitch_we'
        if verbose:
            _method_header_printout(method_name)
        # Check to make sure method was called correctly
        if w_tile is None or e_tile is None:
            _print_method_called_incorrectly('stitch_we')
            return
        # Check to make sure np.append() will not fail due to different grids
        if w_tile.nlat != e_tile.nlat:
            print(method_name + '(): Grid size in Latitude does not match,',
                  'fix this before proceeding')
            return
        inlon, index = self._stitch_radar_variables(w_tile, e_tile,
                                                    ns_flag=False)
        if index == 0:
            print(method_name + '(): No radar vars to stitch! Returning ...')
            return
        self._stitch_metadata(w_tile, e_tile, inlon, ns_flag=False)
        if verbose:
            print('Completed normally')
            _method_footer_printout()

    def _stitch_metadata(self, a_tile=None, b_tile=None, index=None,
                         ns_flag=True):
        """
        ns_flag=True means N-S stitch, False means W-E stitch.
        Uses np.append() to stitch together.
        """
        if ns_flag:
            self.Longitude = np.append(a_tile.Longitude,
                                       b_tile.Longitude[:index, :], axis=0)
            self.Latitude = np.append(a_tile.Latitude,
                                      b_tile.Latitude[:index, :], axis=0)
            if a_tile.Version == 1:
                self.nlat = a_tile.nlat + b_tile.nlat - 1
            if a_tile.Version == 2:
                self.nlat = a_tile.nlat + b_tile.nlat
            self.nlon = a_tile.nlon
        else:
            self.Longitude = np.append(a_tile.Longitude[:, :index],
                                       b_tile.Longitude, axis=1)
            self.Latitude = np.append(a_tile.Latitude[:, :index],
                                      b_tile.Latitude, axis=1)
            if a_tile.Version == 1:
                self.nlon = a_tile.nlon + b_tile.nlon - 1
            if a_tile.Version == 2:
                self.nlon = a_tile.nlon + b_tile.nlon
            self.nlat = a_tile.nlat
        # Populate the other metadata attributes
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
                    print('Stitching', var)
                a_temp = 1.0 * getattr(a_tile, var)
                b_temp = 1.0 * getattr(b_tile, var)
                if ns_flag:
                    if a_tile.Version == 1:
                        inl = np.shape(b_temp)[1] - 1
                    if a_tile.Version == 2:
                        inl = np.shape(b_temp)[1]
                    temp_3d = np.append(a_temp, b_temp[:, :inl, :], axis=1)
                else:
                    if a_tile.Version == 1:
                        inl = np.shape(a_temp)[2] - 1
                    if a_tile.Version == 2:
                        inl = np.shape(a_temp)[2]
                    temp_3d = np.append(a_temp[:, :, :inl], b_temp, axis=2)
                setattr(self, var, temp_3d)
                a_temp = None
                b_temp = None
                temp_3d = None
                index += 1
        return inl, index

###################################################
# MosaicDisplay class
###################################################


class MosaicDisplay(object):

    """
    Class used for plotting MRMS data. To use:
    display = MosaicDisplay(tile), where tile is MosaicTile or Stitch instance
    """

    def __init__(self, mosaic):
        self.mosaic = mosaic

    def plot_horiz(self, var=DEFAULT_VAR, latrange=DEFAULT_LATRANGE,
                   lonrange=DEFAULT_LONRANGE, resolution='l',
                   level=None, parallels=DEFAULT_PARALLELS, area_thresh=10000,
                   meridians=DEFAULT_MERIDIANS, title=None,
                   clevs=DEFAULT_CLEVS, basemap=None, embellish=True,
                   cmap=DEFAULT_CMAP, save=None, show_grid=True,
                   linewidth=DEFAULT_LINEWIDTH, fig=None, ax=None,
                   verbose=False, return_flag=False, colorbar_flag=True,
                   colorbar_loc='bottom'):
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
        show_grid = Set to False to suppress gridlines and lat/lon labels.
        title = Plot title string, None = Basic time & date string as title.
                So if you want a blank title use title='' as keyword.
        clevs = Desired contour levels.
        cmap = Desired color map.
        basemap = Assign to basemap you want to use.
        embellish = Set to false to suppress basemap changes.
        colorbar_loc = Options: 'bottom', 'top', 'right', 'left'
        save = File to save image to. Careful, PS/EPS/PDF can get large!
        verbose = Set to True if you want a lot of text for debugging.
        resolution = Resolution of Basemap instance (e.g., 'c', 'l', 'i', 'h')
        area_thresh = Area threshold to show lakes, etc. (km^2)
        return_flag = Set to True to return plot info.
                      Order is Figure, Axis, Basemap
        """
        method_name = 'plot_horiz'
        ax, fig = self._parse_ax_fig(ax, fig)
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self.mosaic, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        if self.mosaic.nlon <= 1 or self.mosaic.nlat <= 1:
            print('Latitude or Longitude too small to plot')
            if verbose:
                _method_footer_printout()
            return
        if verbose:
            print('Executing plot')
        zdata, slevel = self._get_horizontal_cross_section(var, level, verbose)
        # Removed np.transpose() step from here as it was crashing
        # map proj coordinates under Python 3.
        plon = self.mosaic.Longitude
        plat = self.mosaic.Latitude
        if basemap is None:
            m = self._create_basemap_instance(latrange, lonrange, resolution,
                                              area_thresh)
        else:
            m = basemap
        if embellish:
            m = self._add_gridlines_if_desired(
                m, parallels, meridians, linewidth,
                latrange, lonrange, show_grid)
        x, y = m(plon, plat)  # compute map proj coordinates.
        # Draw filled contours
        # Note the need to transpose for plotting purposes
        cs = m.contourf(x.T, y.T, zdata, clevs, cmap=cmap)
        # cs = m.pcolormesh(x, y, zdata, vmin=np.min(clevs),
        #                   vmax=np.max(clevs), cmap=cmap)
        # Add colorbar, title, and save
        if colorbar_flag:
            cbar = m.colorbar(cs, location=colorbar_loc, pad="7%")
            if var == DEFAULT_VAR:
                cbar.set_label(DEFAULT_VAR_LABEL)
            else:
                # Placeholder for future dual-pol functionality
                cbar.set_label(var)
        if title is None:
            title = epochtime_to_string(self.mosaic.Time) + slevel
        plt.title(title)
        if save is not None:
            plt.savefig(save)
        # Clean up
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax, m

    def plot_vert(self, var=DEFAULT_VAR, lat=None, lon=None,
                  xrange=None, xlabel=None, colorbar_flag=True,
                  zrange=None, zlabel=DEFAULT_ZLABEL, fig=None, ax=None,
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
        title = String for plot title, None = Basic time & date as title.
                So if you want a blank title use title='' as keyword.
        save = File to save image to. Careful, PS/EPS/PDF can get large!
        verbose = Set to True if you want a lot of text for debugging.
        return_flag = Set to True to return Figure, Axis objects
        """
        method_name = 'plot_vert'
        ax, fig = self._parse_ax_fig(ax, fig)
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self.mosaic, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        # Get the cross-section
        vcut, xvar, xrange, xlabel, tlabel = \
            self._get_vertical_slice(var, lat, lon, xrange, xlabel, verbose)
        if vcut is None:
            return
        # Plot details
        if not title:
            title = epochtime_to_string(self.mosaic.Time) + ' ' + tlabel
        if not zrange:
            zrange = [0, np.max(self.mosaic.Height)]
        # Plot execution
        ax, cs = self._plot_vertical_cross_section(
            ax, vcut, xvar, xrange, xlabel, zrange, zlabel, clevs,
            cmap, title, mappable=True)
        if colorbar_flag:
            cbar = fig.colorbar(cs)
            if var == DEFAULT_VAR:
                cbar.set_label(DEFAULT_VAR_LABEL, rotation=90)
            else:
                # Placeholder for future dual-pol functionality
                cbar.set_label(var, rotation=90)
        # Finish up
        if save is not None:
            plt.savefig(save)
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax

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
                       longitude domain of (b). Default lonrange if not set.
                       Similar setup for xrange_c - subplot (c) - except for
                       latitude (i.e., defaults to latrange if not set). The
                       xrange_? variables determine length of crosshairs.
        """
        method_name = 'three_panel_plot'
        plt.close()  # mpl seems buggy if you don't clean up old windows
        if verbose:
            _method_header_printout(method_name)
        if not hasattr(self.mosaic, var):
            _print_variable_does_not_exist(method_name, var)
            if verbose:
                _method_footer_printout()
            return
        if self.mosaic.nlon <= 1 or self.mosaic.nlat <= 1:
            print('Latitude or Longitude too small to plot')
            if verbose:
                _method_footer_printout()
            return
        if lat is None or lon is None:
            print(method_name + '(): Need both constant latitude and',
                  'constant longitude for slices')
            if verbose:
                _method_footer_printout()
            return
        fig = plt.figure()
        fig.set_size_inches(11, 8.5)
        # Horizontal Cross-Section + Color Bar (subplot a)
        ax1 = fig.add_axes(THREE_PANEL_SUBPLOT_A)
        if not title_a:
            slevel, index = self._get_slevel(level, verbose)
            title_a = '(a) ' + epochtime_to_string(self.mosaic.Time) + slevel
        fig, ax1, m = self.plot_horiz(
            var=var, title=title_a, latrange=latrange, lonrange=lonrange,
            level=level, meridians=meridians, parallels=parallels,
            return_flag=True, linewidth=linewidth, show_grid=show_grid,
            clevs=clevs, cmap=cmap, verbose=verbose, area_thresh=area_thresh,
            resolution=resolution)
        if xrange_b is None:
            xrange_b = lonrange
        if not xrange_c:
            xrange_c = latrange
        if show_crosshairs:
            m = self._add_crosshairs(m, lat, lon, xrange_b, xrange_c)
        # Vertical Cross-Section (subplot b)
        if not title_b:
            lat, tlabel2 = self._parse_lat_tlabel(lat)
            title_b = '(b) ' + tlabel2
        ax2 = fig.add_axes(THREE_PANEL_SUBPLOT_B)
        self.plot_vert(var=var, lat=lat, zrange=zrange, xrange=xrange_b,
                       xlabel=lonlabel, zlabel=zlabel, cmap=cmap, clevs=clevs,
                       verbose=verbose, colorbar_flag=False, title=title_b)
        # Vertical Cross-Section (subplot c)
        if not title_c:
            lon, tlabel3 = self._parse_lon_tlabel(lon)
            title_c = '(c) ' + tlabel3
        ax3 = fig.add_axes(THREE_PANEL_SUBPLOT_C)
        self.plot_vert(var=var, lon=lon, zrange=zrange, xrange=xrange_c,
                       xlabel=latlabel, zlabel=zlabel, cmap=cmap, clevs=clevs,
                       verbose=verbose, colorbar_flag=False, title=title_c)
        # Finish up
        if save is not None:
            plt.savefig(save)
        if verbose:
            _method_footer_printout()
        if return_flag:
            return fig, ax1, ax2, ax3, m

    def _get_slevel(self, level, verbose, print_flag=False):
        if level is None:
            slevel = ' Composite '
            index = None
        else:
            if verbose and print_flag:
                print('Attempting to plot cross-section thru', level, 'km MSL')
            if level < np.min(self.mosaic.Height):
                level = np.min(self.mosaic.Height)
            elif level > np.max(self.mosaic.Height):
                level = np.max(self.mosaic.Height)
            index = np.argmin(np.abs(level - self.mosaic.Height))
            level = self.mosaic.Height[index]
            slevel = ' %.1f' % level + ' km MSL'
            if verbose and print_flag:
                print('Actually taking cross-section thru', level, 'km MSL')
        return slevel, index

    def _get_horizontal_cross_section(self, var=DEFAULT_VAR, level=None,
                                      verbose=False):
        slevel, index = self._get_slevel(level, verbose, print_flag=True)
        if index is None:
            if verbose:
                print('No vertical level specified,',
                      'plotting composite reflectivity')
            if not hasattr(self.mosaic, var+'_comp'):
                if verbose:
                    print(var+'_comp does not exist,',
                          'computing it with get_comp()')
                self.mosaic.get_comp(var=var, verbose=verbose)
            zdata = 1.0 * getattr(self.mosaic, var+'_comp')
            zdata = np.transpose(zdata)
        else:
            temp_3d = 1.0 * getattr(self.mosaic, var)
            zdata = temp_3d[index, :, :]
            zdata = np.transpose(zdata)
        return zdata, slevel

    def _create_basemap_instance(self, latrange=None, lonrange=None,
                                 resolution='l', area_thresh=10000):
        # create Basemap instance
        lon_0 = np.mean(lonrange)
        lat_0 = np.mean(latrange)
        m = Basemap(
            projection='merc', lon_0=lon_0, lat_0=lat_0, lat_ts=lat_0,
            llcrnrlat=np.min(latrange), urcrnrlat=np.max(latrange),
            llcrnrlon=np.min(lonrange), urcrnrlon=np.max(lonrange),
            resolution=resolution, area_thresh=area_thresh)
        # Draw coastlines, state and country boundaries, edge of map
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
        return m

    def _add_gridlines_if_desired(self, m=None,
                                  parallels=DEFAULT_PARALLELS,
                                  meridians=DEFAULT_MERIDIANS,
                                  linewidth=DEFAULT_LINEWIDTH,
                                  latrange=None, lonrange=None,
                                  show_grid=True):
        if show_grid:
            # Draw parallels
            vparallels = np.arange(np.floor(np.min(latrange)),
                                   np.ceil(np.max(latrange)), parallels)
            m.drawparallels(vparallels, labels=[1, 0, 0, 0], fontsize=10,
                            linewidth=linewidth)
            # Draw meridians
            vmeridians = np.arange(np.floor(np.min(lonrange)),
                                   np.ceil(np.max(lonrange)), meridians)
            m.drawmeridians(vmeridians, labels=[0, 0, 0, 1], fontsize=10,
                            linewidth=linewidth)
        return m

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

    def _get_vertical_slice(self, var=DEFAULT_VAR, lat=None, lon=None,
                            xrange=None, xlabel=None, verbose=False):
        """Execute slicing, get xvar, vcut"""
        fail = [None, None, None, None, None]
        if lat is None and lon is None:
            print('plot_vert(): Need a constant lat or lon for slice')
            if verbose:
                _method_footer_printout()
            return fail
        elif lat is not None and lon is not None:
            print('plot_vert(): Need either lat or lon for slice, not both!')
            if verbose:
                _method_footer_printout()
            return fail
        else:
            if lon is None:
                if self.mosaic.nlon <= 1:
                    print('Available Longitude range too small to plot')
                    if verbose:
                        _method_footer_printout()
                    return fail
                if verbose:
                    print('Plotting vertical cross-section thru', lat,
                          'deg Latitude')
                if not xrange:
                    xrange = [np.min(self.mosaic.Longitude),
                              np.max(self.mosaic.Longitude)]
                if not xlabel:
                    xlabel = 'Longitude (deg)'
                vcut, xvar, tlabel = \
                    self._get_constant_latitude_cross_section(var, lat)
            if lat is None:
                if self.mosaic.nlat <= 1:
                    print('Available Latitude range too small to plot')
                    if verbose:
                        _method_footer_printout()
                    return fail
                if verbose:
                    print('Plotting vertical cross-section thru', lon,
                          'deg Longitude')
                if not xrange:
                    xrange = [np.min(self.mosaic.Latitude),
                              np.max(self.mosaic.Latitude)]
                if not xlabel:
                    xlabel = 'Latitude (deg)'
                vcut, xvar, tlabel = \
                    self._get_constant_longitude_cross_section(var, lon)
        return vcut, xvar, xrange, xlabel, tlabel

    def _parse_lat_tlabel(self, lat):
        if lat > np.max(self.mosaic.Latitude):
            lat = np.max(self.mosaic.Latitude)
            print('Outside domain, plotting instead thru',
                  lat, ' deg Latitude')
        if lat < np.min(self.mosaic.Latitude):
            lat = np.min(self.mosaic.Latitude)
            print('Outside domain, plotting instead thru',
                  lat, 'deg Latitude')
        return lat, 'Latitude = ' + '%.2f' % lat + ' deg'

    def _get_constant_latitude_cross_section(self, var=DEFAULT_VAR, lat=None):
        lat, tlabel = self._parse_lat_tlabel(lat)
        index = np.round(np.abs(lat-self.mosaic.StartLat) /
                         self.mosaic.LatGridSpacing)
        index = np.int32(index)
        xvar = self.mosaic.Longitude[index, :]
        temp_3d = getattr(self.mosaic, var)
        vcut = temp_3d[:, index, :]
        return vcut, xvar, tlabel

    def _parse_lon_tlabel(self, lon):
        if lon > np.max(self.mosaic.Longitude):
            lon = np.max(self.mosaic.Longitude)
            print('max', lon, np.max(self.mosaic.Longitude))
            print('Outside domain, plotting instead thru',
                  lon, ' deg Longitude')
        if lon < np.min(self.mosaic.Longitude):
            lon = np.min(self.mosaic.Longitude)
            print('min', lon, np.min(self.mosaic.Longitude))
            print('Outside domain, plotting instead thru',
                  lon, 'deg Longitude')
        return lon, 'Longitude = ' + '%.2f' % lon + ' deg'

    def _get_constant_longitude_cross_section(self, var=DEFAULT_VAR, lon=None):
        lon, tlabel = self._parse_lon_tlabel(lon)
        index = np.round(np.abs(lon-self.mosaic.StartLon) /
                         self.mosaic.LonGridSpacing)
        index = np.int32(index)
        xvar = self.mosaic.Latitude[:, index]
        temp_3d = getattr(self.mosaic, var)
        vcut = temp_3d[:, :, index]
        return vcut, xvar, tlabel

    def _plot_vertical_cross_section(self, ax=None, vcut=None, xvar=None,
                                     xrange=None, xlabel=None,
                                     zrange=None, zlabel=None,
                                     clevs=DEFAULT_CLEVS, cmap=DEFAULT_CMAP,
                                     title=None, mappable=False):
        cs = ax.contourf(xvar, self.mosaic.Height, vcut, clevs, cmap=cmap)
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

    def _parse_ax_fig(self, ax=None, fig=None):
        """Parse and return ax and fig parameters. Adapted from Py-ART."""
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        return ax, fig

###################################################
# Independent MMM-Py functions follow
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
    # 1-D stitching, either N-S or W-E
    if np.ndim(map_array) == 1:
        # direction unset or not a string = direction fail
        if direction is None or isinstance(direction, str) is False:
            _print_direction_fail(method_name)
            return
        # E-W Stitching only
        if direction.upper() in ['EW', 'E', 'W', 'WE']:
            result = _stitch_1d_array_we(map_array, verbose)
            if result is None:
                return
        # N-S Stitching only
        elif direction.upper() in ['NS', 'N', 'S', 'SN']:
            result = _stitch_1d_array_ns(map_array, verbose)
            if result is None:
                return
        # everything else = direction fail
        else:
            _print_direction_fail(method_name)
            return
    # map_array fail
    elif np.ndim(map_array) < 1 or np.ndim(map_array) > 2:
        print(method_name+'(): map_array is not right,',
              'use 1- or 2-rank array')
        return
    # 2-D stitching in N-S and W-E
    else:
        if verbose:
            print('Sent a 2-D matrix, attempting to stitch in both',
                  'N-S and E-W directions')
        # Tile number fail
        if not _right_number_of_tiles(map_array):
            return
        # Actual stitching done here
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
    specific grids. For MosaicTiles and Stitches, send the mrefl3d, Latitude,
    Longitude, and Height attributes. Assumes constant grid spacing in horiz.
    Returns volumes of 3-D grid cells (km**3) & areas of 2-D grid cells (km**2)
    """
    re = 6371.1  # km
    vol = 0.0 * dz3d
    latdel = np.abs(lat[0, 0] - lat[1, 0])
    londelr = np.abs(lon[0, 1] - lon[0, 0]) * np.pi / 180.0
    sa = 0.0 * lat
    for j in np.arange(len(lat[:, 0])):
        th1 = np.deg2rad(90.0 + lat[j, 0] - latdel / 2.0)
        th2 = np.deg2rad(90.0 + lat[j, 0] + latdel / 2.0)
        sa[j, :] = re**2 * londelr * (np.cos(th1) - np.cos(th2))
        for k in np.arange(len(height)):
            if k == 0:
                hdel = height[k]
            else:
                hdel = height[k] - height[k-1]
            vol[k, j, :] = hdel * sa[j, 0]
    return vol, sa


def epochtime_to_string(epochtime=None, use_second=False):
    """
    Given an epoch time (seconds since 1/1/1970), return a string useful
    as a plot title. Set the use_second flag to also include seconds in
    the string.
    """
    try:
        if use_second:
            time_string = time.strftime('%m/%d/%Y %H:%M:%S UTC',
                                        time.gmtime(epochtime))
        else:
            time_string = time.strftime('%m/%d/%Y %H:%M UTC',
                                        time.gmtime(epochtime))
    except:
        time_string = ''
    return time_string

###################################################
# MMM-Py internal functions below
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
        print('Sent a 1-D matrix, attempting to stitch in E-W directions')
    for i in np.arange(np.shape(map_array)[0]):
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
        print('Sent a 1-D matrix, attempting to stitch in N-S directions')
    for i in np.arange(np.shape(map_array)[0]):
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
    ns_stitches = []
    for i in np.arange(np.shape(map_array)[1]):
        if not hasattr(map_array[0][i], DEFAULT_VAR) or \
           not hasattr(map_array[1][i], DEFAULT_VAR):
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


def _method_header_printout(method_name=' '):
    print('')
    print('********************')
    print(method_name + '():')


def _method_footer_printout():
    print('********************')
    print('')


def _print_direction_fail(method_name=' '):
    print(method_name + '(): Sent a 1-D array but no direction to stitch!')
    print('Use direction=\'we\' or direction=\'ns\' in argument to fix')


def _print_missing_a_tile(method_name=' '):
    print(method_name + '(): Missing a tile, fix and try again')


def _print_wrong_number_of_tiles(method_name=' '):
    print(method_name + '(): Wrong number of tiles, fix and try again')


def _print_method_done():
    print('Task complete, data contained as class attributes')
    print('Use dir(), help(), or __dict__ to find out what is available')


def _print_method_called_incorrectly(method_name=' '):
    print(method_name + '(): Method called incorrectly, check syntax')


def _print_variable_does_not_exist(method_name=' ', var=DEFAULT_VAR):
    print(method_name+'():', var, 'does not exist, try reading in a file')


def _fill_list(f, size, offset):
    _list = []
    for i in np.arange(size):
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
