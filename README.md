MMM-Py README

Installation:
Put mmmpy.py in your PYTHONPATH

Get MRMS-modified wgrib2 package from here
ftp://ftp.nssl.noaa.gov/projects/MRMS/GRIB2_DECODERS/MRMS_modified_wgrib2_v2.0.1-selectfiles.tgz

Install wgrib2 and note the path to it. Modify the BASE_PATH, TMPDIR, WGRIB2_PATH, and WGRIB2_NAME 
global variables in mmmpy.py as necessary. TMPDIR is where intermediate netCDFs created by wgrib2
will go.

Without wgrib2 MMM-Py can still read legacy MRMS binaries and netCDFs. 

To access everything:
import mmmpy

To see MMM-Py in action, check out the IPython notebooks provided in this distribution.

See LICENSE file for NASA open soure license information.
