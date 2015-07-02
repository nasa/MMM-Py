import mmmpy

def test_read_binary():
    bfile = 'data/MREF3D33L_tile1.20140705.145200.gz'
    tile = mmmpy.MosaicTile(bfile)
    assert tile.Latitude.max() >= 54 and tile.Latitude.max() <= 55
    assert tile.Tile == '1'

def test_read_netcdf():
    nfile = 'data/mosaic3d_tile6_20130531-231500.netcdf'
    tile = mmmpy.MosaicTile(nfile)
    assert tile.Tile == '6'