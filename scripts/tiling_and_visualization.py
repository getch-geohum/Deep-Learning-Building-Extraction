import numpy as np
import rasterio
from rasterio.windows import Window, bounds
from rasterio.features import rasterize
import geopandas as gpd
import os
from tqdm import tqdm
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon
import leafmap
import random
from glob import glob
import matplotlib.pyplot as plt
from skimage.io import imread

def plot_binary_mask(file_path, n_samples, fig_size = 28):
  images = sorted(glob(file_path + "/images/*.tif"))  # assuming the image is in .tif format

  assert len(images) > 0, "No images found in the specified directory."

  if n_samples > len(images):
    n_samples = len(images)

  indexes = random.sample(list(range(0, len(images))), n_samples)

  scale = 2/n_samples
  fig, ax = plt.subplots(n_samples, 2, figsize=(fig_size*scale, fig_size))

  for i, index in enumerate(indexes):
    image = imread(images[index])
    mask = imread(images[index].replace("/images/", "/labels/"))
    ax[i][0].imshow(image)
    ax[i][1].imshow(mask)
    ax[i][0].axis("off")
    ax[i][1].axis("off")
  plt.show()

def tile_visualize(raster_file, tile_height, tile_width, stride_y, stride_x, with_tiles):
  if with_tiles:
    alls = []

    with rasterio.open(raster_file) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            epsg = int(crs.to_epsg())

            for y in tqdm(range(0, height, stride_y)):
                for x in range(0, width, stride_x):
                    
                    # Create window
                    window_ = Window(x, y, tile_width, tile_height)
                    window_bounds_ = bounds(window_, transform)
                    latlon_bounds = transform_bounds(src.crs, f'EPSG:{epsg}', *window_bounds_)
                    min_lon, min_lat, max_lon, max_lat = latlon_bounds
                    poly = Polygon(((min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)))
                    alls.append(poly)

    series = gpd.GeoSeries(alls)
    vals = {'ID': [f'{i}' for i in range(len(series.geometry))],
                'geometry': series.geometry}

    gdf = gpd.GeoDataFrame(vals).set_crs(epsg=epsg)

    m = leafmap.Map()
    m.add_gdf(gdf, layer_name="vector_tile")
    m.add_raster(raster_file, layer_name="raster")

  else:
    m = leafmap.Map()
    m.add_raster(raster_file, layer_name="raster")

  return m


def create_binary_mask_from_shapefile(shapefile_path, image_transform, image_shape, target_crs=None):
    """
    Create a binary mask from a shapefile that matches the image dimensions.
    
    Args:
        shapefile_path: Path to the shapefile
        image_transform: Rasterio transform of the image
        image_shape: Shape of the image (height, width)
        target_crs: Target CRS (if None, will use shapefile's CRS)
    
    Returns:
        Binary mask as numpy array (1 where features exist, 0 elsewhere)
    """
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject if necessary
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    # Create binary mask by rasterizing the geometries
    shapes = [(geom, 1) for geom in gdf.geometry]
    
    mask = rasterize(
        shapes=shapes,
        out_shape=image_shape,
        transform=image_transform,
        fill=0,
        dtype=np.uint8
    )
    
    return mask


def tile_raster(input_raster, output_dir, tile_size, stride, input_mask_file=None, background_value=0):
    """
    Tile an input raster into chips with specified size and stride.
    Discards chips that completely contain background.
    
    Parameters:
    -----------
    input_raster : str
        Path to input raster file
    output_dir : str
        Directory to save output chips
    tile_size : int or tuple
        Size of tiles (height, width). If int, square tiles are created
    stride : int or tuple
        Stride between tiles (y, x). If int, same stride in both directions
    background_value : int/float, optional
        Value considered as background (default: 0)
    
    Returns:
    --------
    list
        List of saved chip filenames
    """


    image_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'labels')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Handle tile_size and stride if they're integers
    if isinstance(tile_size, int):
        tile_height = tile_width = tile_size
    else:
        tile_height, tile_width = tile_size
        
    if isinstance(stride, int):
        stride_y = stride_x = stride
    else:
        stride_y, stride_x = stride
    
    saved_chips = []
    
    with rasterio.open(input_raster) as src:
        # Get raster dimensions
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        
        if input_mask_file is not None:
            mask = create_binary_mask_from_shapefile(
                shapefile_path=input_mask_file, 
                image_transform=transform, 
                image_shape = (height, width), 
                target_crs = crs
            )
        # Calculate number of tiles
        n_tiles_y = ((height - tile_height) // stride_y) + 1
        n_tiles_x = ((width - tile_width) // stride_x) + 1
        
        print(f"Raster dimensions: {height} x {width}")
        print(f"Number of tiles including background: {n_tiles_y} x {n_tiles_x}")
        
        
        # Iterate through tiles
        for y in range(0, height + 1, stride_y):
            for x in range(0, width + 1, stride_x):
                
                # Create window
                window = Window(x, y, tile_width, tile_height)
                
                # Read the tile
                tile = src.read(window=window)
                
                # Check if tile is completely background
                if is_completely_background(tile, background_value):
                    continue
                
                # Update metadata for the chip
                chip_meta = src.meta.copy()
                chip_meta.update({
                    "height": tile_height,
                    "width": tile_width,
                    "transform": src.window_transform(window)
                })
                
                # Generate output filename
                chip_filename = f"chip_{y}_{x}.tif"
                chip_path = os.path.join(image_dir, chip_filename)

                # Save the chip
                with rasterio.open(chip_path, 'w', **chip_meta) as dst:
                    dst.write(tile)

                if input_mask_file is not None:
                    sub_mask = mask[y:y+tile_height, x:x+tile_width]
                    mask_path = os.path.join(mask_dir, chip_filename)

                    mask_meta = src.meta.copy()
                    mask_meta.update({
                        "height": tile_height,
                        "width": tile_width,
                        "count":1,
                        "transform": src.window_transform(window)
                    })

                    with rasterio.open(mask_path, 'w', **mask_meta) as dst:
                        dst.write(sub_mask,1)

                saved_chips.append(chip_filename)

    print(f"\nSuccessfully created {len(saved_chips)} chips")
        
    return None

def is_completely_background(tile, background_value=0):
    """
    Check if the tile is completely filled with background value.
    
    Parameters:
    -----------
    tile : numpy array
        Tile array (bands, height, width)
    background_value : int/float
        Value considered as background
    
    Returns:
    --------
    bool
        True if all pixels are background, False otherwise
    """
    
    # Handle different array shapes
    if tile.ndim == 2:
        return np.all(tile == background_value)
    elif tile.ndim == 3:
        all_background = np.all(tile == background_value, axis=0)
        return np.all(all_background)
        

def tile_raster_with_overlap(input_raster, output_dir, tile_size, input_mask_file=None, overlap=0, background_value=0):
    """
    Alternative function that uses overlap instead of stride.
    Stride = tile_size - overlap
    
    Parameters:
    -----------
    input_raster : str
        Path to input raster file
    input_mask_file: str
        Path to input mask file
    output_dir : str
        Directory to save output chips
    tile_size : int or tuple
        Size of tiles (height, width)
    overlap : int or tuple
        Overlap between tiles (y, x)
    background_value : int/float
        Value considered as background
    """
    if isinstance(tile_size, int):
        tile_height = tile_width = tile_size
    else:
        tile_height, tile_width = tile_size
        
    if isinstance(overlap, int):
        overlap_y = overlap_x = overlap
    else:
        overlap_y, overlap_x = overlap
    
    stride_y = tile_height - overlap_y
    stride_x = tile_width - overlap_x
    

    return tile_raster(input_raster=input_raster,
                       input_mask_file=input_mask_file,
                       output_dir=output_dir,
                       tile_size = (tile_height, tile_width),
                       stride = (stride_y, stride_x),
                       background_value=background_value)


# Main execution example
if __name__ == "__main__":
    # Example usage
    input_raster = "../ESA_training_CEOS/material_and_data/dataset/Kakuma_mosaics/train_scene_112.tif"
    input_mask_file = "../ESA_training_CEOS/masks-20260315T192459Z-3-001/masks/merged_final_aoi_112.shp"
    output_directory = "../ESA_training_CEOS/material_and_data/dataset/CUSTOM_TILED_WITH_MASK_new"
    
    # Parameters
    TILE_SIZE = 640  
    STRIDE = 640     
    BACKGROUND_VALUE = 0  
    
    # Visualize tiling pattern first (optional)
    # visualize_tiling(input_raster, TILE_SIZE, STRIDE, "tiling_pattern.png")
    
    # Tile the raster
    try:
        saved_chips = tile_raster(
            input_raster=input_raster, 
            output_dir=output_directory,
            input_mask_file=input_mask_file, 
            tile_size=TILE_SIZE, 
            stride=STRIDE, 
            background_value=BACKGROUND_VALUE
        )
        
        print(f"\nSuccessfully created {len(saved_chips)} valid chips")
        
    except Exception as e:
        print(f"Error processing raster: {e}")