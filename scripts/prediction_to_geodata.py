import torch
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid
from rasterio import features
import rasterio
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

### Functions for SAMs segmentation psot processing 


def remove_small_holes(polygon, min_hole_area=None, min_hole_ratio=None):
    """
    Remove holes (interior rings) from a polygon based on area or ratio criteria

    Parameters:
    -----------
    polygon : shapely.Polygon or MultiPolygon
        Input polygon geometry
    min_hole_area : float, optional
        Minimum area to keep a hole (holes smaller than this are removed)
    min_hole_ratio : float, optional
        Minimum ratio of hole area to parent polygon area (0-1)
        Holes with ratio smaller than this are removed

    Returns:
    --------
    shapely.Polygon or MultiPolygon
        Polygon with small holes removed
    """

    def process_single_polygon(poly):
        """Process a single polygon (not MultiPolygon)"""
        if poly.is_empty or not poly.interiors:
            return poly

        # Calculate parent polygon area (exterior only)
        exterior_area = Polygon(poly.exterior).area

        # Filter interior rings
        valid_interiors = []
        for interior in poly.interiors:
            hole = Polygon(interior)
            hole_area = hole.area

            # Check if hole should be kept
            keep_hole = True

            if min_hole_area is not None:
                if hole_area < min_hole_area:
                    keep_hole = False

            if min_hole_ratio is not None and keep_hole:
                if hole_area / exterior_area < min_hole_ratio:
                    keep_hole = False

            if keep_hole:
                valid_interiors.append(interior)

        # Return polygon with filtered holes
        if valid_interiors:
            return Polygon(poly.exterior, valid_interiors)
        else:
            return Polygon(poly.exterior)

    # Handle MultiPolygon
    if polygon.geom_type == 'MultiPolygon':
        processed_polygons = [process_single_polygon(p) for p in polygon.geoms]
        # Filter out empty polygons
        processed_polygons = [p for p in processed_polygons if not p.is_empty]
        if len(processed_polygons) == 1:
            return processed_polygons[0]
        elif len(processed_polygons) > 1:
            return MultiPolygon(processed_polygons)
        else:
            return Polygon()  # Empty polygon

    # Handle single polygon
    else:
        return process_single_polygon(polygon)


def binary_raster_to_vector(raster_path, output_path=None, mask_value=1,
                           simplify_tolerance=None, min_area=None,
                           remove_holes=False, min_hole_area=None,
                           min_hole_ratio=None, fix_invalid=True):
    """
    Convert binary raster to vector polygons with option to remove small holes

    Parameters:
    -----------
    raster_path : str
        Path to input binary raster
    output_path : str, optional
        Path to save output vector file (e.g., .gpkg, .shp)
    mask_value : int or float, default=1
        Value in raster representing the feature to vectorize
    simplify_tolerance : float, optional
        Tolerance for simplifying geometries (in map units)
    min_area : float, optional
        Minimum area to keep polygons (filter out small features)
    remove_holes : bool, default=False
        Whether to remove small holes from polygons
    min_hole_area : float, optional
        Minimum area to keep a hole (holes smaller than this are removed)
    min_hole_ratio : float, optional
        Minimum ratio of hole area to parent polygon area (0-1)
    fix_invalid : bool, default=True
        Attempt to fix invalid geometries

    Returns:
    --------
    geopandas.GeoDataFrame
        Vectorized polygons
    """

    # Open the raster
    with rasterio.open(raster_path) as src:
        # Read the first band
        raster_data = src.read(1)

        # Get the transform and CRS
        transform = src.transform
        crs = src.crs

        # Create mask for the feature of interest
        mask = raster_data == mask_value

        # Check if there are any features
        if not np.any(mask):
            print("No features found with the specified mask value")
            return gpd.GeoDataFrame(geometry=[], crs=crs)

        # Vectorize the mask
        results = []

        # Use rasterio.features.shapes to extract polygons
        for geom, value in features.shapes(mask.astype(np.uint8),
                                          mask=mask,
                                          transform=transform):

            # Create shapely geometry
            polygon = shape(geom)

            # Skip if polygon is empty
            if polygon.is_empty:
                continue

            # Fix invalid geometries if requested
            if fix_invalid and not polygon.is_valid:
                try:
                    polygon = polygon.buffer(0)
                    if polygon.is_empty:
                        continue
                except:
                    try:
                        polygon = make_valid(polygon)
                    except:
                        continue

            # Remove small holes if requested
            if remove_holes and polygon.geom_type in ['Polygon', 'MultiPolygon']:
                polygon = remove_small_holes(
                    polygon,
                    min_hole_area=min_hole_area,
                    min_hole_ratio=min_hole_ratio
                )
                if polygon.is_empty:
                    continue

            # Simplify if requested
            if simplify_tolerance:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

            # Filter by area if requested
            if min_area:
                if polygon.geom_type == 'MultiPolygon':
                    # For MultiPolygon, check total area
                    if polygon.area < min_area:
                        continue
                else:
                    if polygon.area < min_area:
                        continue

            results.append({
                'geometry': polygon,
                'value': int(value)
            })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(results, crs=crs)

    # Save if output path provided
    if output_path and len(gdf) > 0:
        gdf.to_file(output_path)
        print(f"Saved {len(gdf)} polygons to {output_path}")

    return gdf


# Additional function for more advanced hole removal
def remove_holes_by_compactness(polygon, max_hole_compactness=0.5):
    """
    Remove holes based on their compactness (holes that are too elongated)

    Compactness = 4π * Area / Perimeter²
    - Circle has compactness = 1
    - Elongated shapes have lower compactness

    Parameters:
    -----------
    polygon : shapely.Polygon
        Input polygon
    max_hole_compactness : float
        Maximum compactness to keep a hole (holes less compact than this are removed)

    Returns:
    --------
    shapely.Polygon
        Polygon with non-compact holes removed
    """
    if polygon.is_empty or not polygon.interiors:
        return polygon

    valid_interiors = []
    exterior_area = Polygon(polygon.exterior).area

    for interior in polygon.interiors:
        hole = Polygon(interior)
        hole_area = hole.area
        hole_perimeter = hole.length

        if hole_perimeter > 0:
            # Calculate compactness (higher = more circular)
            compactness = (4 * np.pi * hole_area) / (hole_perimeter ** 2)

            # Keep holes that are compact enough
            if compactness >= max_hole_compactness:
                valid_interiors.append(interior)

    if valid_interiors:
        return Polygon(polygon.exterior, valid_interiors)
    else:
        return Polygon(polygon.exterior)






##### Functions for detection and instance segmentation postprocessing ####

def yolo_segment_to_shapefile(
    model,
    image_folder,
    ext,
    output_shapefile,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    class_names=None
):
    """
    Perform YOLO instance segmentation inference and save results as shapefile
    
    Args:
        model: trained model which is already loaded
        image_folder: Path to folder containing input image tiles
        output_shapefile: Path for output shapefile (without extension)
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on
        class_names: Optional list of class names for attributes
    """
    
    # Get all image files
    
    image_files = sorted(glob(image_folder + f"/*.{ext}"))
    
    print(f"Found {len(image_files)} image files")
    
    # Store all predictions
    all_polygons = []
    all_confidences = []
    all_class_ids = []
    all_class_names = []
    all_image_sources = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        img_path = str(img_path)
        
        try:
            # Get image geotransform if available
            with rasterio.open(img_path) as src:
                transform = src.transform
                crs = src.crs
                width = src.width
                height = src.height
            
            # Run inference
            results = model.predict(
                img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )[0]
            
            if results.masks is not None:
                # Get masks and convert to polygons
                masks = results.masks.data.cpu().numpy()
                
                for i, mask in enumerate(masks):
                    # Get confidence and class
                    conf = results.boxes.conf[i].item()
                    class_id = int(results.boxes.cls[i].item())
                    class_name = results.names[class_id] if class_names is None else class_names[class_id]
                    
                    # Convert binary mask to polygon
                    polygons = mask_to_polygons(
                        mask, 
                        transform, 
                        width, 
                        height,
                        min_area=10  # Minimum area in pixels to filter small polygons
                    )
                    
                    for polygon in polygons:
                        if polygon is not None and not polygon.is_empty:
                            all_polygons.append(polygon)
                            all_confidences.append(conf)
                            all_class_ids.append(class_id)
                            all_class_names.append(class_name)
                            all_image_sources.append(os.path.basename(img_path))
                            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not all_polygons:
        print("No polygons detected")
        return
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': all_polygons,
        'confidence': all_confidences,
        'class_id': all_class_ids,
        'class_name': all_class_names,
        'source_img': all_image_sources
    }, crs=crs)
    
    # Save to shapefile
    output_path = output_shapefile if output_shapefile.endswith('.shp') else f"{output_shapefile}.shp"
    gdf.to_file(output_path)


    geojson_path = output_path.replace('.shp', '.geojson')
    gdf.to_file(geojson_path, driver='GeoJSON')
    
    print(f"\nResults saved to {output_path}")
    print(f"Total polygons: {len(gdf)}")
    print(f"Classes detected: {gdf['class_name'].unique()}")
    print(f"Confidence range: {gdf['confidence'].min():.3f} - {gdf['confidence'].max():.3f}")
    
    return gdf

def mask_to_polygons(mask, transform, width, height, min_area=1, simplify_tolerance=0.5):
    """
    Convert binary mask to polygons with proper geotransform
    
    Args:
        mask: Binary mask array (height x width)
        transform: Rasterio affine transform
        width: Image width
        height: Image height
        min_area: Minimum area in pixels to keep polygon
        simplify_tolerance: Tolerance for polygon simplification (in pixel units)
    
    Returns:
        List of shapely polygons in world coordinates
    """
    from skimage import measure
    
    polygons = []
    
    # Find contours in the mask
    contours = measure.find_contours(mask, 0.5)
    
    for contour in contours:
        # Convert contour coordinates to world coordinates
        # Note: contour coordinates are (row, col) in image space
        world_coords = []
        for point in contour:
            # Convert row, col to world coordinates
            row, col = point
            world_x, world_y = transform * (col, row) 
            world_coords.append((world_x, world_y))
        
        # Create polygon
        if len(world_coords) >= 3:  # Need at least 3 points for a polygon
            try:
                # Simplify polygon to reduce vertices
                poly = Polygon(world_coords)
                
                # Simplify if tolerance is provided
                if simplify_tolerance > 0:
                    # Convert simplify tolerance to world units
                    world_tolerance = simplify_tolerance * max(transform.a, abs(transform.e))
                    poly = poly.simplify(world_tolerance, preserve_topology=True)
                
                # Filter by area
                if poly.area >= min_area * abs(transform.a * transform.e):
                    polygons.append(poly)
            except Exception as e:
                print(f"Error creating polygon: {e}")
                continue
    return polygons



def yolo_obb_to_shapefile(
    model,
    image_folder,
    ext,
    output_shapefile,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    class_names=None
):
    """
    Perform YOLO oriented object detection inference and save results as shapefile
    
    Args:
        model: trained YOLO OBB model which is already loaded
        image_folder: Path to folder containing input image tiles
        ext: Image file extension (e.g., 'tif', 'jpg', 'png')
        output_shapefile: Path for output shapefile (without extension)
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on
        class_names: Optional list of class names for attributes
    """
    
    # Get all image files
    image_files = sorted(glob(os.path.join(image_folder, f"*.{ext}")))
    
    if not image_files:
        image_files = sorted(glob(os.path.join(image_folder, f"*.{ext.upper()}")))
    
    print(f"Found {len(image_files)} image files")
    
    # Store all predictions
    all_polygons = []
    all_confidences = []
    all_class_ids = []
    all_class_names = []
    all_angles = []
    all_angle_degs = []
    all_image_sources = []
    
    # Initialize CRS variable
    crs = None
    first_image_crs = None
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        img_path = str(img_path)
        
        try:
            # Get image geotransform if available
            with rasterio.open(img_path) as src:
                transform = src.transform
                crs = src.crs
                if first_image_crs is None:
                    first_image_crs = crs
                width = src.width
                height = src.height
            
            # Run inference
            results = model.predict(
                img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )[0]
            
            # Check if oriented boxes exist (YOLOv8 OBB format)
            if hasattr(results, 'obb') and results.obb is not None:
                # Access OBB data - different versions of ultralytics have different access methods
                if hasattr(results.obb, 'xywhr'):
                    # Newer versions
                    xywhr = results.obb.xywhr.cpu().numpy()
                    confs = results.obb.conf.cpu().numpy()
                    cls_ids = results.obb.cls.cpu().numpy()
                    
                    for i in range(len(xywhr)):
                        # Extract parameters
                        x_center, y_center, width_box, height_box, angle = xywhr[i]
                        conf = confs[i]
                        class_id = int(cls_ids[i])
                        
                        # Get class name
                        if class_names is not None:
                            class_name = class_names[class_id]
                        else:
                            class_name = results.names[class_id] if hasattr(results, 'names') else str(class_id)
                        
                        # Convert oriented box to polygon in pixel coordinates
                        polygon_pixel = oriented_box_to_polygon_pixel(
                            float(x_center), 
                            float(y_center), 
                            float(width_box), 
                            float(height_box), 
                            float(angle)
                        )
                        
                        # Transform polygon from pixel to world coordinates
                        polygon_world = transform_polygon_to_world(polygon_pixel, transform)
                        
                        all_polygons.append(polygon_world)
                        all_confidences.append(float(conf))
                        all_class_ids.append(class_id)
                        all_class_names.append(class_name)
                        all_angles.append(float(angle))
                        all_angle_degs.append(float(np.degrees(angle)))
                        all_image_sources.append(os.path.basename(img_path))
                        
                elif hasattr(results.obb, 'data'):
                    # Older versions or different format
                    obb_data = results.obb.data.cpu().numpy()
                    
                    for detection in obb_data:
                        if len(detection) >= 7:
                            x_center, y_center, width_box, height_box, angle, conf, class_id = detection[:7]
                            
                            # Get class name
                            if class_names is not None:
                                class_name = class_names[int(class_id)]
                            else:
                                class_name = results.names[int(class_id)] if hasattr(results, 'names') else str(int(class_id))
                            
                            # Convert oriented box to polygon
                            polygon_pixel = oriented_box_to_polygon_pixel(
                                float(x_center), 
                                float(y_center), 
                                float(width_box), 
                                float(height_box), 
                                float(angle)
                            )
                            
                            polygon_world = transform_polygon_to_world(polygon_pixel, transform)
                            
                            all_polygons.append(polygon_world)
                            all_confidences.append(float(conf))
                            all_class_ids.append(int(class_id))
                            all_class_names.append(class_name)
                            all_angles.append(float(angle))
                            all_angle_degs.append(float(np.degrees(angle)))
                            all_image_sources.append(os.path.basename(img_path))
            
            # Alternative: Check if results have boxes with rotation
            elif hasattr(results, 'boxes') and results.boxes is not None:
                if hasattr(results.boxes, 'xywhr'):
                    # Some versions store rotation in boxes
                    xywhr = results.boxes.xywhr.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy()
                    
                    for i in range(len(xywhr)):
                        x_center, y_center, width_box, height_box, angle = xywhr[i]
                        conf = confs[i]
                        class_id = int(cls_ids[i])
                        
                        if class_names is not None:
                            class_name = class_names[class_id]
                        else:
                            class_name = results.names[class_id]
                        
                        polygon_pixel = oriented_box_to_polygon_pixel(
                            float(x_center), float(y_center), 
                            float(width_box), float(height_box), float(angle)
                        )
                        
                        polygon_world = transform_polygon_to_world(polygon_pixel, transform)
                        
                        all_polygons.append(polygon_world)
                        all_confidences.append(float(conf))
                        all_class_ids.append(class_id)
                        all_class_names.append(class_name)
                        all_angles.append(float(angle))
                        all_angle_degs.append(float(np.degrees(angle)))
                        all_image_sources.append(os.path.basename(img_path))
                            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_polygons:
        print("No oriented objects detected")
        return None
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': all_polygons,
        'confidence': all_confidences,
        'class_id': all_class_ids,
        'class_name': all_class_names,
        'angle_rad': all_angles,
        'angle_deg': all_angle_degs,
        'source_img': all_image_sources
    }, crs=first_image_crs)
    
    # Add area and perimeter (in world units)
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    
    # Save to shapefile
    output_path = output_shapefile if output_shapefile.endswith('.shp') else f"{output_shapefile}.shp"
    gdf.to_file(output_path)
    
    # Also save as GeoJSON for better attribute handling
    geojson_path = output_path.replace('.shp', '.geojson')
    gdf.to_file(geojson_path, driver='GeoJSON')
    
    print(f"\nResults saved to {output_path}")
    print(f"Total oriented objects: {len(gdf)}")
    print(f"Classes detected: {gdf['class_name'].unique()}")
    print(f"Confidence range: {gdf['confidence'].min():.3f} - {gdf['confidence'].max():.3f}")
    print(f"Angle range (deg): {gdf['angle_deg'].min():.1f} - {gdf['angle_deg'].max():.1f}")
    
    return gdf

def oriented_box_to_polygon_pixel(x_center, y_center, width, height, angle_rad):
    """
    Convert oriented bounding box parameters to polygon vertices in pixel coordinates
    
    Args:
        x_center, y_center: Center coordinates in pixels
        width, height: Box dimensions in pixels
        angle_rad: Rotation angle in radians
    
    Returns:
        Shapely Polygon of oriented box in pixel coordinates
    """
    # Half dimensions
    w_half = width / 2
    h_half = height / 2
    
    # Corners in local coordinates (unrotated, centered at origin)
    corners = np.array([
        [-w_half, -h_half],
        [w_half, -h_half],
        [w_half, h_half],
        [-w_half, h_half]
    ])
    
    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Rotate corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate to center
    world_corners = rotated_corners + np.array([x_center, y_center])
    
    # Close the polygon by returning to first point
    return Polygon(world_corners)

def transform_polygon_to_world(polygon_pixel, transform):
    """
    Transform polygon from pixel coordinates to world coordinates using affine transform
    
    Args:
        polygon_pixel: Shapely polygon in pixel coordinates
        transform: Rasterio affine transform
    
    Returns:
        Shapely polygon in world coordinates
    """
    # Get polygon exterior coordinates
    exterior_coords = list(polygon_pixel.exterior.coords)
    
    # Transform each point to world coordinates
    world_coords = []
    for x_pixel, y_pixel in exterior_coords:
        world_x, world_y = transform * (x_pixel, y_pixel)
        world_coords.append((world_x, world_y))
    
    # Create new polygon with world coordinates
    return Polygon(world_coords)

