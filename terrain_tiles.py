#!/usr/bin/env python3
"""
terrain_tiles.py - Generate terrain tiles for resin printing

Creates a grid of tiles that align at edges for modular wall art.

Usage:
    python terrain_tiles.py --lat 52.3139 --lon -2.5947 \
        --tiles 3x3 --tile-radius 1 --tile-size 120 \
        --resolution 1000 --output malvern_tiles/

Dependencies:
    pip install pyproj rasterio numpy numpy-stl requests scipy trimesh
"""

# Standard library
import argparse
import hashlib
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import NamedTuple

# Third party
import numpy as np
import requests
import trimesh
from pyproj import Transformer
from rasterio.io import MemoryFile
from scipy import ndimage
from stl import mesh


# =============================================================================
# Constants
# =============================================================================

# Cache directory for LIDAR data
CACHE_DIR = os.path.expanduser("~/.terrain_tiles_cache")

# EA WCS endpoint - DSM (includes trees, buildings)
WCS_BASE_URL = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs"
COVERAGE_ID = "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"

# API Configuration
API_TIMEOUT_SECONDS = 300
MAX_API_RETRIES = 5

# Data processing
INVALID_ELEVATION_THRESHOLD = -1000  # Elevations below this are considered invalid/nodata


# =============================================================================
# Custom Exceptions
# =============================================================================

class TerrainTileError(Exception):
    """Base exception for terrain tile errors."""
    pass


class APIError(TerrainTileError):
    """Error communicating with the elevation API."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"API request failed with status {status_code}: {message}")


class DataError(TerrainTileError):
    """Error processing elevation data."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

class BoundingBox(NamedTuple):
    """Geographic bounding box in OSGB coordinates."""
    min_easting: float
    min_northing: float
    max_easting: float
    max_northing: float


@dataclass
class TileFetchArgs:
    """Arguments for fetching a single tile's elevation data."""
    tx: int
    ty: int
    centre_lat: float
    centre_lon: float
    tile_radius: float
    tiles_x: int
    tiles_y: int
    use_cache: bool


@dataclass
class TileProcessArgs:
    """Arguments for processing a single tile into a mesh."""
    tx: int
    ty: int
    elevation: np.ndarray
    tile_size_mm: float
    tile_radius_km: float
    base_height: float
    exaggeration: float
    output_dir: str
    output_format: str


@dataclass
class TileResult:
    """Result from processing a single tile."""
    filename: str
    size_mb: float
    max_z: float


# =============================================================================
# Utility Functions
# =============================================================================

def _calculate_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: The current attempt number (0-indexed)

    Returns:
        Delay in seconds to wait before next retry
    """
    return (2 ** attempt) + np.random.random()


# =============================================================================
# Coordinate Functions
# =============================================================================

def lat_lon_to_osgb(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 latitude/longitude to British National Grid coordinates.

    Args:
        lat: Latitude in decimal degrees (WGS84)
        lon: Longitude in decimal degrees (WGS84)

    Returns:
        Tuple of (easting, northing) in OSGB meters
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return transformer.transform(lon, lat)


def get_bbox(lat: float, lon: float, radius_x_km: float, radius_y_km: float) -> BoundingBox:
    """Calculate bounding box in OSGB coordinates from a centre point.

    Args:
        lat: Centre latitude in decimal degrees
        lon: Centre longitude in decimal degrees
        radius_x_km: Half-width of the box in kilometers
        radius_y_km: Half-height of the box in kilometers

    Returns:
        BoundingBox with min/max easting and northing in meters
    """
    easting, northing = lat_lon_to_osgb(lat, lon)
    radius_x_m = radius_x_km * 1000
    radius_y_m = radius_y_km * 1000
    return BoundingBox(
        min_easting=easting - radius_x_m,
        min_northing=northing - radius_y_m,
        max_easting=easting + radius_x_m,
        max_northing=northing + radius_y_m
    )


# =============================================================================
# Data Fetching Functions
# =============================================================================

def fetch_elevation(bbox: BoundingBox, use_cache: bool = True, quiet: bool = False) -> np.ndarray:
    """Fetch elevation data from the Environment Agency LIDAR API.

    Args:
        bbox: Bounding box defining the area to fetch
        use_cache: Whether to use/store cached data
        quiet: Suppress progress output

    Returns:
        2D numpy array of elevation values in meters

    Raises:
        APIError: If the API request fails after all retries
    """
    # Check cache first
    cache_key = hashlib.md5(
        f"{bbox.min_easting},{bbox.min_northing},{bbox.max_easting},{bbox.max_northing}".encode()
    ).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")

    if use_cache and os.path.exists(cache_path):
        if not quiet:
            print("Loading from cache...")
        elevation = np.load(cache_path)
        if not quiet:
            print(f"  Cached size: {elevation.shape[1]}x{elevation.shape[0]} pixels")
            print(f"  Elevation: {np.nanmin(elevation):.1f}m to {np.nanmax(elevation):.1f}m")
        return elevation

    url = (
        f"{WCS_BASE_URL}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&CoverageId={COVERAGE_ID}&format=image/tiff"
        f"&subset=E({bbox.min_easting:.0f},{bbox.max_easting:.0f})"
        f"&subset=N({bbox.min_northing:.0f},{bbox.max_northing:.0f})"
    )

    if not quiet:
        print("Fetching elevation data...")

    # Retry with exponential backoff
    response = None
    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limited
                wait = _calculate_backoff_delay(attempt)
                time.sleep(wait)
            else:
                raise APIError(response.status_code, "WCS request failed")
        except requests.exceptions.ConnectionError:
            if attempt < MAX_API_RETRIES - 1:
                wait = _calculate_backoff_delay(attempt)
                time.sleep(wait)
            else:
                raise

    if response is None or response.status_code != 200:
        status = response.status_code if response else "No response"
        raise APIError(status, f"WCS request failed after {MAX_API_RETRIES} retries")

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            elevation = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan
            elevation[elevation < INVALID_ELEVATION_THRESHOLD] = np.nan

    if not quiet:
        print(f"  Raw size: {elevation.shape[1]}x{elevation.shape[0]} pixels")
        print(f"  Elevation: {np.nanmin(elevation):.1f}m to {np.nanmax(elevation):.1f}m")

    # Save to cache
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, elevation)
        if not quiet:
            print(f"  Cached to {cache_path}")

    return elevation


def fetch_tile_data(args: TileFetchArgs) -> tuple[int, int, np.ndarray]:
    """Fetch elevation data for a single tile.

    Used for parallel fetching where each tile is requested separately.

    Args:
        args: Tile fetch arguments including position and configuration

    Returns:
        Tuple of (tile_x, tile_y, elevation_array)
    """
    # Calculate tile centre in OSGB coordinates
    centre_e, centre_n = lat_lon_to_osgb(args.centre_lat, args.centre_lon)

    # Offset from grid centre to tile centre
    tile_offset_x = (args.tx - (args.tiles_x - 1) / 2) * args.tile_radius * 2 * 1000
    tile_offset_y = (args.ty - (args.tiles_y - 1) / 2) * args.tile_radius * 2 * 1000

    tile_centre_e = centre_e + tile_offset_x
    tile_centre_n = centre_n + tile_offset_y

    # Get bbox for this tile
    radius_m = args.tile_radius * 1000
    bbox = BoundingBox(
        min_easting=tile_centre_e - radius_m,
        min_northing=tile_centre_n - radius_m,
        max_easting=tile_centre_e + radius_m,
        max_northing=tile_centre_n + radius_m
    )

    elevation = fetch_elevation(bbox, use_cache=args.use_cache, quiet=True)
    return args.tx, args.ty, elevation


# =============================================================================
# Data Processing Functions
# =============================================================================

def process_elevation(
    elevation: np.ndarray,
    target_res: int,
    smooth: float,
    quiet: bool = False
) -> np.ndarray:
    """Downsample and smooth the elevation data.

    Args:
        elevation: Raw elevation array from API
        target_res: Target resolution (grid cells per dimension)
        smooth: Gaussian smoothing sigma (0 = no smoothing)
        quiet: Suppress progress output

    Returns:
        Processed elevation array
    """
    height, width = elevation.shape
    factor = max(1, max(height, width) // target_res)

    if factor > 1:
        if not quiet:
            print(f"  Downsampling by {factor}x...")
        new_h, new_w = height // factor, width // factor
        trimmed = elevation[:new_h * factor, :new_w * factor]
        reshaped = trimmed.reshape(new_h, factor, new_w, factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elevation = np.nanmean(reshaped, axis=(1, 3))
        if not quiet:
            print(f"  New size: {elevation.shape[1]}x{elevation.shape[0]}")

    if smooth > 0:
        if not quiet:
            print(f"  Smoothing (sigma={smooth})...")
        elevation = ndimage.gaussian_filter(elevation, sigma=smooth)

    # Flip horizontally to correct orientation
    elevation = np.fliplr(elevation)

    return elevation


# =============================================================================
# Mesh Generation Functions
# =============================================================================

def _generate_surface_triangles(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_values: np.ndarray,
    is_bottom: bool = False
) -> np.ndarray:
    """Generate triangles for a horizontal surface (top terrain or bottom base).

    Args:
        x_coords: Array of X coordinate values
        y_coords: Array of Y coordinate values
        z_values: 2D array of Z values (terrain heights or zeros for bottom)
        is_bottom: If True, reverse winding order for outward-facing normals

    Returns:
        Array of triangles with shape (n_triangles, 3, 3)
    """
    height, width = z_values.shape

    # Create grid indices
    y_idx, x_idx = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing='ij')
    y_idx = y_idx.flatten()
    x_idx = x_idx.flatten()

    n_quads = (height - 1) * (width - 1)
    triangles = np.zeros((n_quads * 2, 3, 3), dtype=np.float32)

    if is_bottom:
        # Bottom surface - reversed winding for outward normals
        triangles[0::2, 0] = np.column_stack([x_coords[x_idx], y_coords[y_idx], z_values[y_idx, x_idx]])
        triangles[0::2, 1] = np.column_stack([x_coords[x_idx], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx]])
        triangles[0::2, 2] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx], z_values[y_idx, x_idx + 1]])
        triangles[1::2, 0] = np.column_stack([x_coords[x_idx], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx]])
        triangles[1::2, 1] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx + 1]])
        triangles[1::2, 2] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx], z_values[y_idx, x_idx + 1]])
    else:
        # Top surface - standard winding
        triangles[0::2, 0] = np.column_stack([x_coords[x_idx], y_coords[y_idx], z_values[y_idx, x_idx]])
        triangles[0::2, 1] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx], z_values[y_idx, x_idx + 1]])
        triangles[0::2, 2] = np.column_stack([x_coords[x_idx], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx]])
        triangles[1::2, 0] = np.column_stack([x_coords[x_idx], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx]])
        triangles[1::2, 1] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx], z_values[y_idx, x_idx + 1]])
        triangles[1::2, 2] = np.column_stack([x_coords[x_idx + 1], y_coords[y_idx + 1], z_values[y_idx + 1, x_idx + 1]])

    return triangles


def _generate_wall_triangles(
    axis_coords: np.ndarray,
    perp_coord: float,
    terrain_edge: np.ndarray,
    wall_type: str
) -> np.ndarray:
    """Generate triangles for a single wall of the mesh.

    Args:
        axis_coords: Coordinate values along the wall
        perp_coord: Fixed coordinate value perpendicular to the wall
        terrain_edge: Terrain Z values along this edge
        wall_type: One of 'south', 'north', 'west', 'east'

    Returns:
        Array of triangles with shape (n_triangles, 3, 3)
    """
    n = len(axis_coords) - 1
    triangles = np.zeros((n * 2, 3, 3), dtype=np.float32)
    idx = np.arange(n)

    if wall_type == 'south':
        # South wall (y = 0, varying x)
        triangles[0::2, 0] = np.column_stack([axis_coords[idx], np.zeros(n), np.zeros(n)])
        triangles[0::2, 1] = np.column_stack([axis_coords[idx], np.zeros(n), terrain_edge[idx]])
        triangles[0::2, 2] = np.column_stack([axis_coords[idx + 1], np.zeros(n), np.zeros(n)])
        triangles[1::2, 0] = np.column_stack([axis_coords[idx + 1], np.zeros(n), np.zeros(n)])
        triangles[1::2, 1] = np.column_stack([axis_coords[idx], np.zeros(n), terrain_edge[idx]])
        triangles[1::2, 2] = np.column_stack([axis_coords[idx + 1], np.zeros(n), terrain_edge[idx + 1]])

    elif wall_type == 'north':
        # North wall (y = perp_coord, varying x)
        triangles[0::2, 0] = np.column_stack([axis_coords[idx], np.full(n, perp_coord), np.zeros(n)])
        triangles[0::2, 1] = np.column_stack([axis_coords[idx + 1], np.full(n, perp_coord), np.zeros(n)])
        triangles[0::2, 2] = np.column_stack([axis_coords[idx], np.full(n, perp_coord), terrain_edge[idx]])
        triangles[1::2, 0] = np.column_stack([axis_coords[idx + 1], np.full(n, perp_coord), np.zeros(n)])
        triangles[1::2, 1] = np.column_stack([axis_coords[idx + 1], np.full(n, perp_coord), terrain_edge[idx + 1]])
        triangles[1::2, 2] = np.column_stack([axis_coords[idx], np.full(n, perp_coord), terrain_edge[idx]])

    elif wall_type == 'west':
        # West wall (x = 0, varying y)
        triangles[0::2, 0] = np.column_stack([np.zeros(n), axis_coords[idx], np.zeros(n)])
        triangles[0::2, 1] = np.column_stack([np.zeros(n), axis_coords[idx + 1], np.zeros(n)])
        triangles[0::2, 2] = np.column_stack([np.zeros(n), axis_coords[idx], terrain_edge[idx]])
        triangles[1::2, 0] = np.column_stack([np.zeros(n), axis_coords[idx + 1], np.zeros(n)])
        triangles[1::2, 1] = np.column_stack([np.zeros(n), axis_coords[idx + 1], terrain_edge[idx + 1]])
        triangles[1::2, 2] = np.column_stack([np.zeros(n), axis_coords[idx], terrain_edge[idx]])

    elif wall_type == 'east':
        # East wall (x = perp_coord, varying y)
        triangles[0::2, 0] = np.column_stack([np.full(n, perp_coord), axis_coords[idx], np.zeros(n)])
        triangles[0::2, 1] = np.column_stack([np.full(n, perp_coord), axis_coords[idx], terrain_edge[idx]])
        triangles[0::2, 2] = np.column_stack([np.full(n, perp_coord), axis_coords[idx + 1], np.zeros(n)])
        triangles[1::2, 0] = np.column_stack([np.full(n, perp_coord), axis_coords[idx + 1], np.zeros(n)])
        triangles[1::2, 1] = np.column_stack([np.full(n, perp_coord), axis_coords[idx], terrain_edge[idx]])
        triangles[1::2, 2] = np.column_stack([np.full(n, perp_coord), axis_coords[idx + 1], terrain_edge[idx + 1]])

    return triangles


def generate_tile_mesh(
    elevation: np.ndarray,
    tile_size_mm: float,
    tile_radius_km: float,
    base_height: float,
    exaggeration: float = 1.0
) -> tuple[mesh.Mesh, float]:
    """Generate a 3D mesh for a single terrain tile.

    Creates a watertight mesh with terrain on top, flat base on bottom,
    and vertical walls connecting them.

    Args:
        elevation: 2D array of elevation values in meters
        tile_size_mm: Output tile size in millimeters
        tile_radius_km: Tile radius in kilometers (half the tile width)
        base_height: Thickness of the base in millimeters
        exaggeration: Vertical exaggeration factor (1.0 = true scale)

    Returns:
        Tuple of (stl_mesh, max_z) where max_z is the maximum height in mm
    """
    height, width = elevation.shape

    # Calculate scales
    tile_width_m = tile_radius_km * 2 * 1000
    horizontal_scale = tile_size_mm / tile_width_m
    vertical_scale = horizontal_scale * exaggeration

    xy_scale = tile_size_mm / (width - 1)
    out_width = (width - 1) * xy_scale
    out_height = (height - 1) * xy_scale

    # Calculate terrain Z values
    min_elev = np.nanmin(elevation)
    terrain_z = base_height + (elevation - min_elev) * vertical_scale
    max_z = np.nanmax(terrain_z)
    terrain_z = np.nan_to_num(terrain_z, nan=base_height)

    # Create coordinate grids
    x_coords = np.arange(width) * xy_scale
    y_coords = np.arange(height) * xy_scale

    # Create bottom surface (all zeros)
    bottom_z = np.zeros_like(terrain_z)

    # Generate all surfaces
    top_triangles = _generate_surface_triangles(x_coords, y_coords, terrain_z, is_bottom=False)
    bottom_triangles = _generate_surface_triangles(x_coords, y_coords, bottom_z, is_bottom=True)

    # Generate walls
    south_triangles = _generate_wall_triangles(x_coords, 0, terrain_z[0, :], 'south')
    north_triangles = _generate_wall_triangles(x_coords, out_height, terrain_z[height - 1, :], 'north')
    west_triangles = _generate_wall_triangles(y_coords, 0, terrain_z[:, 0], 'west')
    east_triangles = _generate_wall_triangles(y_coords, out_width, terrain_z[:, width - 1], 'east')

    # Combine all triangles
    all_triangles = np.vstack([
        top_triangles,
        bottom_triangles,
        south_triangles,
        north_triangles,
        west_triangles,
        east_triangles
    ])

    # Create mesh
    stl_mesh = mesh.Mesh(np.zeros(len(all_triangles), dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = all_triangles

    return stl_mesh, max_z


def process_single_tile(args: TileProcessArgs) -> TileResult:
    """Process a single tile into a mesh file.

    Generates the mesh and saves it to disk in the specified format.

    Args:
        args: Tile processing arguments

    Returns:
        TileResult with filename, size, and max height
    """
    stl_mesh, max_z = generate_tile_mesh(
        args.elevation,
        tile_size_mm=args.tile_size_mm,
        tile_radius_km=args.tile_radius_km,
        base_height=args.base_height,
        exaggeration=args.exaggeration
    )

    ext = "3mf" if args.output_format == "3mf" else "stl"
    filename = f"tile_{args.tx}_{args.ty}.{ext}"
    filepath = os.path.join(args.output_dir, filename)

    if args.output_format == "3mf":
        # Save as STL first, then convert to 3MF
        temp_path = filepath.replace(".3mf", "_temp.stl")
        try:
            stl_mesh.save(temp_path)
            tmesh = trimesh.load(temp_path)
            tmesh.export(filepath)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        stl_mesh.save(filepath)

    size_mb = os.path.getsize(filepath) / 1024 / 1024
    return TileResult(filename=filename, size_mb=size_mb, max_z=max_z)


# =============================================================================
# Parallel Processing Wrappers
# =============================================================================

def _fetch_tile_wrapper(args_tuple: tuple) -> tuple[int, int, np.ndarray]:
    """Wrapper to unpack tuple args for parallel fetch execution."""
    args = TileFetchArgs(*args_tuple)
    return fetch_tile_data(args)


def _process_tile_wrapper(args_tuple: tuple) -> TileResult:
    """Wrapper to unpack tuple args for parallel mesh processing."""
    args = TileProcessArgs(*args_tuple)
    return process_single_tile(args)


# =============================================================================
# Main Function Components
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='UK Terrain Tiles for Resin Printing')
    parser.add_argument('--lat', type=float, required=True, help='Centre latitude')
    parser.add_argument('--lon', type=float, required=True, help='Centre longitude')
    parser.add_argument('--tiles', type=str, required=True, help='Tile grid e.g. 3x3')
    parser.add_argument('--tile-radius', type=float, required=True, help='Radius per tile in km')
    parser.add_argument('--tile-size', type=float, default=120, help='Tile output size in mm')
    parser.add_argument(
        '--exaggeration', type=float, default=2.0,
        help='Vertical exaggeration (1=true scale, 2=2x height)'
    )
    parser.add_argument('--base', type=float, default=2.0, help='Base thickness mm')
    parser.add_argument('--resolution', type=int, default=1000, help='Grid resolution per tile')
    parser.add_argument('--smooth', type=float, default=0.5, help='Smoothing (0=none)')
    parser.add_argument('--output', type=str, default='tiles/', help='Output folder')
    parser.add_argument(
        '--format', type=str, choices=['stl', '3mf'], default='stl',
        help='Output format (3mf is smaller)'
    )
    parser.add_argument('--no-cache', action='store_true', help='Disable LIDAR data caching')
    parser.add_argument(
        '--parallel', type=int, default=0,
        help='Parallel workers for mesh generation (0=sequential)'
    )
    parser.add_argument(
        '--parallel-fetch', type=int, default=0,
        help='Parallel API requests (0=single request, 4-8 recommended)'
    )

    return parser.parse_args()


def print_configuration(
    args: argparse.Namespace,
    tiles_x: int,
    tiles_y: int,
    tile_radius: float
) -> None:
    """Print the configuration summary.

    Args:
        args: Parsed command line arguments
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        tile_radius: Radius of each tile in km
    """
    total_radius_x = tile_radius * tiles_x
    total_radius_y = tile_radius * tiles_y
    scale = (tile_radius * 2 * 1000 * 1000) / args.tile_size

    print(f"\n=== Terrain Tiles ===")
    print(f"Centre: {args.lat}, {args.lon}")
    print(f"Grid: {tiles_x}x{tiles_y} tiles")
    print(f"Per tile: {tile_radius * 2}km × {tile_radius * 2}km → {args.tile_size}mm")
    print(f"Total area: {total_radius_x * 2}km × {total_radius_y * 2}km")
    print(f"Scale: 1:{scale:.0f}")
    print(f"Vertical exaggeration: {args.exaggeration}x")
    print(f"Output format: {args.format.upper()}")
    print()


def fetch_elevations_parallel(
    args: argparse.Namespace,
    tiles_x: int,
    tiles_y: int,
    tile_radius: float
) -> dict[tuple[int, int], np.ndarray]:
    """Fetch elevation data for all tiles using parallel requests.

    Args:
        args: Parsed command line arguments
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        tile_radius: Radius of each tile in km

    Returns:
        Dictionary mapping (tx, ty) to processed elevation arrays
    """
    print(f"Fetching {tiles_x * tiles_y} tiles in parallel ({args.parallel_fetch} workers)...")

    fetch_tasks = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            fetch_tasks.append((
                tx, ty, args.lat, args.lon, tile_radius,
                tiles_x, tiles_y, not args.no_cache
            ))

    tile_elevations = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.parallel_fetch) as executor:
        futures = {executor.submit(_fetch_tile_wrapper, task): task for task in fetch_tasks}
        for future in as_completed(futures):
            tx, ty, elevation = future.result()
            processed = process_elevation(elevation, args.resolution, args.smooth, quiet=True)
            # Flip Y for correct orientation
            tile_elevations[(tx, tiles_y - 1 - ty)] = processed
            completed += 1
            print(f"  Fetched {completed}/{len(fetch_tasks)} tiles", end='\r')

    print(f"  Fetched {completed}/{len(fetch_tasks)} tiles")
    return tile_elevations


def fetch_elevations_bulk(
    args: argparse.Namespace,
    tiles_x: int,
    tiles_y: int,
    tile_radius: float
) -> dict[tuple[int, int], np.ndarray]:
    """Fetch elevation data for all tiles using a single bulk request.

    Args:
        args: Parsed command line arguments
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        tile_radius: Radius of each tile in km

    Returns:
        Dictionary mapping (tx, ty) to elevation arrays
    """
    total_radius_x = tile_radius * tiles_x
    total_radius_y = tile_radius * tiles_y

    bbox = get_bbox(args.lat, args.lon, total_radius_x, total_radius_y)
    full_elevation = fetch_elevation(bbox, use_cache=not args.no_cache)

    # Process full area
    target_res = args.resolution * max(tiles_x, tiles_y)
    full_elevation = process_elevation(full_elevation, target_res, args.smooth)

    # Split into tiles
    height, width = full_elevation.shape
    tile_h = height // tiles_y
    tile_w = width // tiles_x

    tile_elevations = {}
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y_start = (tiles_y - 1 - ty) * tile_h
            y_end = y_start + tile_h
            x_start = tx * tile_w
            x_end = x_start + tile_w
            tile_elevations[(tx, ty)] = full_elevation[y_start:y_end, x_start:x_end].copy()

    return tile_elevations


def generate_meshes(
    tile_elevations: dict[tuple[int, int], np.ndarray],
    args: argparse.Namespace,
    tiles_x: int,
    tiles_y: int,
    tile_radius: float,
    global_min: float,
    global_max: float
) -> None:
    """Generate and save mesh files for all tiles.

    Args:
        tile_elevations: Dictionary mapping (tx, ty) to elevation arrays
        args: Parsed command line arguments
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        tile_radius: Radius of each tile in km
        global_min: Minimum elevation across all tiles
        global_max: Maximum elevation across all tiles
    """
    # Prepare tile tasks for mesh generation
    tile_tasks = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_elev = tile_elevations[(tx, ty)]
            tile_tasks.append((
                tx, ty, tile_elev, args.tile_size, tile_radius,
                args.base, args.exaggeration, args.output, args.format
            ))

    print(f"\nGenerating {len(tile_tasks)} tiles...")

    if args.parallel > 0:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(_process_tile_wrapper, task): task
                for task in tile_tasks
            }
            for future in as_completed(futures):
                result = future.result()
                print(f"  {result.filename}: {result.size_mb:.1f} MB, height: {result.max_z:.1f}mm")
    else:
        # Sequential processing
        for task in tile_tasks:
            result = _process_tile_wrapper(task)
            print(f"  {result.filename}: {result.size_mb:.1f} MB, height: {result.max_z:.1f}mm")


def print_summary(tiles_x: int, tiles_y: int, output_dir: str) -> None:
    """Print completion summary and tile layout.

    Args:
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        output_dir: Output directory path
    """
    print(f"\nDone! {tiles_x * tiles_y} tiles saved to {output_dir}")
    print(f"\nPrint layout (bottom-left origin):")
    for ty in range(tiles_y - 1, -1, -1):
        row = " ".join([f"[{tx},{ty}]" for tx in range(tiles_x)])
        print(f"  {row}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the terrain tiles generator."""
    args = parse_arguments()

    # Parse tile grid
    tiles_x, tiles_y = map(int, args.tiles.lower().split('x'))
    tile_radius = args.tile_radius

    # Print configuration
    print_configuration(args, tiles_x, tiles_y, tile_radius)

    # Create output folder
    os.makedirs(args.output, exist_ok=True)

    # Fetch elevation data
    if args.parallel_fetch > 0:
        tile_elevations = fetch_elevations_parallel(args, tiles_x, tiles_y, tile_radius)
    else:
        tile_elevations = fetch_elevations_bulk(args, tiles_x, tiles_y, tile_radius)

    # Calculate global elevation range
    all_elevs = np.concatenate([e.flatten() for e in tile_elevations.values()])
    global_min = np.nanmin(all_elevs)
    global_max = np.nanmax(all_elevs)
    elevation_range = global_max - global_min

    # Calculate expected relief
    tile_width_m = tile_radius * 2 * 1000
    horizontal_scale = args.tile_size / tile_width_m
    expected_relief = elevation_range * horizontal_scale * args.exaggeration

    print(f"\nGlobal elevation range: {global_min:.1f}m to {global_max:.1f}m ({elevation_range:.0f}m)")
    print(f"Expected relief: {expected_relief:.1f}mm (+ {args.base}mm base)")

    # Generate meshes
    generate_meshes(
        tile_elevations, args, tiles_x, tiles_y, tile_radius, global_min, global_max
    )

    # Print summary
    print_summary(tiles_x, tiles_y, args.output)


if __name__ == "__main__":
    main()
