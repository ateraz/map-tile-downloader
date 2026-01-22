#!/usr/bin/env python3
"""
Download a GeoTIFF by center point, width/height in meters, and zoom level.

Usage:
    python download_geotiff.py --lat 40.7128 --lon -74.0060 --width 1000 --height 1000 --zoom 15
    python download_geotiff.py --lat 40.7128 --lon -74.0060 --width 1000 --height 1000 --zoom 15 --source "Google Satellite"
"""

import argparse
import math
import random
import time
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import mercantile
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds


# Base directory for caching tiles
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / 'tile-cache'
DOWNLOADS_DIR = BASE_DIR / 'downloads'
CONFIG_DIR = BASE_DIR / 'config'
MAP_SOURCES_FILE = CONFIG_DIR / 'map_sources.json'

CACHE_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)


def load_map_sources():
    """Load map sources from config file."""
    if MAP_SOURCES_FILE.exists():
        with open(MAP_SOURCES_FILE, 'r') as f:
            return json.load(f)
    else:
        print("Warning: map_sources.json not found. Using default OSM source.")
        return {"Standard OSM": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"}


def meters_to_degrees(lat, meters):
    """
    Convert meters to degrees at a given latitude.
    Returns (delta_lat, delta_lon) for the given distance in meters.
    """
    # Earth's radius in meters
    earth_radius = 6378137.0

    # Latitude: 1 degree is approximately the same distance everywhere
    delta_lat = meters / earth_radius * (180.0 / math.pi)

    # Longitude: depends on latitude (gets smaller towards poles)
    delta_lon = meters / (earth_radius * math.cos(math.radians(lat))) * (180.0 / math.pi)

    return delta_lat, delta_lon


def calculate_bbox(center_lat, center_lon, width_meters, height_meters):
    """
    Calculate bounding box from center point and dimensions in meters.
    Returns (west, south, east, north) in degrees.
    """
    delta_lat, delta_lon = meters_to_degrees(center_lat, height_meters / 2)
    _, delta_lon_width = meters_to_degrees(center_lat, width_meters / 2)

    north = center_lat + delta_lat
    south = center_lat - delta_lat
    east = center_lon + delta_lon_width
    west = center_lon - delta_lon_width

    return west, south, east, north


def sanitize_style_name(style_name):
    """Convert map style name to a filesystem-safe directory name."""
    style_name = re.sub(r'\s+', '-', style_name)
    style_name = re.sub(r'[^a-zA-Z0-9-_]', '', style_name)
    return style_name


def get_style_cache_dir(style_name):
    """Get the cache directory path for a given map style name."""
    sanitized_name = sanitize_style_name(style_name)
    return CACHE_DIR / sanitized_name


def download_tile(tile, map_style_url, style_cache_dir, max_retries=3, verbose=False):
    """Download a single tile with retries if not in cache."""
    tile_dir = style_cache_dir / str(tile.z) / str(tile.x)
    tile_path = tile_dir / f"{tile.y}.png"

    if tile_path.exists():
        return tile_path

    subdomain = random.choice(['a', 'b', 'c']) if '{s}' in map_style_url else ''
    url = map_style_url.replace('{s}', subdomain).replace('{z}', str(tile.z)).replace('{x}', str(tile.x)).replace('{y}', str(tile.y))
    headers = {'User-Agent': 'MapTileDownloader/1.0'}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tile_dir.mkdir(parents=True, exist_ok=True)
                with open(tile_path, 'wb') as f:
                    f.write(response.content)
                return tile_path
            else:
                if verbose:
                    print(f"\n  Failed ({response.status_code}): {tile.z}/{tile.x}/{tile.y}, attempt {attempt + 1}")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            if verbose:
                print(f"\n  Error: {tile.z}/{tile.x}/{tile.y}, attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)

    return None


def get_tiles_for_bbox(west, south, east, north, zoom):
    """Get all tiles that cover the bounding box at the given zoom level."""
    tiles = list(mercantile.tiles(west, south, east, north, zooms=[zoom]))
    return tiles


def download_tiles(tiles, map_style_url, style_cache_dir, max_workers=5):
    """Download all tiles with parallel execution and progress reporting."""
    total = len(tiles)
    print(f"Downloading {total} tiles...")

    downloaded = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_tile, tile, map_style_url, style_cache_dir): tile for tile in tiles}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                downloaded.append(futures[future])
            else:
                failed += 1

            # Progress bar
            percent = (completed / total) * 100
            bar_len = 40
            filled = int(bar_len * completed // total)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  Progress: |{bar}| {percent:.1f}% ({completed}/{total})", end='', flush=True)

    print()  # New line after progress bar
    print(f"Successfully downloaded/cached {len(downloaded)}/{total} tiles ({failed} failed)")
    return downloaded


def create_geotiff(tiles, style_cache_dir, output_path):
    """Create a GeoTIFF from downloaded tiles by stitching them together."""
    if not tiles:
        return None

    zoom = tiles[0].z

    # Calculate the bounding box and dimensions
    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)

    # Get the geographic bounds
    nw_tile = mercantile.Tile(min_x, min_y, zoom)
    se_tile = mercantile.Tile(max_x, max_y, zoom)
    nw_bounds = mercantile.bounds(nw_tile)
    se_bounds = mercantile.bounds(se_tile)

    west = nw_bounds.west
    north = nw_bounds.north
    east = se_bounds.east
    south = se_bounds.south

    # Calculate dimensions (each tile is 256x256 pixels)
    tile_size = 256
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size

    # Create the output array
    output_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Stitch tiles together
    total_tiles = len(tiles)
    print(f"Stitching {total_tiles} tiles...")
    for i, tile in enumerate(tiles):
        tile_path = style_cache_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png"
        if tile_path.exists():
            with Image.open(tile_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Calculate position in output array
                x_offset = (tile.x - min_x) * tile_size
                y_offset = (tile.y - min_y) * tile_size

                # Place tile in output array
                img_array = np.array(img)
                output_array[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size] = img_array

        # Stitching progress
        percent = ((i + 1) / total_tiles) * 100
        bar_len = 40
        filled = int(bar_len * (i + 1) // total_tiles)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  Progress: |{bar}| {percent:.1f}% ({i + 1}/{total_tiles})", end='', flush=True)

    print()  # New line after progress bar

    # Create transform for WGS84 (EPSG:4326)
    transform = from_bounds(west, south, east, north, width, height)

    # Write the GeoTIFF
    print(f"Writing GeoTIFF to {output_path}...")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,  # RGB
        dtype=rasterio.uint8,
        crs='EPSG:4326',  # WGS84
        transform=transform,
        # compress='lzw'
    ) as dst:
        # Write each band
        for band in range(3):
            dst.write(output_array[:, :, band], band + 1)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Download a GeoTIFF by center point, dimensions in meters, and zoom level.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 1km x 1km area around NYC at zoom 15
  python download_geotiff.py --lat 40.7128 --lon -74.0060 --width 1000 --height 1000 --zoom 15

  # Download using Google Satellite imagery
  python download_geotiff.py --lat 40.7128 --lon -74.0060 --width 500 --height 500 --zoom 17 --source "Google Satellite"

  # List available map sources
  python download_geotiff.py --list-sources
        """
    )

    parser.add_argument('--lat', type=float, help='Center latitude in degrees')
    parser.add_argument('--lon', type=float, help='Center longitude in degrees')
    parser.add_argument('--width', type=float, help='Width in meters')
    parser.add_argument('--height', type=float, help='Height in meters')
    parser.add_argument('--zoom', type=int, help='Zoom level (0-19)')
    parser.add_argument('--source', type=str, default='Esri World Imagery Satellite', help='Map source name (default: Standard OSM)')
    parser.add_argument('--output', type=str, help='Output file path (default: downloads/<source>_<zoom>.tif)')
    parser.add_argument('--list-sources', action='store_true', help='List available map sources')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel download workers (default: 5)')

    args = parser.parse_args()

    # Load map sources
    map_sources = load_map_sources()

    # List sources if requested
    if args.list_sources:
        print("Available map sources:")
        for name in map_sources.keys():
            print(f"  - {name}")
        return

    # Validate required arguments
    if args.lat is None or args.lon is None or args.width is None or args.height is None or args.zoom is None:
        parser.error("--lat, --lon, --width, --height, and --zoom are required (unless using --list-sources)")

    # Validate zoom level
    if args.zoom < 0 or args.zoom > 19:
        parser.error("Zoom level must be between 0 and 19")

    # Validate map source
    if args.source not in map_sources:
        print(f"Error: Unknown map source '{args.source}'")
        print("Available sources:")
        for name in map_sources.keys():
            print(f"  - {name}")
        return

    map_style_url = map_sources[args.source]
    style_cache_dir = get_style_cache_dir(args.source)

    # Calculate bounding box
    west, south, east, north = calculate_bbox(args.lat, args.lon, args.width, args.height)

    print(f"Center: ({args.lat}, {args.lon})")
    print(f"Dimensions: {args.width}m x {args.height}m")
    print(f"Bounding box: W={west:.6f}, S={south:.6f}, E={east:.6f}, N={north:.6f}")
    print(f"Zoom level: {args.zoom}")
    print(f"Map source: {args.source}")

    # Get tiles for the bounding box
    tiles = get_tiles_for_bbox(west, south, east, north, args.zoom)
    print(f"Tiles needed: {len(tiles)}")

    if not tiles:
        print("Error: No tiles found for the specified area")
        return

    # Download tiles
    downloaded_tiles = download_tiles(tiles, map_style_url, style_cache_dir, args.workers)

    if not downloaded_tiles:
        print("Error: Failed to download any tiles")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        sanitized_name = sanitize_style_name(args.source)
        output_path = DOWNLOADS_DIR / f'{sanitized_name}_z{args.zoom}_{args.lat:.4f}_{args.lon:.4f}.tif'

    # Create GeoTIFF
    result = create_geotiff(downloaded_tiles, style_cache_dir, output_path)

    if result:
        print(f"\nGeoTIFF created successfully: {result}")
    else:
        print("\nError: Failed to create GeoTIFF")


if __name__ == '__main__':
    main()
