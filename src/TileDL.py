import subprocess
import sys
import os
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit
import mercantile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import zipfile
import random
import shutil
import re
import time
import json
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import threading
from PIL import Image
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# Base directory for caching tiles, absolute path relative to script location
BASE_DIR = Path(__file__).parent.parent  # Root of map-tile-downloader
CACHE_DIR = BASE_DIR / 'tile-cache' ## Note: Fixed referece location for cached tiles
DOWNLOADS_DIR = BASE_DIR / 'downloads'
CACHE_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

## Note: Moved to 'utils/dependency_installer.py'
# Ensure dependencies are installed
#def install_dependencies():
#    try:
#        with open('requirements.txt', 'r') as f:
#            requirements = f.read().splitlines()
#        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *requirements])
#    except subprocess.CalledProcessError as e:
#        print(f"Failed to install dependencies: {e}")
#        sys.exit(1)
#
## Install dependencies on startup if not already installed
#install_dependencies()

app = Flask(__name__, template_folder='../templates')
socketio = SocketIO(app)


# Load map sources from config file
CONFIG_DIR = Path('config')
MAP_SOURCES_FILE = CONFIG_DIR / 'map_sources.json'
MAP_SOURCES = {}
if MAP_SOURCES_FILE.exists():
    with open(MAP_SOURCES_FILE, 'r') as f:
        MAP_SOURCES = json.load(f)
else:
    print("Warning: map_sources.json not found. No map sources available.")
    sys.exit(1)

# Global event for cancellation
download_event = threading.Event()

def sanitize_style_name(style_name):
    """Convert map style name to a filesystem-safe directory name."""
    style_name = re.sub(r'\s+', '-', style_name)  # Replace spaces with hyphens
    style_name = re.sub(r'[^a-zA-Z0-9-_]', '', style_name)  # Remove non-alphanumeric except hyphens and underscores
    return style_name

def get_style_cache_dir(style_name):
    """Get the cache directory path for a given map style name."""
    sanitized_name = sanitize_style_name(style_name)
    return CACHE_DIR / sanitized_name

def download_tile(tile, map_style, style_cache_dir, convert_to_8bit, max_retries=3):
    """Download a single tile with retries if not cancelled and not in cache, converting to 8-bit if specified."""
    if not download_event.is_set():
        return None
    tile_dir = style_cache_dir / str(tile.z) / str(tile.x)
    tile_path = tile_dir / f"{tile.y}.png"
    if tile_path.exists():
        bounds = mercantile.bounds(tile)
        socketio.emit('tile_skipped', {
            'west': bounds.west,
            'south': bounds.south,
            'east': bounds.east,
            'north': bounds.north
        })
        return tile_path
    subdomain = random.choice(['a', 'b', 'c']) if '{s}' in map_style else ''
    url = map_style.replace('{s}', subdomain).replace('{z}', str(tile.z)).replace('{x}', str(tile.x)).replace('{y}', str(tile.y))
    headers = {'User-Agent': 'MapTileDownloader/1.0'}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tile_dir.mkdir(parents=True, exist_ok=True)
                with open(tile_path, 'wb') as f:
                    f.write(response.content)
                if convert_to_8bit:
                    with Image.open(tile_path) as img:
                        if img.mode != 'P':  # Only convert if not already 8-bit palette
                            img = img.quantize(colors=256)
                            img.save(tile_path)
                bounds = mercantile.bounds(tile)
                socketio.emit('tile_downloaded', {
                    'west': bounds.west,
                    'south': bounds.south,
                    'east': bounds.east,
                    'north': bounds.north
                })
                return tile_path
            else:
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.RequestException:
            time.sleep(2 ** attempt)  # Exponential backoff
    socketio.emit('tile_failed', {
        'tile': f"{tile.z}/{tile.x}/{tile.y}"
    })
    return None

def get_world_tiles():
    """Generate list of tiles for zoom levels 0 to 7 for the entire world."""
    tiles = []
    for z in range(8):  # 0 to 7 inclusive
        for x in range(2**z):
            for y in range(2**z):
                tiles.append(mercantile.Tile(x, y, z))
    return tiles

def get_tiles_for_polygons(polygons_data, min_zoom, max_zoom):
    """Generate list of tiles that intersect with the given polygons for the specified zoom range."""
    polygons = [Polygon([(lng, lat) for lat, lng in poly]) for poly in polygons_data]
    overall_polygon = unary_union(polygons)
    west, south, east, north = overall_polygon.bounds
    all_tiles = []
    for z in range(min_zoom, max_zoom + 1):
        tiles = mercantile.tiles(west, south, east, north, zooms=[z])
        for tile in tiles:
            tile_bbox = mercantile.bounds(tile)
            tile_box = box(tile_bbox.west, tile_bbox.south, tile_bbox.east, tile_bbox.north)
            if any(tile_box.intersects(poly) for poly in polygons):
                all_tiles.append(tile)
    all_tiles.sort(key=lambda tile: (tile.z, -tile.x, tile.y))
    return all_tiles

def download_tiles_with_retries(tiles, map_style, style_cache_dir, convert_to_8bit):
    """Download tiles with efficient retries using parallelism and adaptive backoff."""
    socketio.emit('download_started', {'total_tiles': len(tiles)})
    retry_queue = []
    max_workers = 5
    batch_size = 10
    
    def process_batch(batch):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_tile, tile, map_style, style_cache_dir, convert_to_8bit): tile for tile in batch}
            for future in as_completed(futures):
                if future.result() is None and download_event.is_set():
                    retry_queue.append(futures[future])
    
    while tiles and download_event.is_set():
        for i in range(0, len(tiles), batch_size):
            if not download_event.is_set():
                break
            batch = tiles[i:i + batch_size]
            process_batch(batch)
        tiles = retry_queue if retry_queue else []
        retry_queue = []
        if tiles:
            delay = min(2 ** len(retry_queue), 8)
            time.sleep(delay)
    
    if download_event.is_set():
        socketio.emit('tiles_downloaded')

def create_geotiff(tiles, style_cache_dir, style_name):
    """Create a GeoTIFF from downloaded tiles by stitching them together."""
    if not tiles:
        return None

    # Group tiles by zoom level
    tiles_by_zoom = {}
    for tile in tiles:
        if tile.z not in tiles_by_zoom:
            tiles_by_zoom[tile.z] = []
        tiles_by_zoom[tile.z].append(tile)

    # Use the highest zoom level for the output
    max_zoom = max(tiles_by_zoom.keys())
    tiles_at_max_zoom = tiles_by_zoom[max_zoom]

    # Calculate the bounding box and dimensions
    min_x = min(tile.x for tile in tiles_at_max_zoom)
    max_x = max(tile.x for tile in tiles_at_max_zoom)
    min_y = min(tile.y for tile in tiles_at_max_zoom)
    max_y = max(tile.y for tile in tiles_at_max_zoom)

    # Get the geographic bounds
    nw_tile = mercantile.Tile(min_x, min_y, max_zoom)
    se_tile = mercantile.Tile(max_x, max_y, max_zoom)
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
    for tile in tiles_at_max_zoom:
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

    # Create the GeoTIFF
    sanitized_name = sanitize_style_name(style_name)
    geotiff_path = DOWNLOADS_DIR / f'{sanitized_name}_z{max_zoom}.tif'

    # Create transform for Web Mercator (EPSG:3857)
    transform = from_bounds(west, south, east, north, width, height)

    # Write the GeoTIFF
    with rasterio.open(
        geotiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,  # RGB
        dtype=rasterio.uint8,
        crs='EPSG:4326',  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        # Write each band
        for band in range(3):
            dst.write(output_array[:, :, band], band + 1)

    socketio.emit('geotiff_created', {'path': str(geotiff_path)})
    return str(geotiff_path)

def create_zip(style_cache_dir, style_name):
    """Create a zip file from the style-specific cache directory in the downloads folder."""
    sanitized_name = sanitize_style_name(style_name)
    zip_path = DOWNLOADS_DIR / f'{sanitized_name}.zip'  # Absolute path
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(style_cache_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(style_cache_dir)
                zipf.write(file_path, arcname)
    return str(zip_path)  # Return as string for send_file

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/get_map_sources')
def get_map_sources():
    """Return the list of map sources from the config file."""
    return jsonify(MAP_SOURCES)

@socketio.on('start_download')
def handle_start_download(data):
    """Handle download request for tiles within polygons."""
    try:
        polygons_data = data['polygons']
        min_zoom = data['min_zoom']
        max_zoom = data['max_zoom']
        map_style_url = data['map_style']
        convert_to_8bit = data.get('convert_to_8bit', False)
        style_name = next(name for name, url in MAP_SOURCES.items() if url == map_style_url)
        style_cache_dir = get_style_cache_dir(style_name)
        if min_zoom < 0 or max_zoom > 19 or min_zoom > max_zoom:
            emit('error', {'message': 'Invalid zoom range (must be 0-19, min <= max)'})
            return
        if not polygons_data:
            emit('error', {'message': 'No polygons provided'})
            return
        tiles = get_tiles_for_polygons(polygons_data, min_zoom, max_zoom)
        download_event.set()
        download_tiles_with_retries(tiles, map_style_url, style_cache_dir, convert_to_8bit)
        if download_event.is_set():
            geotiff_path = create_geotiff(tiles, style_cache_dir, style_name)
            if geotiff_path:
                emit('download_complete', {'geotiff_url': f'/download_geotiff?path={geotiff_path}'})
            else:
                emit('error', {'message': 'Failed to create GeoTIFF'})
    except Exception as e:
        print(f"Error processing download: {e}")
        emit('error', {'message': 'An error occurred while processing your request'})

@socketio.on('start_world_download')
def handle_start_world_download(data):
    """Handle download request for world basemap tiles (zoom 0-7)."""
    try:
        map_style_url = data['map_style']
        convert_to_8bit = data.get('convert_to_8bit', False)
        style_name = next(name for name, url in MAP_SOURCES.items() if url == map_style_url)
        style_cache_dir = get_style_cache_dir(style_name)
        tiles = get_world_tiles()
        download_event.set()
        download_tiles_with_retries(tiles, map_style_url, style_cache_dir, convert_to_8bit)
        if download_event.is_set():
            geotiff_path = create_geotiff(tiles, style_cache_dir, style_name)
            if geotiff_path:
                emit('download_complete', {'geotiff_url': f'/download_geotiff?path={geotiff_path}'})
            else:
                emit('error', {'message': 'Failed to create GeoTIFF'})
    except Exception as e:
        print(f"Error processing world download: {e}")
        emit('error', {'message': 'An error occurred while processing your request'})

@socketio.on('cancel_download')
def handle_cancel_download():
    """Handle cancellation of the download."""
    download_event.clear()
    emit('download_cancelled')

@app.route('/download_geotiff')
def download_geotiff():
    """Send the GeoTIFF file to the user."""
    geotiff_path = request.args.get('path')
    while not Path(geotiff_path).exists():  # Wait until the file is created
        time.sleep(0.5)
    return send_file(geotiff_path, as_attachment=True, download_name=Path(geotiff_path).name)

@app.route('/download_zip')
def download_zip():
    """Send the zip file to the user."""
    zip_path = request.args.get('path')
    while not Path(zip_path).exists():  # Wait until the file is created
        time.sleep(0.5)
    return send_file(zip_path, as_attachment=True, download_name=Path(zip_path).name)

@app.route('/export_zip/<style_name>')
def export_zip(style_name):
    """Export cached tiles as a ZIP file."""
    style_cache_dir = get_style_cache_dir(style_name)
    if not style_cache_dir.exists() or not any(style_cache_dir.iterdir()):
        return 'No cached tiles found for this style', 404

    zip_path = create_zip(style_cache_dir, style_name)
    return send_file(zip_path, as_attachment=True, download_name=Path(zip_path).name)

@app.route('/tiles/<style_name>/<int:z>/<int:x>/<int:y>.png')
def serve_tile(style_name, z, x, y):
    """Serve a cached tile if it exists."""
    style_cache_dir = get_style_cache_dir(style_name)
    tile_path = style_cache_dir / str(z) / str(x) / f"{y}.png"
    if tile_path.exists():
        return send_file(tile_path)
    return '', 404

@app.route('/delete_cache/<style_name>', methods=['DELETE'])
def delete_cache(style_name):
    """Delete the cache directory for a specific style."""
    cache_dir = get_style_cache_dir(style_name)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        return '', 204
    return 'Cache not found', 404

@app.route('/get_cached_tiles/<style_name>')
def get_cached_tiles_route(style_name):
    """Return a list of [z, x, y] for cached tiles of the given style."""
    style_cache_dir = get_style_cache_dir(style_name)
    if not style_cache_dir.exists():
        return jsonify([])
    cached_tiles = []
    for z_dir in style_cache_dir.iterdir():
        if z_dir.is_dir():
            try:
                z = int(z_dir.name)
                for x_dir in z_dir.iterdir():
                    if x_dir.is_dir():
                        try:
                            x = int(x_dir.name)
                            for y_file in x_dir.glob('*.png'):
                                try:
                                    y = int(y_file.stem)
                                    cached_tiles.append([z, x, y])
                                except ValueError:
                                    pass
                        except ValueError:
                            pass
            except ValueError:
                pass
    return jsonify(cached_tiles)

if __name__ == '__main__':
    CACHE_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    socketio.run(app, debug=True)
