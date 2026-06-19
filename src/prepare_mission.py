#!/usr/bin/env python3
"""Stage a corridor of imagery + elevation for a long UAV mission.

Takes a flight config (with ``checkpoints``), builds a buffered corridor along the route, and
writes a per-flight directory under ``downloads/`` containing:
  - ``map/``         grayscale single-band EPSG:3857 imagery chunks (corridor-clipped, masked
                     nodata outside, balanced + size-bounded by ``--max-chunk-mb``)
  - ``topography/``  EPSG:4326 DEM chunks, one per map chunk, fetched only for that chunk's
                     bbox, clipped to the corridor, and size-split if they exceed the cap

Both are clipped to the same corridor so coverage matches. Only the per-chunk DEM bboxes are
downloaded (cached), not one big square over the whole mission.

Usage:
    python prepare_mission.py path/to/flight.yml
    python prepare_mission.py path/to/flight.yml --corridor-width-km 3 --zoom 18 --max-chunk-mb 512
    python prepare_mission.py path/to/flight.yml --no-topography

Requires OPENTOPOGRAPHY_API_KEY in the environment for the topography step.
"""

import argparse
import datetime
import json
import math
import os
import shutil
from pathlib import Path

import mercantile
import numpy as np
import rasterio
import yaml
from bmi_topography import Topography
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import transform as shapely_transform, unary_union

from download_geotiff import (
    CACHE_DIR,
    DOWNLOADS_DIR,
    download_tiles,
    get_style_cache_dir,
    load_map_sources,
)

TILE_SIZE = 256  # XYZ tile edge in pixels
WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
# Densify route segments to this spacing (m) before buffering so each piece's latitude --
# and thus its Web-Mercator scale correction -- is locally accurate over a long route.
CORRIDOR_DENSIFY_M = 10000.0
# Cap how far a joint's outer corner is pushed from the vertex, in half-widths. Within the cap
# the corner is the full straight mitre (where the two outer edges meet); sharper turns get a
# straight bevel placed at the cap distance. 4.0 = 2x the corridor width.
CORRIDOR_MITRE_LIMIT = 4.0
TOPO_CACHE_DIR = CACHE_DIR / "topography"

_to_mercator = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform
_to_wgs84 = Transformer.from_crs(WEB_MERCATOR, WGS84, always_xy=True).transform


def _print_progress(done: int, total: int, suffix: str = "") -> None:
    """Render an in-place progress bar (same style as the imagery tile downloader)."""
    bar_len = 40
    filled = int(bar_len * done // total)
    bar = "█" * filled + "░" * (bar_len - filled)
    tail = f" - {suffix}" if suffix else ""
    print(f"\r  Progress: |{bar}| {done / total * 100:.1f}% ({done}/{total}){tail}", end="", flush=True)


def load_route(flight_path: Path) -> tuple[str, list[tuple[float, float]]]:
    """Read flight id and route polyline ([(lat, lon), ...]) from a flight config.

    The route is the starting location followed by the checkpoints, matching
    sim.gen_level.load_route_points.
    """
    with open(flight_path) as f:
        config = yaml.safe_load(f)

    flight_id = config.get("id")
    if not flight_id:
        raise ValueError(f"Flight config {flight_path} is missing required 'id'")

    checkpoints = config.get("checkpoints", [])
    if not checkpoints:
        raise ValueError(f"No checkpoints in {flight_path}")

    points = [(config["starting_lat"], config["starting_lon"])]
    points.extend((cp[0], cp[1]) for cp in checkpoints)
    return flight_id, points


def _half_width_at(lat: float, half_width_m: float) -> float:
    """Half-width in EPSG:3857 units for a given ground half-width at ``lat`` (sec(lat) scale)."""
    return half_width_m / math.cos(math.radians(lat))


def _intersect(p, d, q, e):
    """Point where line ``p + s*d`` meets line ``q + t*e`` (directions must not be parallel)."""
    det = d[0] * (-e[1]) - (-e[0]) * d[1]
    s = ((q[0] - p[0]) * (-e[1]) - (-e[0]) * (q[1] - p[1])) / det
    return (p[0] + s * d[0], p[1] + s * d[1])


def _mitre_fill(vx, vy, u1, u2, half, mitre_limit):
    """Outer-corner fill polygon points for a joint, or None if the segments are collinear.

    ``u1``/``u2`` are the (incoming, outgoing) unit directions at vertex ``(vx, vy)``. The two
    densified pieces meet with flat caps, leaving a wedge gap on the *outside* of the turn.
    Within ``mitre_limit`` half-widths the gap is filled to the full mitre point (where the two
    outer edges naturally meet); beyond it a straight bevel is placed *at the limit distance*
    (not pulled back to the vertex), so the outer corner stays pushed a controlled distance out.
    """
    cross = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(cross) < 1e-9:
        return None  # straight -- pieces already meet, no gap
    # Outward normals: right of travel for a left turn (cross>0), else left of travel.
    if cross > 0:
        n1, n2 = (u1[1], -u1[0]), (u2[1], -u2[0])
    else:
        n1, n2 = (-u1[1], u1[0]), (-u2[1], u2[0])
    pa = (vx + n1[0] * half, vy + n1[1] * half)
    pb = (vx + n2[0] * half, vy + n2[1] * half)
    apex = _intersect(pa, u1, pb, u2)  # where the two outer edges meet (the mitre point)
    apex_dist = math.hypot(apex[0] - vx, apex[1] - vy)
    if apex_dist <= mitre_limit * half:
        return [(vx, vy), pa, apex, pb]  # full straight mitre
    # Bevel: straight cut perpendicular to the bisector, at mitre_limit half-widths from V.
    bis = ((apex[0] - vx) / apex_dist, (apex[1] - vy) / apex_dist)
    cut = (vx + bis[0] * mitre_limit * half, vy + bis[1] * mitre_limit * half)
    perp = (-bis[1], bis[0])
    qa = _intersect(pa, u1, cut, perp)
    qb = _intersect(pb, u2, cut, perp)
    return [(vx, vy), pa, qa, qb, pb]


def build_corridor_polygon(route: list[tuple[float, float]], width_m: float):
    """Build a corridor (WGS84 polygon) of constant *ground* width ``width_m``.

    EPSG:3857 units are true meters only at the equator and stretch by sec(latitude) further
    out, so a plain ``width_m/2`` buffer would be ~width_m*cos(lat) wide on the ground (~37%
    narrow at 50N). To hold the ground width constant over a long, latitude-spanning route,
    each segment is densified and buffered (flat caps) by ``half_width * sec(local latitude)``.
    Each joint is then resolved separately: a per-joint mitre fill (computed from that vertex's
    local width) gives a sharp corner, beveled when the turn is sharper than the mitre limit.
    The two route ends get square caps (extended by their local half-width), matching a buffer.
    """
    half_width_m = width_m / 2.0
    # Dedupe consecutive identical points (e.g. start duplicated as the first checkpoint).
    pts = []
    for lat, lon in route:
        if not pts or (lat, lon) != pts[-1]:
            pts.append((lat, lon))
    merc = [_to_mercator(lon, lat) for lat, lon in pts]

    # Non-degenerate segments as (start_index, ux, uy, length).
    segs = []
    for i in range(len(merc) - 1):
        (ax, ay), (bx, by) = merc[i], merc[i + 1]
        length = math.hypot(bx - ax, by - ay)
        if length >= 1.0:
            segs.append((i, (bx - ax) / length, (by - ay) / length, length))
    if not segs:
        raise ValueError("route has no non-degenerate segments")

    parts = []
    # Body: densified, per-latitude-width, flat-capped pieces.
    for i, _ux, _uy, length in segs:
        (ax, ay), (bx, by) = merc[i], merc[i + 1]
        steps = max(1, math.ceil(length / CORRIDOR_DENSIFY_M))
        for k in range(steps):
            x0, y0 = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
            x1, y1 = ax + (bx - ax) * (k + 1) / steps, ay + (by - ay) * (k + 1) / steps
            _, mid_lat = _to_wgs84((x0 + x1) / 2, (y0 + y1) / 2)
            sh = _half_width_at(mid_lat, half_width_m)
            parts.append(LineString([(x0, y0), (x1, y1)]).buffer(sh, cap_style="flat"))

    # Square end-caps: extend each route end outward by its local half-width.
    (i0, ux0, uy0, _), (iN, uxN, uyN, _) = segs[0], segs[-1]
    sx, sy = merc[i0]
    sh0 = _half_width_at(pts[i0][0], half_width_m)
    parts.append(LineString([(sx - ux0 * sh0, sy - uy0 * sh0), (sx, sy)]).buffer(sh0, cap_style="flat"))
    ex, ey = merc[iN + 1]
    shN = _half_width_at(pts[iN + 1][0], half_width_m)
    parts.append(LineString([(ex, ey), (ex + uxN * shN, ey + uyN * shN)]).buffer(shN, cap_style="flat"))

    # Per-joint mitre/bevel fill at each interior vertex.
    for (_, ux1, uy1, _), (j, ux2, uy2, _) in zip(segs, segs[1:]):
        vx, vy = merc[j]
        sh = _half_width_at(pts[j][0], half_width_m)
        fill = _mitre_fill(vx, vy, (ux1, uy1), (ux2, uy2), sh, CORRIDOR_MITRE_LIMIT)
        if fill is not None:
            parts.append(Polygon(fill))

    # buffer(0) cleans benign self-touches where per-joint fills meet the body pieces; do it
    # in WGS84 (small coords) -- buffer(0) on Mercator's ~3e6 coords hits precision limits.
    return shapely_transform(_to_wgs84, unary_union(parts)).buffer(0)


def select_corridor_tiles(polygon, zoom: int) -> list[mercantile.Tile]:
    """Return the XYZ tiles at ``zoom`` whose footprint intersects the corridor polygon."""
    west, south, east, north = polygon.bounds
    candidates = mercantile.tiles(west, south, east, north, zooms=[zoom])
    tiles = []
    for tile in candidates:
        b = mercantile.bounds(tile)
        if polygon.intersects(box(b.west, b.south, b.east, b.north)):
            tiles.append(tile)
    return tiles


def _balanced_split(tiles: list[mercantile.Tile], key) -> tuple[list, list] | None:
    """Split tiles into two count-balanced halves at a coordinate boundary along ``key``.

    Cutting between distinct coordinate values (not at an arbitrary index) keeps whole
    rows/columns together, so the two halves occupy disjoint, non-overlapping bboxes.
    Returns None if every tile shares one coordinate on this axis (nothing to cut).
    """
    ordered = sorted(tiles, key=key)
    coords = [key(t) for t in ordered]
    n = len(ordered)
    best_i = None
    best_diff = None
    for i in range(1, n):
        if coords[i] == coords[i - 1]:
            continue
        diff = abs(i - (n - i))
        if best_diff is None or diff < best_diff:
            best_diff, best_i = diff, i
    if best_i is None:
        return None
    return ordered[:best_i], ordered[best_i:]


def _split_to_target(tiles: list[mercantile.Tile], target: int) -> list[list[mercantile.Tile]]:
    """Recursively k-d split tiles until each group has <= ``target`` tiles (count-balanced)."""
    if len(tiles) <= target:
        return [tiles]
    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    longer_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    keys = [lambda t: t.x, lambda t: t.y] if longer_x else [lambda t: t.y, lambda t: t.x]
    for key in keys:  # prefer the longer axis; fall back to the other if it can't be cut
        parts = _balanced_split(tiles, key)
        if parts is not None:
            left, right = parts
            return _split_to_target(left, target) + _split_to_target(right, target)
    return [tiles]  # single row/column of tiles -- can't subdivide further


def chunk_corridor_tiles(
    tiles: list[mercantile.Tile], max_chunk_mb: float, bytes_per_px: int = 1
) -> tuple[dict[tuple[int, int], list[mercantile.Tile]], dict]:
    """Split corridor tiles into count-balanced pieces, each at most ~``max_chunk_mb``.

    These are intermediate pieces: the caller stitches them, measures the real LZW size, and
    merges consecutive pieces up to the cap, so final chunk sizes are bounded by actual bytes.
    This only needs pieces fine enough to pack well, so the target uses the uncompressed upper
    bound (``TILE_SIZE^2`` bytes/tile) -- a piece is at most ~the cap before compression and
    comfortably under it after.
    """
    target_tiles = max(1, math.floor(max_chunk_mb * 1024 * 1024 / (TILE_SIZE * TILE_SIZE * bytes_per_px)))
    groups = _split_to_target(tiles, target_tiles)

    chunks: dict[tuple[int, int], list[mercantile.Tile]] = {}
    for group in groups:
        # Key each chunk by its min tile (x, y) -- unique since chunks occupy disjoint bboxes.
        chunks[(min(t.x for t in group), min(t.y for t in group))] = group

    info = {"target_tiles": target_tiles, "n_chunks": len(groups)}
    return chunks, info


def stitch_grayscale_geotiff(
    tiles: list[mercantile.Tile], style_cache_dir: Path, out_path: Path, mask_geom=None
) -> int:
    """Stitch cached tiles into a single-band (grayscale) LZW GeoTIFF. Returns pixel count.

    Pixels that are not real imagery -- missing tiles, and (when ``mask_geom``, a corridor
    polygon in EPSG:3857, is given) anything outside the corridor -- are written as *empty*
    via an internal mask band rather than black, so they read as nodata/transparent. The
    mask also clips the imagery to the smooth corridor edge instead of the whole-tile
    staircase. The file stays single-band, so consumers reading band 1 are unaffected.
    """
    zoom = tiles[0].z
    min_x = min(t.x for t in tiles)
    max_x = max(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_y = max(t.y for t in tiles)

    # XYZ tiles are a Web-Mercator (EPSG:3857) pixel grid, so georeference in 3857: tile
    # rows are linear in mercator Y (a 4326 transform built from lat bounds would distort).
    nw = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, zoom))
    se = mercantile.xy_bounds(mercantile.Tile(max_x, max_y, zoom))
    left, top, right, bottom = nw.left, nw.top, se.right, se.bottom

    width = (max_x - min_x + 1) * TILE_SIZE
    height = (max_y - min_y + 1) * TILE_SIZE
    canvas = np.zeros((height, width), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)  # True only where a real tile was placed

    for tile in tiles:
        tile_path = style_cache_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png"
        if not tile_path.exists():
            continue
        with Image.open(tile_path) as img:
            gray = np.array(img.convert("L"))
        x_off = (tile.x - min_x) * TILE_SIZE
        y_off = (tile.y - min_y) * TILE_SIZE
        canvas[y_off:y_off + TILE_SIZE, x_off:x_off + TILE_SIZE] = gray
        valid[y_off:y_off + TILE_SIZE, x_off:x_off + TILE_SIZE] = True

    transform = from_bounds(left, bottom, right, top, width, height)

    if mask_geom is not None:
        # geometry_mask -> True for pixels OUTSIDE the corridor; mark those empty.
        outside = geometry_mask([mask_geom], out_shape=(height, width), transform=transform)
        valid &= ~outside

    # GDAL_TIFF_INTERNAL_MASK keeps the mask inside the .tif (no sidecar .msk file).
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint8,
            crs=WEB_MERCATOR,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(np.where(valid, canvas, 0), 1)
            dst.write_mask(np.where(valid, 255, 0).astype(np.uint8))

    return width * height


def cached_topography(bounds: tuple[float, float, float, float], dem_type: str) -> tuple[Path, bool]:
    """Return ``(dem_path, was_cached)`` for ``bounds`` + ``dem_type``, fetching on miss.

    Mirrors the XYZ tile cache: DEMs are keyed by dataset + rounded bbox under
    ``tile-cache/topography/`` so repeated runs over the same corridor don't re-fetch.
    """
    west, south, east, north = bounds
    TOPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TOPO_CACHE_DIR / f"{dem_type}_{west:.5f}_{south:.5f}_{east:.5f}_{north:.5f}.tif"
    if cache_path.exists():
        return cache_path, True

    topography = Topography(
        dem_type=dem_type,
        south=south,
        north=north,
        west=west,
        east=east,
        cache_dir=str(TOPO_CACHE_DIR),
    )
    fetched = topography.fetch()
    shutil.move(str(fetched), cache_path)
    return cache_path, False


def chunk_wgs84_bounds(tiles: list[mercantile.Tile], zoom: int) -> tuple[float, float, float, float]:
    """WGS84 (west, south, east, north) bbox covering a chunk's tiles -- the DEM fetch extent."""
    min_x = min(t.x for t in tiles)
    max_x = max(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_y = max(t.y for t in tiles)
    nw = mercantile.bounds(mercantile.Tile(min_x, min_y, zoom))
    se = mercantile.bounds(mercantile.Tile(max_x, max_y, zoom))
    return nw.west, se.south, se.east, nw.north


def write_dem_chunks(
    data: np.ndarray, base_transform, profile: dict, out_dir: Path, prefix: str, max_chunk_mb: float
) -> list[Path]:
    """Write a (clipped) DEM array as one or more GeoTIFFs, each under ``max_chunk_mb``.

    Splits into windowed sub-chunks (preserving georeferencing) only when the DEM exceeds the
    cap; small DEMs (the common SRTM case) stay a single ``{prefix}.tif``.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    count, h, w = data.shape
    bytes_per_px = max(1, np.dtype(profile["dtype"]).itemsize * count)
    step = max(1, int(math.sqrt(max_chunk_mb * 1024 * 1024 / bytes_per_px)))
    single = step >= w and step >= h

    written = []
    for gy, row in enumerate(range(0, h, step)):
        for gx, col in enumerate(range(0, w, step)):
            sub = data[:, row:row + step, col:col + step]
            sub_profile = dict(profile)
            sub_profile.update(
                height=sub.shape[1],
                width=sub.shape[2],
                transform=base_transform * rasterio.Affine.translation(col, row),
                compress="lzw",
            )
            out_path = out_dir / (f"{prefix}.tif" if single else f"{prefix}_{gx}_{gy}.tif")
            with rasterio.open(out_path, "w", **sub_profile) as dst:
                dst.write(sub)
            written.append(out_path)
    return written


def topography_for_chunk(
    bbox: tuple[float, float, float, float], dem_type: str, corridor, out_dir: Path, prefix: str, max_chunk_mb: float
) -> tuple[list[Path], bool]:
    """Fetch the DEM for one map chunk's bbox, clip it to ``corridor``, and size-split it.

    Returns ``(written_files, was_cached)``. Only the small per-chunk bbox is downloaded
    (cached), and clipping marks everything outside the corridor as nodata -- so the DEM
    coverage matches the produced map. ``all_touched=True`` keeps every DEM pixel the
    corridor touches: the DEM is coarse (~90 m), so center-based masking would erode coverage
    up to a pixel inside the fine imagery edge, leaving imagery the DEM doesn't cover.
    """
    dem_path, was_cached = cached_topography(bbox, dem_type)
    with rasterio.open(dem_path) as src:
        nodata = src.nodata if src.nodata is not None else -32768.0
        clipped, clipped_transform = rio_mask(
            src, [mapping(corridor)], crop=True, all_touched=True, nodata=nodata, filled=True
        )
        profile = src.profile.copy()
    profile.update(nodata=nodata)
    return write_dem_chunks(clipped, clipped_transform, profile, out_dir, prefix, max_chunk_mb), was_cached


def resolve_output_dir(downloads_dir: Path, flight_id: str) -> Path:
    """Per-flight output dir; append a date (then time) suffix if the name already exists."""
    base = downloads_dir / flight_id
    if not base.exists():
        return base
    dated = downloads_dir / f"{flight_id}_{datetime.date.today().isoformat()}"
    if not dated.exists():
        return dated
    stamp = datetime.datetime.now().strftime("%H%M%S")
    return downloads_dir / f"{flight_id}_{datetime.date.today().isoformat()}_{stamp}"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a corridor of imagery + elevation chunks for a long UAV mission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("flight_config", type=Path, help="Path to flight config YAML (with checkpoints)")
    parser.add_argument("--corridor-width-km", type=float, default=2.0, help="Total corridor width in km")
    parser.add_argument("--zoom", type=int, default=17, help="XYZ zoom level (0-19)")
    parser.add_argument("--source", type=str, default="Esri World Imagery Satellite", help="Map source name")
    parser.add_argument("--max-chunk-mb", type=float, default=1024.0, help="Max uncompressed chunk size in MB")
    parser.add_argument("--dem-type", type=str, default="SRTMGL3", help="OpenTopography DEM dataset (e.g. SRTMGL3/SRTMGL1/COP30)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel tile download workers")
    parser.add_argument("--downloads-dir", type=Path, default=DOWNLOADS_DIR, help="Base downloads directory")
    parser.add_argument("--skip-topography", action="store_true", help="Skip DEM fetch (imagery only)")
    parser.add_argument(
        "--skip-imagery",
        action="store_true",
        help="Skip imagery download/chunks; still compute corridor chunk geometry and produce "
        "topography (useful for testing the DEM path without fetching thousands of map tiles)",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.zoom < 0 or args.zoom > 19:
        raise SystemExit("Zoom level must be between 0 and 19")

    map_sources = load_map_sources()
    if args.source not in map_sources:
        raise SystemExit(f"Unknown map source '{args.source}'. Available: {', '.join(map_sources)}")

    if not args.skip_topography and "OPENTOPOGRAPHY_API_KEY" not in os.environ:
        raise SystemExit("OPENTOPOGRAPHY_API_KEY env variable is required (or pass --no-topography)")

    flight_id, route = load_route(args.flight_config)
    corridor = build_corridor_polygon(route, args.corridor_width_km * 1000.0)
    corridor_3857 = shapely_transform(_to_mercator, corridor)  # for clipping the 3857 chunks
    bounds = corridor.bounds
    print(f"Flight: {flight_id}  route points: {len(route)}  corridor width: {args.corridor_width_km} km")
    print(f"Corridor bounds (W,S,E,N): {bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f}")

    tiles = select_corridor_tiles(corridor, args.zoom)
    if not tiles:
        raise SystemExit("No tiles intersect the corridor")
    est_mb = len(tiles) * TILE_SIZE * TILE_SIZE / (1024 * 1024)
    print(f"Corridor tiles at zoom {args.zoom}: {len(tiles)} (~{est_mb:.1f} MB grayscale uncompressed)")

    out_dir = resolve_output_dir(args.downloads_dir, flight_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Chunk geometry is pure tile math (no download), so topography can be produced without
    # fetching/stitching any imagery -- handy for testing the DEM path on a long corridor.
    map_chunks, chunk_info = chunk_corridor_tiles(tiles, args.max_chunk_mb, bytes_per_px=1)
    leaf_groups = [grp for _, grp in sorted(map_chunks.items())]  # fine pieces, spatial order
    limit_bytes = int(args.max_chunk_mb * 1024 * 1024)

    map_chunk_info = []
    if args.skip_imagery:
        print(f"Corridor splits into {len(leaf_groups)} piece(s); skipping imagery (--skip-imagery)")
        final_groups = leaf_groups
    else:
        style_cache_dir = get_style_cache_dir(args.source)
        download_tiles(tiles, map_sources[args.source], style_cache_dir, args.workers)
        map_dir = out_dir / "map"
        map_dir.mkdir(parents=True, exist_ok=True)

        # The tiles->MB estimate is conservative and the splitter halves recursively, so it
        # over-splits into pieces well under the cap. Stitch each piece, measure its real LZW
        # size, then greedily merge consecutive pieces while the summed size stays under the
        # cap -- this packs each final chunk close to --max-chunk-mb using actual sizes.
        print(f"Stitching {len(leaf_groups)} piece(s), then merging up to {args.max_chunk_mb:.0f} MB...")
        merged = []  # each: [tiles, summed_bytes, [temp_paths]]
        for grp in leaf_groups:
            tmp = map_dir / f"_piece_{min(t.x for t in grp)}_{min(t.y for t in grp)}.tif"
            stitch_grayscale_geotiff(grp, style_cache_dir, tmp, mask_geom=corridor_3857)
            size = tmp.stat().st_size
            if merged and merged[-1][1] + size <= limit_bytes:
                merged[-1][0].extend(grp)
                merged[-1][1] += size
                merged[-1][2].append(tmp)
            else:
                merged.append([list(grp), size, [tmp]])

        final_groups = []
        print(f"Writing {len(merged)} imagery chunk(s)...")
        for grp, _, tmps in merged:
            gx, gy = min(t.x for t in grp), min(t.y for t in grp)
            out_path = map_dir / f"chunk_{gx}_{gy}.tif"
            if len(tmps) == 1:
                tmps[0].replace(out_path)  # single piece -- no re-stitch needed
            else:
                for t in tmps:
                    t.unlink()
                stitch_grayscale_geotiff(grp, style_cache_dir, out_path, mask_geom=corridor_3857)
            final_groups.append(grp)
            map_chunk_info.append({"name": out_path.name, "tiles": len(grp), "bytes": out_path.stat().st_size})
            print(f"  {out_path.name}: {len(grp)} tiles, {out_path.stat().st_size / 1e6:.1f} MB")

    topo_chunk_info = []
    if not args.skip_topography:
        topo_dir = out_dir / "topography"
        topo_dir.mkdir(parents=True, exist_ok=True)
        total = len(final_groups)
        fetched = 0
        print(f"Fetching + clipping DEM ({args.dem_type}) for {total} chunk(s) (one OpenTopography request each)...")
        for done, grp in enumerate(final_groups, 1):
            gx, gy = min(t.x for t in grp), min(t.y for t in grp)
            bbox = chunk_wgs84_bounds(grp, args.zoom)
            files, was_cached = topography_for_chunk(
                bbox, args.dem_type, corridor, topo_dir, f"chunk_{gx}_{gy}", args.max_chunk_mb
            )
            fetched += not was_cached
            for f in files:
                topo_chunk_info.append({"name": f.name, "bytes": f.stat().st_size})
            _print_progress(done, total, suffix=f"{fetched} fetched, {done - fetched} cached")
        print()  # newline after the progress bar
        print(f"  {len(topo_chunk_info)} topography chunk file(s)")

    summary = {
        "flight_id": flight_id,
        "source": args.source,
        "zoom": args.zoom,
        "corridor_width_km": args.corridor_width_km,
        "corridor_bounds": {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3]},
        "tile_count": len(tiles),
        "max_chunk_mb": args.max_chunk_mb,
        "chunk_grid": chunk_info,
        "map_chunks": map_chunk_info,
        "dem_type": None if args.skip_topography else args.dem_type,
        "topography_chunks": topo_chunk_info,
    }
    with open(out_dir / "prepare_mission.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStaged mission corridor at {out_dir}")
    print(f"  map chunks: {len(map_chunk_info)}  topography chunks: {len(topo_chunk_info)}")


if __name__ == "__main__":
    main()
