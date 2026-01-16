# Terrain Tiles

Generate 3D printable terrain tiles from UK LIDAR data for resin printing.

## What It Does

Creates modular STL/3MF files of real UK terrain from Environment Agency LIDAR data. Tiles align at edges and can be assembled into wall art pieces.

- 12cm × 12cm tiles (configurable)
- Uses DSM data (includes trees, buildings)
- Covers ~99% of England at 1m resolution
- Free data under Open Government Licence

## Installation

```bash
pip install pyproj rasterio numpy numpy-stl requests scipy trimesh
```

## Usage

```bash
python terrain_tiles.py --lat 52.3139 --lon -2.5947 \
  --tiles 3x3 --tile-radius 1 --tile-size 120 \
  --resolution 1000 --output malvern/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--lat` | required | Centre latitude |
| `--lon` | required | Centre longitude |
| `--tiles` | required | Grid size e.g. `3x3` |
| `--tile-radius` | required | Radius per tile in km (1 = 2km × 2km tile) |
| `--tile-size` | 120 | Output size in mm |
| `--resolution` | 1000 | Grid cells per tile |
| `--exaggeration` | 2.0 | Vertical scale multiplier |
| `--base` | 2.0 | Base thickness in mm |
| `--smooth` | 0.5 | Gaussian smoothing sigma |
| `--output` | dist/ | Output folder |
| `--format` | stl | Output format: `stl` or `3mf` |
| `--no-cache` | false | Disable LIDAR data caching |
| `--parallel` | 0 | Parallel workers for mesh generation |
| `--parallel-fetch` | 0 | Parallel API requests (4-8 recommended) |

## Output

Creates numbered tile files in `dist/` (gitignored):

```
dist/
  tile_0_0.stl  (bottom-left)
  tile_1_0.stl  (bottom-right)
  tile_0_1.stl  (top-left)
  tile_1_1.stl  (top-right)
```

## Examples

**Malvern Hills 3×3:**
```bash
python terrain_tiles.py --lat 52.3139 --lon -2.5947 \
  --tiles 3x3 --tile-radius 1 --exaggeration 2 --output malvern/
```

**Lake District single tile:**
```bash
python terrain_tiles.py --lat 54.4609 --lon -3.0886 \
  --tiles 1x1 --tile-radius 2 --exaggeration 1.5 --output lakes/
```

**London (flat terrain, more exaggeration):**
```bash
python terrain_tiles.py --lat 51.5074 --lon -0.1278 \
  --tiles 2x2 --tile-radius 1.5 --exaggeration 4 --output london/
```

**Fast parallel generation:**
```bash
python terrain_tiles.py --lat 52.3139 --lon -2.5947 \
  --tiles 3x3 --tile-radius 1 --parallel-fetch 4 --parallel 4 \
  --format 3mf --output malvern/
```

## Scale

For a 2km × 2km area on a 120mm tile:
- Horizontal scale: 1:16,667
- At `--exaggeration 1`: 100m elevation = 6mm on model
- At `--exaggeration 2`: 100m elevation = 12mm on model

True scale terrain looks flat—exaggeration makes it dramatic.

## Resolution Guide

For 120mm tile covering 2km:
- `--resolution 1000` — good balance (0.12mm per cell)
- `--resolution 500` — faster test prints
- `--resolution 1500` — maximum detail

## Resin Usage

The script calculates and displays resin volume for each tile and total:

```
  tile_0_0.stl: 1.9 MB, height: 22.0mm, resin: 328.1ml
  tile_1_0.stl: 1.9 MB, height: 17.8mm, resin: 214.9ml
  ...
Total resin: 1268.1ml
```

Typical volumes for 180mm × 180mm tiles: 200-450ml depending on terrain relief.

## Data Source

Environment Agency LIDAR Composite DSM (Digital Surface Model) at 1m resolution.
