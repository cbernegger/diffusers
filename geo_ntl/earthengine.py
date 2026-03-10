from pathlib import Path
import json
import time

import requests
import geopandas as gpd
import ee


# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------
PROJECT_ID = "ninth-sol-478514-g1"

# Test-ROI um Kyiv: [xmin, ymin, xmax, ymax]
TEST_ROI_COORDS = [30.15, 50.20, 30.95, 50.75]

# Google-Drive-Export
DRIVE_FOLDER = "EarthEngine"
DRIVE_FILENAME = "kyiv_predictors_130m"

# Lokale Dateien relativ zu dieser .py
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def init_ee():
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine ok:", ee.Number(1).getInfo())


def get_ukraine_boundary():
    api_url = "https://www.geoboundaries.org/api/current/gbOpen/UKR/ADM0/"
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()

    meta = response.json()
    ukr = gpd.read_file(meta["gjDownloadURL"]).to_crs("EPSG:4326")

    boundary_path = RAW_DIR / "ukraine_adm0.geojson"
    ukr.to_file(boundary_path, driver="GeoJSON")
    print("Saved boundary:", boundary_path)

    ukr_geojson = json.loads(ukr.to_json())
    ukr_fc = ee.FeatureCollection(ukr_geojson)
    ukr_geom = ukr_fc.geometry()

    return ukr_geom


def build_layers(ukr_geom):
    gaia = ee.Image("Tsinghua/FROM-GLC/GAIA/v10").select("change_year_index")

    gaia_year = (
        ee.Image.constant(2019)
        .subtract(gaia)
        .rename("gaia_year")
        .updateMask(gaia.gte(1))
        .clip(ukr_geom)
    )

    gaia_impervious_2018 = (
        gaia.gte(1)
        .selfMask()
        .rename("gaia_impervious_2018")
        .clip(ukr_geom)
    )

    ghsl_height_2018 = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .rename("building_height")
        .clip(ukr_geom)
    )

    s2_2018 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ukr_geom)
        .filterDate("2018-05-01", "2018-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(ukr_geom)
    )

    ndvi = s2_2018.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndbi = s2_2018.normalizedDifference(["B11", "B8"]).rename("ndbi")
    mndwi = s2_2018.normalizedDifference(["B3", "B11"]).rename("mndwi")

    dw_2018 = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(ukr_geom)
        .filterDate("2018-05-01", "2018-09-30")
        .select(["built", "water", "trees", "crops"])
        .mean()
        .rename(["dw_built", "dw_water", "dw_trees", "dw_crops"])
        .clip(ukr_geom)
    )

    viirs_2018 = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filterBounds(ukr_geom)
        .filterDate("2018-01-01", "2019-01-01")
    )

    viirs_avg_rad_2018 = (
        viirs_2018.select("avg_rad")
        .mean()
        .rename("bm")
        .clip(ukr_geom)
    )

    urban_mask = gaia_impervious_2018.rename("urban_mask").unmask(0)
    water_mask = dw_2018.select("dw_water").gte(0.5).rename("water_mask").unmask(0)

    train_predictors = ee.Image.cat([
        viirs_avg_rad_2018.unmask(0),           # bm
        gaia_year.rename("gaia_year").unmask(0),
        urban_mask,
        ghsl_height_2018.unmask(0),             # building_height
        water_mask,
        ndvi.rename("ndvi").unmask(0),
        ndbi.rename("ndbi").unmask(0),
        mndwi.rename("mndwi").unmask(0),
    ]).toFloat()

    band_order = [
        "bm",
        "gaia_year",
        "urban_mask",
        "building_height",
        "water_mask",
        "ndvi",
        "ndbi",
        "mndwi",
    ]

    print("Expected export band order:", band_order)
    print("Actual EE band names:", train_predictors.bandNames().getInfo())

    return train_predictors


def export_to_drive(image, region):
    print("Starting Google Drive export...")

    task = ee.batch.Export.image.toDrive(
        image=image.clip(region),
        description=DRIVE_FILENAME,
        folder=DRIVE_FOLDER,
        fileNamePrefix=DRIVE_FILENAME,
        region=region,
        scale=130,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    task.start()
    print("Task started.")
    print("Initial status:", task.status())

    while task.active():
        status = task.status()
        print("Task state:", status.get("state"))
        time.sleep(10)

    final_status = task.status()
    print("Final status:", final_status)

    if final_status.get("state") == "COMPLETED":
        print("\nExport fertig.")
        print(f"Öffne Google Drive > {DRIVE_FOLDER}")
        print(f"Dort sollte liegen: {DRIVE_FILENAME}.tif")
    else:
        print("\nExport fehlgeschlagen.")
        print(final_status)


def main():
    init_ee()
    test_roi = ee.Geometry.Rectangle(TEST_ROI_COORDS)
    ukr_geom = get_ukraine_boundary()
    train_predictors = build_layers(ukr_geom)
    export_to_drive(train_predictors, test_roi)


if __name__ == "__main__":
    main()