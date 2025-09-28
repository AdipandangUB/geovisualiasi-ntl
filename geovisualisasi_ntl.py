"""
Script Analisis Nighttime Light pada Streamlit
FOKUS PADA VISUALISASI GEOSPASIAL NTL
Dengan penghilangan blok hitam, hanya menampilkan pixel terang,
serta masking berdasarkan batas wilayah administrasi
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tempfile
import os
import folium
from streamlit_folium import st_folium
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from rasterio.mask import mask
import geopandas as gpd
import json

# ----------------------------
# FUNGSI VISUALISASI GEOSPASIAL
# ----------------------------

def create_ntl_colormap():
    """Membuat colormap khusus untuk nighttime lights"""
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('ntl_colormap', colors, N=256)

def remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold=0.1, min_pixel_value=0.01):
    """Hilangkan blok hitam & tampilkan hanya pixel terang"""
    data_normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    bright_mask = data_normalized > brightness_threshold
    valid_mask = (data > min_pixel_value) & (~np.isnan(data))
    final_mask = bright_mask & valid_mask
    data_cleaned = np.where(final_mask, data, np.nan)
    return data_cleaned

def create_interactive_ntl_map(raster_path, boundary_path=None, remove_black=True, brightness_threshold=0.1):
    """Membuat peta interaktif Folium dengan masking batas wilayah"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            data[data == src.nodata] = np.nan
            bounds = src.bounds

            if boundary_path:
                gdf = gpd.read_file(boundary_path).to_crs(src.crs)
                out_image, out_transform = mask(src, gdf.geometry, crop=True)
                data = out_image[0]
                bounds = rasterio.transform.array_bounds(
                    out_image.shape[1], out_image.shape[0], out_transform
                )

            if remove_black:
                data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                data = np.nan_to_num(data)

            # Normalisasi
            if np.nanmax(data) > np.nanmin(data):
                data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            else:
                data_norm = data

            # Hitung center
            center_lat = (bounds[3] + bounds[1]) / 2
            center_lon = (bounds[2] + bounds[0]) / 2

            # Peta dasar
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB dark_matter")

            # Simpan PNG sementara
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.imsave(tmp_file.name, data_norm, cmap=create_ntl_colormap())
                folium.raster_layers.ImageOverlay(
                    name="NTL Filtered",
                    image=tmp_file.name,
                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                    opacity=0.7,
                    interactive=True,
                ).add_to(m)

            # Tambah boundary ke peta
            if boundary_path:
                geojson_data = json.loads(gdf.to_json())
                folium.GeoJson(geojson_data, name="Boundary", style_function=lambda x: {
                    "color": "red", "weight": 2, "fillOpacity": 0
                }).add_to(m)

            folium.LayerControl().add_to(m)
            return m

    except Exception as e:
        st.error(f"Error membuat peta interaktif: {str(e)}")
        return None

# ----------------------------
# FUNGSI UTAMA VISUALISASI GEOSPASIAL
# ----------------------------

def setup_geospatial_visualization():
    st.header("🌍 Visualisasi Geospasial Nighttime Lights")
    st.markdown("Analisis spasial data NTL dengan filter piksel terang dan masking batas wilayah")

    # Sidebar kontrol
    st.sidebar.header("⚙️ Pengaturan Filter Pixel")
    remove_black_blocks = st.sidebar.checkbox("Hilangkan Blok Hitam", value=True)
    brightness_threshold = st.sidebar.slider("Threshold Kecerahan Pixel", 0.0, 1.0, 0.1, 0.01)

    # Upload data raster
    raster_file = st.file_uploader("Pilih file raster NTL (TIFF)", type=["tif", "tiff"])

    # Upload boundary (opsional)
    boundary_file = st.file_uploader("Pilih file batas wilayah (Shapefile .shp atau GeoJSON)", type=["geojson", "shp"])

    if raster_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = os.path.join(tmpdir, raster_file.name)
            with open(raster_path, "wb") as f:
                f.write(raster_file.getbuffer())

            boundary_path = None
            if boundary_file:
                boundary_path = os.path.join(tmpdir, boundary_file.name)
                with open(boundary_path, "wb") as f:
                    f.write(boundary_file.getbuffer())

            st.subheader("🗺️ Peta Interaktif Nighttime Lights")
            interactive_map = create_interactive_ntl_map(
                raster_path, boundary_path, remove_black_blocks, brightness_threshold
            )
            if interactive_map:
                st_folium(interactive_map, width=900, height=600)
            else:
                st.error("Gagal membuat peta interaktif")
    else:
        st.info("📁 Silakan unggah file raster NTL (TIFF) untuk mulai")

# ----------------------------
# KONFIGURASI STREAMLIT
# ----------------------------

st.set_page_config(page_title="NTL Geospatial Visualization", layout="wide", page_icon="🌃")

st.title("🌃 Nighttime Lights Geospatial Visualization Tool")
st.markdown("**Dengan fitur filter pixel terang dan masking batas wilayah**")

setup_geospatial_visualization()

# ----------------------------
# INFO DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**NTL Geospatial Visualization Tool - Ver 4.0**")
st.markdown("**Dikembangkan oleh: Firman Afrianto & Adipandang Yudono**")

