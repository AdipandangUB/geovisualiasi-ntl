"""
Script Analisis Nighttime Light pada Streamlit
FOKUS PADA VISUALISASI GEOSPASIAL NTL DENGAN MASKING BATAS ADMINISTRASI
Dengan penghilangan blok hitam dan hanya menampilkan pixel terang
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
from PIL import Image
import io
import geopandas as gpd
from shapely.geometry import mapping
import json

# ----------------------------
# FUNGSI VISUALISASI GEOSPASIAL
# ----------------------------

def create_ntl_colormap():
    """Membuat colormap khusus untuk nighttime lights"""
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('ntl_colormap', colors, N=256)

def mask_with_administrative_boundaries(raster_data, raster_transform, raster_crs, admin_boundary_file, admin_attribute=None, admin_value=None, file_type='shapefile'):
    """
    Melakukan masking data raster dengan batas administrasi wilayah
    
    Parameters:
    - raster_data: array raster
    - raster_transform: transform dari raster
    - raster_crs: CRS dari raster
    - admin_boundary_file: path ke file batas administrasi (shapefile atau geojson)
    - admin_attribute: atribut untuk filtering (opsional)
    - admin_value: nilai atribut untuk filtering (opsional)
    - file_type: tipe file ('shapefile' atau 'geojson')
    
    Returns:
    - masked_data: array raster yang sudah dimasking
    - admin_bounds: bounds dari wilayah administrasi
    """
    try:
        # Baca file batas administrasi berdasarkan tipe file
        if file_type == 'geojson':
            admin_gdf = gpd.read_file(admin_boundary_file)
        else:  # shapefile
            admin_gdf = gpd.read_file(admin_boundary_file)
        
        # Filter berdasarkan atribut jika diberikan
        if admin_attribute and admin_value:
            # Handle case sensitivity and data type issues
            if admin_attribute in admin_gdf.columns:
                # Convert both to string for comparison to avoid type issues
                admin_gdf = admin_gdf[admin_gdf[admin_attribute].astype(str) == str(admin_value)]
        
        # Jika tidak ada data setelah filtering, tampilkan warning
        if len(admin_gdf) == 0:
            st.warning(f"Tidak ada data yang sesuai dengan filter {admin_attribute} = {admin_value}. Menampilkan semua data.")
            admin_gdf = gpd.read_file(admin_boundary_file)  # Reset to original
        
        # Pastikan CRS sama dengan raster
        if admin_gdf.crs != raster_crs:
            admin_gdf = admin_gdf.to_crs(raster_crs)
        
        # Dapatkan bounds dari wilayah administrasi
        admin_bounds = admin_gdf.total_bounds
        
        # Buat mask dari geometri administrasi
        from rasterio.mask import mask
        
        # Convert to geojson-like format
        shapes = [mapping(geom) for geom in admin_gdf.geometry]
        
        # Create temporary file for masked raster
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_raster:
            # Create profile for output
            profile = {
                'driver': 'GTiff',
                'height': raster_data.shape[0],
                'width': raster_data.shape[1],
                'count': 1,
                'dtype': raster_data.dtype,
                'crs': raster_crs,
                'transform': raster_transform,
            }
            
            # Write raster data to temporary file
            with rasterio.open(tmp_raster.name, 'w', **profile) as dst:
                dst.write(raster_data, 1)
            
            # Perform masking
            with rasterio.open(tmp_raster.name) as src:
                try:
                    out_image, out_transform = mask(src, shapes, crop=False, filled=True)
                    masked_data = out_image[0]
                    
                    # Set area outside mask to NaN
                    masked_data[out_image[0] == src.nodata] = np.nan
                except Exception as mask_error:
                    st.warning(f"Masking tidak berhasil: {mask_error}. Menggunakan data asli.")
                    masked_data = raster_data
        
        # Clean up temporary file
        try:
            os.unlink(tmp_raster.name)
        except:
            pass
        
        return masked_data, admin_bounds
        
    except Exception as e:
        st.error(f"Error dalam masking batas administrasi: {str(e)}")
        return raster_data, None

def remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold=0.1, min_pixel_value=0.01):
    """
    Menghilangkan blok hitam dan hanya mempertahankan pixel terang
    """
    # Normalisasi data ke range 0-1
    if np.nanmax(data) > np.nanmin(data):
        data_normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    else:
        data_normalized = data
    
    # Buat mask untuk pixel terang
    bright_mask = data_normalized > brightness_threshold
    
    # Mask untuk nilai yang valid (bukan NaN dan di atas minimum value)
    valid_mask = (data > min_pixel_value) & (~np.isnan(data))
    
    # Gabungkan mask - hanya pertahankan pixel yang terang DAN valid
    final_mask = bright_mask & valid_mask
    
    # Buat array hasil dengan nilai asli untuk pixel terang, NaN untuk lainnya
    data_cleaned = np.where(final_mask, data, np.nan)
    
    return data_cleaned

def read_geojson_file(geojson_file):
    """
    Membaca file GeoJSON dengan error handling yang lebih baik
    """
    try:
        # Coba baca dengan geopandas
        gdf = gpd.read_file(geojson_file)
        return gdf, None
    except Exception as e:
        return None, f"Error membaca GeoJSON dengan geopandas: {str(e)}"

def read_shapefile_files(shp_file, companion_files):
    """
    Membaca shapefile dengan error handling yang lebih baik
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simpan file utama
            shp_path = os.path.join(tmpdir, shp_file.name)
            with open(shp_path, "wb") as f:
                f.write(shp_file.getbuffer())
            
            # Simpan file pendamping
            for comp_file in companion_files:
                comp_path = os.path.join(tmpdir, comp_file.name)
                with open(comp_path, "wb") as f:
                    f.write(comp_file.getbuffer())
            
            # Baca shapefile
            gdf = gpd.read_file(shp_path)
            return gdf, None
            
    except Exception as e:
        return None, f"Error membaca shapefile: {str(e)}"

def plot_geospatial_ntl(raster_path, title="Nighttime Lights", remove_black=True, 
                       brightness_threshold=0.1, admin_boundary_file=None, 
                       admin_attribute=None, admin_value=None, boundary_file_type='shapefile'):
    """Visualisasi geospasial raster NTL dengan Matplotlib"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            crs = src.crs
            
            # Handle no data values
            data[data == src.nodata] = np.nan
            
            # Lakukan masking dengan batas administrasi jika diberikan
            if admin_boundary_file:
                data, admin_bounds = mask_with_administrative_boundaries(
                    data, transform, crs, admin_boundary_file, admin_attribute, admin_value, boundary_file_type
                )
            
            # Hapus blok hitam dan pertahankan hanya pixel terang jika diminta
            if remove_black:
                data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot raster dengan colormap khusus NTL
            ntl_cmap = create_ntl_colormap()
            im = ax.imshow(data, cmap=ntl_cmap, 
                          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Radiance (nW/cm²/sr)')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    except Exception as e:
        st.error(f"Error dalam visualisasi raster: {str(e)}")
        return None

def create_interactive_ntl_map(raster_paths, year_labels=None, remove_black=True, 
                              brightness_threshold=0.1, admin_boundary_file=None,
                              admin_attribute=None, admin_value=None, boundary_file_type='shapefile'):
    """Membuat peta interaktif dengan Folium untuk data NTL"""
    try:
        if not raster_paths:
            return None
            
        # Gunakan raster pertama untuk center map
        with rasterio.open(raster_paths[0]) as src:
            bounds = src.bounds
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
        
        # Buat peta dasar
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=8, 
                      tiles='CartoDB dark_matter')
        
        # Tambahkan tile layers alternatif
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Tambahkan batas administrasi jika diberikan
        if admin_boundary_file:
            try:
                if boundary_file_type == 'geojson':
                    admin_gdf = gpd.read_file(admin_boundary_file)
                else:
                    admin_gdf = gpd.read_file(admin_boundary_file)
                
                # Filter berdasarkan atribut jika diberikan
                if admin_attribute and admin_value:
                    if admin_attribute in admin_gdf.columns:
                        admin_gdf = admin_gdf[admin_gdf[admin_attribute].astype(str) == str(admin_value)]
                
                # Tambahkan batas administrasi ke peta
                folium.GeoJson(
                    admin_gdf,
                    name='Batas Administrasi',
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'color': 'red',
                        'weight': 2,
                        'fillOpacity': 0
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=[admin_attribute] if admin_attribute and admin_attribute in admin_gdf.columns else [],
                        aliases=[admin_attribute] if admin_attribute and admin_attribute in admin_gdf.columns else []
                    )
                ).add_to(m)
            except Exception as e:
                st.warning(f"Tidak dapat menambahkan batas administrasi ke peta: {str(e)}")
        
        # Untuk setiap raster, tambahkan sebagai overlay
        for i, raster_path in enumerate(raster_paths):
            try:
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    transform = src.transform
                    crs = src.crs
                    year_label = year_labels[i] if year_labels and i < len(year_labels) else f"Year {i+1}"
                    
                    # Convert raster to PNG untuk overlay
                    data = src.read(1)
                    data[data == src.nodata] = 0
                    
                    # Lakukan masking dengan batas administrasi jika diberikan
                    if admin_boundary_file:
                        data, _ = mask_with_administrative_boundaries(
                            data, transform, crs, admin_boundary_file, admin_attribute, admin_value, boundary_file_type
                        )
                    
                    # Hapus blok hitam dan pertahankan hanya pixel terang jika diminta
                    if remove_black:
                        data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                        data = np.nan_to_num(data)  # Convert NaN to 0 untuk visualisasi
                    
                    # Normalisasi data untuk visualisasi
                    if np.nanmax(data) > np.nanmin(data):
                        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                    else:
                        data_norm = data
                    
                    # Simpan sebagai PNG sementara
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        plt.imsave(tmp_file.name, data_norm, cmap=create_ntl_colormap())
                        
                        # Add raster overlay ke peta
                        img_overlay = folium.raster_layers.ImageOverlay(
                            name=year_label,
                            image=tmp_file.name,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.7,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)
            except Exception as e:
                st.warning(f"Gagal memproses raster {raster_path}: {str(e)}")
                continue
        
        # Tambahkan layer control
        folium.LayerControl().add_to(m)
        
        # Tambahkan measure control
        folium.plugins.MeasureControl(position='bottomleft').add_to(m)
        
        # Tambahkan fullscreen control
        folium.plugins.Fullscreen(position='topright').add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error membuat peta interaktif: {str(e)}")
        return None

# [Fungsi lainnya tetap sama seperti sebelumnya...]

def setup_geospatial_visualization():
    """Setup utama untuk visualisasi geospasial"""
    
    st.header("🌍 Visualisasi Geospasial Nighttime Lights dengan Masking Administrasi")
    st.markdown("Analisis spasial dan temporal data nighttime lights dengan masking batas administrasi wilayah")
    
    # Kontrol untuk penghilangan blok hitam
    st.sidebar.header("⚙️ Pengaturan Filter Pixel")
    remove_black_blocks = st.sidebar.checkbox("Hilangkan Blok Hitam", value=True, 
                                            help="Hanya tampilkan pixel terang saja")
    
    brightness_threshold = st.sidebar.slider("Threshold Kecerahan Pixel", 
                                           min_value=0.0, max_value=1.0, 
                                           value=0.1, step=0.01,
                                           help="Nilai threshold untuk menentukan pixel terang (0-1)")
    
    # Upload data raster
    raster_files = st.file_uploader(
        "Pilih file TIFF raster NTL untuk visualisasi", 
        type=["tif", "tiff"], 
        accept_multiple_files=True,
        help="Unggah beberapa file raster NTL untuk analisis geospasial",
        key="geospatial_upload"
    )
    
    # Upload batas administrasi
    st.sidebar.header("🗺️ Pengaturan Batas Administrasi")
    
    # Pilihan tipe file batas administrasi
    boundary_file_type = st.sidebar.radio(
        "Pilih tipe file batas administrasi:",
        ["Shapefile", "GeoJSON"],
        help="Pilih format file batas administrasi yang akan diunggah"
    )
    
    admin_attribute = None
    admin_value = None
    admin_boundary_path = None
    admin_gdf = None

    if boundary_file_type == "Shapefile":
        # Upload shapefile batas administrasi
        admin_files = st.sidebar.file_uploader(
            "Upload Shapefile Batas Administrasi (.shp + companion files)",
            type=["shp", "shx", "dbf", "prj"],
            accept_multiple_files=True,
            help="Upload semua file shapefile (.shp, .shx, .dbf, .prj)"
        )
        
        if admin_files:
            # Pisahkan file berdasarkan ekstensi
            shp_file = None
            companion_files = []
            
            for file in admin_files:
                if file.name.endswith('.shp'):
                    shp_file = file
                else:
                    companion_files.append(file)
            
            if shp_file:
                admin_gdf, error = read_shapefile_files(shp_file, companion_files)
                
                if error:
                    st.sidebar.error(error)
                elif admin_gdf is not None:
                    st.sidebar.success(f"✅ Shapefile berhasil diunggah: {len(admin_gdf)} fitur")
                    
                    # Simpan file sementara
                    with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp_file:
                        admin_gdf.to_file(tmp_file.name, driver='ESRI Shapefile')
                        admin_boundary_path = tmp_file.name
                    
                    # Pilih atribut untuk filtering
                    if not admin_gdf.empty:
                        attributes = list(admin_gdf.columns)
                        # Exclude geometry columns
                        attributes = [attr for attr in attributes if attr != 'geometry']
                        
                        if attributes:
                            admin_attribute = st.sidebar.selectbox(
                                "Pilih atribut untuk filtering",
                                options=attributes,
                                index=0
                            )
                            
                            if admin_attribute:
                                unique_values = admin_gdf[admin_attribute].astype(str).unique()
                                admin_value = st.sidebar.selectbox(
                                    "Pilih nilai atribut",
                                    options=unique_values
                                )
    
    else:  # GeoJSON
        # Upload file GeoJSON
        geojson_file = st.sidebar.file_uploader(
            "Upload File GeoJSON Batas Administrasi",
            type=["geojson", "json"],
            help="Upload file GeoJSON yang berisi batas administrasi"
        )
        
        if geojson_file:
            admin_gdf, error = read_geojson_file(geojson_file)
            
            if error:
                st.sidebar.error(error)
            elif admin_gdf is not None:
                st.sidebar.success(f"✅ GeoJSON berhasil diunggah: {len(admin_gdf)} fitur")
                
                # Simpan file sementara
                with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp_file:
                    admin_gdf.to_file(tmp_file.name, driver='GeoJSON')
                    admin_boundary_path = tmp_file.name
                
                # Pilih atribut untuk filtering
                if not admin_gdf.empty:
                    attributes = list(admin_gdf.columns)
                    # Exclude geometry columns
                    attributes = [attr for attr in attributes if attr != 'geometry']
                    
                    if attributes:
                        admin_attribute = st.sidebar.selectbox(
                            "Pilih atribut untuk filtering",
                            options=attributes,
                            index=0
                        )
                        
                        if admin_attribute:
                            unique_values = admin_gdf[admin_attribute].astype(str).unique()
                            admin_value = st.sidebar.selectbox(
                                "Pilih nilai atribut",
                                options=unique_values
                            )

    # [Bagian visualisasi tetap sama seperti sebelumnya...]
    if raster_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_paths = []
            for i, uploaded_file in enumerate(raster_files):
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raster_paths.append(file_path)
            
            st.success(f"✅ {len(raster_paths)} file raster berhasil diunggah")
            
            # Tampilkan informasi filter
            if remove_black_blocks:
                st.info(f"🔧 Filter aktif: Hanya menampilkan pixel dengan kecerahan di atas {brightness_threshold}")
            
            if admin_boundary_path and admin_gdf is not None:
                file_type = 'geojson' if boundary_file_type == "GeoJSON" else 'shapefile'
                filter_info = f"({admin_attribute}: {admin_value})" if admin_attribute and admin_value else "(semua data)"
                st.info(f"🗺️ Masking aktif: Batas administrasi {filter_info} - Format: {boundary_file_type}")
            
            # Kontrol visualisasi
            col1, col2 = st.columns(2)
            with col1:
                viz_type = st.selectbox(
                    "Tipe Visualisasi",
                    ["Peta Interaktif", "Grid Comparison", "Analisis Statistik", "Single View"]
                )
            
            # Visualisasi berdasarkan pilihan
            if viz_type == "Peta Interaktif":
                st.subheader("🗺️ Peta Interaktif Nighttime Lights dengan Batas Administrasi")
                years = [f"Tahun {2020+i}" for i in range(len(raster_paths))]
                interactive_map = create_interactive_ntl_map(
                    raster_paths, years, remove_black_blocks, brightness_threshold,
                    admin_boundary_path, admin_attribute, admin_value, 
                    'geojson' if boundary_file_type == "GeoJSON" else 'shapefile'
                )
                if interactive_map:
                    st_folium(interactive_map, width=900, height=600)
                else:
                    st.error("Gagal membuat peta interaktif")

    else:
        st.info("📁 Silakan unggah file TIFF raster NTL untuk memulai visualisasi")

# ----------------------------
# KONFIGURASI STREAMLIT UTAMA
# ----------------------------

st.set_page_config(
    page_title="NTL Geospatial Visualization with Administrative Masking", 
    layout="wide",
    page_icon="🌃"
)

st.title("🌃 Nighttime Lights Geospatial Visualization Tool")
st.markdown("**Dengan Fitur Masking Batas Administrasi dan Filter Pixel Terang**")

# Langsung menampilkan visualisasi geospasial tanpa tabs
setup_geospatial_visualization()

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Geospatial Visualization Tool - Ver 5.1**")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Analysis Expert) & Adipandang Yudono (WebGIS NTL Analytics Developer)**")
