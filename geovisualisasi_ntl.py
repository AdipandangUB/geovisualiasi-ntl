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

def mask_with_administrative_boundaries(raster_data, raster_transform, raster_crs, admin_boundary_path, admin_attribute=None, admin_value=None):
    """
    Melakukan masking data raster dengan batas administrasi wilayah
    """
    try:
        # Baca file batas administrasi
        admin_gdf = gpd.read_file(admin_boundary_path)
        
        # Filter berdasarkan atribut jika diberikan
        if admin_attribute and admin_value and admin_attribute in admin_gdf.columns:
            admin_gdf = admin_gdf[admin_gdf[admin_attribute].astype(str) == str(admin_value)]
        
        # Jika tidak ada data setelah filtering, tampilkan warning
        if len(admin_gdf) == 0:
            st.warning(f"Tidak ada data yang sesuai dengan filter {admin_attribute} = {admin_value}. Menampilkan semua data.")
            admin_gdf = gpd.read_file(admin_boundary_path)
        
        # Pastikan CRS sama dengan raster
        if admin_gdf.crs != raster_crs:
            admin_gdf = admin_gdf.to_crs(raster_crs)
        
        # Buat mask dari geometri administrasi
        from rasterio.mask import mask
        
        # Convert to geojson-like format
        shapes = [mapping(geom) for geom in admin_gdf.geometry]
        
        # Create temporary file for original raster
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
        
        return masked_data
        
    except Exception as e:
        st.error(f"Error dalam masking batas administrasi: {str(e)}")
        return raster_data

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

def save_uploaded_files(uploaded_files, temp_dir):
    """Menyimpan uploaded files ke temporary directory dan return paths"""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def process_shapefile_upload(shp_file, companion_files, temp_dir):
    """Memproses upload shapefile dan return GeoDataFrame"""
    try:
        # Simpan semua file shapefile
        all_files = [shp_file] + companion_files
        saved_paths = save_uploaded_files(all_files, temp_dir)
        
        # Cari path ke file .shp
        shp_path = next((path for path in saved_paths if path.endswith('.shp')), None)
        
        if shp_path:
            # Baca shapefile
            gdf = gpd.read_file(shp_path)
            return gdf, None, shp_path
        else:
            return None, "File .shp tidak ditemukan", None
            
    except Exception as e:
        return None, f"Error membaca shapefile: {str(e)}", None

def process_geojson_upload(geojson_file, temp_dir):
    """Memproses upload GeoJSON dan return GeoDataFrame"""
    try:
        saved_paths = save_uploaded_files([geojson_file], temp_dir)
        geojson_path = saved_paths[0]
        gdf = gpd.read_file(geojson_path)
        return gdf, None, geojson_path
    except Exception as e:
        return None, f"Error membaca GeoJSON: {str(e)}", None

def plot_geospatial_ntl(raster_path, title="Nighttime Lights", remove_black=True, 
                       brightness_threshold=0.1, admin_boundary_path=None, 
                       admin_attribute=None, admin_value=None):
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
            if admin_boundary_path:
                data = mask_with_administrative_boundaries(
                    data, transform, crs, admin_boundary_path, admin_attribute, admin_value
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
                              brightness_threshold=0.1, admin_boundary_path=None,
                              admin_attribute=None, admin_value=None):
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
        if admin_boundary_path:
            try:
                admin_gdf = gpd.read_file(admin_boundary_path)
                
                # Filter berdasarkan atribut jika diberikan
                if admin_attribute and admin_value and admin_attribute in admin_gdf.columns:
                    admin_gdf = admin_gdf[admin_gdf[admin_attribute].astype(str) == str(admin_value)]
                
                # Konversi ke JSON yang compatible dengan Folium
                geojson_data = admin_gdf.__geo_interface__
                
                # Tambahkan batas administrasi ke peta
                folium.GeoJson(
                    geojson_data,
                    name='Batas Administrasi',
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'color': 'red',
                        'weight': 2,
                        'fillOpacity': 0
                    }
                ).add_to(m)
                
            except Exception as e:
                st.warning(f"Tidak dapat menambahkan batas administrasi ke peta: {str(e)}")
        
        # Untuk setiap raster, tambahkan sebagai overlay
        for i, raster_path in enumerate(raster_paths):
            try:
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    data = src.read(1)
                    
                    # Handle no data values
                    data[data == src.nodata] = 0
                    
                    # Lakukan masking dengan batas administrasi jika diberikan
                    if admin_boundary_path:
                        data = mask_with_administrative_boundaries(
                            data, src.transform, src.crs, admin_boundary_path, admin_attribute, admin_value
                        )
                    
                    # Hapus blok hitam dan pertahankan hanya pixel terang jika diminta
                    if remove_black:
                        data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                    
                    # Replace NaN dengan 0 untuk visualisasi
                    data = np.nan_to_num(data)
                    
                    # Normalisasi data untuk visualisasi
                    if np.max(data) > np.min(data):
                        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
                    else:
                        data_norm = data
                    
                    # Apply colormap
                    ntl_cmap = create_ntl_colormap()
                    colored_data = ntl_cmap(data_norm)
                    
                    # Simpan sebagai PNG sementara
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        plt.imsave(tmp_file.name, colored_data)
                        
                        # Add raster overlay ke peta
                        img_overlay = folium.raster_layers.ImageOverlay(
                            name=year_labels[i] if year_labels and i < len(year_labels) else f"Year {i+1}",
                            image=tmp_file.name,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.7,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)
                        
                        # Clean up temporary file setelah peta dibuat
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                            
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
        help="Unggah beberapa file raster NTL untuk analisis geospasial"
    )
    
    # Upload batas administrasi
    st.sidebar.header("🗺️ Pengaturan Batas Administrasi")
    
    # Pilihan tipe file batas administrasi
    boundary_file_type = st.sidebar.radio(
        "Pilih tipe file batas administrasi:",
        ["Shapefile", "GeoJSON"],
        help="Pilih format file batas administrasi yang akan diunggah"
    )
    
    # Inisialisasi session state variables
    if 'admin_gdf' not in st.session_state:
        st.session_state.admin_gdf = None
    if 'admin_boundary_path' not in st.session_state:
        st.session_state.admin_boundary_path = None
    if 'admin_attribute' not in st.session_state:
        st.session_state.admin_attribute = None
    if 'admin_value' not in st.session_state:
        st.session_state.admin_value = None

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
            
            if shp_file and companion_files:
                with tempfile.TemporaryDirectory() as tmpdir:
                    admin_gdf, error, admin_boundary_path = process_shapefile_upload(
                        shp_file, companion_files, tmpdir
                    )
                    
                    if error:
                        st.sidebar.error(error)
                    elif admin_gdf is not None:
                        st.session_state.admin_gdf = admin_gdf
                        st.session_state.admin_boundary_path = admin_boundary_path
                        st.sidebar.success(f"✅ Shapefile berhasil diunggah: {len(admin_gdf)} fitur")
    
    else:  # GeoJSON
        # Upload file GeoJSON
        geojson_file = st.sidebar.file_uploader(
            "Upload File GeoJSON Batas Administrasi",
            type=["geojson", "json"],
            help="Upload file GeoJSON yang berisi batas administrasi"
        )
        
        if geojson_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                admin_gdf, error, admin_boundary_path = process_geojson_upload(geojson_file, tmpdir)
                
                if error:
                    st.sidebar.error(error)
                elif admin_gdf is not None:
                    st.session_state.admin_gdf = admin_gdf
                    st.session_state.admin_boundary_path = admin_boundary_path
                    st.sidebar.success(f"✅ GeoJSON berhasil diunggah: {len(admin_gdf)} fitur")

    # Tampilkan opsi filtering jika ada data administrasi
    if st.session_state.admin_gdf is not None and not st.session_state.admin_gdf.empty:
        attributes = [col for col in st.session_state.admin_gdf.columns if col != 'geometry']
        
        if attributes:
            st.session_state.admin_attribute = st.sidebar.selectbox(
                "Pilih atribut untuk filtering",
                options=attributes,
                index=0
            )
            
            if st.session_state.admin_attribute:
                unique_values = st.session_state.admin_gdf[st.session_state.admin_attribute].astype(str).unique()
                st.session_state.admin_value = st.sidebar.selectbox(
                    "Pilih nilai atribut",
                    options=unique_values
                )

    # Proses visualisasi jika ada raster files
    if raster_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_paths = save_uploaded_files(raster_files, tmpdir)
            
            st.success(f"✅ {len(raster_paths)} file raster berhasil diunggah")
            
            # Tampilkan informasi filter
            if remove_black_blocks:
                st.info(f"🔧 Filter aktif: Hanya menampilkan pixel dengan kecerahan di atas {brightness_threshold}")
            
            if st.session_state.admin_boundary_path and st.session_state.admin_gdf is not None:
                filter_info = f"({st.session_state.admin_attribute}: {st.session_state.admin_value})" if st.session_state.admin_attribute and st.session_state.admin_value else "(semua data)"
                st.info(f"🗺️ Masking aktif: Batas administrasi {filter_info} - Format: {boundary_file_type}")
            
            # Kontrol visualisasi
            col1, col2 = st.columns(2)
            with col1:
                viz_type = st.selectbox(
                    "Tipe Visualisasi",
                    ["Peta Interaktif", "Grid Comparison", "Single View"]
                )
            
            # Visualisasi berdasarkan pilihan
            if viz_type == "Peta Interaktif":
                st.subheader("🗺️ Peta Interaktif Nighttime Lights dengan Batas Administrasi")
                years = [f"Tahun {2020+i}" for i in range(len(raster_paths))]
                
                interactive_map = create_interactive_ntl_map(
                    raster_paths, 
                    years, 
                    remove_black_blocks, 
                    brightness_threshold,
                    st.session_state.admin_boundary_path,
                    st.session_state.admin_attribute,
                    st.session_state.admin_value
                )
                
                if interactive_map:
                    st_folium(interactive_map, width=900, height=600)
                else:
                    st.error("Gagal membuat peta interaktif")
                    
            elif viz_type == "Single View" and raster_paths:
                st.subheader("📊 Single View Nighttime Lights")
                fig = plot_geospatial_ntl(
                    raster_paths[0],
                    "Nighttime Lights",
                    remove_black_blocks,
                    brightness_threshold,
                    st.session_state.admin_boundary_path,
                    st.session_state.admin_attribute,
                    st.session_state.admin_value
                )
                if fig:
                    st.pyplot(fig)

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

# Langsung menampilkan visualisasi geospasial
setup_geospatial_visualization()

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Geospatial Visualization Tool - Ver 5.2**")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Analysis Expert) & Adipandang Yudono (WebGIS NTL Analytics Developer)**")
