"""
Script Analisis Nighttime Light pada Streamlit
FOKUS PADA VISUALISASI GEOSPASIAL NTL
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

# ----------------------------
# FUNGSI VISUALISASI GEOSPASIAL
# ----------------------------

def create_ntl_colormap():
    """Membuat colormap khusus untuk nighttime lights"""
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('ntl_colormap', colors, N=256)

def remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold=0.1, min_pixel_value=0.01):
    """
    Menghilangkan blok hitam dan hanya mempertahankan pixel terang
    
    Parameters:
    - data: array raster
    - brightness_threshold: threshold untuk menentukan pixel terang (0-1)
    - min_pixel_value: nilai minimum absolut untuk dianggap sebagai pixel valid
    
    Returns:
    - data_cleaned: array dengan blok hitam dihilangkan dan hanya pixel terang yang ditampilkan
    """
    # Normalisasi data ke range 0-1
    data_normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    
    # Buat mask untuk pixel terang
    bright_mask = data_normalized > brightness_threshold
    
    # Mask untuk nilai yang valid (bukan NaN dan di atas minimum value)
    valid_mask = (data > min_pixel_value) & (~np.isnan(data))
    
    # Gabungkan mask - hanya pertahankan pixel yang terang DAN valid
    final_mask = bright_mask & valid_mask
    
    # Buat array hasil dengan nilai asli untuk pixel terang, NaN untuk lainnya
    data_cleaned = np.where(final_mask, data, np.nan)
    
    return data_cleaned

def plot_geospatial_ntl(raster_path, title="Nighttime Lights", remove_black=True, brightness_threshold=0.1):
    """Visualisasi geospasial raster NTL dengan Matplotlib"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Handle no data values
            data[data == src.nodata] = np.nan
            
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

def create_interactive_ntl_map(raster_paths, year_labels=None, remove_black=True, brightness_threshold=0.1):
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
        
        # Untuk setiap raster, tambahkan sebagai overlay
        for i, raster_path in enumerate(raster_paths):
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                year_label = year_labels[i] if year_labels and i < len(year_labels) else f"Year {i+1}"
                
                # Convert raster to PNG untuk overlay
                data = src.read(1)
                data[data == src.nodata] = 0
                
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

def plot_ntl_comparison(raster_paths, titles=None, remove_black=True, brightness_threshold=0.1):
    """Plot komparasi multiple NTL raster dalam grid"""
    n_rasters = len(raster_paths)
    n_cols = min(3, n_rasters)
    n_rows = (n_rasters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rasters == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (ax, raster_path) in enumerate(zip(axes.flat, raster_paths)):
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                data[data == src.nodata] = np.nan
                bounds = src.bounds
                
                # Hapus blok hitam dan pertahankan hanya pixel terang jika diminta
                if remove_black:
                    data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                
                ntl_cmap = create_ntl_colormap()
                im = ax.imshow(data, cmap=ntl_cmap,
                              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                
                title = titles[i] if titles and i < len(titles) else f"NTL {i+1}"
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar untuk setiap subplot
                plt.colorbar(im, ax=ax, shrink=0.8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Error Loading Data")
    
    # Sembunyikan axes yang tidak terpakai
    for j in range(i+1, n_rows*n_cols):
        axes.flat[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def generate_ntl_statistics(raster_paths, remove_black=True, brightness_threshold=0.1):
    """Generate statistics untuk data NTL"""
    stats_data = []
    
    for i, path in enumerate(raster_paths):
        with rasterio.open(path) as src:
            data = src.read(1)
            data[data == src.nodata] = np.nan
            
            # Hapus blok hitam dan pertahankan hanya pixel terang jika diminta
            if remove_black:
                data_cleaned = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                bright_pixel_count = np.sum(~np.isnan(data_cleaned))
                total_pixel_count = np.sum(~np.isnan(data))
                bright_pixel_ratio = bright_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
            else:
                data_cleaned = data
                bright_pixel_count = np.sum(data_cleaned > np.nanpercentile(data_cleaned, 90))  # Top 10% sebagai pixel terang
                total_pixel_count = np.sum(~np.isnan(data_cleaned))
                bright_pixel_ratio = bright_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
            
            stats = {
                'Dataset': f"NTL {i+1}",
                'Min': np.nanmin(data_cleaned),
                'Max': np.nanmax(data_cleaned),
                'Mean': np.nanmean(data_cleaned),
                'Std': np.nanstd(data_cleaned),
                'Total Area (px)': total_pixel_count,
                'Bright Area (px)': bright_pixel_count,
                'Bright Pixel Ratio (%)': bright_pixel_ratio * 100
            }
            stats_data.append(stats)
    
    return pd.DataFrame(stats_data)

# ----------------------------
# FUNGSI UTAMA VISUALISASI GEOSPASIAL
# ----------------------------

def setup_geospatial_visualization():
    """Setup utama untuk visualisasi geospasial"""
    
    st.header("🌍 Visualisasi Geospasial Nighttime Lights")
    st.markdown("Analisis spasial dan temporal data nighttime lights dengan visualisasi interaktif")
    
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
            
            # Kontrol visualisasi
            col1, col2 = st.columns(2)
            with col1:
                viz_type = st.selectbox(
                    "Tipe Visualisasi",
                    ["Peta Interaktif", "Grid Comparison", "Analisis Statistik", "Single View"]
                )
            with col2:
                opacity = st.slider("Opacity Peta", 0.1, 1.0, 0.7)
            
            # Visualisasi berdasarkan pilihan
            if viz_type == "Peta Interaktif":
                st.subheader("🗺️ Peta Interaktif Nighttime Lights")
                years = [f"Tahun {2020+i}" for i in range(len(raster_paths))]
                interactive_map = create_interactive_ntl_map(raster_paths, years, 
                                                           remove_black_blocks, brightness_threshold)
                if interactive_map:
                    st_folium(interactive_map, width=900, height=600)
                else:
                    st.error("Gagal membuat peta interaktif")
            
            elif viz_type == "Grid Comparison":
                st.subheader("📊 Perbandingan Multi-Temporal")
                titles = [f"Data {i+1} ({uploaded_file.name})" for i, uploaded_file in enumerate(raster_files)]
                comp_fig = plot_ntl_comparison(raster_paths, titles, remove_black_blocks, brightness_threshold)
                st.pyplot(comp_fig)
            
            elif viz_type == "Analisis Statistik":
                st.subheader("📈 Statistik Spasial NTL")
                
                stats_df = generate_ntl_statistics(raster_paths, remove_black_blocks, brightness_threshold)
                st.dataframe(stats_df.style.format({
                    'Min': '{:.4f}',
                    'Max': '{:.4f}', 
                    'Mean': '{:.4f}',
                    'Std': '{:.4f}',
                    'Total Area (px)': '{:,}',
                    'Bright Area (px)': '{:,}',
                    'Bright Pixel Ratio (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # Visualisasi trend
                if len(raster_paths) > 1:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Data preparation dengan filter
                    metrics_data = []
                    for path in raster_paths:
                        with rasterio.open(path) as src:
                            data = src.read(1)
                            data[data == src.nodata] = np.nan
                            if remove_black_blocks:
                                data = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                            metrics_data.append(data)
                    
                    metrics = {
                        'Mean Radiance': [np.nanmean(data) for data in metrics_data],
                        'Max Radiance': [np.nanmax(data) for data in metrics_data],
                        'Illuminated Area': [np.sum(~np.isnan(data)) for data in metrics_data],
                        'Std Dev': [np.nanstd(data) for data in metrics_data]
                    }
                    
                    for idx, (title, values) in enumerate(metrics.items()):
                        ax = axes[idx//2, idx%2]
                        x_range = list(range(len(values)))
                        ax.plot(x_range, values, 'o-', linewidth=2, markersize=6)
                        ax.set_title(title)
                        ax.set_xlabel('Dataset')
                        ax.set_ylabel('Nilai')
                        ax.grid(True, alpha=0.3)
                        ax.set_xticks(x_range)
                        ax.set_xticklabels([f'DS{i+1}' for i in x_range])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif viz_type == "Single View":
                st.subheader("🔍 Detail Visualisasi per Dataset")
                
                selected_idx = st.selectbox(
                    "Pilih dataset",
                    options=list(range(len(raster_files))),
                    format_func=lambda x: f"Dataset {x+1} - {raster_files[x].name}"
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = plot_geospatial_ntl(raster_paths[selected_idx], 
                                            f"Nighttime Lights - {raster_files[selected_idx].name}",
                                            remove_black_blocks, brightness_threshold)
                    if fig:
                        st.pyplot(fig)
                
                with col2:
                    # Statistik dataset terpilih
                    with rasterio.open(raster_paths[selected_idx]) as src:
                        data = src.read(1)
                        data[data == src.nodata] = np.nan
                        
                        if remove_black_blocks:
                            data_cleaned = remove_black_blocks_and_keep_bright_pixels(data, brightness_threshold)
                            bright_pixel_count = np.sum(~np.isnan(data_cleaned))
                            total_pixel_count = np.sum(~np.isnan(data))
                            bright_ratio = (bright_pixel_count / total_pixel_count * 100) if total_pixel_count > 0 else 0
                        else:
                            data_cleaned = data
                            bright_pixel_count = np.sum(data_cleaned > np.nanpercentile(data_cleaned, 90))
                            total_pixel_count = np.sum(~np.isnan(data_cleaned))
                            bright_ratio = (bright_pixel_count / total_pixel_count * 100) if total_pixel_count > 0 else 0
                        
                        st.metric("Radiansi Minimum", f"{np.nanmin(data_cleaned):.4f}")
                        st.metric("Radiansi Maksimum", f"{np.nanmax(data_cleaned):.4f}")
                        st.metric("Radiansi Rata-rata", f"{np.nanmean(data_cleaned):.4f}")
                        st.metric("Total Area (pixels)", f"{total_pixel_count:,}")
                        st.metric("Area Terang (pixels)", f"{bright_pixel_count:,}")
                        st.metric("Persentase Area Terang", f"{bright_ratio:.2f}%")
                        
                        # Histogram
                        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                        valid_data = data_cleaned[~np.isnan(data_cleaned)]
                        if len(valid_data) > 0:
                            ax_hist.hist(valid_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
                            ax_hist.set_xlabel('Radiansi')
                            ax_hist.set_ylabel('Frekuensi')
                            ax_hist.set_title('Distribusi Radiansi (Pixel Terang)')
                            ax_hist.grid(True, alpha=0.3)
                        else:
                            ax_hist.text(0.5, 0.5, 'Tidak ada data\nyang memenuhi filter', 
                                       ha='center', va='center', transform=ax_hist.transAxes)
                        st.pyplot(fig_hist)
    else:
        st.info("📁 Silakan unggah file TIFF raster NTL untuk memulai visualisasi")

# ----------------------------
# KONFIGURASI STREAMLIT UTAMA - HANYA VISUALISASI GEOSPASIAL
# ----------------------------

st.set_page_config(
    page_title="NTL Geospatial Visualization", 
    layout="wide",
    page_icon="🌃"
)

st.title("🌃 Nighttime Lights Geospatial Visualization Tool")
st.markdown("**Dengan Fitur Penghilangan Blok Hitam dan Filter Pixel Terang**")

# Langsung menampilkan visualisasi geospasial tanpa tabs
setup_geospatial_visualization()

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Geospatial Visualization Tool - Ver 3.0**")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Analysis Expert) & Adipandang Yudono (WebGIS NTL Analytics Developer)**")
