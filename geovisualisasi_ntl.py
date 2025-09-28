# -*- coding: utf-8 -*-
"""
Script Analisis Nighttime Light pada Streamlit
FOKUS PADA VISUALISASI GEOSPASIAL NTL
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

# ----------------------------
# FUNGSI VISUALISASI GEOSPASIAL
# ----------------------------

def create_ntl_colormap():
    """Membuat colormap khusus untuk nighttime lights"""
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('ntl_colormap', colors, N=256)

def plot_geospatial_ntl(raster_path, title="Nighttime Lights"):
    """Visualisasi geospasial raster NTL dengan Matplotlib"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Handle no data values
            data[data == src.nodata] = np.nan
            
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

def create_interactive_ntl_map(raster_paths, year_labels=None):
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
                
                # Normalisasi data untuk visualisasi
                data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                
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

def plot_ntl_comparison(raster_paths, titles=None):
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

def generate_ntl_statistics(raster_paths):
    """Generate statistics untuk data NTL"""
    stats_data = []
    
    for i, path in enumerate(raster_paths):
        with rasterio.open(path) as src:
            data = src.read(1)
            data[data == src.nodata] = np.nan
            
            stats = {
                'Dataset': f"NTL {i+1}",
                'Min': np.nanmin(data),
                'Max': np.nanmax(data),
                'Mean': np.nanmean(data),
                'Std': np.nanstd(data),
                'Area (px)': np.sum(~np.isnan(data))
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
                interactive_map = create_interactive_ntl_map(raster_paths, years)
                if interactive_map:
                    st_folium(interactive_map, width=900, height=600)
                else:
                    st.error("Gagal membuat peta interaktif")
            
            elif viz_type == "Grid Comparison":
                st.subheader("📊 Perbandingan Multi-Temporal")
                titles = [f"Data {i+1} ({uploaded_file.name})" for i, uploaded_file in enumerate(raster_files)]
                comp_fig = plot_ntl_comparison(raster_paths, titles)
                st.pyplot(comp_fig)
            
            elif viz_type == "Analisis Statistik":
                st.subheader("📈 Statistik Spasial NTL")
                
                stats_df = generate_ntl_statistics(raster_paths)
                st.dataframe(stats_df.style.format({
                    'Min': '{:.2f}',
                    'Max': '{:.2f}', 
                    'Mean': '{:.2f}',
                    'Std': '{:.2f}'
                }), use_container_width=True)
                
                # Visualisasi trend
                if len(raster_paths) > 1:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Data preparation
                    metrics = {
                        'Mean Radiance': [np.nanmean(rasterio.open(path).read(1)) for path in raster_paths],
                        'Max Radiance': [np.nanmax(rasterio.open(path).read(1)) for path in raster_paths],
                        'Illuminated Area': [np.sum(~np.isnan(rasterio.open(path).read(1))) for path in raster_paths],
                        'Std Dev': [np.nanstd(rasterio.open(path).read(1)) for path in raster_paths]
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
                                            f"Nighttime Lights - {raster_files[selected_idx].name}")
                    if fig:
                        st.pyplot(fig)
                
                with col2:
                    # Statistik dataset terpilih
                    with rasterio.open(raster_paths[selected_idx]) as src:
                        data = src.read(1)
                        data[data == src.nodata] = np.nan
                        
                        st.metric("Radiansi Minimum", f"{np.nanmin(data):.2f}")
                        st.metric("Radiansi Maksimum", f"{np.nanmax(data):.2f}")
                        st.metric("Radiansi Rata-rata", f"{np.nanmean(data):.2f}")
                        st.metric("Area Terang (pixels)", f"{np.sum(~np.isnan(data)):,}")
                        
                        # Histogram
                        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                        ax_hist.hist(data[~np.isnan(data)].flatten(), bins=50, alpha=0.7, edgecolor='black')
                        ax_hist.set_xlabel('Radiansi')
                        ax_hist.set_ylabel('Frekuensi')
                        ax_hist.set_title('Distribusi Radiansi')
                        ax_hist.grid(True, alpha=0.3)
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

# Langsung menampilkan visualisasi geospasial tanpa tabs
setup_geospatial_visualization()

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Geospatial Visualization Tool - Ver 3.0**")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Analysis Expert) & Adipandang Yudono (WebGIS NTL Analytics Developer)**")
