import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# 1. KONFIGURASI HALAMAN & BRANDING UNAND
st.set_page_config(page_title="Portal Presensi Informatika", page_icon="🎓", layout="wide")

# Efek Auto-Refresh: Halaman akan memuat ulang data otomatis setiap 3000 milidetik (3 detik)
st_autorefresh(interval=3000, key="data_refresh")

# CSS Custom untuk UI/UX bernuansa hijau khas kampus
st.markdown("""
    <style>
    .stMetric {
        background-color: #1A1C23;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #2FA572; /* Hijau Khas */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"]:nth-child(3) > div {
        border-left-color: #E53935; /* Merah untuk Keterlambatan */
    }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNGSI DATABASE & LOGIKA BISNIS
@st.cache_data(ttl=2) # Cache dipercepat agar auto-refresh sinkron
def ambil_data():
    try:
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT id, nama, tanggal, waktu FROM attendance ORDER BY id DESC", conn)
        conn.close()
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['tanggal'] + ' ' + df['waktu'])
            # LOGIKA KETERLAMBATAN: Batas jam masuk kelas/lab 08:30 pagi
            batas_waktu = pd.to_datetime(df['tanggal'] + ' 08:30:00')
            df['status'] = ['Tepat Waktu' if waktu <= batas else 'Terlambat' 
                            for waktu, batas in zip(df['datetime'], batas_waktu)]
        return df
    except Exception as e:
        return pd.DataFrame(columns=["id", "nama", "tanggal", "waktu", "datetime", "status"])

df = ambil_data()

# 3. SIDEBAR (KONTROL ADMINISTRATOR LAB)
st.sidebar.title("🎓 Universitas Andalas")
st.sidebar.markdown("**Sistem Presensi Lab Informatika**")
st.sidebar.divider()

st.sidebar.subheader("🎛️ Filter Analitik")
if not df.empty:
    daftar_nama = ["Semua Mahasiswa/Staff"] + list(df['nama'].unique())
    filter_nama = st.sidebar.selectbox("Cari Entitas:", daftar_nama)
    
    filter_status = st.sidebar.radio("Status Kedatangan:", ["Semua", "Tepat Waktu", "Terlambat"])
    
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    if min_date == max_date:
        filter_tanggal = st.sidebar.date_input("Periode Data:", [min_date, max_date])
    else:
        filter_tanggal = st.sidebar.date_input("Periode Data:", [min_date, max_date], min_value=min_date, max_value=max_date)

    df_filtered = df.copy()
    if filter_nama != "Semua Mahasiswa/Staff":
        df_filtered = df_filtered[df_filtered['nama'] == filter_nama]
    if filter_status != "Semua":
        df_filtered = df_filtered[df_filtered['status'] == filter_status]
    if len(filter_tanggal) == 2:
        start_date, end_date = filter_tanggal
        df_filtered = df_filtered[(df_filtered['datetime'].dt.date >= start_date) & (df_filtered['datetime'].dt.date <= end_date)]
else:
    df_filtered = df
    st.sidebar.warning("Database masih kosong. Menunggu pemindaian wajah...")

st.sidebar.divider()
st.sidebar.caption("© 2026 Dept. Informatika Universitas Andalas")

# 4. HEADER & KPI METRIK (LIVE DASHBOARD)
st.title("🔴 Live Monitoring Kehadiran")
st.markdown("Pemantauan *real-time* berbasis Computer Vision & Liveness Detection.")

tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")

if not df.empty:
    df_hari_ini = df[df['tanggal'] == tanggal_hari_ini]
    hadir_hari_ini = len(df_hari_ini)
    tepat_waktu_hari_ini = len(df_hari_ini[df_hari_ini['status'] == 'Tepat Waktu'])
    terlambat_hari_ini = len(df_hari_ini[df_hari_ini['status'] == 'Terlambat'])
else:
    hadir_hari_ini = tepat_waktu_hari_ini = terlambat_hari_ini = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Total Kehadiran (Hari Ini)", f"{hadir_hari_ini} Orang")
col2.metric("✅ Tepat Waktu", f"{tepat_waktu_hari_ini} Orang")
col3.metric("⚠️ Terlambat", f"{terlambat_hari_ini} Orang")
col4.metric("🗃️ Total Rekaman", f"{len(df_filtered)} Entri")

st.divider()

# 5. GRAFIK ANALITIK (PLOTLY)
if not df_filtered.empty:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Tren Disiplin Kehadiran")
        tren_harian = df_filtered.groupby(['tanggal', 'status']).size().reset_index(name='jumlah')
        fig_line = px.bar(tren_harian, x='tanggal', y='jumlah', color='status', 
                          color_discrete_map={'Tepat Waktu': '#2FA572', 'Terlambat': '#E53935'},
                          barmode='group')
        fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_chart2:
        st.subheader("📊 Distribusi Kedatangan")
        persentase = df_filtered['status'].value_counts().reset_index()
        persentase.columns = ['status', 'jumlah']
        fig_pie = px.pie(persentase, values='jumlah', names='status', hole=0.4,
                         color='status', color_discrete_map={'Tepat Waktu': '#2FA572', 'Terlambat': '#E53935'})
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

# 6. TABEL DATA & EKSPOR
st.subheader("📋 Log Scan Kamera Utama")

if not df_filtered.empty:
    tabel_tampil = df_filtered[['id', 'nama', 'tanggal', 'waktu', 'status']]
    st.dataframe(tabel_tampil, use_container_width=True, hide_index=True)
    
    csv = tabel_tampil.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Unduh Rekap Data Laporan (CSV)",
        data=csv,
        file_name=f"Rekap_Kehadiran_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )
else:
    st.info("Menunggu data masuk...")