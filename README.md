# Vision AI — Sistem Presensi Cerdas Berbasis Web

**Face Recognition & Hybrid Liveness Detection untuk Mencegah Manipulasi Kehadiran**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com/)
[![Dlib](https://img.shields.io/badge/Dlib-19.24-orange)](http://dlib.net/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)]()

> Tugas Besar Mata Kuliah Image Processing — Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Andalas (2026)

---

## 📖 Tentang Proyek

**Vision AI** adalah sistem presensi berbasis web yang menggantikan mekanisme absensi konvensional (manual, kartu, RFID, PIN) dengan **pengenalan wajah berbasis CNN (Dlib)** yang dipadukan dengan **Hybrid Liveness Detection**. Sistem ini dirancang untuk memberantas praktik *titip absen* dengan memastikan bahwa hanya wajah asli — dan hidup — dari individu terdaftar yang dapat mencatatkan kehadiran.

Dua lapisan keamanan bekerja secara simultan:

- **Active Liveness Detection** — pengguna diminta menyelesaikan tantangan acak (kedip mata, buka mulut, atau toleh kepala) yang dianalisis secara real-time.
- **Passive Anti-Spoofing** — analisis tekstur piksel (Laplacian Variance) untuk menolak foto cetak maupun tampilan layar perangkat.

Hasil pengujian menunjukkan **akurasi keseluruhan 95,3%**, **FAR 1,2%**, dan **FRR 3,5%** pada 200 skenario uji.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🧑‍💻 **Face Recognition CNN (Dlib)** | Deteksi & pengenalan wajah 128-dimensi menggunakan model ResNet-34 pre-trained |
| 🔄 **Face Alignment (Affine Transform)** | Koreksi orientasi wajah otomatis sebelum ekstraksi encoding |
| 👁️ **Active Liveness (EAR / MAR / Head Pose)** | Tantangan kedip mata, buka mulut, dan toleh kepala secara acak |
| 🛡️ **Passive Anti-Spoofing (Laplacian Variance)** | Menolak foto cetak & tampilan layar HP sebagai media presensi |
| 📡 **Real-Time Streaming (SSE)** | Video & status liveness diperbarui tanpa reload halaman |
| 📊 **Dashboard Analitik** | Statistik kehadiran interaktif berbasis Chart.js dengan ekspor CSV |
| 🔐 **Autentikasi Admin** | Manajemen sesi & role (superadmin / HRD) |
| 🌐 **One-Shot Learning** | Pendaftaran wajah baru hanya dengan 1 foto per individu |

---

## 🏗️ Arsitektur Sistem

Sistem menggunakan arsitektur berlapis (layered architecture):

```
Layer 1: Akuisisi Data        → Kamera (OpenCV VideoCapture)
Layer 2: Pemrosesan Citra & AI → Face Recognition (Dlib CNN + Affine Transform)
                                  Hybrid Liveness (EAR, MAR, Head Pose, Laplacian)
Layer 3: Logika Bisnis & DB    → Flask backend + SQLAlchemy + SQLite
Layer 4: Presentasi Web        → HTML5, Bootstrap 5, Chart.js (via SSE)
```

**Alur singkat:** Kamera → Deteksi Wajah (CNN) → Passive Anti-Spoofing (Laplacian Variance) → Tantangan Active Liveness (EAR/MAR/Head Pose) → Face Alignment & Encoding → Pencocokan Euclidean Distance (< 0,6) → Cek Duplikasi → Catat Presensi ke SQLite.

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi | Versi |
|---|---|---|
| Bahasa Pemrograman | Python | 3.10+ |
| Web Framework | Flask | 3.0.0 |
| ORM | Flask-SQLAlchemy | 3.1.1 |
| Computer Vision | OpenCV | 4.8.1 (headless) |
| Face Detection & Landmark | Dlib | 19.24.2 |
| Face Recognition Wrapper | face_recognition | 1.3.0 |
| Database | SQLite | — |
| Frontend | HTML5, Bootstrap 5 | 5.3.3 |
| Visualisasi Data | Chart.js | 4.4.3 |
| Text-to-Speech | pyttsx3 | 2.90 |
| Production Server | Gunicorn | 21.2.0 |
| Tunneling (Demo Publik) | Ngrok | 3.x (Static Domain) |

---

## 📂 Struktur Proyek

```
attendancesystemai/
├── app.py                  # Aplikasi utama Flask (routes, model DB, pipeline AI)
├── requirement.txt         # Daftar dependensi Python
├── dataset_wajah/          # Penyimpanan foto wajah terdaftar (dibuat otomatis)
├── attendance_modern.db    # Database SQLite (dibuat otomatis saat runtime)
└── templates/
    ├── index.html          # Landing page
    ├── scanner.html        # Halaman Live Scanner (presensi)
    ├── dashboard.html       # Dashboard analitik admin
    └── login.html           # Halaman login admin
```

---

## 🚀 Instalasi & Menjalankan

### Prasyarat

- Python 3.10 atau lebih baru
- Webcam (built-in atau eksternal, minimal 720p)
- CMake & compiler C++ (dibutuhkan untuk membangun `dlib`)

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/suryaandika3005/attendancesystemai.git
cd attendancesystemai

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependensi
pip install -r requirement.txt

# 4. Jalankan aplikasi
python app.py
```

Aplikasi akan berjalan pada `http://127.0.0.1:5000`.

### Kredensial Admin Default

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | superadmin |
| `hrd` | `hrd123` | hrd |

> ⚠️ **Penting:** Ganti kredensial default ini sebelum digunakan di luar lingkungan pengembangan/demo.

### Deployment Publik (Opsional, via Ngrok)

```bash
ngrok http --domain=<your-static-domain> 5000
```

---

## 🧪 Pengujian & Metrik Performa

Pengujian dilakukan menggunakan metode **Black-Box Testing** pada 200 skenario mencakup variasi pencahayaan, sudut wajah, dan jenis serangan spoofing.

| Metrik | Nilai |
|---|---|
| Overall Accuracy | 95,30% |
| Precision (Face Recognition) | 94,80% |
| Recall (Face Recognition) | 96,10% |
| F1-Score | 95,44% |
| Akurasi Anti-Spoofing (Passive) | 97,50% |
| Akurasi Liveness Detection (Active) | 96,80% |
| False Acceptance Rate (FAR) | 1,20% |
| False Rejection Rate (FRR) | 3,50% |
| Waktu Inferensi Rata-rata / Frame | 248 ms |
| Throughput | ± 4 fps |

---

## 🔮 Pengembangan ke Depan

- **Model deteksi wajah yang lebih ringan** (MediaPipe Face Mesh / YOLO-Face) untuk meningkatkan throughput di atas 4 fps.
- **Dukungan kamera Infra Merah (IR)** untuk meningkatkan robustness anti-spoofing pada kondisi pencahayaan rendah.
- **Migrasi basis data ke cloud** (PostgreSQL / Firebase Firestore) untuk mendukung skala enterprise dengan konkurensi tinggi.

---

## 📚 Referensi Utama

1. Horn Boe, C. et al. (2024). *An Automated Face Detection and Recognition for Class Attendance.* JOIV.
2. Santoso, J.T. et al. (2024). *Optimizing Attendance System: Integrating Liveness Detection and Deep Learning for Reliable Face Recognition.* JUITA.
3. Dewi, C. et al. (2022). *Adjusting Eye Aspect Ratio for Strong Eye Blink Detection Based on Facial Landmarks.* PeerJ Computer Science.
4. Turhal, U. et al. (2024). *A New Face Presentation Attack Detection Method Based on Face-Weighted Multi-Color Multi-Level Texture Features.* The Visual Computer.

Daftar pustaka lengkap tersedia pada laporan tugas besar.

---

## 👤 Penulis

**Surya Andika**
NIM: 2311533005
Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Andalas

📄 Laporan lengkap: *Vision AI: Sistem Presensi Cerdas Berbasis Web Menggunakan Face Recognition dan Hybrid Liveness Detection* (2026)
🎥 Video demonstrasi: [youtu.be/XK_2x4oag0o](https://youtu.be/XK_2x4oag0o)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis sebagai pemenuhan tugas besar mata kuliah Image Processing. Silakan gunakan sebagai referensi dengan menyertakan atribusi yang sesuai.
