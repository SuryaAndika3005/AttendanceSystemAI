import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import threading
import math
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog, ttk
import sqlite3
import pyttsx3
import csv

# ==========================================
# 1. KONFIGURASI UI (CustomTkinter)
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# ==========================================
# 2. INISIALISASI DATABASE (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT UNIQUE, waktu_daftar DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT, tanggal TEXT, waktu TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS admins (username TEXT PRIMARY KEY, password TEXT)''')
    
    cursor.execute("SELECT * FROM admins WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")
    
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. FUNGSI SUARA (TEXT-TO-SPEECH)
# ==========================================
def ucapkan_pesan(teks):
    def jalankan_tts():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(teks)
        engine.runAndWait()
    threading.Thread(target=jalankan_tts, daemon=True).start()

# ==========================================
# 4. KELAS MULTI-THREADING UNTUK KAMERA
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# 5. FUNGSI LIVENESS & PENGENALAN WAJAH
# ==========================================
def hitung_jarak(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def eye_aspect_ratio(eye):
    A = hitung_jarak(eye[1], eye[5])
    B = hitung_jarak(eye[2], eye[4])
    C = hitung_jarak(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

EAR_THRESHOLD = 0.22
DATASET_DIR = "dataset_wajah"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

known_face_encodings = []
known_face_names = []

def load_registered_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(DATASET_DIR, filename)
            name = os.path.splitext(filename)[0]
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

# ==========================================
# 6. KELAS APLIKASI UTAMA (DASHBOARD)
# ==========================================
class AttendanceApp(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("AI Attendance System - GPU Accelerated (RTX 3050)")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.parent = parent

        self.wajah_sudah_absen = {}
        self.tanggal_terakhir_absen = datetime.now().date()
        self.hasil_gambar_terakhir = []
        self.mata_tertutup_sebelumnya = False
        
        load_registered_faces()
        self.muat_absensi_hari_ini()

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.info_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#212121")
        self.info_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.clock_label = ctk.CTkLabel(self.info_frame, text="00:00:00", font=("Helvetica", 42, "bold"), text_color="#2FA572")
        self.clock_label.pack(pady=(25, 5))
        self.date_label = ctk.CTkLabel(self.info_frame, text="Tanggal", font=("Helvetica", 16), text_color="#B0B0B0")
        self.date_label.pack(pady=(0, 25))

        self.register_btn = ctk.CTkButton(
            self.info_frame, text="➕ Registrasi Wajah", height=45, 
            font=("Helvetica", 14, "bold"), fg_color="#D4AF37", 
            text_color="#121212", hover_color="#B5952F", command=self.register_new_face
        )
        self.register_btn.pack(fill="x", padx=25, pady=10)

        self.analytics_btn = ctk.CTkButton(
            self.info_frame, text="📊 Dashboard Analytics", height=45, 
            font=("Helvetica", 14, "bold"), fg_color="#2FA572", 
            text_color="#121212", hover_color="#248259", command=self.show_analytics
        )
        self.analytics_btn.pack(fill="x", padx=25, pady=(0, 10))

        self.log_label = ctk.CTkLabel(self.info_frame, text="📋 Log Presensi Hari Ini:", font=("Helvetica", 14, "bold"), text_color="#E0E0E0")
        self.log_label.pack(anchor="w", padx=25, pady=(15, 5))
        
        self.log_box = ctk.CTkTextbox(
            self.info_frame, state="disabled", font=("Consolas", 13), 
            fg_color="#121212", text_color="#2FA572", border_width=1, border_color="#333333"
        )
        self.log_box.pack(expand=True, fill="both", padx=25, pady=(0, 25))

        self.refresh_log_ui()
        self.vs = WebcamStream(src=0).start()
        self.update_clock()
        self.update_video()

    def muat_absensi_hari_ini(self):
        self.wajah_sudah_absen.clear()
        tgl_sekarang = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT nama, waktu FROM attendance WHERE tanggal = ?", (tgl_sekarang,))
        for row in cursor.fetchall():
            self.wajah_sudah_absen[row[0]] = row[1]
        conn.close()

    def catat_kehadiran_db(self, nama):
        now = datetime.now()
        tanggal, jam = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (nama, tanggal, waktu) VALUES (?, ?, ?)", (nama, tanggal, jam))
        conn.commit()
        conn.close()
        return jam

    def show_analytics(self):
        analytics_window = ctk.CTkToplevel(self)
        analytics_window.title("Dashboard Analytics - Riwayat Presensi")
        analytics_window.geometry("750x500")
        analytics_window.grab_set()

        ctk.CTkLabel(analytics_window, text="Rekapitulasi Data Kehadiran", font=("Helvetica", 20, "bold"), text_color="#2FA572").pack(pady=(20, 10))

        def export_csv():
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance")
            data = cursor.fetchall()
            conn.close()
            filename = f"Rekap_Presensi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Nama Lengkap", "Tanggal", "Jam Masuk"])
                writer.writerows(data)
            messagebox.showinfo("Sukses", f"Data berhasil diekspor ke {filename}", parent=analytics_window)

        ctk.CTkButton(analytics_window, text="⬇️ Ekspor Laporan (.csv)", font=("Helvetica", 12, "bold"), command=export_csv).pack(pady=(0, 15))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#212121", foreground="#E0E0E0", rowheight=30, fieldbackground="#212121", borderwidth=0)
        style.map('Treeview', background=[('selected', '#2FA572')])
        style.configure("Treeview.Heading", background="#121212", foreground="#2FA572", relief="flat", font=("Helvetica", 12, "bold"))
        
        tree_frame = ctk.CTkFrame(analytics_window)
        tree_frame.pack(expand=True, fill="both", padx=20, pady=(0, 20))
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side="right", fill="y")

        tree = ttk.Treeview(tree_frame, columns=("ID", "Nama Lengkap", "Tanggal", "Jam Masuk"), show="headings", yscrollcommand=tree_scroll.set)
        for col in tree["columns"]: tree.heading(col, text=col)
        tree.column("ID", width=50, anchor="center")
        tree.column("Nama Lengkap", width=300, anchor="w")
        tree.column("Tanggal", width=120, anchor="center")
        tree.column("Jam Masuk", width=120, anchor="center")
        tree.pack(expand=True, fill="both")
        tree_scroll.config(command=tree.yview)

        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, nama, tanggal, waktu FROM attendance ORDER BY id DESC")
        for row in cursor.fetchall(): tree.insert("", "end", values=row)
        conn.close()

    def update_clock(self):
        now = datetime.now()
        self.clock_label.configure(text=now.strftime("%H:%M:%S"))
        self.date_label.configure(text=now.strftime("%A, %d %b %Y"))
        self.after(1000, self.update_clock)

    def refresh_log_ui(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        if not self.wajah_sudah_absen:
            self.log_box.insert("end", "Belum ada data hari ini.\n")
        else:
            for nama, jam in reversed(self.wajah_sudah_absen.items()):
                self.log_box.insert("end", f"[{jam}] {nama} hadir\n")
        self.log_box.configure(state="disabled")

    def register_new_face(self):
        frame = self.vs.read()
        if frame is None: return
        new_name = simpledialog.askstring("Registrasi", "Masukkan Nama Anda:", parent=self)
        if new_name:
            cv2.imwrite(os.path.join(DATASET_DIR, f"{new_name}.jpg"), frame)
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (nama) VALUES (?)", (new_name,))
                conn.commit()
            except sqlite3.IntegrityError: pass 
            conn.close()
            load_registered_faces()
            messagebox.showinfo("Sukses", f"Wajah {new_name} berhasil didaftarkan!", parent=self)

    def update_video(self):
        frame = self.vs.read()
        if frame is not None:
            tgl_sekarang = datetime.now().date()
            if tgl_sekarang > self.tanggal_terakhir_absen:
                self.wajah_sudah_absen.clear()
                self.tanggal_terakhir_absen = tgl_sekarang
                self.refresh_log_ui()

            # AI SEKARANG MEMPROSES SETIAP FRAME (TANPA FRAME SKIPPING)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            try:
                face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            except:
                face_landmarks_list = [{}] * len(face_locations)

            self.hasil_gambar_terakhir = []

            for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                name, color, display_text = "Unknown", (0, 0, 255), "Tidak Dikenal"
                ear = 1.0

                if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                    ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0

                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if face_distances[best_match_index] < 0.45:
                        name = known_face_names[best_match_index]
                        if name in self.wajah_sudah_absen:
                            color, display_text = (255, 255, 0), f"{name} (Absen pkl {self.wajah_sudah_absen[name]})"
                        else:
                            color, display_text = (0, 255, 0), f"Wajah: {name}"
                            
                            # Deteksi Liveness / Kedipan
                            if ear < EAR_THRESHOLD:
                                self.mata_tertutup_sebelumnya = True
                            elif ear >= EAR_THRESHOLD and self.mata_tertutup_sebelumnya:
                                self.mata_tertutup_sebelumnya = False
                                if name not in self.wajah_sudah_absen:
                                    jam_masuk = self.catat_kehadiran_db(name)
                                    self.wajah_sudah_absen[name] = jam_masuk
                                    self.refresh_log_ui()
                                    ucapkan_pesan(f"Presensi atas nama {name}, berhasil direkam.")
                                    threading.Thread(target=lambda n=name: messagebox.showinfo("Presensi Sukses", f"Kehadiran '{n}' tercatat!"), daemon=True).start()
                            
                            if not self.mata_tertutup_sebelumnya and name not in self.wajah_sudah_absen:
                                display_text = f"{name} - Berkedip utk Absen!"

                self.hasil_gambar_terakhir.append((left, top, right, bottom, color, display_text))

            # Render Grafis
            for (left, top, right, bottom, color, display_text) in self.hasil_gambar_terakhir:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), color, cv2.FILLED)
                cv2.putText(frame, display_text, (left + 5, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize((640, 480)) 
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

        # UI DIREFRESH SETIAP 15ms (60 FPS SUPER MULUS)
        self.after(15, self.update_video)

    def on_closing(self):
        self.vs.stop()
        self.destroy()
        self.parent.destroy()

# ==========================================
# 7. KELAS LOGIN 
# ==========================================
class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Login Sistem - AI Attendance")
        self.geometry("400x500")
        self.resizable(False, False)
        
        self.frame = ctk.CTkFrame(self, corner_radius=15)
        self.frame.pack(pady=40, padx=40, fill="both", expand=True)
        
        ctk.CTkLabel(self.frame, text="Sistem Presensi AI", font=("Helvetica", 24, "bold"), text_color="#2FA572").pack(pady=(40, 10))
        ctk.CTkLabel(self.frame, text="Login Administrator", font=("Helvetica", 14)).pack(pady=(0, 30))
        
        self.entry_user = ctk.CTkEntry(self.frame, placeholder_text="Username", width=250, height=40)
        self.entry_user.pack(pady=10)
        
        self.entry_pass = ctk.CTkEntry(self.frame, placeholder_text="Password", show="*", width=250, height=40)
        self.entry_pass.pack(pady=10)
        
        self.login_btn = ctk.CTkButton(self.frame, text="Masuk", width=250, height=45, font=("Helvetica", 14, "bold"), fg_color="#2FA572", hover_color="#248259", command=self.cek_login)
        self.login_btn.pack(pady=30)
        
    def cek_login(self):
        user = self.entry_user.get()
        pwd = self.entry_pass.get()
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username=? AND password=?", (user, pwd))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.withdraw()
            app = AttendanceApp(self)
        else:
            messagebox.showerror("Error", "Username atau Password salah!")

if __name__ == "__main__":
    login_app = LoginWindow()
    login_app.mainloop()