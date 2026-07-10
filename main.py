import os
import pickle
import time
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# konfigurasi global
DATASET_DIR = "dataset"
DB_FILE = "feature_database.pkl"
UKURAN_GAMBAR = (256, 256)

# parameter LBP
LBP_RADIUS = 1
LBP_TITIK = 8
LBP_METODE = "uniform"

# parameter histogram warna
BINS_WARNA = 64


def ekstrak_lbp(gambar_gray):
    # ekstraksi tekstur dengan local binary pattern
    lbp_map = local_binary_pattern(gambar_gray, LBP_TITIK, LBP_RADIUS, method=LBP_METODE)
    
    jumlah_bin = LBP_TITIK + 2
    hist, _ = np.histogram(lbp_map.ravel(), bins=jumlah_bin, range=(0, jumlah_bin))
    
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
        
    return hist, lbp_map

def ekstrak_warna(gambar_bgr):
    # memisahkan channel warna bgr
    channels = cv2.split(gambar_bgr)
    hist_gabungan = []
    
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [BINS_WARNA], [0, 256])
        hist = hist.flatten().astype(float)
        
        if hist.sum() > 0:
            hist /= hist.sum()
            
        hist_gabungan.append(hist)
        
    return np.concatenate(hist_gabungan)

def proses_gambar(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None, None, None, None
        
    # konversi dan resize
    img = cv2.resize(img, UKURAN_GAMBAR, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # mendapatkan fitur histogram
    hist_lbp, map_lbp = ekstrak_lbp(gray)
    hist_warna = ekstrak_warna(img)
    
    # penggabungan fitur (feature fusion)
    fitur = np.concatenate([hist_lbp, hist_warna])
    
    # normalisasi l2
    norm = np.linalg.norm(fitur)
    if norm > 0:
        fitur = fitur / norm
        
    return fitur, img, gray, map_lbp


def buat_database():
    print("Memproses fitur gambar dari dataset...")
    
    kumpulan_path = []
    kumpulan_label = []
    kumpulan_fitur = []
    
    for motif in sorted(os.listdir(DATASET_DIR)):
        folder_motif = os.path.join(DATASET_DIR, motif)
        if not os.path.isdir(folder_motif):
            continue
            
        for file in os.listdir(folder_motif):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path_lengkap = os.path.join(folder_motif, file)
                fitur, _, _, _ = proses_gambar(path_lengkap)
                
                if fitur is not None:
                    kumpulan_path.append(path_lengkap)
                    kumpulan_label.append(motif)
                    kumpulan_fitur.append(fitur)
                    
    db = {
        "paths": kumpulan_path,
        "labels": kumpulan_label,
        "features": np.array(kumpulan_fitur)
    }
    
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
        
    print(f"Database selesai dibuat. Total data: {len(kumpulan_path)}")
    return db

def hitung_jarak_chisquare(a, b):
    # perhitungan chi-square distance
    atas = (a - b) ** 2
    bawah = a + b + 1e-10
    return 0.5 * np.sum(atas / bawah)

def cari_gambar_mirip(path_query, db):
    waktu_mulai = time.time()
    
    fitur_query, bgr_q, _, map_lbp_q = proses_gambar(path_query)
    if fitur_query is None:
        return None
        
    semua_fitur = db["features"]
    semua_path = db["paths"]
    semua_label = db["labels"]
    
    # menghitung jarak query terhadap semua gambar pada database
    jarak = [hitung_jarak_chisquare(fitur_query, f) for f in semua_fitur]
    urutan = np.argsort(jarak)
    
    hasil_pencarian = []
    for idx in urutan:
        # mengecualikan gambar query jika berasal dari dataset
        if os.path.abspath(semua_path[idx]) == os.path.abspath(path_query):
            continue
            
        nilai_jarak = jarak[idx]
        similarity = (1.0 / (1.0 + nilai_jarak)) * 100.0
        
        hasil_pencarian.append({
            "path": semua_path[idx],
            "label": semua_label[idx],
            "distance": nilai_jarak,
            "similarity": similarity
        })
        
    waktu_proses = time.time() - waktu_mulai
    
    # perhitungan metrik evaluasi (Precision, Recall, F1) pada Top-20
    top_1 = hasil_pencarian[0]
    nama_folder = os.path.basename(os.path.dirname(os.path.abspath(path_query)))
    label_asli = nama_folder if nama_folder in semua_label else top_1["label"]
    
    k = 20
    top_k_labels = [h["label"] for h in hasil_pencarian[:k]]
    true_positive = sum(1 for lbl in top_k_labels if lbl == label_asli)
    total_relevan = sum(1 for lbl in semua_label if lbl == label_asli)
    
    precision = true_positive / k if k > 0 else 0
    recall = true_positive / total_relevan if total_relevan > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "query_bgr": bgr_q,
        "query_lbp": map_lbp_q,
        "hasil": hasil_pencarian,
        "waktu": waktu_proses,
        "metrik": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "k": k
        }
    }


# antarmuka gui
ctk.set_appearance_mode("light")

class AplikasiCBIR(ctk.CTk):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.title("Batik Retrieval - LBP & Warna")
        self.geometry("900x600")
        self.resizable(False, False)
        
        self.path_gambar = None
        self.data_hasil = None
        self.cache_gambar = []
        
        self.buat_tampilan()
        
    def siapkan_gambar_gui(self, img_bgr, ukuran=280):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((ukuran, ukuran))
        return ctk.CTkImage(light_image=img_pil, size=img_pil.size)

    def buat_tampilan(self):
        judul = ctk.CTkLabel(self, text="Content-Based Batik Image Retrieval using Local Binary Pattern and Color Histogram", font=("Arial", 22, "bold"))
        judul.pack(pady=15)
        
        frame_tengah = ctk.CTkFrame(self, fg_color="transparent")
        frame_tengah.pack(fill="both", expand=True, padx=20)
        
        # panel kiri (query)
        panel_kiri = ctk.CTkFrame(frame_tengah, fg_color="#f0f0f0", corner_radius=10)
        panel_kiri.pack(side="left", fill="both", expand=True, padx=10)
        
        ctk.CTkLabel(panel_kiri, text="Gambar Query", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.lbl_query = ctk.CTkLabel(panel_kiri, text="Belum ada gambar", width=280, height=280, fg_color="#e0e0e0")
        self.lbl_query.pack(pady=10)
        
        ctk.CTkButton(panel_kiri, text="Pilih Gambar", command=self.klik_upload).pack(pady=10)
        
        # panel kanan (hasil retrieval)
        panel_kanan = ctk.CTkFrame(frame_tengah, fg_color="#f0f0f0", corner_radius=10)
        panel_kanan.pack(side="right", fill="both", expand=True, padx=10)
        
        ctk.CTkLabel(panel_kanan, text="Hasil Paling Mirip (Top 1)", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.lbl_hasil = ctk.CTkLabel(panel_kanan, text="Belum dicari", width=280, height=280, fg_color="#e0e0e0", cursor="hand2")
        self.lbl_hasil.pack(pady=10)
        self.lbl_hasil.bind("<Button-1>", self.klik_detail)
        
        self.btn_cari = ctk.CTkButton(panel_kanan, text="Cari Motif", command=self.klik_cari, state="disabled")
        self.btn_cari.pack(pady=10)
        
        self.lbl_info = ctk.CTkLabel(panel_kanan, text="", font=("Arial", 14, "bold"))
        self.lbl_info.pack()
        
        self.btn_detail = ctk.CTkButton(panel_kanan, text="Lihat Analisis Fitur", command=lambda: self.klik_detail(None), fg_color="#6c757d", hover_color="#5a6268")
        self.btn_detail.pack(pady=10)
        self.btn_detail.pack_forget() # sembunyikan dulu sampai ada hasil

    def klik_upload(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
            
        self.path_gambar = path
        img = cv2.imread(path)
        img_ctk = self.siapkan_gambar_gui(img)
        self.cache_gambar.append(img_ctk)
        
        self.lbl_query.configure(image=img_ctk, text="")
        self.lbl_hasil.configure(image=None, text="Siap mencari...")
        self.lbl_info.configure(text="")
        self.btn_cari.configure(state="normal")
        self.btn_detail.pack_forget()
        self.data_hasil = None

    def klik_cari(self):
        self.btn_cari.configure(state="disabled")
        self.lbl_info.configure(text="Proses pencarian...", text_color="orange")
        self.update()
        
        self.data_hasil = cari_gambar_mirip(self.path_gambar, self.db)
        
        if not self.data_hasil or not self.data_hasil["hasil"]:
            self.lbl_info.configure(text="Gagal menemukan kecocokan", text_color="red")
            return
            
        top_1 = self.data_hasil["hasil"][0]
        
        img_hasil = cv2.imread(top_1["path"])
        img_ctk = self.siapkan_gambar_gui(img_hasil)
        self.cache_gambar.append(img_ctk)
        
        self.lbl_hasil.configure(image=img_ctk, text="")
        
        info_text = f"Motif: {top_1['label'].replace('_', ' ')}\nSimilarity: {top_1['similarity']:.2f}%\nWaktu: {self.data_hasil['waktu']:.2f} dtk"
        self.lbl_info.configure(text=info_text, text_color="green")
        self.btn_cari.configure(state="normal")
        self.btn_detail.pack(pady=10)

    def klik_detail(self, event):
        if not self.data_hasil:
            return
            
        popup = ctk.CTkToplevel(self)
        popup.title("Analisis Ekstraksi Fitur")
        popup.geometry("900x700")
        
        top_1 = self.data_hasil["hasil"][0]
        
        # fungsi untuk membuat grafik histogram menggunakan matplotlib
        def plot_histogram(hist_data, color_line, hist_type='bar', color_fill=None):
            fig, ax = plt.subplots(figsize=(2.5, 2), dpi=100)
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            if hist_type == 'bar':
                ax.bar(range(len(hist_data)), hist_data, color=color_fill, edgecolor=color_line, width=0.8)
            else:
                ax.plot(hist_data, color=color_line, linewidth=2)
                if color_fill:
                    ax.fill_between(range(len(hist_data)), hist_data, color=color_fill, alpha=0.3)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.tight_layout(pad=0)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf.seek(0)
            img_pil = Image.open(buf)
            plt.close(fig)
            return img_pil

        # menggunakan gambar query untuk memvisualisasikan proses ekstraksi
        img_bgr = self.data_hasil["query_bgr"]
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lbp_hist, lbp_map = ekstrak_lbp(gray_img)
        
        b, g, r = cv2.split(img_bgr)
        hist_b = cv2.calcHist([b], [0], None, [64], [0, 256]).flatten()
        hist_g = cv2.calcHist([g], [0], None, [64], [0, 256]).flatten()
        hist_r = cv2.calcHist([r], [0], None, [64], [0, 256]).flatten()

        scroll_frame = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # fungsi untuk menampilkan gambar ke dalam popup
        def tambah_gambar(parent, judul, img_source, is_gray=False, is_pil=False):
            kolom = ctk.CTkFrame(parent, fg_color="transparent")
            kolom.pack(side="left", expand=True, padx=10)
            ctk.CTkLabel(kolom, text=judul, font=("Arial", 12, "bold")).pack()
            
            if is_pil:
                pil_img = img_source
            else:
                if is_gray:
                    disp = (img_source / img_source.max() * 255).astype(np.uint8) if img_source.max() > 0 else img_source.astype(np.uint8)
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
                else:
                    disp = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(disp)
                pil_img.thumbnail((160, 160))
                
            img_ctk = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
            self.cache_gambar.append(img_ctk)
            ctk.CTkLabel(kolom, image=img_ctk, text="").pack(pady=10)

        # bagian ekstraksi fitur (tekstur)
        ctk.CTkLabel(scroll_frame, text="Hasil Ekstraksi Fitur", font=("Arial", 20, "bold"), anchor="w").pack(fill="x", pady=(10, 5))
        
        frame_41 = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        frame_41.pack(fill="x", pady=5)
        
        tambah_gambar(frame_41, "Original Image", img_bgr)
        tambah_gambar(frame_41, "Grayscale", gray_img, is_gray=True)
        tambah_gambar(frame_41, "LBP Image", lbp_map, is_gray=True)
        
        pil_lbp = plot_histogram(lbp_hist, color_line='gray', hist_type='bar', color_fill='silver')
        tambah_gambar(frame_41, "Histogram LBP", pil_lbp, is_pil=True)
        
        teks_41 = ("Gambar batik diubah menjadi grayscale sebelum dilakukan ekstraksi tekstur menggunakan Local Binary Pattern. "
                   "Hasil LBP kemudian dikonversi menjadi histogram sebagai representasi fitur tekstur.")
        ctk.CTkLabel(scroll_frame, text=teks_41, justify="left", font=("Arial", 13), wraplength=800).pack(fill="x", padx=10, pady=(0, 20))
        
        # garis pemisah
        frame_line = ctk.CTkFrame(scroll_frame, height=2, fg_color="#d0d0d0")
        frame_line.pack(fill="x", pady=20)

        # bagian ekstraksi warna
        ctk.CTkLabel(scroll_frame, text="Hasil Ekstraksi Warna", font=("Arial", 20, "bold"), anchor="w").pack(fill="x", pady=(10, 5))
        
        frame_42 = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        frame_42.pack(fill="x", pady=5)
        
        tambah_gambar(frame_42, "Original", img_bgr)
        
        pil_r = plot_histogram(hist_r, color_line='red', hist_type='plot', color_fill='red')
        tambah_gambar(frame_42, "Histogram R", pil_r, is_pil=True)
        
        pil_g = plot_histogram(hist_g, color_line='green', hist_type='plot', color_fill='green')
        tambah_gambar(frame_42, "Histogram G", pil_g, is_pil=True)
        
        pil_b = plot_histogram(hist_b, color_line='blue', hist_type='plot', color_fill='blue')
        tambah_gambar(frame_42, "Histogram B", pil_b, is_pil=True)
        
        teks_42 = "Histogram RGB digunakan untuk merepresentasikan distribusi warna citra."
        ctk.CTkLabel(scroll_frame, text=teks_42, justify="left", font=("Arial", 13), wraplength=800).pack(fill="x", padx=10, pady=(0, 20))
        
        # garis pemisah
        frame_line_2 = ctk.CTkFrame(scroll_frame, height=2, fg_color="#d0d0d0")
        frame_line_2.pack(fill="x", pady=20)
        
        # bagian 4.3 evaluasi sistem
        ctk.CTkLabel(scroll_frame, text="4.3 Evaluasi Metrik Pencarian", font=("Arial", 20, "bold"), anchor="w").pack(fill="x", pady=(10, 5))
        
        metrik = self.data_hasil["metrik"]
        frame_43 = ctk.CTkFrame(scroll_frame, fg_color="#f8f9fa", corner_radius=8)
        frame_43.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(frame_43, text=f"Berdasarkan Top-{metrik['k']} Gambar Teratas", font=("Arial", 14, "italic")).pack(pady=(10, 5))
        
        frame_metrik = ctk.CTkFrame(frame_43, fg_color="transparent")
        frame_metrik.pack(pady=10)
        
        ctk.CTkLabel(frame_metrik, text=f"Precision: {metrik['precision'] * 100:.2f}%", font=("Arial", 16, "bold"), text_color="#0056b3").pack(side="left", padx=20)
        ctk.CTkLabel(frame_metrik, text=f"Recall: {metrik['recall'] * 100:.2f}%", font=("Arial", 16, "bold"), text_color="#198754").pack(side="left", padx=20)
        ctk.CTkLabel(frame_metrik, text=f"F1-Score: {metrik['f1'] * 100:.2f}%", font=("Arial", 16, "bold"), text_color="#dc3545").pack(side="left", padx=20)
        
        teks_43 = ("Metrik dievaluasi dengan membandingkan label kelas pada top K hasil pencarian terhadap kelas aktual gambar kueri. "
                   "Nilai F1-Score digunakan sebagai indikator keandalan utama sistem (harmonic mean).")
        ctk.CTkLabel(scroll_frame, text=teks_43, justify="left", font=("Arial", 13), wraplength=800).pack(fill="x", padx=10, pady=(10, 20))


if __name__ == "__main__":
    # memuat database, dan mengekstrak ulang jika file tidak ditemukan
    if not os.path.exists(DB_FILE):
        db = buat_database()
    else:
        with open(DB_FILE, "rb") as f:
            db = pickle.load(f)
            
    app = AplikasiCBIR(db)
    app.mainloop()
