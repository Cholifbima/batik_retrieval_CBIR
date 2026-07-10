# Content-Based Batik Image Retrieval

Implementasi Content-Based Image Retrieval (CBIR) untuk pencarian gambar batik berdasarkan kemiripan visual.

Mengacu pada paper: *Enhanced Content-Based Image Retrieval through Integrated Local Average Binary Patterns and Joint Color Probabilities* (2024), dengan penyederhanaan metode.

## Metode

| Komponen | Paper | Implementasi |
|---|---|---|
| Fitur Tekstur | LABP (Local Average Binary Pattern) | LBP (Local Binary Pattern) |
| Fitur Warna | Joint Probability Color Distribution | RGB Histogram |
| Jarak | Extended Canberra Distance | Chi-Square Distance |
| Fusion | Feature Concatenation + Normalisasi | Feature Concatenation + L2 Norm |

## Dataset

Batik Nusantara Dataset (Kaggle) — 20 motif batik Indonesia, 800 gambar.

## Cara Pakai

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Siapkan dataset (gabungkan train + test):
```
python setup_dataset.py
```

3. Bangun feature database:
```
python build_database.py
```

4. Jalankan aplikasi:
```
python main.py
```

## Struktur File

| File | Fungsi |
|---|---|
| `utils.py` | Preprocessing gambar (resize, konversi warna) |
| `extract_feature.py` | Ekstraksi fitur LBP + Color Histogram |
| `build_database.py` | Membangun feature database dari dataset |
| `retrieval.py` | Perhitungan jarak dan pencarian gambar mirip |
| `gui.py` | Antarmuka GUI (CustomTkinter) |
| `main.py` | Entry point aplikasi |
| `setup_dataset.py` | Script persiapan dataset |
