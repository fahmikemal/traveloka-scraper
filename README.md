# Analisis Sentimen Komentar Traveloka di X
### Menggunakan Metode Multinomial Naive Bayes

Proyek ini menganalisis sentimen komentar pengguna tentang **semua layanan Traveloka** (pesawat, hotel, kereta, bus, rental, aktivitas, paylater, dll) yang diambil dari media sosial **X (Twitter)**.

---

## Sebelum Mulai — Syarat yang Harus Disiapkan

Sebelum menjalankan aplikasi ini, pastikan sudah punya:

1. **Python** versi 3.10 ke atas — download di https://www.python.org/downloads/
   - Saat install, centang **"Add Python to PATH"**
2. **Akun X (Twitter)** — untuk login dan mengambil data
3. **Koneksi internet**

---

## Langkah Pertama Kali (Dari Nol)

Ikuti urutan ini **satu kali saja** saat pertama kali setup:

```
Install Python  →  Download proyek  →  Buat virtual env
→  Install library  →  Install browser  →  Siap pakai
```

---

## Struktur Folder

```
scraper-x/
├── main.py                  ← File utama (semua proses ada di sini)
├── requirements.txt         ← Daftar library yang dibutuhkan
│
├── data/
│   ├── raw/                 ← Hasil scraping mentah (.csv)
│   ├── processed/           ← Hasil preprocessing (.csv)
│   └── labeled/             ← Data berlabel untuk training (.csv)
│
├── models/                  ← Model terlatih (.pkl)
├── results/                 ← Confusion matrix & metrik (.png, .csv)
└── session/
    └── cookies.json         ← Session login X (dibuat otomatis)
```

---

## Cara Install (Lakukan Sekali Saja)

### 1. Cek Python sudah terinstall
Buka **Command Prompt** atau **PowerShell**, ketik:
```bash
python --version
```
Harus muncul versi Python, contoh: `Python 3.11.0`

> Belum punya Python? Download di https://www.python.org/downloads/ — pilih versi terbaru, saat install **centang "Add Python to PATH"**

---

### 2. Buka folder proyek di terminal
Klik kanan folder `scraper-x` → **Open in Terminal** (atau PowerShell)

Atau ketik manual:
```bash
cd "D:\scraper-x"
```
> Sesuaikan path dengan lokasi folder proyek di komputer Anda

---

### 3. Buat virtual environment
```bash
python -m venv myenv
```
Ini membuat folder `myenv` yang berisi Python khusus untuk proyek ini.

---

### 4. Aktifkan virtual environment
```bash
myenv\Scripts\activate
```
Kalau berhasil, di depan terminal akan muncul tulisan `(myenv)`:
```
(myenv) PS D:\scraper-x>
```

> **Penting:** Setiap kali buka terminal baru, selalu aktifkan dulu dengan perintah di atas sebelum menjalankan apapun.

---

### 5. Install semua library
```bash
pip install -r requirements.txt
```
Tunggu sampai selesai (bisa beberapa menit tergantung koneksi internet).

---

### 6. Install browser Chromium untuk Playwright
```bash
playwright install chromium
```
Ini mengunduh browser khusus yang dipakai untuk mengambil data dari X.

---

## Cara Pakai

Semua proses dijalankan lewat satu file: `main.py`

### Opsi yang tersedia

| Opsi | Fungsi |
|---|---|
| `--login` | Login ke X dan simpan session |
| `--scrape` | Scrape komentar Traveloka dari X |
| `--preprocess` | Preprocessing teks (cleaning + stemming) |
| `--autolabel` | Labeling otomatis berbasis keyword |
| `--train` | Latih model Multinomial Naive Bayes |
| `--all` | Jalankan semua tahap sekaligus |
| `--predict` | Prediksi sentimen file baru |

### Parameter tambahan

| Parameter | Fungsi | Default |
|---|---|---|
| `--max 150` | Jumlah tweet per query | 100 |
| `--test-size 0.2` | Rasio data uji | 0.2 (20%) |
| `--input file.csv` | Tentukan file input manual | - |
| `--labeled file.csv` | File berlabel untuk training | - |
| `--model file.pkl` | Model untuk prediksi | - |

---

## Alur Lengkap Step by Step

> Setiap kali membuka terminal baru, **aktifkan virtual environment dulu**:
> ```bash
> myenv\Scripts\activate
> ```

---

### STEP 1 — Simpan Session Login X (lakukan sekali saja)

X tidak mengizinkan login otomatis, jadi session diambil secara manual dari browser biasa menggunakan ekstensi. Ikuti langkah berikut:

**1. Install ekstensi Cookie-Editor di Chrome/Edge**
- Buka Chrome Web Store
- Cari **"Cookie-Editor"** (ikon biru-putih)
- Klik **Add to Chrome** → **Add Extension**

**2. Login ke X di browser**
- Buka https://x.com
- Login dengan akun X Anda seperti biasa

**3. Export cookies**
- Setelah berhasil masuk ke beranda X, klik ikon ekstensi **Cookie-Editor** di pojok kanan atas browser
- Klik tombol **Export** (ikon panah ke bawah) di pojok kanan bawah
- Cookies otomatis tersalin ke clipboard

**4. Simpan ke file**
- Buka folder proyek `scraper-x/session/`
- Buat file baru bernama `cookies.json`
- Paste hasil export tadi ke dalam file tersebut
- Simpan file

> **Catatan:** Jika session expired (biasanya setelah beberapa hari atau logout), ulangi langkah di atas dari nomor 2.

---

### STEP 2 — Scraping Komentar

```bash
python main.py --scrape --max 100
```

- Mengambil komentar dari 17 keyword layanan Traveloka
- Hasil disimpan di `data/raw/traveloka_raw_TANGGAL.csv`
- Kolom hasil: `id, username, date, text, likes, retweets, replies, query`

Contoh dengan lebih banyak data:
```bash
python main.py --scrape --max 300
```

---

### STEP 3 — Preprocessing Teks

```bash
python main.py --preprocess --input data/raw/traveloka_raw_TANGGAL.csv
```

Proses yang dilakukan:
1. Hapus URL, mention (@user), hashtag (#)
2. Hapus emoji dan karakter non-ASCII
3. Hapus angka dan tanda baca
4. Lowercase semua teks
5. Tokenisasi
6. Hapus stopword (Bahasa Indonesia + Inggris)
7. Stemming menggunakan PySastrawi

Hasil disimpan di `data/processed/`

---

### STEP 4 — Auto Labeling

```bash
python main.py --autolabel --input data/processed/traveloka_raw_TANGGAL_processed.csv
```

- Memberi label otomatis berdasarkan keyword positif/negatif
- Label `0` (negatif), `1` (netral), `2` (positif)
- Hasil disimpan di `data/labeled/`

> **Disarankan:** Buka file hasil di Excel, cek beberapa baris dan koreksi label yang kurang tepat sebelum lanjut ke training.

---

### STEP 5 — Training Model

```bash
python main.py --train --labeled data/labeled/traveloka_raw_TANGGAL_labeled.csv
```

Proses yang dilakukan:
1. **Oversampling** — menyeimbangkan jumlah data tiap kelas
2. **TF-IDF Vectorizer** — mengubah teks menjadi angka (fitur)
3. **GridSearch** — mencari nilai alpha terbaik otomatis
4. **ComplementNB** — melatih model Naive Bayes
5. **Evaluasi** — accuracy, precision, recall, F1-score, cross-validation

Hasil tersimpan di:
- `models/mnb_model_TANGGAL.pkl` — model terlatih
- `results/confusion_matrix_TANGGAL.png` — grafik confusion matrix
- `results/metrics_TANGGAL.csv` — ringkasan metrik

---

### Semua Sekaligus (STEP 2–5)

Setelah login, bisa jalankan semua tahap dengan satu perintah:

```bash
python main.py --all --max 150
```

---

### Prediksi Data Baru

Jika ingin memprediksi file baru menggunakan model yang sudah ada:

```bash
python main.py --predict --input data/processed/file_baru.csv
```

Atau dengan model tertentu:
```bash
python main.py --predict --model models/mnb_model_xxx.pkl --input data/processed/file_baru.csv
```

---

## Ringkasan Urutan Perintah (Copy-Paste Siap Pakai)

Jalankan satu per satu dari atas ke bawah:

```bash
# Aktifkan virtual environment (wajib setiap buka terminal baru)
myenv\Scripts\activate

# STEP 2 — Ambil data komentar Traveloka dari X
# (Pastikan cookies.json sudah ada di folder session/ — lihat STEP 1)
python main.py --scrape --max 150

# STEP 3 — Preprocessing teks
# Ganti nama file sesuai hasil scraping di folder data/raw/
python main.py --preprocess --input data/raw/traveloka_raw_TANGGAL.csv

# STEP 4 — Auto labeling
# Ganti nama file sesuai hasil preprocessing di folder data/processed/
python main.py --autolabel --input data/processed/traveloka_raw_TANGGAL_processed.csv

# STEP 5 — Training model
# Ganti nama file sesuai hasil labeling di folder data/labeled/
python main.py --train --labeled data/labeled/traveloka_raw_TANGGAL_labeled.csv
```

> **Tips:** Nama file hasil scraping, preprocessing, dan labeling menggunakan tanggal & jam otomatis.
> Cek folder `data/raw/`, `data/processed/`, dan `data/labeled/` untuk melihat nama file yang benar.

---

## Library yang Digunakan

| Library | Fungsi |
|---|---|
| `playwright` | Scraping browser otomatis |
| `pandas` | Manipulasi data |
| `numpy` | Operasi numerik |
| `scikit-learn` | Model machine learning (MNB, TF-IDF) |
| `nltk` | Tokenisasi teks |
| `PySastrawi` | Stemming Bahasa Indonesia |
| `matplotlib` & `seaborn` | Visualisasi confusion matrix |
| `wordcloud` | Visualisasi kata (opsional) |
| `tqdm` | Progress bar |

---

## Troubleshooting

**Session expired saat scraping:**
```bash
python main.py --login
```

**Error saat install scikit-learn (Python 3.14):**
```bash
pip install scikit-learn --no-build-isolation
```

**Playwright browser tidak terbuka:**
```bash
playwright install chromium
```
