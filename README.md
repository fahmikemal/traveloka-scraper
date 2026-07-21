# Analisis Sentimen Komentar Traveloka di X
### Menggunakan Metode Multinomial Naive Bayes

Proyek ini menganalisis sentimen komentar pengguna tentang **semua layanan Traveloka**
(pesawat, hotel, kereta, bus, rental, aktivitas, paylater, dll) yang diambil dari
media sosial **X (Twitter)**.

Metode utama: **Multinomial Naive Bayes (MNB)**, dengan **ComplementNB** sebagai
pembanding.

---

## Daftar Isi

1. [Yang Perlu Disiapkan](#yang-perlu-disiapkan)
2. [Instalasi](#instalasi-lakukan-sekali-saja)
3. [Setup Session X](#setup-session-x-wajib)
4. [Alur Lengkap](#alur-lengkap-step-by-step)
5. [Daftar Opsi](#daftar-opsi)
6. [Struktur Folder](#struktur-folder)
7. [Catatan Metodologi](#catatan-metodologi-penting-untuk-skripsi)
8. [Rincian Teknis & Rumus](#rincian-teknis--rumus-sesuai-kode)
9. [Panduan Penyelarasan Naskah](#panduan-penyelarasan-naskah-dengan-kode)
10. [Troubleshooting](#troubleshooting)

---

## Yang Perlu Disiapkan

1. **Python 3.11 ke atas** — https://www.python.org/downloads/
   (saat install di Windows, centang **"Add Python to PATH"**)
2. **Akun X (Twitter)** — untuk mengambil data
3. **Browser Chrome/Edge** — untuk mengekspor cookie
4. **Koneksi internet**

---

## Instalasi (Lakukan Sekali Saja)

### 1. Buat virtual environment

**Linux / macOS**
```bash
cd /path/ke/scraper-x
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```powershell
cd D:\scraper-x
python -m venv myenv
myenv\Scripts\activate
```

Kalau berhasil, akan muncul `(.venv)` atau `(myenv)` di depan prompt terminal.

> **Penting:** setiap kali membuka terminal baru, aktifkan dulu virtual
> environment sebelum menjalankan apapun.

### 2. Install library

```bash
pip install -r requirements.txt
```

Semua versi sudah dikunci di `requirements.txt` agar hasil penelitian dapat
direproduksi.

### 3. Install browser untuk Playwright

```bash
playwright install chromium
```

### 4. Pastikan lexicon InSet tersedia

Folder `lexicon/` harus berisi `positive.tsv` dan `negative.tsv`. Kalau belum ada:

```bash
mkdir -p lexicon && cd lexicon
curl -LO https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv
curl -LO https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv
cd ..
```

---

## Setup Session X (Wajib)

X tidak mengizinkan login otomatis, jadi session diambil dari cookie browser.

1. Pasang ekstensi **Cookie-Editor** di Chrome/Edge (cari di Chrome Web Store)
2. Buka https://x.com dan **login seperti biasa**
3. Setelah masuk beranda, klik ikon **Cookie-Editor** di pojok kanan atas
4. Klik tombol **Export** (ikon panah ke bawah) — cookie tersalin ke clipboard
5. Buat file `session/cookies.json`, paste isinya, lalu simpan

**Amankan file tersebut** (berisi `auth_token` yang setara akses penuh ke akun Anda):

```bash
chmod 600 session/cookies.json     # Linux/macOS
```

> Jika session expired (biasanya setelah beberapa hari atau setelah logout),
> ulangi langkah 2–5.

---

## Alur Lengkap Step by Step

### STEP 1 — Scraping

```bash
python main.py --scrape --max 130
```

- Mengambil tweet dari **33 kata kunci** layanan Traveloka
- Hasil: `data/raw/traveloka_raw_TANGGAL.csv`
- Kolom: `id, username, date, text, likes, retweets, replies, query`

> **Perlu waktu lama.** Ada jeda 2,5 detik tiap scroll. 33 query x 130 tweet
> bisa memakan 1–2 jam. Browser akan terbuka — jangan ditutup.

**Berapa banyak yang dibutuhkan?** Sekitar **45-50% hasil scraping bukan
Bahasa Indonesia** dan akan dibuang di STEP 2. Untuk mendapat ~1.000 data
Indonesia, scrape minimal 2.000 tweet mentah (`--max 130` ke atas).

> **Catatan penting soal query.** Operator `lang:id` **tidak dipakai**. Diuji
> 21 Juli 2026, operator itu membuat X mengembalikan indeks lama — tweet
> 2013–2014, 19 dari 20 berasal dari akun resmi `@traveloka`. Tanpa operator
> tersebut, hasil pencarian adalah tweet hari itu juga. Penyaringan bahasa
> dilakukan di STEP 2 memakai `langdetect`.
>
> Query juga memakai `-from:traveloka` untuk membuang balasan customer service
> akun resmi, yang isinya teks PR dan akan mencemari analisis sentimen
> pelanggan. Balasan dari pengguna biasa tetap diambil.

---

### STEP 2 — Preprocessing

```bash
python main.py --preprocess
```

Otomatis mengambil file raw terbaru. Untuk menentukan file secara manual:

```bash
python main.py --preprocess --input data/raw/traveloka_raw_TANGGAL.csv
```

Proses yang dijalankan:

1. Hapus URL, mention (`@user`), hashtag (`#`)
2. Hapus karakter non-ASCII, angka, dan tanda baca
3. Lowercase
4. **Filter bahasa** — buang dokumen non-Indonesia (langdetect, confidence >= 0.70)
5. Tokenisasi
6. Hapus stopword — **kata negasi sengaja DIPERTAHANKAN**
7. Stemming dengan PySastrawi

Hasil: `data/processed/`

> **Perhatikan output-nya.** Akan tercetak distribusi bahasa dan berapa baris
> yang dibuang. Kalau data Indonesia tersisa < 300, akan muncul peringatan —
> scrape lagi sebelum melanjutkan.

Untuk melewati filter bahasa (**tidak disarankan**): tambahkan `--no-lang-filter`

---

### STEP 3 — Auto Labeling (InSet)

```bash
python main.py --autolabel
```

Menggunakan lexicon **InSet** (positive.tsv 3.607 + negative.tsv 6.606, bobot
-5 sampai +5) ditambah `lexicon/custom.tsv` untuk istilah domain:

- Pencocokan berbasis **token**, bukan substring
- Mendukung **frasa multi-kata** (677 entri), pencocokan terpanjang didahulukan
- **Penanganan negasi**: kata sentimen yang didahului kata negasi dalam jarak
  3 token akan dibalik polaritasnya
- Label: `0` = negatif, `1` = netral, `2` = positif
- Ambang netral: `|skor| <= 2` (ubah dengan `--threshold`)

> **Penting — InSet mengandung kontradiksi internal.** 1.142 kata muncul di
> `positive.tsv` **dan** `negative.tsv` dengan bobot berlawanan (mis. `bagus`
> = +2 dan −4). Proyek ini mendahulukan `positive.tsv`, dipilih setelah menguji
> tiga strategi terhadap 20 kata umum: positive didahulukan **16/20**,
> |bobot| maksimum 15/20, dijumlahkan 14/20.
>
> Selain itu beberapa istilah penting **tidak ada di InSet sama sekali**
> (`kecewa`, `tipu`, `penipuan`, `refund`, `komplain`), dan beberapa salah
> polaritas (`lambat` dan `mahal` bernilai +1). Keduanya dikoreksi lewat
> `lexicon/custom.tsv`, yang menaikkan akurasi polaritas pada 20 kata uji
> tersebut dari 14/20 menjadi **20/20**.
>
> `custom.tsv` **wajib dilaporkan di skripsi** sebagai *"InSet augmented with
> domain terms"*, lengkap dengan daftar kata dan alasan penambahannya.

Hasil: `data/labeled/`, dengan kolom tambahan `sentiment_score` dan `lexicon_hits`.

> Output juga melaporkan berapa label "netral" yang sebenarnya **tidak terdeteksi
> lexicon** — angka ini penting untuk dibahas di skripsi.

**Perlindungan data:** perintah ini **menolak menimpa** file berlabel yang sudah
ada, agar koreksi manual Anda tidak hilang. Gunakan `--force` bila memang ingin
menimpa (backup dulu).

---

### STEP 4 — Validasi Label (JANGAN DILEWATI)

Pelabelan otomatis **bukan ground truth**. Validitasnya wajib diukur.

**4a. Ambil sampel untuk dilabeli manual**

```bash
python main.py --sample-gold 200      # angka opsional, default 200
```

Hasil: `data/gold/gold_sample_TANGGAL.csv` (stratified per kelas)

**4b. Isi label manual**

Buka file tersebut di Excel/LibreOffice, isi kolom **`label_manual`**:
`0` = negatif, `1` = netral, `2` = positif

> Baca kolom `text` (teks asli). **Jangan lihat kolom `label_auto`** dulu agar
> penilaian Anda tidak terpengaruh. Simpan kembali sebagai CSV.

**4c. Ukur kesepakatan**

```bash
python main.py --kappa
```

Menghasilkan **Cohen's Kappa** beserta interpretasi Landis & Koch (1977):

| Kappa | Interpretasi |
|---|---|
| < 0.00 | lebih buruk dari tebakan acak |
| 0.00 – 0.20 | sangat rendah (slight) |
| 0.20 – 0.40 | rendah (fair) |
| 0.40 – 0.60 | sedang (moderate) |
| 0.60 – 0.80 | baik (substantial) |
| 0.80 – 1.00 | sangat baik (almost perfect) |

> **Jika Kappa < 0.40**, label otomatis tidak cukup valid. Gunakan label manual
> untuk training, atau perbaiki ambang/lexicon terlebih dahulu.

---

### STEP 5 — Training

```bash
python main.py --train
```

Proses yang dijalankan:

1. **Split dulu** (stratified), data uji dibiarkan timpang sesuai kondisi nyata
2. **Oversampling hanya pada data latih** — mencegah data leakage
3. **Baseline kelas mayoritas** dihitung sebagai acuan minimum
4. **TF-IDF** (unigram + bigram, `sublinear_tf`, `max_df=0.95`)
5. **GridSearch** alpha dengan `scoring="f1_macro"`
6. Melatih **MultinomialNB** (utama) dan **ComplementNB** (pembanding)
7. **Cross-validation 10-fold** dengan oversampling **di dalam tiap fold**
8. Evaluasi: accuracy, macro-F1, weighted-F1, macro-precision, macro-recall

Hasil:
- `models/mnb_model_TANGGAL.pkl` dan `models/cnb_model_TANGGAL.pkl`
- `results/confusion_matrix_multinomialnb_TANGGAL.png` (dan `complementnb`)
- `results/metrics_TANGGAL.csv` — satu baris per model, lengkap dengan baseline

---

### Prediksi Data Baru

```bash
python main.py --predict --input data/processed/file_baru.csv
```

Dengan model tertentu:

```bash
python main.py --predict --model models/mnb_model_xxx.pkl --input data/processed/file_baru.csv
```

---

## Ringkasan Perintah (Copy-Paste)

```bash
# Aktifkan virtual environment dulu
source .venv/bin/activate          # Windows: myenv\Scripts\activate

# Pastikan session/cookies.json sudah diisi (lihat Setup Session X)

python main.py --scrape --max 130
python main.py --preprocess
python main.py --autolabel
python main.py --sample-gold 200
#   -> isi kolom 'label_manual' di Excel, simpan sebagai CSV
python main.py --kappa
python main.py --train
```

> **Jangan gunakan `--all`** untuk penelitian. Opsi itu melompati validasi
> Cohen's Kappa, padahal validasi tersebut yang membuat hasil penelitian dapat
> dipertanggungjawabkan.

---

## Daftar Opsi

### Tahapan

| Opsi | Fungsi |
|---|---|
| `--scrape` | Scrape komentar Traveloka dari X |
| `--preprocess` | Preprocessing teks + filter bahasa |
| `--autolabel` | Pelabelan otomatis dengan lexicon InSet |
| `--sample-gold` | Ambil sampel untuk dilabeli manual |
| `--kappa` | Ukur Cohen's Kappa (otomatis vs manual) |
| `--train` | Latih MultinomialNB + ComplementNB |
| `--all` | Scrape + Preprocess + AutoLabel + Train |
| `--predict` | Prediksi sentimen file baru |

### Parameter

| Parameter | Fungsi | Default |
|---|---|---|
| `--max N` | Tweet per query | 100 |
| `--test-size R` | Rasio data uji | 0.2 |
| `--threshold N` | Ambang netral skor InSet | 2 |
| `--input FILE` | File input CSV manual | otomatis |
| `--labeled FILE` | File berlabel untuk `--train` | otomatis |
| `--gold FILE` | File gold untuk `--kappa` | otomatis |
| `--model FILE` | Model `.pkl` untuk `--predict` | terbaru |
| `--force` | Izinkan menimpa file berlabel | mati |
| `--headless` | Browser tanpa tampilan (untuk server) | mati |
| `--no-lang-filter` | Jangan buang teks non-Indonesia | mati |

---

## Struktur Folder

```
scraper-x/
├── main.py                  ← Semua proses ada di sini
├── requirements.txt         ← Library (versi terkunci)
├── README.md
│
├── lexicon/                 ← Lexicon sentimen
│   ├── positive.tsv         (3.607 kata, dari InSet)
│   ├── negative.tsv         (6.606 kata, dari InSet)
│   └── custom.tsv           (istilah domain + koreksi polaritas)
│
├── data/
│   ├── raw/                 ← Hasil scraping mentah
│   ├── processed/           ← Hasil preprocessing
│   ├── labeled/             ← Data berlabel otomatis
│   └── gold/                ← Sampel untuk label manual
│
├── models/                  ← Model terlatih (.pkl)
├── results/                 ← Confusion matrix & metrik
├── referensi/               ← Paper pendukung
└── session/
    └── cookies.json         ← Session X (dibuat manual)
```

---

## Catatan Metodologi (Penting untuk Skripsi)

### Mengapa split dilakukan sebelum oversampling

Oversampling sebelum pembagian data menyebabkan **data leakage**: baris hasil
duplikasi kelas minoritas bocor ke data uji, sehingga akurasi yang dilaporkan
adalah hafalan, bukan generalisasi. Pada dataset awal proyek ini, 49,2% baris
data uji ternyata identik dengan baris di data latih.

Karena itu urutannya: **split → oversampling hanya pada data latih**. Data uji
dibiarkan timpang agar mencerminkan distribusi nyata.

### Mengapa macro-F1, bukan accuracy

Pada data timpang, accuracy menyesatkan. Model yang selalu menebak kelas
mayoritas bisa mendapat accuracy tinggi tanpa mempelajari apapun. Karena itu
setiap run mencetak **baseline kelas mayoritas** — model hanya berguna bila
melampauinya, dan **macro-F1** dipakai sebagai metrik utama.

### Mengapa kata negasi tidak dihapus

Menghapus "tidak", "bukan", "gak" membuat `"tidak bagus"` menyusut menjadi
`"bagus"`, sehingga model tidak mungkin mempelajari pembalikan polaritas.
Daftar stopword bawaan Sastrawi memuat kata-kata ini, jadi dikurangkan secara
eksplisit di `NEGATION_WORDS`.

### Mengapa dokumen non-Indonesia dibuang

Filter `lang:id` milik X tidak dapat diandalkan — banyak tweet promosi berbahasa
Inggris tetap lolos. Menganalisis teks Inggris memakai Sastrawi (stemmer
Indonesia) dan InSet (lexicon Indonesia) menghasilkan skor yang tidak bermakna.
`langdetect` dipakai dengan `DetectorFactory.seed = 0` agar hasilnya dapat
direproduksi.

---

## Rincian Teknis — Rumus Sesuai Kode

> Semua rumus di bawah **sudah diverifikasi secara numerik** terhadap
> scikit-learn 1.9.0 yang benar-benar dipakai, bukan disalin dari buku teks.

### 1. Dataset

| | Keterangan |
|---|---|
| Sumber | Media sosial X (Twitter), tab *Latest* |
| Alat | Playwright (Chromium), session cookie |
| Kata kunci | 33 query netral (nama layanan + frasa pemakaian) |
| Kolom | `id, username, date, text, likes, retweets, replies, query` |
| Filter query | `-from:traveloka` (buang balasan CS resmi) |
| Filter bahasa | `langdetect`, hanya `id` dengan confidence ≥ 0.70 |
| Target | ≥ 500 dokumen Bahasa Indonesia |

### 2. Lexicon InSet

InSet (Koto & Rahmaningtyas, 2017) — kamus sentimen Bahasa Indonesia:

| File | Jumlah | Bobot |
|---|---|---|
| `positive.tsv` | 3.607 kata | +1 … +5 |
| `negative.tsv` | 6.606 kata | −5 … −1 |
| `custom.tsv` | 16 entri | koreksi domain |
| **Efektif** | **9.082 entri** | 677 frasa multi-kata |

**Skor sentimen dokumen:**

```
skor(d) = Σ  w(t) · neg(t)
         t∈d
```

- `w(t)` = bobot kata/frasa `t` menurut lexicon
- `neg(t)` = −1 bila ada kata negasi dalam **3 token** sebelum `t`, selain itu +1
- Pencocokan berbasis **token**, frasa terpanjang didahulukan (maks 4 kata)

**Aturan label:**

```
label = 2 (positif)  bila  skor >  2
        0 (negatif)  bila  skor < -2
        1 (netral)   bila  -2 ≤ skor ≤ 2
```

### 3. TF-IDF — ⚠️ Bukan Rumus Buku Teks

scikit-learn **tidak** memakai `log(N/df)`. Yang benar-benar dihitung:

**Term Frequency** (`sublinear_tf=True`):
```
tf(t,d) = 1 + ln( f(t,d) )        f = frekuensi mentah t dalam d
```

**Inverse Document Frequency** (`smooth_idf=True`, default):
```
idf(t) = ln( (1 + n) / (1 + df(t)) ) + 1
```

**Bobot & normalisasi** (`norm='l2'`):
```
w(t,d) = tf(t,d) × idf(t)                      lalu dinormalisasi:
w_norm(t,d) = w(t,d) / √( Σ w(t',d)² )
                        t'∈d
```

**Bukti numerik** (n=4 dokumen, `traveloka` df=3):

| Rumus | Hasil |
|---|---|
| Buku teks `log₁₀(N/df)` | 0,124939 |
| **sklearn `ln((1+n)/(1+df))+1`** | **1,223144** ✅ |
| `idf_` aktual dari sklearn | 1,223144 |

> 📌 **Penting untuk skripsi.** Bila Bab II Anda menuliskan `IDF = log(N/DF)`,
> itu **tidak sesuai** dengan yang dihitung program. Pilih salah satu: perbaiki
> rumus di Bab II agar sesuai sklearn, **atau** sebutkan eksplisit bahwa
> perhitungan manual memakai rumus klasik sedangkan implementasi memakai
> varian *smoothed* milik scikit-learn.

**Parameter aktif:**

| Parameter | Nilai | Arti |
|---|---|---|
| `ngram_range` | (1, 2) | unigram + bigram |
| `min_df` | 1 | kata muncul ≥ 1 dokumen |
| `max_df` | 0.95 | **buang** kata yang muncul di > 95% dokumen |
| `sublinear_tf` | True | redam frekuensi dengan log |
| `norm` | l2 | tiap dokumen dinormalisasi |

> `max_df=0.95` berarti kata **"traveloka" sendiri kemungkinan besar dibuang**,
> karena muncul di hampir semua dokumen sehingga tidak membedakan kelas.

### 4. Multinomial Naive Bayes

**Klasifikasi:**
```
ĉ = argmax [ log P(c) + Σ log P(t|c) ]
      c∈C              t∈d
```

**Prior** (`fit_prior=True`):
```
P(c) = N_c / N          N_c = jumlah dokumen kelas c
```

**Likelihood dengan Lidstone smoothing:**
```
P(t|c) = ( N_tc + α ) / ( N_c + α·V )
```
- `N_tc` = total bobot kata `t` pada kelas `c`
- `N_c` = total bobot semua kata pada kelas `c`
- `V` = jumlah kata unik (ukuran vocabulary)
- `α` = parameter smoothing; `α=1` disebut **Laplace smoothing**

`α` dicari otomatis lewat GridSearch: `[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]`,
5-fold, `scoring="f1_macro"`.

### 5. ComplementNB (Pembanding)

Rennie dkk. (2003). Alih-alih menghitung statistik **kelas c**, ia menghitung
statistik **komplemen** — semua kelas *selain* c:

```
             ( N_t~c + α )
θ_~c,t  =  ───────────────────
            ( N_~c + α·V )

w_ct = log θ_~c,t          lalu dinormalisasi:  w_ct / Σ|w_ct|

ĉ = argmin Σ  f_t · w_ct        (argMIN, bukan argmax)
      c    t∈d
```

Dirancang khusus untuk data timpang — estimasi parameternya lebih stabil karena
kelas komplemen selalu punya lebih banyak sampel daripada kelas itu sendiri.

### 6. Penanganan Ketidakseimbangan

**Urutan wajib:** split → oversampling **hanya data latih**.

```
Random Oversampling: tiap kelas minoritas diduplikasi acak (dengan pengembalian)
                     sampai jumlahnya = kelas terbanyak
```

Data uji **tidak disentuh** agar tetap mencerminkan distribusi nyata.

### 7. Evaluasi

**Confusion matrix 3×3** — `C[i][j]` = jumlah dokumen kelas asli `i` yang
diprediksi sebagai `j`.

```
Accuracy = Σ C[i][i] / Σ C[i][j]              (diagonal / total)
            i           i,j

Precision_i = C[i][i] / Σ C[j][i]             (kolom)
                          j

Recall_i    = C[i][i] / Σ C[i][j]             (baris)
                          j

F1_i = 2 · (Precision_i × Recall_i) / (Precision_i + Recall_i)
```

**Macro-F1 — metrik utama:**
```
Macro-F1 = (1/k) · Σ F1_i          k = 3 kelas
                    i
```
Rata-rata **tanpa bobot**, sehingga kelas kecil punya pengaruh sama besar
dengan kelas besar. Inilah sebabnya macro-F1 dipakai, bukan weighted-F1 yang
justru menyembunyikan kegagalan pada kelas minoritas.

**Baseline kelas mayoritas** — acuan minimum yang wajib dilampaui:
```
Baseline_acc = N_mayoritas / N_total
```

**Cross-validation 10-fold**, stratified, dengan oversampling **di dalam tiap
fold latih** (turun otomatis bila kelas terkecil < 10 sampel).

### 8. Cohen's Kappa — Validasi Label

Mengukur kesepakatan label otomatis vs label manual, **dikoreksi terhadap
kesepakatan yang terjadi secara kebetulan**:

```
        p_o − p_e
κ  =  ─────────────
         1 − p_e
```

- `p_o` = proporsi label yang sama (kesepakatan teramati)
- `p_e` = proporsi kesepakatan yang diharapkan **secara kebetulan**:
  ```
  p_e = Σ ( n_manual,i / N ) × ( n_auto,i / N )
        i
  ```

**Contoh terverifikasi** (10 sampel): `p_o = 0,800` · `p_e = 0,340` →
κ = (0,800 − 0,340) / (1 − 0,340) = **0,6970** — identik dengan
`sklearn.metrics.cohen_kappa_score`.

**Kenapa perlu?** Accuracy mentah menipu. Kalau 80% data berlabel netral,
menebak "netral" terus sudah memberi 80% kecocokan tanpa pemahaman apa pun.
Kappa mengoreksi hal ini: κ = 0 berarti **tidak lebih baik daripada menebak**.

Interpretasi Landis & Koch (1977): < 0,20 sangat rendah · 0,20–0,40 rendah ·
0,40–0,60 sedang · 0,60–0,80 baik · > 0,80 sangat baik.

---

## Library dan Fungsinya

| Library | Versi | Dipakai untuk | Fungsi/kelas spesifik |
|---|---|---|---|
| `playwright` | 1.61.0 | Otomasi browser Chromium untuk scraping X | `sync_playwright`, `page.query_selector_all` |
| `pandas` | 3.0.3 | Baca/tulis CSV, manipulasi tabel | `read_csv`, `DataFrame`, `drop_duplicates` |
| `numpy` | 2.5.1 | Operasi numerik rata-rata & simpangan baku | `mean`, `std` |
| `scikit-learn` | 1.9.0 | Ekstraksi fitur, model, evaluasi | lihat rincian di bawah |
| `nltk` | 3.10.0 | Tokenisasi teks | `word_tokenize`, korpus `punkt` |
| `PySastrawi` | 1.2.1 | Stemming & stopword Bahasa Indonesia | `StemmerFactory`, `StopWordRemoverFactory` |
| `langdetect` | 1.0.9 | Deteksi bahasa dokumen | `detect_langs`, `DetectorFactory.seed` |
| `matplotlib` | 3.11.1 | Render gambar confusion matrix | `pyplot.subplots`, `savefig` |
| `seaborn` | 0.13.2 | Heatmap confusion matrix | `heatmap` |
| `tqdm` | 4.69.0 | Progress bar | `tqdm`, `progress_apply` |
| `openpyxl` | 3.1.5 | Dukungan file Excel (opsional) | — |
| `wordcloud` | 1.9.6 | Visualisasi kata (opsional) | — |

### Rincian scikit-learn

| Modul | Fungsi dalam penelitian ini |
|---|---|
| `TfidfVectorizer` | Ubah teks → matriks bobot TF-IDF (unigram + bigram) |
| `MultinomialNB` | **Model utama** sesuai judul penelitian |
| `ComplementNB` | Model pembanding untuk data timpang |
| `Pipeline` | Rangkai TF-IDF + classifier jadi satu objek |
| `train_test_split` | Bagi data 80/20, `stratify` menjaga proporsi kelas |
| `GridSearchCV` | Cari `alpha` terbaik, 5-fold, `scoring="f1_macro"` |
| `StratifiedKFold` | Cross-validation 10-fold berimbang |
| `resample` | Random oversampling **hanya pada data latih** |
| `accuracy_score` | Proporsi prediksi benar |
| `precision_score` / `recall_score` | Presisi & recall (`average="macro"`) |
| `f1_score` | **Macro-F1**, metrik utama |
| `classification_report` | Tabel metrik per kelas |
| `confusion_matrix` | Matriks 3×3 aktual vs prediksi |
| `cohen_kappa_score` | **Validasi label** otomatis vs manual |
| `clone` | Salin pipeline bersih tiap fold CV |

---

---

## Panduan Penyelarasan Naskah dengan Kode

Bagian ini untuk memastikan naskah skripsi menjelaskan **apa yang benar-benar
dijalankan program**. Ketidakcocokan antara naskah dan kode adalah hal pertama
yang dicari penguji, dan paling sulit dijelaskan saat sidang.

Gunakan sebagai daftar periksa sebelum menyerahkan naskah.

### A. Rumus yang wajib dicek ulang

| Bagian naskah | Sering keliru ditulis | Yang benar-benar dihitung |
|---|---|---|
| IDF | `log(N/DF)` atau `log₂(D/df)` | `ln((1+n)/(1+df)) + 1` — varian *smoothed* sklearn |
| TF | frekuensi mentah / total kata | `1 + ln(f)` karena `sublinear_tf=True` |
| Bobot akhir | `TF × IDF` saja | `TF × IDF` **lalu dinormalisasi L2** |
| Accuracy | `(TP+TN)/(...)` gaya 2 kelas | `Σ diagonal / Σ semua sel` (3 kelas) |
| Metrik utama | accuracy | **macro-F1** + baseline kelas mayoritas |

> Bila Bab II memakai rumus klasik untuk perhitungan manual, **tetap boleh** —
> asalkan ditambahkan satu kalimat: *"perhitungan manual menggunakan rumus
> klasik untuk ilustrasi, sedangkan implementasi menggunakan varian smoothed
> pada scikit-learn."* Tanpa kalimat itu, angka manual dan angka program tidak
> akan pernah cocok bila diverifikasi penguji.

### B. Konsistensi angka

Pastikan angka berikut sama di naskah dan kode:

| Hal | Nilai di kode |
|---|---|
| Jumlah kata kunci pencarian | **33** |
| Rasio data latih : uji | **80 : 20** |
| Ambang netral skor lexicon | **\|skor\| ≤ 2** |
| Jendela deteksi negasi | **3 token** |
| Rentang bobot InSet | **−5 … +5** |
| Cross-validation | **10-fold** stratified |
| Grid pencarian alpha | `0.01 · 0.05 · 0.1 · 0.3 · 0.5 · 1.0 · 2.0` |
| Confidence minimum deteksi bahasa | **0.70** |

Semua nilai di atas dapat dikonfirmasi langsung dari `main.py`.

### C. Yang perlu ditambahkan ke Bab Metodologi

Tahapan berikut dijalankan program tetapi sering belum tertulis di naskah:

1. **Penyaringan bahasa** — beserta alasan operator `lang:id` tidak dipakai
2. **Penyaringan akun resmi** (`-from:traveloka`) — beserta alasannya
3. **Oversampling hanya pada data latih** — dan mengapa urutannya tidak boleh dibalik
4. **Baseline kelas mayoritas** sebagai acuan minimum
5. **Macro-F1** sebagai metrik utama, bukan accuracy
6. **Cohen's Kappa** untuk validasi pelabelan otomatis
7. **`custom.tsv`** — dilaporkan sebagai *"InSet augmented with domain terms"*,
   lengkap dengan daftar kata dan alasan penambahannya
8. **Perbandingan MultinomialNB vs ComplementNB**

### D. Lampiran yang sebaiknya disertakan

- Tangkapan layar hasil crawling — **dari program ini**, bukan dari tool lain.
  Kolom yang benar: `id, username, date, text, likes, retweets, replies, query`
- Bagian *"Dilewati / gagal"* dari laporan scraping — sebagai keterbatasan
  pengumpulan data
- Distribusi bahasa sebelum dan sesudah penyaringan
- Nilai Cohen's Kappa beserta interpretasinya
- Tabel `results/metrics_*.csv` — memuat kedua model dan baseline sekaligus

### E. Penandaan contoh vs hasil

Tabel yang berisi **contoh perhitungan manual** wajib diberi keterangan tegas,
misalnya *"Contoh Perhitungan Manual (bukan hasil penelitian)"*.

Ini penting karena perhitungan manual dengan satu atau dua data uji akan
menghasilkan accuracy, precision, recall, dan F1 bernilai **1,00 (100%)** —
angka yang mustahil pada data sebenarnya. Tanpa keterangan, angka tersebut
sangat mudah disalahpahami sebagai hasil penelitian.

Hal yang sama berlaku untuk tabel berisi nama pengguna dan ulasan rekaan yang
dipakai sebagai ilustrasi tahapan — beri label **"Data Ilustrasi"**.

### F. Batasan yang jujur lebih kuat daripada angka tinggi

Nilai yang wajar untuk dilaporkan apa adanya:

- Cohen's Kappa rendah — menunjukkan keterbatasan pelabelan berbasis lexicon,
  bukan kegagalan penelitian
- Macro-F1 di kisaran 0,6 pada data timpang — realistis untuk 3 kelas
- Jumlah dokumen yang gagal diambil saat scraping

Angka tinggi tanpa penjelasan asal-usulnya jauh lebih berisiko di sidang
daripada angka sedang yang dapat dipertanggungjawabkan.

---

### Batas yang perlu diakui

- Pelabelan InSet **bukan ground truth** — wajib divalidasi dengan Cohen's Kappa
- InSet mengandung 1.142 kata dengan polaritas bertentangan di kedua filenya,
  dan tidak memuat sebagian istilah penting domain ini — sebagian dikoreksi
  lewat `custom.tsv`, tetapi sisa kesalahan pasti masih ada
- `langdetect` kurang akurat pada teks pendek; sebagian tweet Indonesia gaul
  mungkin ikut terbuang
- Dataset < 1.000 dokumen menghasilkan interval kepercayaan yang sangat lebar
- Scraping X melanggar Terms of Service X; gunakan hanya untuk keperluan
  akademis, dan pertimbangkan anonimisasi kolom `username` pada lampiran skripsi

---

---|
| `playwright` | Scraping browser otomatis |
| `pandas`, `numpy` | Manipulasi data |
| `scikit-learn` | MultinomialNB, ComplementNB, TF-IDF, evaluasi |
| `nltk` | Tokenisasi |
| `PySastrawi` | Stemming Bahasa Indonesia |
| `langdetect` | Deteksi bahasa |
| `matplotlib`, `seaborn` | Visualisasi confusion matrix |
| `wordcloud` | Visualisasi kata (opsional) |
| `tqdm` | Progress bar |

---

## Troubleshooting

**`ERROR: Session tidak ditemukan`**
File `session/cookies.json` belum ada. Lihat [Setup Session X](#setup-session-x-wajib).

**`Session expired atau tidak valid!`**
Cookie sudah kedaluwarsa. Ambil ulang lewat Cookie-Editor.

**`ERROR: Lexicon InSet tidak ditemukan`**
Unduh `positive.tsv` dan `negative.tsv` ke folder `lexicon/` (lihat Instalasi #4).

**`BERHENTI: File berlabel sudah ada`**
Perlindungan agar koreksi manual tidak tertimpa. Backup dulu, lalu tambahkan `--force`.

**Browser tidak terbuka saat scraping**
```bash
playwright install chromium
```
Scraper membutuhkan tampilan layar — tidak bisa dijalankan lewat SSH tanpa display.

**Hasil preprocessing tinggal sedikit**
Wajar — sekitar 57% hasil scraping bukan Bahasa Indonesia. Scrape lebih banyak
dengan `--max` yang lebih besar.

---

## Referensi

- Rennie, J.D.M., Shih, L., Teevan, J., & Karger, D.R. (2003). *Tackling the Poor
  Assumptions of Naive Bayes Text Classifiers.* ICML, 616–623.
- Koto, F., & Rahmaningtyas, G.Y. (2017). *InSet Lexicon: Evaluation of a Word List
  for Indonesian Sentiment Analysis in Microblogs.* IALP.
  https://github.com/fajri91/InSet
- Landis, J.R., & Koch, G.G. (1977). *The Measurement of Observer Agreement for
  Categorical Data.* Biometrics, 33(1), 159–174.
- Dokumentasi scikit-learn: https://scikit-learn.org/stable/modules/naive_bayes.html
