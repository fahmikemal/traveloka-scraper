"""
main.py  —  Analisis Sentimen Komentar Traveloka di X
Menggunakan Metode Multinomial Naive Bayes

Opsi:
    --scrape         Scrape komentar Traveloka dari X
    --preprocess     Preprocessing teks (cleaning + filter bahasa + stemming)
    --autolabel      Pelabelan otomatis dengan lexicon InSet
    --sample-gold    Ambil sampel untuk dilabeli manual
    --kappa          Ukur Cohen's Kappa (label otomatis vs manual)
    --train          Latih Multinomial Naive Bayes (+ ComplementNB pembanding)
    --all            Jalankan scrape + preprocess + autolabel + train
    --predict        Prediksi sentimen file baru tanpa training ulang

Session X diambil dari cookie browser (lihat SETUP SESSION di bawah).

Contoh:
    python main.py --scrape --max 250
    python main.py --preprocess
    python main.py --autolabel
    python main.py --sample-gold 200   &&  python main.py --kappa
    python main.py --train
    python main.py --predict --model models/mnb_model_xxx.pkl --input data/processed/xxx.csv

SETUP SESSION (sekali, ulangi bila expired):
    1. Pasang ekstensi "Cookie-Editor" di Chrome/Edge
    2. Login ke https://x.com seperti biasa
    3. Klik ikon Cookie-Editor -> tombol Export (tersalin ke clipboard)
    4. Paste ke file: session/cookies.json
"""

import argparse
import os
import sys
import json
import re
import string
import time
import pickle
import glob
from datetime import datetime
from urllib.parse import quote

# ──────────────────────────────────────────────────────────
#  CONFIG INLINE
# ──────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE    = os.path.join(BASE_DIR, "session", "cookies.json")
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
LABELED_DIR     = os.path.join(BASE_DIR, "data", "labeled")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

X_BASE_URL  = "https://x.com"
X_SEARCH_URL = "https://x.com/search"

# Query dipilih NETRAL secara sentimen: nama layanan dan frasa pemakaian
# sehari-hari. Kata bermuatan seperti "traveloka lemot" atau "traveloka bagus"
# SENGAJA dihindari karena akan menanam label pada sampel dan membuat
# distribusi kelas menjadi artefak desain query, bukan temuan.
#
# Layanan diverifikasi masih aktif per Juli 2026. "traveloka eats" TIDAK
# dipakai: layanan itu resmi ditutup 31 Oktober 2022 bersama Send dan Mart,
# sehingga query tersebut hanya akan memanen data 2021-2022 yang basi.
SEARCH_QUERIES = [
    # layanan
    "traveloka",
    "traveloka pesawat",
    "traveloka tiket pesawat",
    "traveloka hotel",
    "traveloka kereta api",
    "traveloka bus",
    "traveloka travel",
    "traveloka rental mobil",
    "traveloka aktivitas",
    "traveloka atraksi",
    "traveloka xperience",
    "traveloka paylater",
    "traveloka asuransi",
    "traveloka aplikasi",
    "traveloka voucher",
    "traveloka promo",
    # layanan baru sejak Traveloka 5.0
    "traveloka kapal pesiar",
    "traveloka paket wisata",
    # proses transaksi
    "traveloka refund",
    "traveloka reschedule",
    "traveloka booking",
    "traveloka bayar",
    "traveloka akun",
    "traveloka tiket",
    # frasa pemakaian sehari-hari (paling produktif, paling netral)
    "pakai traveloka",
    "pesan di traveloka",
    "beli di traveloka",
    "lewat traveloka",
    "dari traveloka",
    "booking di traveloka",
    "tiket traveloka",
    # singkatan & sebutan
    "tvlk",
    "@traveloka",
]

SCROLL_PAUSE = 2.5
LABEL_MAP    = {0: "negatif", 1: "netral", 2: "positif"}

# ── Filter query pencarian X ──
#
# "lang:id" SENGAJA TIDAK DIPAKAI. Diuji pada 21 Juli 2026, operator itu
# membuat X mengembalikan indeks lama: query "traveloka lang:id" menghasilkan
# tweet tahun 2013-2014 (19 dari 20 berasal dari akun resmi @traveloka),
# sedangkan "traveloka" tanpa operator tersebut menghasilkan tweet hari itu
# juga. Kombinasi "lang:id since:2026-01-01" bahkan mengembalikan 0 hasil.
#
# Penyaringan bahasa dipindahkan ke tahap --preprocess memakai langdetect,
# yang lebih andal dan dapat diverifikasi. Pada uji tersebut hasil pencarian
# tanpa "lang:id" ternyata sudah 8/8 berbahasa Indonesia.
#
# "-from:traveloka" membuang balasan customer service akun resmi. Isinya teks
# PR ("Terima kasih telah mempercayai Traveloka") yang akan mencemari analisis
# sentimen PELANGGAN. Balasan dari pengguna biasa tetap diambil karena justru
# di situ banyak keluhan muncul.
QUERY_FILTER = "-from:traveloka"


# ──────────────────────────────────────────────────────────
#  STEP 2 — SCRAPING
# ──────────────────────────────────────────────────────────

def _normalize_cookies(raw: list) -> list:
    """Konversi format Chrome extension ke format Playwright."""
    SAMESITE_MAP = {
        "unspecified": "None",
        "no_restriction": "None",
        "lax": "Lax",
        "strict": "Strict",
    }
    result = []
    for c in raw:
        same_site = SAMESITE_MAP.get(str(c.get("sameSite", "None")).lower(), "None")
        secure = bool(c.get("secure", False))
        # Playwright menolak sameSite="None" bila secure=False.
        if same_site == "None" and not secure:
            same_site = "Lax"

        cookie = {
            "name":     c["name"],
            "value":    c["value"],
            "domain":   c["domain"],
            "path":     c.get("path", "/"),
            "httpOnly": bool(c.get("httpOnly", False)),
            "secure":   secure,
            "sameSite": same_site,
        }
        # expirationDate (Chrome ext) atau expires (Playwright).
        # Pakai "is not None": nilai 0 valid dan tidak boleh dianggap kosong.
        exp = c.get("expirationDate")
        if exp is None:
            exp = c.get("expires")
        if exp is not None:
            try:
                cookie["expires"] = int(float(exp))
            except (TypeError, ValueError):
                pass
        result.append(cookie)
    return result


COOKIE_HELP = (
    "Cara mengambil session X:\n"
    "  1. Pasang ekstensi 'Cookie-Editor' di Chrome/Edge\n"
    "  2. Login ke https://x.com seperti biasa\n"
    "  3. Klik ikon Cookie-Editor -> tombol Export (tersalin ke clipboard)\n"
    f"  4. Paste ke file: {SESSION_FILE}"
)


def _load_cookies(context):
    if not os.path.exists(SESSION_FILE):
        sys.exit(f"ERROR: Session tidak ditemukan.\n\n{COOKIE_HELP}")
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    context.add_cookies(_normalize_cookies(raw))


def _parse_count(text):
    """Ubah teks metrik X ("1.234", "12,5 rb", "3.4K") menjadi integer.

    X menampilkan angka mengikuti locale, jadi pemisah ribuan bisa "." (ID)
    atau "," (EN). Versi lama mengganti semua "," menjadi "." sehingga
    "1,234" berubah jadi "1.234" dan int() gagal -> metrik 4 digit ke atas
    diam-diam tercatat 0.
    """
    if not text:
        return 0
    t = str(text).strip().upper()
    if not t:
        return 0

    multipliers = [("RB", 1_000), ("JT", 1_000_000),
                   ("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000)]
    for suffix, mult in multipliers:
        if t.endswith(suffix):
            num = t[: -len(suffix)].strip()
            # Di sini pemisah desimal: "3.4K" (EN) atau "3,4 rb" (ID)
            num = num.replace(",", ".")
            try:
                return int(float(num) * mult)
            except ValueError:
                return 0

    # Tanpa sufiks: semua "." dan "," adalah pemisah ribuan
    digits = re.sub(r"[.,\s]", "", t)
    try:
        return int(digits)
    except ValueError:
        return 0


def _extract_id(url):
    m = re.search(r"/status/(\d+)", url or "")
    return m.group(1) if m else ""


def _goto_with_retry(page, url, attempts=3, timeout=30_000):
    """Buka URL dengan percobaan ulang + backoff. True bila berhasil."""
    from playwright.sync_api import Error as PwError
    from playwright.sync_api import TimeoutError as PwTimeout

    for i in range(1, attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            return True
        except (PwTimeout, PwError) as e:
            kind = type(e).__name__
            if i < attempts:
                wait = SCROLL_PAUSE * (2 ** i)      # 5s, 10s, ...
                print(f"    [{kind}] percobaan {i}/{attempts} gagal, "
                      f"ulangi dalam {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"    [{kind}] gagal setelah {attempts} percobaan")
    return False


def _extract_tweet(art, query, collected):
    """Ambil satu tweet dari elemen article.

    Mengembalikan (data|None, alasan_gagal|None) sehingga pemanggil bisa
    MENGHITUNG kegagalan, bukan menelannya diam-diam seperti versi lama.
    """
    text_el = art.query_selector('[data-testid="tweetText"]')
    text = text_el.inner_text() if text_el else ""
    if not text:
        return None, "tanpa_teks"

    link_el = art.query_selector('a[href*="/status/"]')
    tid = _extract_id(link_el.get_attribute("href") if link_el else "")
    if not tid:
        return None, "tanpa_id"
    if tid in collected:
        return None, "duplikat"

    username = ""
    user_el = art.query_selector('[data-testid="User-Name"]')
    if user_el:
        for sp in user_el.query_selector_all("span"):
            t = sp.inner_text().strip()
            if t.startswith("@"):
                username = t
                break

    time_el = art.query_selector("time")
    date = time_el.get_attribute("datetime") if time_el else ""

    def metric(name):
        el = art.query_selector(f'[data-testid="{name}"]')
        return _parse_count(el.inner_text()) if el else 0

    return {
        "id": tid, "username": username, "date": date, "text": text,
        "likes": metric("like"), "retweets": metric("retweet"),
        "replies": metric("reply"), "query": query,
    }, None


def _scrape_query(page, query, max_tweets, stats):
    from playwright.sync_api import Error as PwError
    from tqdm import tqdm

    url = f"{X_SEARCH_URL}?q={quote(f'{query} {QUERY_FILTER}'.strip())}&src=typed_query&f=live"
    print(f"\n  Query: {query}")
    if not _goto_with_retry(page, url):
        stats["query_gagal"] += 1
        return []
    time.sleep(3)

    collected = {}
    no_new = 0
    max_no_new = 8          # lebih longgar: X kadang lambat memuat batch berikutnya

    with tqdm(total=max_tweets, desc="  Terkumpul", unit="tw", leave=False) as pbar:
        while len(collected) < max_tweets:
            try:
                articles = page.query_selector_all('article[data-testid="tweet"]')
            except PwError as e:
                stats["error_dom"] += 1
                print(f"    [DOM] {type(e).__name__} saat membaca daftar tweet")
                break

            new = 0
            for art in articles:
                try:
                    data, reason = _extract_tweet(art, query, collected)
                except PwError:
                    # Elemen hilang dari DOM (X memvirtualisasi daftar).
                    # Dihitung, tidak ditelan diam-diam.
                    stats["elemen_stale"] += 1
                    continue
                except Exception as e:
                    stats["error_tak_terduga"] += 1
                    stats["contoh_error"].setdefault(type(e).__name__, str(e)[:120])
                    continue

                if data is None:
                    stats[reason] = stats.get(reason, 0) + 1
                    continue

                collected[data["id"]] = data
                new += 1
                pbar.update(1)
                if len(collected) >= max_tweets:
                    break

            if new == 0:
                no_new += 1
                if no_new >= max_no_new:
                    break
            else:
                no_new = 0

            try:
                page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            except PwError:
                stats["error_scroll"] += 1
                break
            time.sleep(SCROLL_PAUSE)

    if len(collected) < max_tweets:
        print(f"    (berhenti di {len(collected)}/{max_tweets} - hasil habis)")
    return list(collected.values())


def step_scrape(max_tweets, headless=False):
    import pandas as pd
    from playwright.sync_api import sync_playwright

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(RAW_DATA_DIR, f"traveloka_raw_{timestamp}.csv")

    print("\n[STEP 2] SCRAPING KOMENTAR TRAVELOKA")
    print("=" * 50)
    print(f"Total query : {len(SEARCH_QUERIES)}")
    print(f"Max/query   : {max_tweets} tweet")

    cols = ["id", "username", "date", "text", "likes", "retweets", "replies", "query"]
    all_tweets, seen = [], set()
    stats = {
        "tanpa_teks": 0, "tanpa_id": 0, "duplikat": 0, "elemen_stale": 0,
        "error_dom": 0, "error_scroll": 0, "error_tak_terduga": 0,
        "query_gagal": 0, "contoh_error": {},
    }
    per_query = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless, slow_mo=30,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/141.0.0.0 Safari/537.36"
            ),
        )
        # Sembunyikan penanda otomasi (sebelumnya hanya ada di alur login)
        context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            "window.chrome={runtime:{}};"
        )
        _load_cookies(context)
        page = context.new_page()

        # ── Validasi session ──
        # Cek URL saja tidak cukup: X bisa menampilkan halaman challenge tanpa
        # kata "login" di URL. Yang menentukan adalah ada/tidaknya elemen
        # khas beranda pengguna yang sudah masuk.
        print("\nMemeriksa session...")
        if not _goto_with_retry(page, X_BASE_URL):
            browser.close()
            sys.exit("Tidak bisa membuka x.com. Periksa koneksi internet.")
        time.sleep(3)

        url_now = page.url.lower()
        logged_in = False
        if not any(k in url_now for k in ("/login", "/i/flow", "account/access")):
            for sel in ('[data-testid="SideNav_AccountSwitcher_Button"]',
                        '[data-testid="AppTabBar_Home_Link"]',
                        '[data-testid="primaryColumn"]'):
                try:
                    if page.wait_for_selector(sel, timeout=8_000):
                        logged_in = True
                        break
                except Exception:
                    continue

        if not logged_in:
            browser.close()
            sys.exit(
                f"Session tidak valid / expired.\n"
                f"URL saat ini: {page.url}\n\n{COOKIE_HELP}"
            )
        print("Session valid.\n")

        for q in SEARCH_QUERIES:
            try:
                tweets = _scrape_query(page, q, max_tweets, stats)
            except KeyboardInterrupt:
                print("\n[DIBATALKAN] Menyimpan data yang sudah terkumpul...")
                break
            except Exception as e:
                stats["query_gagal"] += 1
                stats["contoh_error"].setdefault(type(e).__name__, str(e)[:120])
                print(f"  [ERROR] {q}: {type(e).__name__}: {e}")
                tweets = []

            baru = 0
            for tw in tweets:
                if tw["id"] not in seen:
                    seen.add(tw["id"])
                    all_tweets.append(tw)
                    baru += 1
            per_query[q] = baru
            print(f"  Diambil {len(tweets)}, unik baru {baru}, total {len(all_tweets)}")

            # ── Simpan progres setiap selesai satu query ──
            # Penting di Google Colab: runtime bisa terputus kapan saja.
            # Tanpa ini, seluruh hasil scraping berjam-jam akan hilang.
            if all_tweets:
                pd.DataFrame(all_tweets, columns=cols).to_csv(
                    output, index=False, encoding="utf-8-sig"
                )
            time.sleep(3)

        browser.close()

    df = pd.DataFrame(all_tweets, columns=cols)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    # ── Laporan integritas: apa yang GAGAL, bukan hanya yang berhasil ──
    print("\n" + "=" * 50)
    print("LAPORAN SCRAPING")
    print("=" * 50)
    print(f"Tweet unik tersimpan : {len(df)}")
    print(f"Tersimpan            : {output}")

    kosong = [q for q, n in per_query.items() if n == 0]
    if kosong:
        print(f"\nQuery tanpa hasil ({len(kosong)}):")
        for q in kosong:
            print(f"  - {q}")

    print("\nDilewati / gagal:")
    print(f"  duplikat antar-query : {stats['duplikat']}")
    print(f"  tanpa teks           : {stats['tanpa_teks']}")
    print(f"  tanpa id             : {stats['tanpa_id']}")
    print(f"  elemen hilang (DOM)  : {stats['elemen_stale']}")
    print(f"  error DOM / scroll   : {stats['error_dom']} / {stats['error_scroll']}")
    print(f"  error tak terduga    : {stats['error_tak_terduga']}")
    print(f"  query gagal total    : {stats['query_gagal']}")
    if stats["contoh_error"]:
        print("  contoh error:")
        for name, msg in stats["contoh_error"].items():
            print(f"    {name}: {msg}")

    hilang = stats["elemen_stale"] + stats["error_tak_terduga"]
    if hilang:
        total = len(df) + hilang
        print(f"\n  PERHATIAN: {hilang} tweet hilang karena error "
              f"({hilang / total * 100:.1f}% dari yang sempat terlihat).")
        print("  Angka ini WAJIB dilaporkan di skripsi sebagai keterbatasan"
              " pengumpulan data.")

    print(f"\nCatatan: sekitar 45-50% hasil biasanya bukan Bahasa Indonesia dan")
    print("akan dibuang pada tahap --preprocess.")
    return output


# ──────────────────────────────────────────────────────────
#  STEP 3 — PREPROCESSING
# ──────────────────────────────────────────────────────────

# Kata negasi WAJIB dipertahankan: menghapusnya membuat "tidak bagus"
# menyusut jadi "bagus" sehingga model tidak mungkin mempelajari pembalikan
# polaritas. Sastrawi memasukkan sebagian kata ini ke daftar stopword bawaan,
# jadi harus dikurangkan secara eksplisit.
NEGATION_WORDS = {
    "tidak", "tdk", "tak", "ga", "gak", "gk", "nggak", "ngga", "nda", "ndak",
    "enggak", "engga", "bukan", "bukanlah", "jangan", "janganlah", "belum",
    "blm", "belom", "tanpa", "kurang", "no", "not", "never", "dont", "cant",
}


def _init_nlp():
    import nltk
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)

    stemmer = StemmerFactory().create_stemmer()
    sw_base = set(StopWordRemoverFactory().get_stop_words())
    sw_extra = {
        "yg","ny","nya","gw","gue","lo","lu","sy","aku","kamu","dia","kita",
        "kami","mereka","ini","itu","juga","sudah","suda","udah","udh","dah",
        "aja","saja","banget","bgt","mau","mw","emang","memang","dong","sih",
        "deh","nih","loh","lah","kan","ya","yaa","haha","hehe","wkwk",
        "wkwkwk","hm","hmm","eh","ah","oh","oke","ok","iya","iyaa",
        "sampe","kayak","kaya","kek","bisa","bs","tapi","tp","trus","terus",
        "habis","abis","lagi","lg","pake","pakai","sama","sm","ke","di",
        "dari","dan","atau","dengan","untuk","dlm","dalam","kalau","klo",
        "kalo","lebih","udah","sudah",
    }
    return stemmer, (sw_base | sw_extra) - NEGATION_WORDS


def _clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _detect_lang(text):
    """Deteksi bahasa satu dokumen. Mengembalikan (kode_bahasa, kepercayaan)."""
    from langdetect import detect_langs

    t = str(text).strip()
    if len(t) < 10:                      # terlalu pendek untuk diandalkan
        return "unknown", 0.0
    try:
        best = detect_langs(t)[0]
        return best.lang, float(best.prob)
    except Exception:
        return "unknown", 0.0


def step_preprocess(input_file, filter_lang=True, lang_conf=0.70):
    import pandas as pd
    from nltk.tokenize import word_tokenize
    from tqdm import tqdm
    from langdetect import DetectorFactory

    # Tanpa seed, langdetect memberi hasil BERBEDA tiap dijalankan.
    # Untuk penelitian, hasil wajib dapat direproduksi.
    DetectorFactory.seed = 0

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stemmer, stopwords = _init_nlp()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output = os.path.join(PROCESSED_DIR, f"{basename}_processed.csv")

    print("\n[STEP 3] PREPROCESSING TEKS")
    print("=" * 50)
    print(f"Input : {input_file}")

    if isinstance(input_file, (list, tuple)):
        # Gabungkan beberapa sesi scraping, buang duplikat berdasarkan id tweet
        parts = []
        for f in input_file:
            d = pd.read_csv(f, encoding="utf-8-sig")
            print(f"  + {os.path.basename(f)}: {len(d)} baris")
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        sebelum = len(df)
        df = df.drop_duplicates(subset=["id"])
        print(f"  Gabungan: {sebelum} -> {len(df)} baris setelah buang duplikat id")
        basename = f"traveloka_gabungan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = os.path.join(PROCESSED_DIR, f"{basename}_processed.csv")
    else:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"Total : {len(df)} baris")

    tqdm.pandas(desc="  Cleaning")
    df["clean_text"] = df["text"].progress_apply(_clean)

    # ── Filter bahasa ──
    # Filter lang:id milik X tidak dapat diandalkan: banyak tweet promosi
    # berbahasa Inggris tetap lolos. Menganalisis teks Inggris memakai
    # Sastrawi (stemmer Indonesia) dan InSet (lexicon Indonesia) menghasilkan
    # skor yang tidak bermakna, jadi dokumen non-Indonesia harus dibuang.
    if filter_lang:
        tqdm.pandas(desc="  Deteksi bahasa")
        detected = df["clean_text"].progress_apply(_detect_lang)
        df["lang"]      = [c for c, _ in detected]
        df["lang_conf"] = [p for _, p in detected]

        before = len(df)
        dist = df["lang"].value_counts()
        print("\n  Distribusi bahasa terdeteksi:")
        for code, cnt in dist.head(6).items():
            print(f"    {code}: {cnt} ({cnt / before * 100:.1f}%)")

        df = df[(df["lang"] == "id") & (df["lang_conf"] >= lang_conf)]
        dropped = before - len(df)
        print(f"\n  Dibuang non-Indonesia    : {dropped} baris "
              f"({dropped / before * 100:.1f}%)")
        print(f"  Tersisa (lang=id, conf>={lang_conf}): {len(df)} baris")
        if len(df) < 300:
            print(f"\n  PERINGATAN: hanya {len(df)} dokumen Indonesia tersisa.")
            print("  Ini terlalu sedikit untuk klasifikasi 3 kelas yang andal.")
            print("  Scrape lebih banyak data sebelum menarik kesimpulan.")

    def full_process(text):
        tokens  = word_tokenize(text)
        tokens  = [t for t in tokens if t not in stopwords and len(t) > 1]
        stemmed = [stemmer.stem(t) for t in tokens]
        return " ".join(stemmed)

    tqdm.pandas(desc="  Stemming")
    df["stemmed_text"] = df["clean_text"].progress_apply(full_process)

    df = df[df["stemmed_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["stemmed_text"])

    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nSelesai preprocessing: {len(df)} baris")
    print(f"Tersimpan            : {output}")
    return output


# ──────────────────────────────────────────────────────────
#  STEP 3b — AUTO LABELING (lexicon InSet + penanganan negasi)
# ──────────────────────────────────────────────────────────
#
#  Pelabelan memakai InSet (Indonesia Sentiment Lexicon, Koto & Rahmaningtyas
#  2017): 3.609 kata positif dan 6.609 kata negatif dengan bobot -5..+5.
#
#  Tiga perbaikan atas pendekatan keyword sebelumnya:
#    1. Pencocokan berbasis TOKEN, bukan substring. "selamat" tidak lagi
#       terhitung negatif gara-gara mengandung "lama".
#    2. Bobot skalar, bukan hitungan biner, sehingga "sangat mengecewakan"
#       tidak setara bobotnya dengan "agak lambat".
#    3. Penanganan NEGASI: kata sentimen yang didahului kata negasi dalam
#       jarak NEGATION_WINDOW token dibalik polaritasnya.
#
#  PERINGATAN METODOLOGIS: pelabelan otomatis BUKAN ground truth. Ia hanya
#  perkiraan. Validitasnya wajib diukur terhadap label manual memakai
#  Cohen's Kappa (jalankan --sample-gold lalu --kappa).

LEXICON_DIR      = os.path.join(BASE_DIR, "lexicon")
GOLD_DIR         = os.path.join(BASE_DIR, "data", "gold")
NEUTRAL_THRESHOLD = 2    # |skor| <= ambang -> netral
NEGATION_WINDOW   = 3    # jangkauan kata negasi ke depan
MAX_PHRASE_LEN    = 4    # entri terpanjang InSet


def _load_inset(verbose=False):
    """Muat lexicon InSet menjadi dict kata/frasa -> bobot.

    CATATAN PENTING — kontradiksi di dalam InSet:
    1.142 kata muncul di positive.tsv DAN negative.tsv dengan bobot berlawanan
    (contoh: "bagus" = +2 dan -4; "senang" = +5 dan -4). Ini sifat asli lexicon
    hasil anotasi crowd, bukan kesalahan unduhan.

    Aturan penggabungan: untuk kata yang bertentangan, POSITIVE.TSV DIDAHULUKAN.
    Dipilih setelah membandingkan tiga strategi terhadap 20 kata umum yang
    polaritasnya tidak diperdebatkan penutur asli:

        positive didahulukan : 16/20 benar   <- dipakai
        bobot |maksimum|     : 15/20 benar
        bobot dijumlahkan    : 14/20 benar

    Alasannya konsisten dengan komposisi lexicon: negative.tsv jauh lebih besar
    (6.606 vs 3.607) dan lebih berisik -- memuat kata seperti "gemar" pada -1.

    Sebelum perbaikan ini, hasil bergantung pada urutan muat file sehingga
    seluruh 1.142 kata itu diam-diam menjadi negatif.

    Batas yang harus diakui di skripsi:
    - Sebagian kata umum tetap salah polaritas (mis. "lambat" dan "mahal"
      bernilai +1 di InSet).
    - Sejumlah istilah penting untuk domain ini TIDAK ADA di InSet sama sekali:
      "kecewa", "tipu", "penipuan", "refund", "komplain". Karena itu tersedia
      lexicon tambahan opsional di lexicon/custom.tsv.
    Konsekuensinya, validasi Cohen's Kappa terhadap label manual bersifat
    WAJIB, bukan opsional.
    """
    import csv as _csv

    files = {}
    for fname in ("positive.tsv", "negative.tsv"):
        path = os.path.join(LEXICON_DIR, fname)
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: Lexicon InSet tidak ditemukan: {path}\n"
                "Unduh dulu dengan:\n"
                "  mkdir -p lexicon && cd lexicon\n"
                "  curl -LO https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv\n"
                "  curl -LO https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
            )
        entries = {}
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(_csv.reader(f, delimiter="\t")):
                if i == 0 or len(row) < 2:
                    continue
                try:
                    entries[row[0].strip().lower()] = int(row[1])
                except ValueError:
                    continue
        files[fname] = entries

    pos, neg = files["positive.tsv"], files["negative.tsv"]
    ambiguous = set(pos) & set(neg)

    lex = dict(neg)
    lex.update(pos)          # positive.tsv menimpa negative.tsv bila bertentangan

    # Lexicon tambahan opsional untuk istilah domain yang tidak ada di InSet.
    # Format sama (kata<TAB>bobot) dan diprioritaskan di atas keduanya.
    n_custom = 0
    custom_path = os.path.join(LEXICON_DIR, "custom.tsv")
    if os.path.exists(custom_path):
        with open(custom_path, encoding="utf-8") as f:
            for i, row in enumerate(_csv.reader(f, delimiter="\t")):
                if i == 0 or len(row) < 2 or row[0].startswith("#"):
                    continue
                try:
                    lex[row[0].strip().lower()] = int(row[1])
                    n_custom += 1
                except ValueError:
                    continue

    if verbose:
        print(f"Lexicon       : {len(lex)} entri "
              f"({sum(1 for k in lex if ' ' in k)} frasa multi-kata)")
        print(f"  positive.tsv={len(pos)}  negative.tsv={len(neg)}")
        print(f"  kata bertentangan di kedua file: {len(ambiguous)} "
              f"-> positive.tsv didahulukan")
        if n_custom:
            print(f"  custom.tsv (istilah domain): {n_custom} entri")
    return lex


def _score_inset(text, lex, window=NEGATION_WINDOW):
    """Hitung skor sentimen sebuah teks. Mengembalikan (skor, daftar_kecocokan).

    Pencocokan frasa terpanjang lebih dulu agar "tidak bisa" menang atas "bisa".
    """
    tokens = str(text).lower().split()
    n = len(tokens)
    score, hits, i = 0, [], 0

    while i < n:
        matched = False
        for length in range(min(MAX_PHRASE_LEN, n - i), 0, -1):
            phrase = " ".join(tokens[i:i + length])
            if phrase in lex:
                weight = lex[phrase]
                # Negasi dalam `window` token sebelum kata sentimen -> balik polaritas
                preceding = tokens[max(0, i - window):i]
                negated = any(t in NEGATION_WORDS for t in preceding)
                if negated:
                    weight = -weight
                score += weight
                hits.append((phrase, weight, negated))
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return score, hits


def _autolabel_text(text, lex, threshold=NEUTRAL_THRESHOLD):
    score, _ = _score_inset(text, lex)
    if score > threshold:
        return 2
    if score < -threshold:
        return 0
    return 1


def step_autolabel(input_file, threshold=NEUTRAL_THRESHOLD, force=False):
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(LABELED_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    basename = basename.replace("_processed", "")
    output = os.path.join(LABELED_DIR, f"{basename}_labeled.csv")

    print("\n[STEP 3b] AUTO LABELING (InSet lexicon)")
    print("=" * 50)
    print(f"Input : {input_file}")

    # ── Lindungi koreksi label manual dari penimpaan ──
    if os.path.exists(output) and not force:
        sys.exit(
            f"BERHENTI: File berlabel sudah ada:\n  {output}\n\n"
            "File ini mungkin berisi koreksi label MANUAL Anda. Menimpanya\n"
            "akan menghapus pekerjaan tersebut secara permanen.\n\n"
            "Pilihan:\n"
            "  1. Backup dulu, lalu jalankan ulang dengan --force\n"
            "  2. Ganti nama file lama bila ingin menyimpan keduanya"
        )

    lex = _load_inset(verbose=True)
    print(f"Ambang netral : |skor| <= {threshold}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    src_col = "clean_text" if "clean_text" in df.columns else "text"

    tqdm.pandas(desc="  Labeling")
    scored = df[src_col].progress_apply(lambda t: _score_inset(t, lex))
    df["sentiment_score"] = [s for s, _ in scored]
    df["lexicon_hits"]    = [len(h) for _, h in scored]
    df["label"] = df["sentiment_score"].apply(
        lambda s: 2 if s > threshold else (0 if s < -threshold else 1)
    )

    label_names = {0: "negatif", 1: "netral", 2: "positif"}
    print("\nDistribusi label otomatis:")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        print(f"  {label_names[lbl]}: {cnt} ({cnt / len(df) * 100:.1f}%)")

    # Transparansi: netral karena seimbang, atau karena tak terdeteksi?
    undetected = int((df["lexicon_hits"] == 0).sum())
    netral_total = int((df["label"] == 1).sum())
    print(f"\nDari {netral_total} dokumen netral:")
    print(f"  {undetected} tidak mengandung kata lexicon sama sekali (tak terdeteksi)")
    print(f"  {netral_total - undetected} benar-benar seimbang/lemah sentimennya")
    if netral_total:
        print(f"  -> {undetected / netral_total * 100:.1f}% label netral sebenarnya "
              "'tidak diketahui', bukan 'netral'")

    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nFile berlabel disimpan: {output}")
    print("\nLANGKAH WAJIB BERIKUTNYA — validasi label:")
    print("  1. python main.py --sample-gold 200")
    print("  2. Isi kolom 'label_manual' di file gold secara manual")
    print("  3. python main.py --kappa")
    return output


# ──────────────────────────────────────────────────────────
#  STEP 3c — VALIDASI LABEL (gold sample + Cohen's Kappa)
# ──────────────────────────────────────────────────────────

def step_sample_gold(input_file, n=200, random_state=42):
    """Ambil sampel acak untuk dilabeli MANUAL sebagai pembanding."""
    import pandas as pd

    os.makedirs(GOLD_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(GOLD_DIR, f"gold_sample_{timestamp}.csv")

    print("\n[STEP 3c] SAMPEL GOLD UNTUK LABEL MANUAL")
    print("=" * 50)

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    if "label" not in df.columns:
        sys.exit("ERROR: File input harus sudah punya kolom 'label' (jalankan --autolabel dulu).")

    n = min(n, len(df))
    # Stratified agar tiap kelas terwakili, bukan didominasi kelas mayoritas.
    # groupby.apply dihindari: pada pandas 3.x kolom grouping ikut terbuang.
    parts = []
    for _, g in df.groupby("label"):
        k = min(len(g), max(1, round(n * len(g) / len(df))))
        parts.append(g.sample(k, random_state=random_state))
    sample = pd.concat(parts).sample(frac=1, random_state=random_state)

    out = sample[["id", "text", "clean_text", "label"]].copy()
    out = out.rename(columns={"label": "label_auto"})
    out["label_manual"] = ""      # <- diisi manusia
    out.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Sampel     : {len(out)} baris (stratified)")
    print(f"Tersimpan  : {output}")
    print("\nCARA MENGISI:")
    print("  1. Buka file di Excel/LibreOffice")
    print("  2. Isi kolom 'label_manual': 0=negatif, 1=netral, 2=positif")
    print("     Baca kolom 'text' (teks asli), JANGAN lihat 'label_auto' dulu")
    print("     agar penilaian Anda tidak terpengaruh (menghindari bias).")
    print("  3. Simpan sebagai CSV, lalu jalankan: python main.py --kappa")
    return output


def step_kappa(gold_file):
    """Ukur kesepakatan label otomatis vs label manual (Cohen's Kappa)."""
    import pandas as pd
    from sklearn.metrics import (
        cohen_kappa_score, accuracy_score,
        classification_report, confusion_matrix,
    )

    print("\n[STEP 3c] VALIDASI LABEL — COHEN'S KAPPA")
    print("=" * 50)
    print(f"File gold: {gold_file}")

    df = pd.read_csv(gold_file, encoding="utf-8-sig")
    for col in ("label_auto", "label_manual"):
        if col not in df.columns:
            sys.exit(f"ERROR: Kolom '{col}' tidak ditemukan di {gold_file}")

    df = df[pd.to_numeric(df["label_manual"], errors="coerce").notna()]
    if df.empty:
        sys.exit(
            "ERROR: Kolom 'label_manual' masih kosong.\n"
            "Isi dulu labelnya secara manual (0/1/2), lalu jalankan lagi."
        )

    auto   = df["label_auto"].astype(int)
    manual = df["label_manual"].astype(int)

    kappa = cohen_kappa_score(manual, auto)
    acc   = accuracy_score(manual, auto)

    print(f"\nSampel tervalidasi: {len(df)} baris")
    print(f"Persentase cocok  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Cohen's Kappa     : {kappa:.4f}")

    # Interpretasi Landis & Koch (1977)
    if   kappa < 0.00: tafsir = "lebih buruk dari tebakan acak"
    elif kappa < 0.20: tafsir = "sangat rendah (slight)"
    elif kappa < 0.40: tafsir = "rendah (fair)"
    elif kappa < 0.60: tafsir = "sedang (moderate)"
    elif kappa < 0.80: tafsir = "baik (substantial)"
    else:              tafsir = "sangat baik (almost perfect)"
    print(f"Interpretasi      : {tafsir}  [Landis & Koch, 1977]")

    if kappa < 0.40:
        print("\nPERINGATAN: kesepakatan di bawah 0.40.")
        print("Pelabelan otomatis TIDAK cukup valid untuk dipakai apa adanya.")
        print("Gunakan label manual untuk training, atau perbaiki dulu")
        print("ambang/lexicon sebelum melanjutkan.")

    labels_sorted = sorted(LABEL_MAP)
    print("\nLabel manual (acuan) vs label otomatis:")
    print(classification_report(
        manual, auto, labels=labels_sorted,
        target_names=[LABEL_MAP[i] for i in labels_sorted],
        zero_division=0,
    ))
    print("Confusion matrix (baris=manual, kolom=otomatis):")
    print(confusion_matrix(manual, auto, labels=labels_sorted))
    return kappa


# ──────────────────────────────────────────────────────────
#  STEP 4 — TRAINING MNB
# ──────────────────────────────────────────────────────────

def _oversample(X, y, random_state):
    """Samakan jumlah tiap kelas ke kelas terbanyak.

    Hanya boleh dipanggil pada data latih, tidak pernah pada data uji.
    """
    import pandas as pd
    from sklearn.utils import resample

    df = pd.DataFrame({"X": X, "y": y})
    max_count = df["y"].value_counts().max()
    parts = []
    for lbl in df["y"].unique():
        subset = df[df["y"] == lbl]
        if len(subset) < max_count:
            subset = resample(
                subset, replace=True, n_samples=max_count,
                random_state=random_state,
            )
        parts.append(subset)
    out = (pd.concat(parts)
             .sample(frac=1, random_state=random_state)
             .reset_index(drop=True))
    return out["X"], out["y"]


def _cv_no_leak(pipeline, X, y, random_state, n_splits=10):
    """Cross-validation macro-F1 tanpa kebocoran data.

    Oversampling dijalankan di dalam setiap fold dan hanya pada bagian
    latih fold tersebut, sehingga fold validasi tetap murni. Ini setara
    dengan imblearn.pipeline.Pipeline, ditulis manual agar alurnya
    transparan dan tidak menambah dependensi.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # StratifiedKFold gagal bila ada kelas yang anggotanya lebih sedikit
    # daripada jumlah fold. Turunkan otomatis dan beri tahu penggunanya.
    min_class = int(y.value_counts().min())
    if min_class < n_splits:
        n_splits = max(2, min_class)
        print(f"  (CV diturunkan ke {n_splits}-fold: kelas terkecil "
              f"hanya {min_class} sampel)")
    if min_class < 2:
        print("  (CV dilewati: ada kelas dengan < 2 sampel)")
        return float("nan"), float("nan")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = _oversample(X.iloc[tr_idx], y.iloc[tr_idx], random_state)
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        pred = model.predict(X.iloc[va_idx])
        scores.append(
            f1_score(y.iloc[va_idx], pred, average="macro", zero_division=0)
        )
    return float(np.mean(scores)), float(np.std(scores))


def step_train(input_file, test_size=0.2, random_state=42):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")          # backend non-interaktif: aman tanpa display
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score,
    )
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.pipeline import Pipeline

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n[STEP 4] TRAINING MULTINOMIAL NAIVE BAYES")
    print("=" * 50)
    print(f"Input: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    if "stemmed_text" not in df.columns or "label" not in df.columns:
        sys.exit(
            "ERROR: File harus punya kolom 'stemmed_text' dan 'label'.\n"
            "Tambahkan kolom label (0=negatif, 1=netral, 2=positif) secara manual."
        )

    df = df.dropna(subset=["stemmed_text", "label"])
    df = df[df["stemmed_text"].str.strip() != ""]
    df["label"] = df["label"].astype(int)

    print(f"Total data awal: {len(df)}")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        print(f"  {LABEL_MAP.get(lbl, lbl)}: {cnt}")

    # ══════════════════════════════════════════════════════════════
    #  SPLIT DULU, BARU OVERSAMPLING.
    #  Oversampling sebelum split menyebabkan data leakage: baris hasil
    #  duplikasi kelas minoritas bocor ke data uji, sehingga akurasi
    #  yang dilaporkan menjadi hafalan, bukan generalisasi.
    #  Data uji sengaja dibiarkan timpang agar mencerminkan kondisi nyata.
    # ══════════════════════════════════════════════════════════════
    X, y = df["stemmed_text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nData latih: {len(X_train)}  |  Data uji: {len(X_test)} (dibiarkan timpang)")

    X_train, y_train = _oversample(X_train, y_train, random_state)
    print(f"Data latih setelah oversampling: {len(X_train)}")
    for lbl, cnt in y_train.value_counts().sort_index().items():
        print(f"  {LABEL_MAP.get(lbl, lbl)}: {cnt}")

    # ── Baseline kelas mayoritas: acuan minimum yang harus dilampaui ──
    majority = y_train.value_counts().idxmax()
    base_pred = pd.Series([majority] * len(y_test), index=y_test.index)
    base_acc = accuracy_score(y_test, base_pred)
    base_f1m = f1_score(y_test, base_pred, average="macro", zero_division=0)
    print(f"\nBASELINE (selalu menebak '{LABEL_MAP.get(majority, majority)}')")
    print(f"  Accuracy: {base_acc:.4f}  |  Macro-F1: {base_f1m:.4f}")
    print("  Model dianggap berguna hanya bila melampaui angka ini.")

    labels_sorted = sorted(LABEL_MAP)
    target_names  = [LABEL_MAP[i] for i in labels_sorted]
    param_grid    = {"nb__alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}

    def build(estimator):
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                analyzer="word",
            )),
            ("nb", estimator),
        ])

    # MultinomialNB = metode utama sesuai judul; ComplementNB = pembanding.
    candidates = [
        ("MultinomialNB", MultinomialNB()),
        ("ComplementNB",  ComplementNB()),
    ]

    rows, artifacts = [], []
    for name, estimator in candidates:
        print("\n" + "=" * 55)
        print(f"MODEL: {name}")
        print("=" * 55)

        grid = GridSearchCV(
            build(estimator), param_grid, cv=5,
            scoring="f1_macro", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_alpha = grid.best_params_["nb__alpha"]
        print(f"Alpha terbaik (scoring=f1_macro): {best_alpha}")

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wgt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="macro", zero_division=0)
        cv_mean, cv_std = _cv_no_leak(build(estimator), X, y, random_state)

        print(f"\nAccuracy       : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"Macro-F1       : {f1_mac:.4f}   <- metrik utama (data timpang)")
        print(f"Weighted-F1    : {f1_wgt:.4f}")
        print(f"Macro-Precision: {prec:.4f}")
        print(f"Macro-Recall   : {rec:.4f}")
        print(f"CV-10 Macro-F1 : {cv_mean:.4f} +/- {cv_std:.4f}  (oversampling di dalam fold)")
        print(f"\nSelisih vs baseline: Acc {acc-base_acc:+.4f}  |  Macro-F1 {f1_mac-base_f1m:+.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, labels=labels_sorted,
            target_names=target_names, zero_division=0,
        ))

        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}\nAnalisis Sentimen Traveloka")
        plt.tight_layout()
        cm_path = os.path.join(
            RESULTS_DIR, f"confusion_matrix_{name.lower()}_{timestamp}.png"
        )
        plt.savefig(cm_path, dpi=150); plt.close()

        prefix = "mnb" if name == "MultinomialNB" else "cnb"
        mdl_path = os.path.join(MODELS_DIR, f"{prefix}_model_{timestamp}.pkl")
        with open(mdl_path, "wb") as f:
            pickle.dump(model, f)

        rows.append({
            "timestamp": timestamp, "model": name,
            "total_data": len(df), "train_size": len(X_train),
            "test_size": len(X_test), "best_alpha": best_alpha,
            "accuracy": round(acc, 4), "macro_f1": round(f1_mac, 4),
            "weighted_f1": round(f1_wgt, 4),
            "macro_precision": round(prec, 4), "macro_recall": round(rec, 4),
            "cv10_macro_f1_mean": round(cv_mean, 4),
            "cv10_macro_f1_std": round(cv_std, 4),
            "baseline_accuracy": round(base_acc, 4),
            "baseline_macro_f1": round(base_f1m, 4),
        })
        artifacts.append((name, mdl_path, cm_path))

    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print("PERBANDINGAN AKHIR")
    print("=" * 55)
    print(f"{'Model':<16}{'Accuracy':>10}{'Macro-F1':>11}{'Weighted-F1':>13}")
    print("-" * 50)
    print(f"{'Baseline':<16}{base_acc:>10.4f}{base_f1m:>11.4f}{'-':>13}")
    for r in rows:
        print(f"{r['model']:<16}{r['accuracy']:>10.4f}"
              f"{r['macro_f1']:>11.4f}{r['weighted_f1']:>13.4f}")

    print()
    for name, mdl_path, cm_path in artifacts:
        print(f"Model {name:<14}: {mdl_path}")
        print(f"  Confusion matrix : {cm_path}")
    print(f"Metrik            : {metrics_path}")

    # Model utama sesuai judul penelitian = MultinomialNB
    return artifacts[0][1]


# ──────────────────────────────────────────────────────────
#  STEP OPSIONAL — PREDIKSI FILE BARU
# ──────────────────────────────────────────────────────────

def step_predict(model_path, input_file):
    import pandas as pd

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n[PREDICT] PREDIKSI SENTIMEN")
    print("=" * 50)

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    if "stemmed_text" not in df.columns:
        sys.exit("ERROR: Kolom 'stemmed_text' tidak ditemukan di file input.")

    df["sentimen"]       = pipeline.predict(df["stemmed_text"])
    df["sentimen_label"] = df["sentimen"].map(LABEL_MAP)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output = os.path.join(RESULTS_DIR, f"{basename}_predicted.csv")
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Distribusi prediksi:")
    for lbl, cnt in df["sentimen_label"].value_counts().items():
        print(f"  {lbl}: {cnt}")
    print(f"\nHasil disimpan: {output}")


# ──────────────────────────────────────────────────────────
#  ARGPARSE & MAIN
# ──────────────────────────────────────────────────────────

def latest_file(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    # Urut berdasarkan waktu modifikasi: benar walau pola nama berbeda-beda
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Analisis Sentimen Traveloka — Multinomial Naive Bayes",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--scrape",     action="store_true", help="Scrape komentar Traveloka")
    parser.add_argument("--preprocess", action="store_true", help="Preprocessing teks")
    parser.add_argument("--autolabel",  action="store_true", help="Auto labeling dengan lexicon InSet")
    parser.add_argument("--sample-gold", nargs="?", type=int, const=200,
                        default=None, metavar="N",
                        help="Ambil N sampel untuk dilabeli MANUAL (default: 200)")
    parser.add_argument("--kappa",      action="store_true", help="Ukur Cohen's Kappa (auto vs manual)")
    parser.add_argument("--train",      action="store_true", help="Latih model MNB (+ CNB pembanding)")
    parser.add_argument("--all",        action="store_true", help="Scrape + Preprocess + AutoLabel + Train sekaligus")
    parser.add_argument("--predict",    action="store_true", help="Prediksi sentimen file baru")

    parser.add_argument("--max",          type=int,   default=100,  help="Tweet per query (default: 100)")
    parser.add_argument("--test-size",    type=float, default=0.2,  help="Rasio data uji (default: 0.2)")
    parser.add_argument("--threshold",    type=int,   default=NEUTRAL_THRESHOLD,
                        help=f"Ambang netral |skor| InSet (default: {NEUTRAL_THRESHOLD})")
    parser.add_argument("--input",        type=str,   default=None, help="File input CSV (manual)")
    parser.add_argument("--model",        type=str,   default=None, help="Path model .pkl untuk --predict")
    parser.add_argument("--labeled",      type=str,   default=None, help="File CSV berlabel untuk --train")
    parser.add_argument("--gold",         type=str,   default=None, help="File gold berlabel manual untuk --kappa")
    parser.add_argument("--force",        action="store_true",      help="Izinkan menimpa file berlabel yang sudah ada")
    parser.add_argument("--merge",        action="store_true",
                        help="Gabungkan SEMUA file di data/raw/ saat --preprocess")
    parser.add_argument("--headless",     action="store_true",
                        help="Jalankan browser tanpa tampilan (untuk server)")
    parser.add_argument("--no-lang-filter", action="store_true",
                        help="Jangan buang dokumen non-Indonesia (TIDAK disarankan)")

    args = parser.parse_args()

    if not any([args.scrape, args.preprocess, args.autolabel,
                args.sample_gold is not None, args.kappa, args.train,
                args.all, args.predict]):
        parser.print_help()
        print("\n" + COOKIE_HELP)
        return

    raw_file       = None
    processed_file = None
    model_path     = None

    # ── ALL = scrape + preprocess + autolabel + train ──
    if args.all:
        args.scrape = args.preprocess = args.autolabel = args.train = True

    # ── SCRAPE ──
    if args.scrape:
        raw_file = step_scrape(max_tweets=args.max, headless=args.headless)

    # ── PREPROCESS ──
    if args.preprocess:
        if args.merge:
            src = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "traveloka_raw_*.csv")))
            if not src:
                sys.exit("ERROR: Tidak ada file di data/raw/.")
            print(f"\nMode gabungan: {len(src)} file hasil scraping")
        else:
            src = raw_file or args.input or latest_file(RAW_DATA_DIR, "traveloka_raw_*.csv")
            if not src:
                sys.exit("ERROR: Tidak ada file raw. Jalankan --scrape dulu atau berikan --input.")
        processed_file = step_preprocess(src, filter_lang=not args.no_lang_filter)

    # ── AUTO LABEL ──
    if args.autolabel:
        src = processed_file or args.input or latest_file(PROCESSED_DIR, "*_processed.csv")
        if not src:
            sys.exit("ERROR: Tidak ada file processed. Jalankan --preprocess dulu atau berikan --input.")
        labeled_file = step_autolabel(src, threshold=args.threshold, force=args.force)
    else:
        labeled_file = None

    # ── SAMPEL GOLD untuk label manual ──
    if args.sample_gold is not None:
        src = args.labeled or labeled_file or latest_file(LABELED_DIR, "*_labeled.csv")
        if not src:
            sys.exit("ERROR: Tidak ada file berlabel. Jalankan --autolabel dulu.")
        step_sample_gold(src, n=args.sample_gold)
        return

    # ── COHEN'S KAPPA ──
    if args.kappa:
        src = args.gold or latest_file(GOLD_DIR, "gold_sample_*.csv")
        if not src:
            sys.exit("ERROR: Tidak ada file gold. Jalankan --sample-gold dulu.")
        step_kappa(src)
        return

    # ── TRAIN ──
    if args.train:
        src = args.labeled or labeled_file or latest_file(LABELED_DIR, "*.csv")
        if not src:
            sys.exit(
                "ERROR: Tidak ada file berlabel di data/labeled/.\n"
                "Buka file processed, tambahkan kolom 'label' (0/1/2), simpan di data/labeled/"
            )
        model_path = step_train(src, test_size=args.test_size)

    # ── PREDICT ──
    if args.predict:
        mdl = args.model or latest_file(MODELS_DIR, "mnb_model_*.pkl")
        src = args.input or processed_file or latest_file(PROCESSED_DIR, "*_processed.csv")
        if not mdl:
            sys.exit("ERROR: Model tidak ditemukan. Jalankan --train dulu atau berikan --model.")
        if not src:
            sys.exit("ERROR: File input tidak ditemukan. Berikan --input.")
        step_predict(mdl, src)

    print("\nSemua tahap selesai.")


if __name__ == "__main__":
    main()
