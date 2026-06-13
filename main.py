"""
main.py  —  Analisis Sentimen Komentar Traveloka di X
Menggunakan Metode Multinomial Naive Bayes

Opsi:
    --login          Simpan session login X (browser terbuka)
    --scrape         Scrape komentar Traveloka dari X
    --preprocess     Preprocessing teks (cleaning + stemming)
    --train          Latih model Multinomial Naive Bayes
    --all            Jalankan semua tahap sekaligus (kecuali --login)
    --predict        Prediksi sentimen file baru tanpa training ulang

Contoh:
    python main.py --login
    python main.py --scrape --max 150
    python main.py --all --max 150 --test-size 0.2
    python main.py --scrape --preprocess
    python main.py --predict --model models/mnb_model_xxx.pkl --input data/processed/xxx.csv
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
X_LOGIN_URL = "https://x.com/i/flow/login"
X_SEARCH_URL = "https://x.com/search"

SEARCH_QUERIES = [
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
    "traveloka refund",
    "traveloka reschedule",
    "traveloka aplikasi",
    "@traveloka",
]

SCROLL_PAUSE = 2.5
LANG_FILTER  = "lang:id"
LABEL_MAP    = {0: "negatif", 1: "netral", 2: "positif"}


# ──────────────────────────────────────────────────────────
#  STEP 1 — LOGIN & SIMPAN SESSION
# ──────────────────────────────────────────────────────────

def step_login():
    from playwright.sync_api import sync_playwright

    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    print("\n[STEP 1] LOGIN X — Simpan Session")
    print("=" * 50)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=50,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--start-maximized",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
            locale="id-ID",
        )

        # Hapus tanda webdriver agar tidak terdeteksi bot
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            window.chrome = { runtime: {} };
        """)

        page = context.new_page()
        page.goto("https://x.com", wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)
        page.goto(X_LOGIN_URL, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)

        print("\nBrowser sudah terbuka di halaman login X.")
        print("Silakan login secara manual (email → password → selesai).")
        print("Tunggu sampai beranda X muncul, lalu tekan ENTER di sini.\n")
        input(">> Tekan ENTER setelah beranda X tampil... ")

        cookies = context.cookies()
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2, ensure_ascii=False)

        browser.close()

    print(f"Session disimpan: {SESSION_FILE}")


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
        cookie = {
            "name":     c["name"],
            "value":    c["value"],
            "domain":   c["domain"],
            "path":     c.get("path", "/"),
            "httpOnly": c.get("httpOnly", False),
            "secure":   c.get("secure", False),
            "sameSite": SAMESITE_MAP.get(
                str(c.get("sameSite", "None")).lower(), "None"
            ),
        }
        # expirationDate (Chrome ext) atau expires (Playwright)
        exp = c.get("expirationDate") or c.get("expires")
        if exp:
            cookie["expires"] = int(exp)
        result.append(cookie)
    return result


def _load_cookies(context):
    if not os.path.exists(SESSION_FILE):
        sys.exit(
            "ERROR: Session tidak ditemukan. Jalankan dulu:\n"
            "  python main.py --login"
        )
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    context.add_cookies(_normalize_cookies(raw))


def _parse_count(text):
    if not text:
        return 0
    text = text.strip().replace(",", ".")
    for suffix, mult in [("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000)]:
        if text.upper().endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except ValueError:
                return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _extract_id(url):
    m = re.search(r"/status/(\d+)", url or "")
    return m.group(1) if m else ""


def _scrape_query(page, query, max_tweets):
    from playwright.sync_api import TimeoutError as PwTimeout
    from tqdm import tqdm

    url = f"{X_SEARCH_URL}?q={quote(f'{query} {LANG_FILTER}')}&src=typed_query&f=live"
    print(f"\n  Query: {query}")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except PwTimeout:
        print("  [TIMEOUT] Dilewati.")
        return []
    time.sleep(3)

    collected = {}
    no_new = 0

    with tqdm(total=max_tweets, desc="  Terkumpul", unit="tw", leave=False) as pbar:
        while len(collected) < max_tweets:
            articles = page.query_selector_all('article[data-testid="tweet"]')
            new = 0
            for art in articles:
                try:
                    text_el = art.query_selector('[data-testid="tweetText"]')
                    text = text_el.inner_text() if text_el else ""
                    if not text:
                        continue
                    link_el = art.query_selector('a[href*="/status/"]')
                    tid = _extract_id(link_el.get_attribute("href") if link_el else "")
                    if not tid or tid in collected:
                        continue
                    user_el = art.query_selector('[data-testid="User-Name"]')
                    username = ""
                    if user_el:
                        for sp in user_el.query_selector_all("span"):
                            t = sp.inner_text().strip()
                            if t.startswith("@"):
                                username = t
                                break
                    time_el = art.query_selector("time")
                    date = time_el.get_attribute("datetime") if time_el else ""

                    def metric(tid_):
                        el = art.query_selector(f'[data-testid="{tid_}"]')
                        return _parse_count(el.inner_text()) if el else 0

                    collected[tid] = {
                        "id": tid, "username": username, "date": date,
                        "text": text, "likes": metric("like"),
                        "retweets": metric("retweet"), "replies": metric("reply"),
                        "query": query,
                    }
                    new += 1
                    pbar.update(1)
                    if len(collected) >= max_tweets:
                        break
                except Exception:
                    continue

            if new == 0:
                no_new += 1
                if no_new >= 5:
                    break
            else:
                no_new = 0
            page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            time.sleep(SCROLL_PAUSE)

    return list(collected.values())


def step_scrape(max_tweets):
    import pandas as pd
    from playwright.sync_api import sync_playwright

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(RAW_DATA_DIR, f"traveloka_raw_{timestamp}.csv")

    print("\n[STEP 2] SCRAPING KOMENTAR TRAVELOKA")
    print("=" * 50)
    print(f"Total query : {len(SEARCH_QUERIES)}")
    print(f"Max/query   : {max_tweets} tweet")

    all_tweets, seen = [], set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, slow_mo=30,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        _load_cookies(context)
        page = context.new_page()

        print("\nMemeriksa session...")
        page.goto(X_BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(2)
        if "login" in page.url.lower():
            browser.close()
            sys.exit("Session expired! Jalankan: python main.py --login")
        print("Session valid.\n")

        for q in SEARCH_QUERIES:
            try:
                tweets = _scrape_query(page, q, max_tweets)
                for tw in tweets:
                    if tw["id"] not in seen:
                        seen.add(tw["id"])
                        all_tweets.append(tw)
                print(f"  Subtotal unik: {len(all_tweets)}")
            except Exception as e:
                print(f"  [ERROR] {q}: {e}")
            time.sleep(3)

        browser.close()

    cols = ["id", "username", "date", "text", "likes", "retweets", "replies", "query"]
    df = pd.DataFrame(all_tweets, columns=cols)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\nSelesai scraping: {len(df)} tweet unik")
    print(f"Tersimpan       : {output}")
    return output


# ──────────────────────────────────────────────────────────
#  STEP 3 — PREPROCESSING
# ──────────────────────────────────────────────────────────

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
        "aja","saja","banget","bgt","mau","mw","ga","gak","nggak","ngga",
        "nda","ndak","tidak","tdk","enggak","emang","memang","dong","sih",
        "deh","nih","loh","lah","kan","ya","yaa","haha","hehe","wkwk",
        "wkwkwk","hm","hmm","eh","ah","oh","oke","ok","iya","iyaa",
        "sampe","kayak","kaya","kek","bisa","bs","tapi","tp","trus","terus",
        "habis","abis","lagi","lg","pake","pakai","sama","sm","ke","di",
        "dari","dan","atau","dengan","untuk","dlm","dalam","kalau","klo",
        "kalo","lebih","udah","sudah",
    }
    return stemmer, sw_base | sw_extra


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


def step_preprocess(input_file):
    import pandas as pd
    from nltk.tokenize import word_tokenize
    from tqdm import tqdm

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stemmer, stopwords = _init_nlp()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output = os.path.join(PROCESSED_DIR, f"{basename}_processed.csv")

    print("\n[STEP 3] PREPROCESSING TEKS")
    print("=" * 50)
    print(f"Input : {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"Total : {len(df)} baris")

    tqdm.pandas(desc="  Cleaning")
    df["clean_text"] = df["text"].progress_apply(_clean)

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
#  STEP 3b — AUTO LABELING (keyword-based)
# ──────────────────────────────────────────────────────────

KEYWORDS_POSITIF = {
    # Indonesia
    "bagus", "baik", "mantap", "keren", "hebat", "luar biasa", "memuaskan",
    "puas", "senang", "suka", "cinta", "terbaik", "mudah", "gampang",
    "cepat", "lancar", "aman", "nyaman", "murah", "terjangkau", "hemat",
    "promo", "diskon", "untung", "berhasil", "sukses", "terima kasih",
    "makasih", "membantu", "teratasi", "beres", "mantul", "worth",
    "aplikasi bagus", "pelayanan bagus", "responsif", "ramah", "profesional",
    "rekomen", "rekomendasi", "top notch", "luar biasa", "memuaskan",
    "menyenangkan", "sempurna", "terpercaya", "andalan", "favorite",
    "favorit", "praktis", "efisien", "canggih", "inovatif", "terjamin",
    # Inggris
    "good", "great", "nice", "best", "love", "recommended", "helpful",
    "solved", "thanks", "thank you", "wow", "amazing", "excellent",
    "satisfied", "happy", "awesome", "fantastic", "perfect", "wonderful",
    "brilliant", "outstanding", "superb", "smooth", "fast", "easy",
    "convenient", "reliable", "trustworthy", "affordable", "worth it",
    "highly recommend", "five star", "5 star", "well done", "impressed",
    "no problem", "works well", "works fine", "love it", "great app",
    "good service", "quick response", "professional",
}

KEYWORDS_NEGATIF = {
    # Indonesia
    "buruk", "jelek", "parah", "kecewa", "mengecewakan", "gagal", "error",
    "bug", "crash", "lemot", "lambat", "lama", "mahal", "kemahalan",
    "ribet", "susah", "sulit", "tidak bisa", "gabisa", "nggak bisa",
    "tidak jelas", "bingung", "tipu", "bohong", "penipuan", "cancel",
    "dibatalkan", "hangus", "hilang", "kehilangan", "rugi", "komplain",
    "keluhan", "klaim", "masalah", "kendala", "gangguan", "tidak berfungsi",
    "tidak aktif", "ditipu", "menipu", "tertipu", "kesal", "marah",
    "sebal", "jengkel", "minta tolong", "tolong", "bantuan", "urgent",
    "lambat sekali", "tidak respon", "tidak responsif", "uang tidak kembali",
    "uang hilang", "nggak bisa", "ga bisa", "gak bisa", "zonk", "kapok",
    "nyesel", "menyesal", "percuma", "sia sia", "gagal terus", "down terus",
    "tidak profesional", "mengecewakan", "tidak memuaskan", "tidak aman",
    "berbahaya", "merugikan", "penipuan", "bohong", "dibohongi",
    # Inggris
    "bad", "worst", "horrible", "terrible", "awful", "disappointed",
    "frustrating", "frustrated", "scam", "fraud", "refund", "chargeback",
    "not working", "doesn't work", "not work", "broken", "useless",
    "waste", "waste of money", "waste of time", "slow", "laggy",
    "problem", "issue", "complaint", "terrible service", "poor service",
    "bad service", "no response", "unresponsive", "ignored", "rude",
    "unprofessional", "misleading", "fake", "lie", "lied", "cheated",
    "money lost", "not received", "never arrived", "cancelled",
    "disappointed", "regret", "avoid", "do not use", "don't use",
    "beware", "warning", "negative", "awful experience", "worst app",
    "terrible app", "not recommended", "zero star", "1 star", "one star",
}

def _autolabel_text(text: str) -> int:
    t = str(text).lower()
    pos = sum(1 for k in KEYWORDS_POSITIF if k in t)
    neg = sum(1 for k in KEYWORDS_NEGATIF if k in t)
    if pos > neg:
        return 2
    elif neg > pos:
        return 0
    else:
        return 1


def step_autolabel(input_file):
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(LABELED_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    # Hilangkan suffix _processed jika ada
    basename = basename.replace("_processed", "")
    output = os.path.join(LABELED_DIR, f"{basename}_labeled.csv")

    print("\n[STEP 3b] AUTO LABELING")
    print("=" * 50)
    print(f"Input : {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Gunakan clean_text jika ada, fallback ke text asli
    src_col = "clean_text" if "clean_text" in df.columns else "text"

    tqdm.pandas(desc="  Labeling")
    df["label"] = df[src_col].progress_apply(_autolabel_text)

    dist = df["label"].value_counts().sort_index()
    label_names = {0: "negatif", 1: "netral", 2: "positif"}
    print("\nDistribusi label otomatis:")
    for lbl, cnt in dist.items():
        print(f"  {label_names[lbl]}: {cnt} ({cnt/len(df)*100:.1f}%)")

    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nFile berlabel disimpan: {output}")
    print("Silakan cek & koreksi kolom 'label' yang kurang tepat sebelum training.")
    return output


# ──────────────────────────────────────────────────────────
#  STEP 4 — TRAINING MNB
# ──────────────────────────────────────────────────────────

def step_train(input_file, test_size=0.2, random_state=42):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.utils import resample

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

    # ── Oversampling: samakan jumlah tiap kelas ke kelas terbanyak ──
    max_count = df["label"].value_counts().max()
    parts = []
    for lbl in df["label"].unique():
        subset = df[df["label"] == lbl]
        if len(subset) < max_count:
            subset = resample(subset, replace=True, n_samples=max_count, random_state=random_state)
        parts.append(subset)
    df_balanced = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nSetelah oversampling: {len(df_balanced)}")
    for lbl, cnt in df_balanced["label"].value_counts().sort_index().items():
        print(f"  {LABEL_MAP.get(lbl, lbl)}: {cnt}")

    X, y = df_balanced["stemmed_text"], df_balanced["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nData latih: {len(X_train)}  |  Data uji: {len(X_test)}")

    # ── GridSearch: cari alpha terbaik ──
    print("\nMencari parameter terbaik (GridSearch)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("mnb", ComplementNB()),
    ])
    param_grid = {"mnb__alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_alpha = grid.best_params_["mnb__alpha"]
    pipeline   = grid.best_estimator_
    print(f"Alpha terbaik: {best_alpha}")

    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cv   = cross_val_score(pipeline, X, y, cv=10, scoring="accuracy")

    print("\nHASIL EVALUASI")
    print("-" * 40)
    print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"CV-10 Acc : {cv.mean():.4f} ± {cv.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABEL_MAP))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        yticklabels=[LABEL_MAP[i] for i in sorted(LABEL_MAP)], ax=ax
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Multinomial Naive Bayes\nAnalisis Sentimen Traveloka")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, dpi=150); plt.close()

    # Simpan model
    model_path = os.path.join(MODELS_DIR, f"mnb_model_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Simpan metrik
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{timestamp}.csv")
    pd.DataFrame([{
        "timestamp": timestamp, "total_data": len(df),
        "train_size": len(X_train), "test_size": len(X_test),
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1_score": round(f1, 4),
        "cv_mean": round(cv.mean(), 4), "cv_std": round(cv.std(), 4),
    }]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"Model           : {model_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"Metrik          : {metrics_path}")
    return model_path


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
    files = sorted(glob.glob(os.path.join(directory, pattern)), reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(
        description="Analisis Sentimen Traveloka — Multinomial Naive Bayes",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--login",      action="store_true", help="Login X dan simpan session")
    parser.add_argument("--scrape",     action="store_true", help="Scrape komentar Traveloka")
    parser.add_argument("--preprocess", action="store_true", help="Preprocessing teks")
    parser.add_argument("--autolabel",  action="store_true", help="Auto labeling berbasis keyword")
    parser.add_argument("--train",      action="store_true", help="Latih model MNB")
    parser.add_argument("--all",        action="store_true", help="Scrape + Preprocess + AutoLabel + Train sekaligus")
    parser.add_argument("--predict",    action="store_true", help="Prediksi sentimen file baru")

    parser.add_argument("--max",          type=int,   default=100,  help="Tweet per query (default: 100)")
    parser.add_argument("--test-size",    type=float, default=0.2,  help="Rasio data uji (default: 0.2)")
    parser.add_argument("--input",        type=str,   default=None, help="File input CSV (manual)")
    parser.add_argument("--model",        type=str,   default=None, help="Path model .pkl untuk --predict")
    parser.add_argument("--labeled",      type=str,   default=None, help="File CSV berlabel untuk --train")

    args = parser.parse_args()

    if not any([args.login, args.scrape, args.preprocess, args.autolabel, args.train, args.all, args.predict]):
        parser.print_help()
        return

    raw_file       = None
    processed_file = None
    model_path     = None

    # ── LOGIN ──
    if args.login:
        step_login()
        return

    # ── ALL = scrape + preprocess + autolabel + train ──
    if args.all:
        args.scrape = args.preprocess = args.autolabel = args.train = True

    # ── SCRAPE ──
    if args.scrape:
        raw_file = step_scrape(max_tweets=args.max)

    # ── PREPROCESS ──
    if args.preprocess:
        src = raw_file or args.input or latest_file(RAW_DATA_DIR, "traveloka_raw_*.csv")
        if not src:
            sys.exit("ERROR: Tidak ada file raw. Jalankan --scrape dulu atau berikan --input.")
        processed_file = step_preprocess(src)

    # ── AUTO LABEL ──
    if args.autolabel:
        src = processed_file or args.input or latest_file(PROCESSED_DIR, "*_processed.csv")
        if not src:
            sys.exit("ERROR: Tidak ada file processed. Jalankan --preprocess dulu atau berikan --input.")
        labeled_file = step_autolabel(src)
    else:
        labeled_file = None

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
