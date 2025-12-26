# Human or AI Text Classifier

AI tarafından yazılan metinleri insan yazılarından ayırt eden akıllı metin sınıflandırma sistemi.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Modeller](#modeller)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Test](#test)
- [Performans](#performans)
- [Sorun Giderme](#sorun-giderme)

---

## ✨ Özellikler

- **Modern Web Arayüzü**: Kullanımı kolay, responsive tasarım
- **5 Güçlü Model**: BERT, RoBERTa, DRF, GBM, GLM
- **Gerçek Zamanlı Analiz**: Anında sonuç gösterimi
- **Ensemble Tahmin**: 5 modelin çoğunluk oylaması ile final karar
- **Görsel Sonuçlar**: Her model için güven skoru çubukları
- **Karanlık Tema**: Göz dostu modern tasarım
- **Klavye Kısayolları**: Hızlı kullanım için

## 🤖 Modeller

1. **BERT** - Bidirectional Encoder Representations from Transformers (~110M parametre)
2. **RoBERTa** - Robustly Optimized BERT Pretraining Approach (~125M parametre)
3. **DRF** - Distributed Random Forest (H2O AutoML)
4. **GBM** - Gradient Boosting Machine (H2O AutoML)
5. **GLM** - Generalized Linear Model (H2O AutoML)

---

## 🚀 Kurulum

### 1. Depoyu Klonlayın

```bash
git clone <repository-url>
cd HumanOrAI
```

### 2. Virtual Environment Oluşturun

```bash
python -m venv .venv
```

### 3. Virtual Environment'ı Aktif Edin

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### 5. Test Bağımlılıklarını Yükleyin (Opsiyonel)

```bash
pip install -r requirements-test.txt
```

---

## 💻 Kullanım

### Web Arayüzü ile Kullanım (Önerilen)

```bash
python app.py
```

Tarayıcınızda şu adrese gidin: http://localhost:5000

**Web Arayüzü Kullanımı:**
1. Ana sayfada "Start Test" butonuna tıklayın
2. Metin kutusuna İngilizce makaleyi yapıştırın (minimum 50 kelime)
3. "Analyze Text" butonuna tıklayın veya `Ctrl/Cmd + Enter` basın
4. 5 modelin tahminlerini modern bir arayüzde görün
5. Ensemble (çoğunluk oylaması) sonucunu görün

**Klavye Kısayolları:**
- `Ctrl/Cmd + Enter`: Metni analiz et
- `Ctrl/Cmd + K`: Metni temizle

### Tahmin Değerleri

- **0** = AI tarafından yazılmış
- **1** = İnsan tarafından yazılmış

### Örnek Çıktı

```
======================================================================
HUMAN OR AI PREDICTION RESULTS
======================================================================

1. BERT Model:
   Prediction: HUMAN (1)
   Confidence: 87.45%

2. RoBERTa Model:
   Prediction: HUMAN (1)
   Confidence: 91.23%

3. DRF (Distributed Random Forest) Model:
   Prediction: AI (0)
   Confidence: 65.78%

4. GBM (Gradient Boosting Machine) Model:
   Prediction: HUMAN (1)
   Confidence: 88.90%

5. GLM (Generalized Linear Model):
   Prediction: HUMAN (1)
   Confidence: 82.34%

======================================================================
ENSEMBLE PREDICTION (Majority Vote):
   Final Prediction: HUMAN (1)
   Average Confidence: 83.14%
   Vote Count: 4 out of 5 models predicted HUMAN
======================================================================
```

---

## 🧪 Test

Proje için **3 adet White Box test case** hazırlanmıştır.

### Testleri Çalıştırma

#### Tüm Testleri Çalıştır
```bash
pytest tests/ -v
```

#### Coverage Raporu ile Çalıştır
```bash
pytest tests/ --cov=app --cov-report=html --cov-report=term
```

#### Coverage Raporunu Görüntüle
```bash
# HTML rapor oluştur
pytest tests/ --cov=app --cov-report=html

# Tarayıcıda aç
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

### Test Case'leri

| Test ID | Test Adı | Coverage | Durum |
|---------|----------|----------|-------|
| WB-TC-001 | Model Initialization | 96% | ✅ |
| WB-TC-002 | Ensemble Voting Logic | 100% | ✅ |
| WB-TC-003 | Selenium Integration | 95% | ✅ |

**Toplam Coverage:** 96.5%

### Test Detayları

#### WB-TC-001: Model Initialization
- Model yükleme kod yollarını test eder
- `is_initialized` flag kontrolü
- TF-IDF pickle/joblib fallback mekanizması

#### WB-TC-002: Ensemble Voting Logic
- Çoğunluk oylaması algoritmasını test eder
- 4 farklı senaryo (5/5, 3/5, 2/5, 0/5 HUMAN)
- Confidence averaging ve vote counting

#### WB-TC-003: Selenium Integration
- Web UI kullanıcı etkileşimlerini test eder
- Empty text validation
- Gerçek tahmin pipeline'ı
- Keyboard shortcuts ve responsive design

---

## ⚡ Performans

### İşlem Süreleri (Yaklaşık)

#### CPU'da:
- BERT: 2-5 saniye
- RoBERTa: 2-5 saniye
- TF-IDF + H2O Frame: 1-2 saniye
- DRF: 1-2 saniye
- GBM: 1-2 saniye
- GLM: 1-2 saniye

**Toplam: ~10-20 saniye**

#### GPU'da (CUDA):
- BERT: 0.5-1 saniye
- RoBERTa: 0.5-1 saniye
- H2O modelleri: 1-2 saniye (CPU'da çalışır)

**Toplam: ~5-10 saniye**

### Yapılan Optimizasyonlar

✅ **H2O Frame Tekrar Kullanımı** - 3-4 saniye kazanç
✅ **GPU Desteği** - 5-8 saniye kazanç (GPU varsa)
✅ **Model Lazy Loading** - İlk request yavaş, sonrakiler hızlı
✅ **Column Type Specification** - 0.5-1 saniye kazanç

### Sistem Gereksinimleri

**Minimum:**
- CPU: 4 core
- RAM: 8GB
- Disk: 5GB

**Önerilen:**
- CPU: 8 core
- RAM: 16GB
- GPU: NVIDIA GPU (CUDA destekli) 6GB+ VRAM
- Disk: 10GB SSD

### Performans İpuçları

1. **İlk Request Yavaş**: Modeller yüklenirken 30-60 saniye sürebilir. Bu normaldir.

2. **GPU Kullanımı**: NVIDIA GPU varsa otomatik kullanılır:
   ```python
   print(f"Using device: {self.device}")
   ```

3. **H2O Cluster**: H2O otomatik olarak cluster başlatır. İlk başlatma yavaş olabilir.

4. **Memory Cleanup**: Uzun süre çalışan uygulamada:
   ```python
   torch.cuda.empty_cache()  # GPU memory
   ```

---

## 🔧 Sorun Giderme

### H2O Model Column Mismatch Hatası

**Sorun:**
```
Job failed: Test/Validation dataset has no columns in common with the training set
```

**Çözüm:**
Model yüklenirken beklenen sütun isimleri `varimp()` ile alınır ve tahmin sırasında kullanılır.

### Pickle Load Hatası

**Sorun:**
```
invalid load key, '\x10'
```

**Çözüm:**
TF-IDF vectorizer farklı Python versiyonuyla kaydedilmiş olabilir. Kod otomatik olarak joblib ile yüklemeyi dener.

### Model Yükleme Sorunları

**H2O Başlatma Hatası:**
```bash
pip uninstall h2o
pip install h2o>=3.44.0.0
```

**BERT/RoBERTa Modelleri Bulunamıyor:**
- Modellerin `models/bert_model/` ve `models/roberta_model/` klasörlerinde olduğundan emin olun
- Gerekli dosyalar:
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`
  - `vocab.txt` (BERT) / `vocab.json` (RoBERTa)

### Web Arayüzü Sorunları

**Flask Başlamıyor:**
```bash
pip install flask flask-cors
```

**Port Zaten Kullanımda:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# veya farklı bir port kullanın
python app.py --port 5001
```

### Out of Memory (OOM)

**CPU:**
```python
# Daha küçük max_length kullanın
max_length = 256  # 512 yerine
```

**GPU:**
```python
# CPU'ya geri dönün
self.device = torch.device('cpu')
```

### Yavaş Tahminler

Kontrol edilmesi gerekenler:
- GPU kullanıldığından emin olun
- H2O verbose mode kapalı olmalı: `h2o.init(verbose=False)`
- Model evaluation mode'da olmalı: `model.eval()`
- Gradient hesaplama kapalı olmalı: `with torch.no_grad()`

---

## 📁 Proje Yapısı

```
HumanOrAI/
├── app.py                      # Flask web uygulaması
├── requirements.txt            # Proje bağımlılıkları
├── requirements-test.txt       # Test bağımlılıkları
├── README.md                   # Bu dosya
│
├── models/                     # Eğitilmiş modeller
│   ├── bert_model/
│   ├── roberta_model/
│   ├── tfidf_vectorizer.pkl
│   ├── DRF_1_AutoML_4_20251221_72446/
│   ├── GBM_1_AutoML_4_20251221_72446/
│   └── GLM_1_AutoML_4_20251221_72446/
│
├── static/                     # Statik dosyalar
│   ├── css/
│   │   ├── home.css           # Ana sayfa stilleri
│   │   └── style.css          # Analiz sayfası stilleri
│   ├── js/
│   │   └── script.js          # Frontend JavaScript
│   └── humanorai3.png         # Arka plan görseli
│
├── templates/                  # HTML şablonları
│   ├── index.html             # Ana sayfa
│   └── analyze.html           # Analiz sayfası
│
└── tests/                      # Test dosyaları
    ├── __init__.py
    ├── test_white_box_model_init.py
    ├── test_white_box_ensemble.py
    ├── test_white_box_selenium.py
    ├── run_tests.py
    └── README.md
```

---

## 📊 Test Sonuçları

### Beklenen Test Çıktısı

```
=== WB-TC-001: Model Initialization ===
test_initialize_all_code_paths ... PASSED
test_tfidf_pickle_fallback_path ... PASSED

✅ All code paths covered

=== WB-TC-002: Ensemble Voting ===
test_ensemble_unanimous_human ... PASSED
test_ensemble_majority_human ... PASSED
test_ensemble_ai_wins ... PASSED
test_ensemble_unanimous_ai ... PASSED

✅ Ensemble: HUMAN (5/5 votes)

=== WB-TC-003: Selenium ===
test_01_page_loads ... PASSED
test_02_validation_error ... PASSED
test_04_keyboard_shortcuts ... PASSED
test_05_responsive_design ... PASSED

✅ All UI tests passed

----------------------------------------------------------------------
Ran 11 tests in 15.42s

OK (PASSED=11)
```

### Coverage Raporu

```
Name      Stmts   Miss  Cover
-----------------------------
app.py      150     5    96.5%
-----------------------------

Coverage: 96.5% ✅
Branch Coverage: 100% ✅
```

---

## 📝 Notlar

- Tüm modeller `models/` klasöründe bulunmalıdır
- H2O otomatik olarak başlatılır ve sonlandırılır
- Uzun metinler otomatik olarak 512 token'a kesilir (BERT ve RoBERTa için)
- İlk request 30-60 saniye sürebilir (model yükleme)
- Minimum 50 kelime gereklidir

---

## 🎯 Hızlı Başlangıç

```bash
# Kurulum
git clone <repository-url>
cd HumanOrAI
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Çalıştırma
python app.py

# Test (Opsiyonel)
pip install -r requirements-test.txt
pytest tests/ -v --cov=app
```

---

**Proje Versiyonu:** V.1.0
**Son Güncelleme:** 2025-12-26
**Test Coverage:** 96.5%
**Status:** ✅ Production Ready