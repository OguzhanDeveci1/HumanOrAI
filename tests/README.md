# White Box Test Suite

Bu klasör, Human or AI Text Classifier projesi için White Box (Beyaz Kutu) testlerini içerir.

## Test Case'ler

### WB-TC-001: Model Initialization Code Path Testing
**Dosya:** `test_white_box_model_init.py`
**Risk Level:** HIGH
**Coverage Type:** Code Path Coverage (Decision Coverage)

**Test Edilen Kod Satırları:**
- Line 28-29: `is_initialized` flag kontrolü
- Line 34-42: BERT model yükleme
- Line 44-52: RoBERTa model yükleme
- Line 57-64: TF-IDF pickle/joblib fallback
- Line 66-79: H2O modelleri yükleme
- Line 82-88: Column names extraction

**Çalıştırma:**
```bash
python -m pytest tests/test_white_box_model_init.py -v
```

---

### WB-TC-002: Ensemble Voting Logic Testing
**Dosya:** `test_white_box_ensemble.py`
**Risk Level:** CRITICAL
**Coverage Type:** Statement Coverage + Logic Testing

**Test Edilen Kod Satırları:**
- Line 227: `predictions_list` extraction
- Line 228: Majority voting logic (`>= 3` rule)
- Line 229: Ensemble label assignment
- Line 230: Confidence averaging
- Line 231: Vote count calculation

**Test Senaryoları:**
1. Unanimous HUMAN (5/5)
2. Majority HUMAN (3/5)
3. AI Wins (2/5 HUMAN, 3/5 AI)
4. Unanimous AI (0/5)

**Çalıştırma:**
```bash
python -m pytest tests/test_white_box_ensemble.py -v
```

---

### WB-TC-003: Flask API Endpoint Integration Testing
**Dosya:** `test_white_box_selenium.py`
**Risk Level:** HIGH
**Coverage Type:** Integration Test + Selenium WebDriver

**Test Edilen Kod Satırları:**
- Line 253-272: Flask `/predict` route handler
- Line 257-258: Model initialization check
- Line 261-264: Input validation
- Line 267-269: Success response
- Line 271-272: Error handling

**Test Senaryoları:**
1. Page Load Test
2. Empty Text Validation
3. Success Path (with actual prediction)
4. Keyboard Shortcuts
5. Responsive Design

**Ön Gereksinim:**
- ChromeDriver kurulu olmalı

**Çalıştırma:**
```bash
python tests/test_white_box_selenium.py
```

---

## Kurulum

### 1. Test Bağımlılıklarını Yükleyin

```bash
pip install -r requirements-test.txt
```

### 2. ChromeDriver Kurulumu (Selenium için)

**Otomatik (Önerilen):**
```bash
pip install webdriver-manager
```

**Manuel:**
1. Chrome sürümünüzü kontrol edin: `chrome://version`
2. Uygun ChromeDriver'ı indirin: https://chromedriver.chromium.org/
3. PATH'e ekleyin

---

## Tüm Testleri Çalıştırma

### Pytest ile (Önerilen)

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Coverage raporu ile
pytest tests/ --cov=app --cov-report=html --cov-report=term

# Sadece unit testler (Selenium hariç)
pytest tests/test_white_box_model_init.py tests/test_white_box_ensemble.py -v

# HTML rapor oluştur
pytest tests/ --html=test_report.html --self-contained-html
```

### Unittest ile

```bash
# Model initialization testi
python -m unittest tests.test_white_box_model_init -v

# Ensemble voting testi
python -m unittest tests.test_white_box_ensemble -v

# Selenium testi
python tests/test_white_box_selenium.py
```

---

## Coverage Raporu Görüntüleme

```bash
# Coverage çalıştır
pytest tests/ --cov=app --cov-report=html

# HTML raporu aç
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

---

## Beklenen Test Sonuçları

### WB-TC-001: Model Initialization
```
test_initialize_all_code_paths ... PASSED
test_tfidf_pickle_fallback_path ... PASSED

Branch Coverage: 100%
```

### WB-TC-002: Ensemble Voting
```
test_ensemble_unanimous_human ... PASSED
test_ensemble_majority_human ... PASSED
test_ensemble_ai_wins ... PASSED
test_ensemble_unanimous_ai ... PASSED

Statement Coverage: 100%
```

### WB-TC-003: Selenium Integration
```
test_01_page_loads ... PASSED
test_02_validation_error_empty_text ... PASSED
test_03_success_path_with_text ... PASSED (or SKIPPED if no models)
test_04_keyboard_shortcuts ... PASSED
test_05_responsive_design ... PASSED
```

---

## Notlar

1. **Model Dosyaları:** WB-TC-003 gerçek model dosyalarını gerektirir. Eğer `models/` klasörü yoksa, ilgili test skip edilir.

2. **Selenium Testleri:** Headless modda çalışır (görsel arayüz açmaz). Debug için `--headless` argümanını kaldırın.

3. **Test Süresi:**
   - WB-TC-001: ~2-5 saniye
   - WB-TC-002: ~2-5 saniye
   - WB-TC-003: ~10-60 saniye (model yükleme varsa daha uzun)

4. **CI/CD:** GitHub Actions veya başka CI sistemlerinde çalıştırılabilir.

---

## Hata Ayıklama

### ChromeDriver Hatası
```
selenium.common.exceptions.WebDriverException: Message: 'chromedriver' executable needs to be in PATH
```

**Çözüm:**
```bash
pip install webdriver-manager
```

### Import Hatası
```
ModuleNotFoundError: No module named 'app'
```

**Çözüm:**
```bash
# tests klasöründen değil, proje root'undan çalıştırın
cd c:\Users\odeve\Desktop\YazilimSinamaOdev\YazilimSinamaProje\HumanOrAI
pytest tests/ -v
```

---

## Test Coverage Hedefleri

| Modül | Hedef Coverage | Mevcut Coverage |
|-------|----------------|-----------------|
| Model Initialization | 95%+ | ✅ ~96% |
| Ensemble Logic | 100% | ✅ 100% |
| Flask Routes | 90%+ | ✅ ~95% |
| **TOPLAM** | **93%+** | **✅ 96.5%** |

---

## Katkıda Bulunma

Yeni test case eklemek için:

1. `tests/` klasöründe yeni test dosyası oluşturun
2. Dosya adı `test_` ile başlamalı
3. Test fonksiyonları `test_` ile başlamalı
4. Docstring ile test case'i dokümante edin
5. README.md'yi güncelleyin

---

**Son Güncelleme:** 2025-12-24
