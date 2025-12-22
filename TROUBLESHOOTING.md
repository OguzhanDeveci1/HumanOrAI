# Troubleshooting Guide

## H2O Model Column Mismatch Hatası

### Sorun
```
Job failed: Test/Validation dataset has no columns in common with the training set
```

### Çözüm
Bu hata, H2O modellerinin beklediği sütun isimleriyle gönderilen veri sütunlarının eşleşmemesinden kaynaklanır.

**Yapılan Düzeltmeler:**

1. Model yüklenirken, modelin beklediği sütun isimleri `varimp()` fonksiyonu ile alınıyor
2. Tahmin yaparken, bu sütun isimleri kullanılarak DataFrame oluşturuluyor
3. Bu sayede eğitim ve tahmin sırasındaki sütun isimleri eşleşiyor

### Kod İyileştirmeleri

**Önce:**
```python
df = pd.DataFrame(tfidf_features, columns=[f'C{i}' for i in range(n_features)])
```

**Sonra:**
```python
# Model yüklenirken
varimp = self.models['DRF'].varimp(use_pandas=True)
self.h2o_column_names = varimp['variable'].tolist()

# Tahmin yaparken
df = pd.DataFrame(tfidf_features, columns=self.h2o_column_names[:n_features])
```

## Pickle Load Hatası

### Sorun
```
invalid load key, '\x10'
```

### Çözüm
TF-IDF vectorizer farklı bir Python versiyonuyla kaydedilmiş olabilir.

**Çözüm Kodu:**
```python
try:
    with open(tfidf_path, 'rb') as f:
        self.tfidf_vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading with pickle: {e}")
    print("Trying with joblib...")
    import joblib
    self.tfidf_vectorizer = joblib.load(tfidf_path)
```

## Model Yükleme Sorunları

### H2O Başlatma Hatası

**Sorun:** H2O başlatılamıyor

**Çözüm:**
```bash
# H2O'yu yeniden yükleyin
pip uninstall h2o
pip install h2o>=3.44.0.0
```

### BERT/RoBERTa Modelleri Bulunamıyor

**Sorun:** `model.safetensors` bulunamıyor

**Çözüm:**
- Modellerin `models/bert_model/` ve `models/roberta_model/` klasörlerinde olduğundan emin olun
- Gerekli dosyalar:
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`
  - `vocab.txt` (BERT için) veya `vocab.json` (RoBERTa için)

## Web Arayüzü Sorunları

### Flask Başlamıyor

**Sorun:** `ModuleNotFoundError: No module named 'flask'`

**Çözüm:**
```bash
pip install flask flask-cors
```

### Port Zaten Kullanımda

**Sorun:** `Address already in use`

**Çözüm:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# veya farklı bir port kullanın
python app.py --port 5001
```

## Performans İyileştirmeleri

### Yavaş Model Yükleme

**İyileştirme:**
- İlk request'te modeller yükleniyor (lazy loading)
- Sonraki request'ler hızlı

### H2O Cluster Kapatma

**Otomatik kapatma:**
```python
# Flask app sonlandığında
import atexit

@atexit.register
def cleanup():
    h2o.cluster().shutdown()
```

## Diagnostic Tool Kullanımı

### Model Dosyalarını Kontrol Etme

```bash
python check_models.py
```

Bu script:
- Tüm model dosyalarının varlığını kontrol eder
- TF-IDF vectorizer'ın yüklenebilir olduğunu test eder
- H2O modellerinin çalıştığını doğrular
- Hangi yükleme yönteminin (pickle/joblib) kullanılması gerektiğini gösterir

### H2O Model Detayları

```bash
python debug_h2o.py
```

Bu script:
- H2O modelinin beklediği sütun isimlerini gösterir
- Model özelliklerini listeler
- Variable importance bilgilerini gösterir

## Sık Karşılaşılan Hatalar

### 1. Out of Memory (OOM)

**Belirtiler:** Program çöküyor, CUDA OOM hatası

**Çözüm:**
```python
# CPU'da çalıştır
device = torch.device('cpu')
model.to(device)

# veya batch size'ı küçült
max_length = 256  # 512 yerine
```

### 2. Slow Predictions

**Belirtiler:** Her tahmin 30+ saniye sürüyor

**Çözüm:**
- H2O verbose mode'u kapatıldı: `h2o.init(verbose=False)`
- Model evaluation mode'u aktif: `model.eval()`
- Gradient hesaplama kapalı: `with torch.no_grad():`

### 3. Inconsistent Results

**Belirtiler:** Aynı metin farklı sonuçlar veriyor

**Çözüm:**
```python
# Deterministic mode
torch.manual_seed(42)
np.random.seed(42)

# Evaluation mode
model.eval()
```

## İletişim

Sorun devam ederse:
1. `check_models.py` çıktısını kontrol edin
2. Error log'larını kaydedin
3. Model dosyalarının tam olup olmadığını kontrol edin
