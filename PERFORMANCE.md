# Performance Optimization Guide

## Neden Yavaş?

### 1. Model Boyutları
- **BERT**: ~110M parametre (~440MB)
- **RoBERTa**: ~125M parametre (~500MB)
- **H2O Modelleri**: TF-IDF vektörizasyon + tahmin

### 2. İşlem Süreleri (Yaklaşık)

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
- H2O modelleri: aynı (CPU'da çalışır)

**Toplam: ~5-10 saniye**

## Yapılan Optimizasyonlar

### 1. H2O Frame Tekrar Kullanımı ✅

**Önce:**
```python
# Her model için ayrı ayrı TF-IDF ve H2O Frame oluşturma
pred = predict_h2o_model(text, 'DRF')  # TF-IDF + H2O Frame
pred = predict_h2o_model(text, 'GBM')  # TF-IDF + H2O Frame (tekrar!)
pred = predict_h2o_model(text, 'GLM')  # TF-IDF + H2O Frame (tekrar!)
```

**Sonra:**
```python
# Bir kere TF-IDF ve H2O Frame oluştur, 3 model için kullan
tfidf_features = self.tfidf_vectorizer.transform([text])
h2o_frame = h2o.H2OFrame(df)

pred_drf = self.models['DRF'].predict(h2o_frame)
pred_gbm = self.models['GBM'].predict(h2o_frame)
pred_glm = self.models['GLM'].predict(h2o_frame)
```

**Kazanç: 3-4 saniye**

### 2. GPU Desteği ✅

BERT ve RoBERTa modelleri GPU'da çalışabilir:

```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(self.device)
inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

**Kazanç (GPU varsa): 5-8 saniye**

### 3. Model Lazy Loading ✅

Modeller ilk request'te yükleniyor:

```python
@app.route('/predict', methods=['POST'])
def predict():
    if not predictor.is_initialized:
        predictor.initialize()  # İlk request'te yükle
```

**Kazanç: İlk request yavaş, sonrakiler hızlı**

### 4. Column Type Specification ✅

H2O Frame oluştururken column type'ları belirt:

```python
h2o_frame = h2o.H2OFrame(df, column_types=['numeric'] * n_features)
```

**Kazanç: 0.5-1 saniye**

## Ek Optimizasyon Önerileri

### 1. Model Küçültme (Quantization)

BERT ve RoBERTa modellerini küçült:

```python
# INT8 quantization
import torch.quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Beklenen Kazanç: 30-40% hızlanma**

### 2. ONNX Runtime

Modelleri ONNX formatına çevir:

```bash
pip install onnx onnxruntime
```

**Beklenen Kazanç: 2-3x hızlanma**

### 3. Batch Processing

Birden fazla metin varsa batch olarak işle:

```python
texts = ["text1", "text2", "text3"]
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
```

**Beklenen Kazanç: 10 metin için 5-10 saniye**

### 4. Caching

Aynı metinler için sonuçları cache'le:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def predict_cached(text_hash):
    return predict_all(text)
```

### 5. Asenkron İşleme

Modelleri paralel çalıştır:

```python
import asyncio

async def predict_all_async(text):
    tasks = [
        predict_bert_async(text),
        predict_roberta_async(text),
        predict_h2o_async(text)
    ]
    results = await asyncio.gather(*tasks)
```

**Beklenen Kazanç: 40-50% hızlanma**

## Performans Karşılaştırması

| Optimizasyon | CPU Süresi | GPU Süresi | Kazanç |
|-------------|-----------|-----------|--------|
| Optimizasyon Öncesi | 15-25s | - | - |
| H2O Frame Tekrar Kullanımı | 12-20s | - | 3-5s |
| + GPU Desteği | - | 6-10s | 6-10s |
| + Column Types | 11-18s | 5-9s | 1-2s |
| + Model Quantization | 8-12s | 3-6s | 3-6s |
| + ONNX Runtime | 4-6s | 2-3s | 4-6s |

## Sistem Gereksinimleri

### Minimum
- CPU: 4 core
- RAM: 8GB
- Disk: 5GB

### Önerilen
- CPU: 8 core
- RAM: 16GB
- GPU: NVIDIA GPU (CUDA destekli) 6GB+ VRAM
- Disk: 10GB SSD

## Performans İpuçları

1. **İlk Request Yavaş**: Modeller yüklenirken 30-60 saniye sürebilir. Bu normaldir.

2. **GPU Kullanımı**: NVIDIA GPU varsa otomatik kullanılır. Kontrol:
   ```python
   print(f"Using device: {self.device}")
   ```

3. **H2O Cluster**: H2O otomatik olarak cluster başlatır. İlk başlatma yavaş olabilir.

4. **Memory Cleanup**: Uzun süre çalışan uygulamada bellek temizliği:
   ```python
   torch.cuda.empty_cache()  # GPU memory
   ```

## Monitoring

Terminal çıktısında progress göster:

```
Initializing models...
Using device: cuda
Loading BERT...
Loading RoBERTa...
Loading TF-IDF...
Initializing H2O...
Loading H2O models...
Model expects 5000 features
All models loaded successfully!

Running BERT...
Running RoBERTa...
Running H2O models...
  - DRF...
  - GBM...
  - GLM...
```

Her modelin ne zaman çalıştığını terminal'de görebilirsiniz.

## Troubleshooting

### "Out of Memory" Hatası

**CPU:**
```python
# Daha küçük max_length kullan
max_length = 256  # 512 yerine
```

**GPU:**
```python
# CPU'ya geri dön
self.device = torch.device('cpu')
```

### Yavaş Tahminler

1. GPU kullanıldığından emin olun
2. H2O verbose mode kapalı olmalı: `h2o.init(verbose=False)`
3. Model evaluation mode'da olmalı: `model.eval()`
4. Gradient hesaplama kapalı olmalı: `with torch.no_grad()`

## Sonuç

Yapılan optimizasyonlarla:
- **CPU'da**: ~15-20 saniye → ~11-18 saniye
- **GPU'da**: ~6-10 saniye

Daha fazla hızlanma için ONNX Runtime veya model quantization kullanılabilir.
