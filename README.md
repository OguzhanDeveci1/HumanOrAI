# Human or AI Text Classifier

Bu proje, İngilizce makalelerin insan mı yoksa AI tarafından mı yazıldığını tahmin eden 5 farklı makine öğrenmesi modeli kullanır.

## Özellikler

- **Modern Web Arayüzü**: Kullanımı kolay, responsive tasarım
- **5 Güçlü Model**: BERT, RoBERTa, DRF, GBM, GLM
- **Gerçek Zamanlı Analiz**: Anında sonuç gösterimi
- **Ensemble Tahmin**: 5 modelin çoğunluk oylaması ile final karar
- **Görsel Sonuçlar**: Her model için güven skoru çubukları
- **Karanlık Tema**: Göz dostu modern tasarım
- **Klavye Kısayolları**: Hızlı kullanım için

## Modeller

1. **BERT** - Bidirectional Encoder Representations from Transformers
2. **RoBERTa** - Robustly Optimized BERT Pretraining Approach
3. **DRF** - Distributed Random Forest (H2O AutoML)
4. **GBM** - Gradient Boosting Machine (H2O AutoML)
5. **GLM** - Generalized Linear Model (H2O AutoML)

## Kurulum

### 1. Virtual Environment Oluşturma

```bash
python -m venv .venv
```

### 2. Virtual Environment'ı Aktif Etme

Windows:
```bash
.venv\Scripts\activate
```

Linux/Mac:
```bash
source .venv/bin/activate
```

### 3. Gerekli Paketleri Yükleme

```bash
pip install -r requirements.txt
```

## Kullanım

### Web Arayüzü ile Kullanım (Önerilen)

```bash
python app.py
```

Tarayıcınızda şu adrese gidin: http://localhost:5000

Web arayüzünde:
1. Metin kutusuna İngilizce makaleyi yapıştırın
2. "Analyze Text" butonuna tıklayın
3. 5 modelin tahminlerini modern bir arayüzde görün
4. Ensemble (çoğunluk oylaması) sonucunu görün

**Klavye Kısayolları:**
- `Ctrl/Cmd + Enter`: Metni analiz et
- `Ctrl/Cmd + K`: Metni temizle

### Komut Satırı ile Kullanım

```bash
python predict.py
```

Program çalıştığında:
1. İngilizce makaleyi/metni girin
2. Metni girdikten sonra iki kez Enter'a basın
3. 5 modelin tahminlerini ve doğruluk değerlerini göreceksiniz

## Çıktı Formatı

Her model için:
- **Prediction**: HUMAN (1) veya AI (0)
- **Confidence**: Modelin tahmin güven skoru (%)

Ayrıca ensemble (çoğunluk oylaması) sonucu da gösterilir.

## Tahmin Değerleri

- **0** = AI tarafından yazılmış
- **1** = İnsan tarafından yazılmış

## Örnek Çıktı

```
======================================================================
HUMAN OR AI PREDICTION RESULTS
======================================================================

Input text preview: This is an example article text...

----------------------------------------------------------------------

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

## Notlar

- Tüm modeller `models/` klasöründe bulunmalıdır
- H2O otomatik olarak başlatılır ve sonlandırılır
- Uzun metinler otomatik olarak 512 token'a kesilir (BERT ve RoBERTa için)
