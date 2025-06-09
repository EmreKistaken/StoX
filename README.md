# 📊 E-Ticaret Satış Analizi Uygulaması

Bu proje, e-ticaret işletmelerinin satış verilerini analiz etmek, müşteri davranışlarını anlamak ve gelecekteki satışları tahmin etmek için geliştirilmiş kapsamlı bir analiz uygulamasıdır.

## 🚀 Özellikler

### 📈 Analiz Özellikleri
- **Satış Trend Analizi**: Günlük, haftalık, aylık satış trendleri
- **Müşteri Segmentasyonu**: RFM (Recency, Frequency, Monetary) analizi
- **Zaman Serisi Analizi**: Mevsimsellik ve trend analizi
- **Ürün Performans Analizi**: En çok satan ürünler, kategori analizi
- **Stok Optimizasyonu**: Stok seviyesi önerileri
- **Satış Tahmini**: Prophet ve ARIMA modelleri ile gelecek tahminleri

### 📊 Görselleştirme
- İnteraktif grafikler (Plotly)
- Isı haritaları
- Pasta ve çubuk grafikler
- Zaman serisi grafikleri
- Müşteri segmentasyonu görselleştirmeleri

### 📄 Raporlama
- Otomatik PDF rapor oluşturma
- HTML formatında interaktif raporlar
- Kapsamlı analiz özetleri
- Pazarlama strateji önerileri

## 🛠️ Teknolojiler

- **Frontend**: Streamlit
- **Veri İşleme**: Pandas, NumPy
- **Makine Öğrenmesi**: Scikit-learn, Prophet, ARIMA
- **Görselleştirme**: Plotly, Seaborn
- **Raporlama**: ReportLab, Jinja2
- **Veri Formatları**: Excel, CSV, JSON

## 📋 Gereksinimler

### Sistem Gereksinimleri
- Python 3.8+
- 4GB RAM (minimum)
- 2GB disk alanı

### Python Paketleri
```
streamlit==1.32.0
pandas==1.5.3
openpyxl==3.1.2
plotly==5.19.0
numpy==1.23.5
scikit-learn==1.2.2
prophet==1.1.5
statsmodels==0.13.5
seaborn==0.12.2
pmdarima==2.0.3
reportlab==4.0.4
jinja2==3.1.2
weasyprint==58.1
requests==2.28.2
numba==0.56.4
pyarrow==11.0.0
fastparquet==2023.4.0
python-dateutil==2.8.2
```

## 🚀 Kurulum

### 1. Projeyi Klonlayın
```bash
git clone <repository-url>
cd veri_ajans
```

### 2. Sanal Ortam Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

## 📊 Veri Formatı

Uygulama aşağıdaki sütunları içeren veri dosyalarını destekler:

### Zorunlu Sütunlar
- `tarih`: Satış tarihi (YYYY-MM-DD formatında)
- `urun_adi`: Ürün adı
- `miktar`: Satış miktarı
- `satis_tutari`: Satış tutarı (TL)

### Opsiyonel Sütunlar
- `musteri_id`: Müşteri kimlik numarası (RFM analizi için)
- `siparis_id`: Sipariş kimlik numarası (RFM analizi için)
- `kategori`: Ürün kategorisi (kategori analizi için)

### Desteklenen Dosya Formatları
- Excel (.xlsx, .xls)
- CSV (.csv)
- JSON (.json)

## 📈 Kullanım Kılavuzu

### 1. Veri Yükleme
- Uygulamayı açın
- "Dosya Yükle" bölümünden veri dosyanızı seçin
- Dosya formatını kontrol edin ve yükleyin

### 2. Veri Doğrulama
- Sistem otomatik olarak veri formatını kontrol eder
- Eksik sütunlar varsa uyarı verir
- Tarih formatı otomatik olarak dönüştürülür

### 3. Analiz Seçenekleri

#### 📊 Genel Analiz
- Toplam satış istatistikleri
- Satış trendleri
- En çok satan ürünler
- Kategori performansı

#### 👥 Müşteri Analizi
- RFM segmentasyonu
- Müşteri davranış analizi
- Segment bazlı pazarlama önerileri

#### 🔮 Tahminleme
- 7-90 gün arası satış tahmini
- Prophet ve ARIMA modelleri
- Güven aralıkları

#### 📦 Stok Optimizasyonu
- Stok seviyesi önerileri
- Ürün bazlı stok durumu
- Yeniden sipariş önerileri

### 4. Rapor Oluşturma
- "Kapsamlı Rapor" bölümünden rapor oluşturun
- HTML formatında indirin
- PDF formatında kaydedin

## 🏗️ Proje Yapısı

```
.../
├── app.py                 # Ana Streamlit uygulaması
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── backend/              # Backend modülleri
│   ├── analysis/         # Analiz fonksiyonları
│   ├── api/             # API endpoints
│   └── utils/           # Yardımcı fonksiyonlar
├── frontend/             # Frontend bileşenleri
│   └── src/             # Kaynak dosyalar
├── data/                 # Veri dosyaları
│   └── sales_data.xlsx   # Örnek veri
├── analysis/             # Analiz sonuçları
│   └── summary.py       # Özet raporlar
├── reports/              # Oluşturulan raporlar
│   └── reports.pdf      # PDF raporlar
└── visuals/              # Görselleştirmeler
```

## 🔧 Özelleştirme

### Analiz Parametrelerini Değiştirme
`app.py` dosyasında aşağıdaki parametreleri özelleştirebilirsiniz:

```python
# RFM analizi parametreleri
RFM_SCORES = {
    'recency_bins': [0, 30, 60, 90, 180, float('inf')],
    'frequency_bins': [1, 2, 3, 5, 10, float('inf')],
    'monetary_bins': [0, 100, 500, 1000, 5000, float('inf')]
}

# Tahmin parametreleri
FORECAST_DAYS = 30
CONFIDENCE_LEVEL = 0.95
```

### Yeni Analiz Fonksiyonları Ekleme
1. `app.py` dosyasında yeni fonksiyon tanımlayın
2. Streamlit arayüzüne yeni bölüm ekleyin
3. Gerekli import'ları ekleyin

## 📊 Örnek Kullanım Senaryoları

### Senaryo 1: Aylık Satış Raporu
1. Aylık satış verilerini yükleyin
2. "Genel Analiz" bölümünden trendleri inceleyin
3. "Kapsamlı Rapor" oluşturun
4. PDF raporu indirin

### Senaryo 2: Müşteri Segmentasyonu
1. Müşteri ID'li veri yükleyin
2. "Müşteri Analizi" bölümüne gidin
3. RFM skorlarını inceleyin
4. Segment bazlı stratejileri uygulayın

### Senaryo 3: Satış Tahmini
1. Geçmiş satış verilerini yükleyin
2. "Satış Tahmini" bölümünden tahmin oluşturun
3. Prophet ve ARIMA sonuçlarını karşılaştırın
4. Stok planlaması yapın

## 🐛 Sorun Giderme

### Yaygın Sorunlar

#### 1. Tarih Formatı Hatası
```
Hata: Tarih dönüşümünde hata
Çözüm: Tarih sütununun formatını kontrol edin (YYYY-MM-DD)
```

#### 2. Eksik Sütun Hatası
```
Hata: Eksik sütunlar: tarih, urun_adi
Çözüm: Veri dosyanızda gerekli sütunların olduğundan emin olun
```

#### 3. Bellek Hatası
```
Hata: MemoryError
Çözüm: Veri boyutunu küçültün veya RAM'i artırın
```

#### 4. Prophet Kurulum Hatası
```bash
# Windows için
conda install -c conda-forge prophet

# Linux/macOS için
pip install prophet --no-deps
pip install pystan
```

## 📈 Performans İpuçları

### Veri Optimizasyonu
- Büyük dosyalar için veri örnekleme kullanın
- Gereksiz sütunları kaldırın
- Tarih aralığını sınırlayın

### Bellek Yönetimi
- Sanal ortam kullanın
- Gereksiz değişkenleri silin
- Büyük veri setleri için chunking kullanın

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

- **Geliştirici**: [Yunus Emre Kahraman]
- **E-posta**: [yemrekah04@gmail.com]

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak kütüphaneleri kullanmaktadır:
- Streamlit
- Pandas
- Plotly
- Prophet
- Scikit-learn
- ReportLab

---

**Not**: Bu uygulama eğitim ve ticari amaçlar için geliştirilmiştir. Veri güvenliği ve gizliliği konularında gerekli önlemleri almayı unutmayın. 
