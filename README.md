# **Teachable Machine Inference API**

Bu proje, **Teachable Machine** ile eğitilmiş modelleri kullanarak resim ve ses dosyaları üzerinde tahmin yapmak için geliştirilmiş bir **FastAPI** tabanlı RESTful API'dir. Öğrenciler, bu API aracılığıyla kendi eğittikleri modelleri çalıştırabilir ve tahmin sonuçlarını alabilirler.

---

## **Projenin Amacı**

Bu API'nin temel amacı, öğrencilerin **Teachable Machine** ile eğittikleri modelleri kolayca kullanabilmelerini sağlamaktır. API, resim ve ses dosyalarını işleyebilir ve bu dosyalar üzerinde tahminler yaparak sonuçları JSON formatında döndürür.

---

## **Özellikler**

- **Resim ve Ses Tahmini**: Resim (JPEG, PNG) ve ses (WAV, MP3) dosyaları üzerinde tahmin yapma.
- **Esnek Model Yükleme**: Modeller ve etiketler dinamik olarak yüklenir.
- **Hata Yönetimi**: Geçersiz dosya türleri, eksik modeller ve diğer hatalar için açıklayıcı hata mesajları.
- **Dokümantasyon**: Otomatik olarak oluşturulan Swagger UI ve ReDoc dokümantasyonu.
- **Çoklu İstek Desteği**: Aynı anda birden fazla tahmin isteği işlenebilir.

---

## **Kurulum**

### **1. Gereksinimler**

Projeyi çalıştırmak için aşağıdaki yazılımların yüklü olması gerekmektedir:

- **Python 3.12 veya üzeri** (Tercihen 3.12.3)
- **pip** (Python paket yöneticisi)

### **2. Sanal Ortam Oluşturma**

Projeyi çalıştırmak için bir sanal ortam oluşturmanız önerilir:

```bash
python -m venv venv
```

Sanal ortamı etkinleştirin:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### **3. Gerekli Paketlerin Yüklenmesi**

Proje bağımlılıklarını yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

---

## **Çalıştırma**

API'yi çalıştırmak için aşağıdaki komutu kullanın:

```bash
python main.py
```

Varsayılan olarak, API `http://127.0.0.1:8080` adresinde çalışacaktır. Farklı bir IP veya port kullanmak için komut satırı argümanlarını kullanabilirsiniz:

```bash
python main.py --host 127.0.0.1 --port 8000
```

---

## **API Dokümantasyonu**

API'nin interaktif dokümantasyonuna aşağıdaki adreslerden erişebilirsiniz:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

---

## **Örnek Kullanım**

### **1. Resim Tahmini**

- **Model Adı**: `cats_and_dogs`
- **Model Türü**: `image`
- **Dosya**: Bir kedi veya köpek resmi (JPEG veya PNG formatında).

**Örnek İstek**:
```bash
curl -X POST "http://localhost:8080/predict?model_name=cats_and_dogs&model_type=image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@kedi.jpg"
```

**Örnek Yanıt**:
```json
{
    "prediction": {
        "Dog": 0.08,
        "Cat": 0.92
    }
}
```

### **2. Ses Tahmini** (WIP)

- **Model Adı**: `ses_tanima_modeli`
- **Model Türü**: `audio`
- **Dosya**: Bir ses dosyası (WAV veya MP3 formatında).

**Örnek İstek**:
```bash
curl -X POST "http://localhost:8080/predict?model_name=ses_tanima_modeli&model_type=audio" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ses_dosyasi.wav"
```

**Örnek Yanıt**:
```json
{
    "prediction": {
        "müzik": 0.75,
        "konuşma": 0.25
    }
}
```

---

## **Hata Durumları**

- **400 Bad Request**: Geçersiz dosya türü veya model türü.
- **404 Not Found**: Model veya etiket dosyası bulunamadı.
- **500 Internal Server Error**: Sunucu tarafında bir hata oluştu.

---

## **Katkıda Bulunma**

Bu proje açık kaynaklıdır. Katkıda bulunmak isterseniz lütfen bir **Pull Request** gönderin.

---

## **Lisans**

Bu proje **MIT Lisansı** altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.