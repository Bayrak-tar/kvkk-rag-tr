# Faz 0 — Plan ve proje yapısı

## Proje özeti

- **Ad:** `kvkk-rag-tr`
- **Amaç:** KVKK ve ilgili Türk mevzuatı üzerinde kaynak atıflı soru-cevap sunan; chunking, embedding, retrieval ve reranking seçeneklerini deneylerle karşılaştıran açık kaynak bir RAG sistemi.
- **Veri ilkesi:** Yalnızca herkese açık mevzuat metinleri ve KVKK Kurumunun kamuya açık rehber/kararları kullanılacak. Kişisel veri veya izinsiz/telifli içerik veri kümesine eklenmeyecek.
- **Ölçüm ilkesi:** Yalnızca gerçekten çalıştırılmış deneylerin sonuçları yayımlanacak. Önceden bilinmeyen değerler varsayım olarak işaretlenecek, sonuç gibi sunulmayacak.

## Uygulama sırası

1. **Faz 1 — Veri ve chunking:** Kaynakları alma, temizleme ve madde/fıkra bazlı yapılandırma; kaynak URL'si ve mevzuat metadata'sı; sabit boyut, madde/başlık bazlı ve semantik chunking; temel birim testleri.
2. **Faz 2 — Temel RAG:** Yapılandırılabilir embedding modelleri, vektör indeksleme ve getirilen bağlamla sınırlı, kaynak gösteren uçtan uca cevaplama.
3. **Faz 3 — Retrieval seçenekleri:** BM25, vektör arama ve hibrit arama; isteğe bağlı Türkçe ön işleme ve reranking.
4. **Faz 4 — Değerlendirme:** Elle doldurulabilir 40–60 soruluk değerlendirme şablonu; Recall@k, MRR ve nDCG@k; basit faithfulness kontrolü ve elle işaretlenebilir hata analizi; config üzerinden çalışıp CSV/JSON çıktısı üreten deney araçları. Sorular kamuya açık, doğrulanabilir kaynaklara dayandırılacak.
5. **Faz 5 — Servis ve paketleme:** FastAPI, Gradio veya Streamlit arayüzü, Docker/Compose, `.env.example`, sırları dışlayan `.gitignore` ve GitHub Actions CI.
6. **Faz 6 — Dokümantasyon:** Kurulum, mimari diyagramı, gerçekten ölçülmüş sonuç tablosu, Türkçede aramanın güçlükleri, sınırlılıklar/gelecek işler ve hukuki uyarı.

Her fazın sonunda sonuç ve doğrulama durumu özetlenecek, sonraki faza geçmeden kullanıcı onayı beklenecek.

## Önerilen klasör yapısı

```text
kvkk-rag-tr/
  configs/                  # Uygulama ve deney ayarları
  data/
    raw/                     # İndirilen kamuya açık kaynaklar
    processed/               # Temizlenmiş ve yapılandırılmış kayıtlar
  src/
    ingestion/               # Kaynak alma ve normalizasyon
    chunking/                # Chunk stratejileri
    embeddings/              # Embedding sağlayıcıları
    retrieval/               # Vektör, BM25 ve hibrit arama
    rerank/                  # İsteğe bağlı yeniden sıralama
    generation/              # LLM soyutlaması ve cevap üretimi
    api/                     # FastAPI
    ui/                      # Gradio veya Streamlit
  eval/
    questions/               # Değerlendirme seti şablonu/verisi
    metrics/                  # Retrieval ve üretim metrikleri
    runners/                  # Deney çalıştırıcıları
  results/                   # Gerçek deneylerden CSV/JSON sonuçları
  tests/                     # Birim ve entegrasyon testleri
  docs/                      # Plan ve mimari dokümantasyonu
  .env.example
  .gitignore
  Dockerfile
  docker-compose.yml
  pyproject.toml              # Paket, Ruff ve pytest ayarları
  README.md
```

## Başlangıç varsayımları ve karar durumu

- **Veri kapsamı:** Faz 1 başlangıcında onaylandı; 6698 sayılı Kanun ve KVKK alanıyla doğrudan ilgili yönetmelikler. İlk manifestte dört resmî kaynak var; kaynak listesi `docs/data-sources.md` içindedir.
- **LLM varsayılanı önerisi:** Yerel model; API tabanlı sağlayıcılar config ile seçilebilir. Henüz kararlaştırılmadı.
- **Arayüz önerisi:** Gradio ile başlamak. Henüz kararlaştırılmadı.
- **Python ve araçlar:** Python 3.11+, type hints ve docstring; Ruff ile lint/format, pytest ile test. Bağımlılıklar `pyproject.toml` içinde tutulur.

Resmî kaynakların URL ve biçimleri veri alımında kontrol edilmiştir. Kaynak yanıtlarının güncellenebileceği ve her çalıştırmada yeniden indirileceği belgelenmiştir.

## Açık kararlar

1. Yerel LLM varsayılan, API sağlayıcıları config üzerinden seçilebilir şekilde ilerleyelim mi?
2. İlk arayüz Gradio olsun mu?

## Faz 0 tamamlanma kaydı

Bu belge, mevcut boş proje klasörüne Faz 0 planını ve önerilen yapıyı kaydetmek için oluşturuldu. Faz 0 kapsamında uygulama kodu, deney sonucu veya doğrulanmamış metrik eklenmedi.
