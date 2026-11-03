# Faz 5 — API, arayüz ve Docker

## API

FastAPI uygulaması `src/api/app.py` içinde `/health` ve `/ask` endpoint'lerini sunar. `/ask` JSON gövdesi `{"question":"..."}` kabul eder, cevabı ve yapılandırılmış madde atıflarını döndürür. Soru boş olamaz ve en fazla 2000 karakterdir. RAG modelleri ilk geçerli soruda yüklenir; health kontrolü model başlatmaz.

Yerel API:

```powershell
python -m pip install -e ".[web,retrieval]"
uvicorn src.api.app:app --reload
```

`KVKK_RAG_CONFIG` ile alternatif RAG config dosyası seçilebilir. Ollama URL'si `OLLAMA_BASE_URL` üzerinden override edilebilir. OpenAI uyumlu API anahtarı `.env` içindeki `OPENAI_API_KEY` değişkeninden okunur.

## Gradio

`src/ui/app.py` API'ye HTTP üzerinden bağlanır; böylece UI kendi model kopyasını yüklemez. Yerelde API ve UI'ı ayrı terminallerde başlatın:

```powershell
python -m src.ui.app
```

Arayüz varsayılan olarak `http://127.0.0.1:7861` adresinde açılır. API adresi `KVKK_API_URL` ile ayarlanabilir; varsayılanı `http://127.0.0.1:8001/ask` olur.

## Docker Compose

İlk hazırlıkta host'ta mevcut kaynak kayıtlarını ve indeksleri oluşturun. Docker kullanımında aynı işlemler bind mount içindeki dosyaları üretir:

```powershell
Copy-Item .env.example .env
docker compose build
docker compose run --rm api python -m src.ingestion.cli --offline
docker compose run --rm api python -m src.retrieval.index_cli --config configs/rag.json
docker compose up
```

API `http://localhost:8001/docs`, Gradio ise `http://localhost:7861` adresindedir. Compose, Ollama'yı host üzerinde `http://host.docker.internal:11434` adresinde bekler; farklı adres için compose servisindeki `OLLAMA_BASE_URL` değerini düzenleyin. API ve UI ayrı container'lardır; Chroma ve model cache'i API servisine bağlanır.

Gerçek API anahtarını `.env` içinde tutun; `.env` ve yerel `data/` içeriği Git ve Docker build context dışında bırakılmıştır. Uygulama halka açık mevzuat metinleri içindir; kullanıcıların özel/kişisel verilerini soru olarak göndermemesi gerekir.

## Sınırlar

API/UI entegrasyon testleri sahte RAG servisi kullanır. Bu fazda Docker image build'i, Hugging Face ağırlık indirme, Chroma indeksleme veya canlı LLM çağrısı çalıştırılmadı. İlk gerçek kullanım öncesi veri ve BM25/vektör indeksleri hazırlanmalı, Ollama modeli indirilmeli veya API sağlayıcısı yapılandırılmalıdır.
