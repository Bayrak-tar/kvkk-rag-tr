# Faz 2 — Embedding, vektör indeksi ve temel RAG

## Bileşenler

- `src/embeddings/`: Sentence Transformers adaptörü ve model ayarları.
- `src/retrieval/`: JSONL corpus'u chunk'layan, embedding üreten ve Chroma'ya upsert eden indeksleme akışı.
- `src/generation/`: Ollama veya OpenAI uyumlu sohbet sağlayıcısı; retrieval bağlamını kullanan RAG cevabı.
- `configs/embedding.json`: varsayılan `intfloat/multilingual-e5-base` modeli.
- `configs/embedding-bge-m3.json`: `BAAI/bge-m3` alternatifi.
- `configs/rag*.json`: deney ayarları ve her model için ayrı Chroma koleksiyon seçimi.

E5 adaptörü sorgulara `query:`, mevzuat parçalarına `passage:` ön eki ekler. Bu, E5 model kartındaki asimetrik retrieval talimatıdır. BGE-M3 adaptörü talimat ön eki eklemez. [Multilingual E5 model kartı](https://huggingface.co/intfloat/multilingual-e5-base) · [BGE-M3 model kartı](https://huggingface.co/BAAI/bge-m3)

Chroma yerel kalıcı istemcisi `data/chroma` altında tutulur. Koleksiyon adı model kimliğinden türetilir; böylece farklı vektör boyutlu embedding modelleri aynı koleksiyona yazılmaz. [Chroma PersistentClient](https://docs.trychroma.com/docs/run-chroma/cloud-client?lang=typescript)

## Kurulum ve çalıştırma

```powershell
python -m pip install -e ".[retrieval,dev]"
python -m src.ingestion.cli --offline
python -m src.retrieval.index_cli --config configs/rag.json
python -m src.generation.cli "İlgili kişinin hakları nelerdir?" --config configs/rag.json
```

İkinci modeli indekslemek ve onunla sorgulamak için `configs/rag-bge-m3.json` kullanın. İlk çalıştırmada Hugging Face model ağırlıkları indirilir; model ayarındaki `device` varsayılanı `cpu`'dur. Daha hızlı yerel kullanım için yapılandırma `cuda` olarak değiştirilebilir.

## LLM sağlayıcısı

Varsayılan sağlayıcı Ollama'dır. `configs/generation.json` içindeki `provider`, `model` ve `base_url` ile yerel servisi seçin. Ollama sağlayıcısı resmi `/api/chat` endpoint'ini çağırır ve stream'i kapalı tutar. [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)

OpenAI Chat Completions biçimli bir API kullanmak için `provider` değerini `openai-compatible` yapın, `base_url` değerini sağlayıcının `/v1` taban URL'siyle güncelleyin ve gerçek API anahtarını `.env` içindeki `OPENAI_API_KEY` değişkenine koyun. `.env` Git dışında tutulur; `configs/generation.json` yalnızca değişken adını taşır.

## Sınırlar

- İndeksleme, model inference ve Ollama/API canlı çağrısı bu kod adımında otomatik çalıştırılmadı; opsiyonel ML paketleri ve model ağırlıkları makineye göre kurulmalıdır.
- Cevap prompt'u yalnızca getirilen bağlamı kullanmayı, madde kaynaklarını belirtmeyi ve bağlam yetersizse “bulunamadı” demeyi ister. Üretici modelin bu kurala uyumu mutlak garanti değildir; faithfulness ölçümü Faz 4'te yapılacaktır.
- Bu aşamada gerçek retrieval metriği veya model karşılaştırma sonucu üretilmedi.
