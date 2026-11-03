# Faz 3 — BM25, hibrit arama ve reranking

## Arama seçenekleri

`configs/rag.json` içindeki `retrieval.strategy` alanı `vector`, `bm25` veya `hybrid` olabilir. `hybrid` seçeneği embedding sıralamasıyla BM25 sıralamasını Reciprocal Rank Fusion (RRF) üzerinden birleştirir. `candidate_k` her iki aramanın RRF'e verdiği aday sayısını, `rrf_k` ise RRF sabitini belirler. BM25'in `k1` ve `b` parametreleri de config'ten ayarlanabilir.

İndeksleme komutu vektör koleksiyonuna ek olarak chunk'ları `bm25_index_path` ile belirtilen JSONL dosyasına yazar. `bm25` modu sorgu sırasında bu dosyayı kullanır; embedding modeli ve Chroma bağlantısı açılmaz.

İsteğe bağlı CrossEncoder için `retrieval.reranker.enabled` değerini `true` yapın. Varsayılan model `BAAI/bge-reranker-v2-m3`'tür; CrossEncoder bağımlılığı `sentence-transformers` retrieval ekstra paketinin içindedir. Model ilk kullanımda Hugging Face'den indirilir. Model sorgu-belge çiftlerini puanlayarak adayları tekrar sıralar. [Resmî model kartı](https://huggingface.co/BAAI/bge-reranker-v2-m3)

Örnek hibrit ayar:

```json
{
  "strategy": "hybrid",
  "candidate_k": 30,
  "rrf_k": 60,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "reranker": {
    "enabled": true,
    "model_name": "BAAI/bge-reranker-v2-m3"
  }
}
```

## Türkçe işleme kapsamı

BM25 tokenizer'ı Unicode sözcükleri ve Türkçe `I/İ` küçük harf dönüşümünü korur. Bu fazda stemming veya lemmatization uygulanmıyor; morfolojik işlemenin etkisi ayrı bir deney olarak ölçülmelidir. Zemberek'in Java tabanlı yapısı ve bakım durumu ek operasyon yükü doğurduğundan varsayılan retrieval yoluna dahil edilmedi. [Zemberek deposu](https://github.com/ahmetaa/zemberek-nlp)

## Kullanım

Önce corpus'u ve iki indeksi oluşturun:

```powershell
python -m src.ingestion.cli --offline
python -m src.retrieval.index_cli --config configs/rag.json
```

`configs/rag.json` içinde `strategy` değerini değiştirip sorguyu çalıştırın:

```powershell
python -m src.generation.cli "İlgili kişinin hakları nelerdir?" --config configs/rag.json
```

## Değerlendirme durumu

Bu faz BM25, RRF ve CrossEncoder yapılandırmasını ekler; gerçek retrieval ölçümü veya deney sonucu üretmez. Recall@k, MRR ve nDCG@k sonuçları Faz 4'te soru seti üzerinde çalıştırılan deneylerden raporlanacaktır. Kayıtlı BM25 skorları arama sıralaması içindir; `VectorHit.distance` alanında skorun negatifi taşınır ve bu alan iki farklı retrieval yönteminin metrik karşılaştırması olarak yorumlanmamalıdır.
