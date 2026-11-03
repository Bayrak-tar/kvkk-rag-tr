# Mimari ve veri akışı

```mermaid
flowchart LR
    A[Mevzuat Bilgi Sistemi / Resmî Gazete / KVKK] --> B[src.ingestion]
    B --> C[data/processed/legal_units.jsonl]
    C --> D[src.chunking]
    D --> E[Chunk metadata: kaynak, kanun, madde, fıkra]
    E --> F[Sentence Transformers embedding]
    E --> G[BM25 JSONL]
    F --> H[Chroma]
    G --> I[BM25Retriever]
    H --> J[VectorRetriever]
    I --> K[BM25]
    J --> L[Vektör]
    I --> M[RRF hibrit]
    J --> M
    K --> N[İsteğe bağlı CrossEncoder]
    L --> N
    M --> N
    N --> O[src.generation.rag]
    O --> P[Ollama / OpenAI uyumlu API]
    P --> X[Atıf etiketlerini filtrele]
    X --> Y[LLM destek denetimi]
    Y --> Q[FastAPI]
    Q --> R[Gradio]
    K --> S[eval.run]
    L --> S
    M --> S
    S --> T[Recall@k / MRR@k / nDCG@k]
    O --> U[Atıf kapsama ve kaynak destek kontrolü]
    T --> V[results/*.csv ve *.json]
    U --> V
    V --> W[eval.compare: Markdown / SVG]
```

## Sorumluluklar

- `src/ingestion/`: resmî kaynak manifestinden indirme, metin temizleme ve kanun/madde/fıkra birimleri.
- `src/chunking/`: sabit boyut, madde ve semantik bölümleme; chunk metadata'sı kaynak atfını taşır.
- `src/embeddings/`: E5 ve BGE-M3 Sentence Transformers adaptörleri.
- `src/retrieval/`: Chroma vektör store, Okapi BM25 JSONL indeksi, varsayılan RRF hibrit arama ve isteğe bağlı CrossEncoder.
- `src/generation/`: Ollama/OpenAI uyumlu sağlayıcı soyutlaması, ortak RAG runtime, yanıtın kullandığı atıfların süzülmesi ve yapılandırılabilir LLM kaynak destek denetimi.
- `src/api/`, `src/ui/`: FastAPI REST katmanı ve API'ye HTTP üzerinden bağlanan Gradio istemcisi.
- `eval/`: draft/validated soru bankası, retrieval metrikleri, atıf kapsama ölçümü, LLM destek denetimi, config runner ve sonuç karşılaştırma aracı.

Model ağırlıkları veya üretilmiş indeksler kaynak kod deposuna konmaz. Deney çıktıları yalnız runner gerçekten çalıştığında yazılır; sonuç JSON'u config, corpus ve soru seti hash'lerini saklar.

Kaynak destek denetimi aynı sohbet modeline ikinci bir görev gönderir. Geçerli kaynak etiketi olmayan, denetimden geçmeyen veya denetim yanıtı okunamayan cevaplarda kaynak kartları gösterilmez ve sistem güvenli biçimde yanıt vermekten kaçınır. Bu model tabanlı denetim insan veya hukukçu incelemesinin yerini almaz; doğruluk oranı yalnız elle doğrulanmış evaluation sorularıyla ölçülmelidir.
