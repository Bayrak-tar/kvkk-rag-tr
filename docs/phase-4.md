# Faz 4 — Değerlendirme ve deney çalıştırma

## Soru seti

`eval/questions/kvkk_seed.jsonl` içinde 40 elle gözden geçirilebilir soru taslağı vardır. Her satırda soru, `source_id:madde_no` biçiminde altın madde kimliği, referans cevap, resmî kaynak URL'leri, konu kategorisi ve durum alanı bulunur. Soruların kapsamı 6698 sayılı Kanun'un tanımlar, ilkeler, işleme şartları, özel nitelikli veriler, aktarımlar, aydınlatma, haklar, güvenlik, başvuru/şikâyet ve VERBİS maddelerinden seçilmiştir. İlgili resmî açıklamalar: [Aydınlatma Yükümlülüğü](https://www.kvkk.gov.tr/Icerik/2033/Aydinlatma-Yukumlulugu-), [Açık Rıza](https://www.kvkk.gov.tr/Icerik/2037/Acik-Riza-Alirken-Dikkat-Edilecek-Hususlar), [Kanun kapsamındaki hak ve yükümlülükler](https://www.kvkk.gov.tr/Icerik/4192/Kanun-Kapsamindaki-Hak-ve-Yukumlulukler) ve [Kurul karar özetleri](https://www.kvkk.gov.tr/Icerik/8896/kisisel-verileri-koruma-kurulunun-yeni-yayinlanan-karar-ozetleri).

Tüm başlangıç kayıtlarının durumu `draft`'tır. Referans cevapları ve gold maddeleri mevzuatın geçerli sürümü, indirilen corpus ve kaynak URL'leriyle uzman gözden geçirmesinden geçirin; yalnız sonra `status` alanını `validated` yapın. Runner varsayılan olarak `validated` kayıtlar dışındakileri ölçüme almaz. Yayımlanmış metrikler için soru setinin hash'i sonuç metadata'sına yazılır. JSON Schema açıklaması `eval/questions/schema.json` içindedir; `eval.dataset.load_questions` aynı temel gereksinimleri bağımlılık eklemeden doğrular.

## Retrieval metrikleri

Her sonuç listesi önce tekrarlanan chunk'ları `source_id:article_no` seviyesinde tekilleştirir.

- **Recall@k:** İlk k benzersiz madde içinde bulunan gold maddelerin, tüm gold maddelere oranı.
- **MRR@k:** İlk ilgili gold maddenin reciprocal rank değeri; ilgili madde ilk k içinde yoksa 0.
- **nDCG@k:** Gold madde üyeliğine dayalı binary relevance ile normalize edilmiş DCG.

Bir soru birden fazla gold maddeye sahip olabilir. Aggregation soru başına makro ortalamadır. Bunlar chunk değil madde düzeyinde ölçümlerdir; karşılaştırılan sistemlerde aynı gold seti ve aynı soru seti kullanılmalıdır.

## Deney çalıştırma

`configs/eval-vector.json`, `eval-bm25.json`, `eval-hybrid.json` ve `eval-hybrid-rerank.json` başlangıç deney config'leridir. Her config `rag_config`, soru dosyası, kabul edilen kayıt durumları, k değerleri ve retrieval override'larını belirler. İlk üçü embedding/vektör indeksi veya BM25 chunk indeksini; reranker deneyi ayrıca CrossEncoder ağırlıklarını gerektirir.

```powershell
python -m src.ingestion.cli --offline
python -m src.retrieval.index_cli --config configs/rag.json
python -m eval.run --config configs/eval-bm25.json
python -m eval.compare --results-dir results --output-dir results/comparison
```

`eval.run` retrieval ve opsiyonel cevap üretim modunu config'ten çalıştırır. Üretim değerlendirmesi için config'te `run_generation` değerini `true` yapın; LLM çağrıları maliyet/zaman doğurabilir. Her deney soru bazında ve toplam metrikleri JSON'a, Excel uyumlu CSV'ye yazar. JSON; UTC çalıştırma zamanı, corpus/config/soru seti SHA-256 değerleri, retrieval stratejisi, embedding, chunking ve reranker ayarlarını da saklar.

`eval.compare` var olan `results/*.json` dosyalarından `comparison.csv`, `comparison.md` ve SVG bar grafik üretir. Eksik sonuç üretmez; gerçek deney çalıştırılmadıysa karşılaştırma dosyası oluşturulamaz. Deneyler tamamlandığında README sonuç tablosuna `comparison.md` içeriği aktarılabilir.

## Üretim kontrolü ve hata incelemesi

`citation_support_check` cevap cümlelerinde geçerli `[S1]` benzeri kaynak etiketlerinin bulunma oranını ölçen basit bir heuristic'tir. Kaynağın iddiayı anlamsal olarak desteklediğini doğrulamaz; bu nedenle metric adı `citation_coverage` olarak tutulur ve tek başına faithfulness kanıtı sayılmaz. Bulunamadı yanıtları citation coverage ortalamasına katılmaz; JSON sonucunda abstention olarak ayrı işaretlenir. `eval/error_analysis_template.csv`, yanlış/eksik retrieval, kaynak uyuşmazlığı, desteksiz iddia, eksik atıf, yanlış abstention ve diğer hata türleri için elle inceleme kaydı sağlar.

## Deney dürüstlüğü ve sınırlar

Bu depoda 40 soru taslağı vardır; hiçbiri insan doğrulaması yapılmış evaluation örneği olarak sunulmaz. Faz 4 kodlama/doğrulama sırasında gerçek embedding, reranker veya LLM deneyi çalıştırılmadı; dolayısıyla README'ye aktarılabilecek performans metriği henüz yoktur. Kaynak mevzuat metni değişebileceğinden corpus hash'i ve soru seti hash'i deneyle birlikte saklanmalıdır. Değerlendirme hukuki doğruluk sertifikası veya hukuki tavsiye değildir.
