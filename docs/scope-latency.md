# Korpus kapsamı ve gecikme

`configs/sources.json` şu anda dört resmî mevzuat kaynağı ve KVKK Kurumunun yedi konu sayfasını listeler. Kurumun konu sayfaları; aydınlatma, veri güvenliği, özel nitelikli veriler, yurt dışına aktarım, ilgili kişi hakları ve başvurular ile açık rıza konularını kapsar. Bunlar rehber niteliğinde kaynaklardır; sistem atıflarında “Bölüm” olarak gösterilir, kanun maddesi gibi sunulmaz.

Yeni kaynak manifestini yerel corpus'a almak için:

```powershell
python -m src.ingestion.cli
python -m src.retrieval.index_cli --config configs/rag.json
```

İlk komut resmî sayfaları `data/raw/` içine indirip `data/processed/legal_units.jsonl` dosyasını yeniler. İkinci komut embeddingleri, Chroma vektör koleksiyonunu ve BM25 indeksini yeniden üretir. API ve arayüz süreçlerini yeniden başlatın. İndirme yalnızca resmî kurum URL'lerinden yapılır; ham ve işlenmiş veriler repoya eklenmez.

Bu çalışma ortamında terminalden dış ağa erişim işletim sistemi tarafından engellendiği için yeni sayfaların indirilmesi ve indeksin gerçek verilerle yeniden kurulması tamamlanamadı. Bu nedenle projede bulunan mevcut corpus yeni 7 sayfayı henüz içermiyor. İndirme hatası sessizce başarı gibi gösterilmez.

## Gecikme

- `configs/generation.json`: Ollama/Qwen için `think: false` düşünme tokenlarını kapatır; `keep_alive: 30m` modelin her istekten sonra bellekten atılmasını önlemeyi amaçlar; `num_predict: 384` yanıt uzunluğunu sınırlar.
- `configs/rag.json`: hibrit arama 4 parçaya düşürüldü, böylece modele giden bağlam ve generation prompt kısalır.
- Kaynak destek denetimi açık kalır ve yanıt üretiminin ardından ikinci LLM çağrısı yapar. Doğrulamayı kapatmak gecikmeyi azaltabilir ancak atıf kontrolü güvencesini de azaltacağı için varsayılanı değiştirilmedi.
- `/ask` yanıtındaki `latency_ms`, API isteğinin servis çağrısından sonraki yaklaşık süresini verir. Soğuk başlangıç, makine/GPU ve model farkları nedeniyle bu ayarlar için ölçülmüş bir hızlanma oranı iddia edilmez.

## Sınırlamalar

KVKK Kurumunun konu sayfaları, burada rehber türünde ikincil açıklama kaynağı olarak indekslenir. Resmî mevzuat metinleri bağlayıcı birincil kaynak olmaya devam eder. Karar özetleri henüz eklenmedi; yayımlanmış karar sayfaları olay ve kişi bağlamı içerebildiğinden, kişisel veri kısıtına uygun seçme ve anonimlik incelemesi yapılmadan topluca içeri alınmamalıdır.
