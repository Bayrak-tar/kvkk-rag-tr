# Faz 1 veri kaynakları

Bu fazın ilk manifesti `configs/sources.json` içinde yer alır. Yalnızca resmî kamu kaynakları eklenmiştir:

| Kaynak | Yayıncı | URL |
| --- | --- | --- |
| 6698 sayılı Kişisel Verilerin Korunması Kanunu | T.C. Mevzuat Bilgi Sistemi | <https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=1&MevzuatNo=6698&MevzuatTertip=5> |
| Yurt dışına aktarım yönetmeliği (10.07.2024) | T.C. Resmî Gazete | <https://www.resmigazete.gov.tr/eskiler/2024/07/20240710-2.htm> |
| Silme, yok etme veya anonim hale getirme yönetmeliği | Kişisel Verileri Koruma Kurumu | <https://www.kvkk.gov.tr/Icerik/5441/KISISEL-VERILERIN-SILINMESI-YOK-EDILMESI-VEYA-ANONIM-HALE-GETIRILMESI-HAKKINDA-YONETMELIK> |
| Veri Sorumluları Sicili Hakkında Yönetmelik | Kişisel Verileri Koruma Kurumu | <https://www.kvkk.gov.tr/Icerik/5442/VERI-SORUMLULARI-SICILI-HAKKINDA-YONETMELIK> |

Kaynak listesi başlangıç kapsamıdır; KVKK sitesindeki mevzuat dizini başka yönetmelik ve tebliğleri de içerir. Her ek kaynakta resmî yayıncı, doğrudan URL ve metin biçimi manifestte açıkça belirtilmelidir. Mevzuat Bilgi Sistemi bağlantısı ağ koşullarına göre zaman aşımına uğrayabileceğinden indirme tekrar denenebilir; indirme başarısızsa parser boş/başarılı sonuç taklidi yapmaz.

## İndirme ve çıktı

```powershell
python -m src.ingestion.cli
```

Ham yanıtlar `data/raw/`, madde/fıkra kayıtları `data/processed/legal_units.jsonl` içine yazılır. Bu dosyalar `.gitignore` ile dışarıda tutulur; her kullanıcı resmî kaynaktan yeniden üretebilir. Kaynak URL'si, kanun/yönetmelik adı ve madde/fıkra numarası her yapılandırılmış kayıtta bulunur.

Ham dosyalar mevcutsa ayrıştırmayı yeniden ağ kullanmadan çalıştırın:

```powershell
python -m src.ingestion.cli --offline
```

Kaynak metinlerin değişebileceğini göz önünde bulundurun. Üretim deneyleri için indirme tarihini ve kaynak yanıtının hash'ini de sonraki fazda deney metadata'sına eklemek gerekir.

## Chunking

`configs/chunking.json` varsayılan stratejiyi ve üç stratejinin ayarlarını tutar. `src.chunking.config.load_chunker_config()` seçilen stratejiyi yükler. Semantic strateji, embedding modelini Faz 2'de seçebilmek için bir benzerlik fonksiyonunu dışarıdan alır; bu fazda chunker sahte embedding üretmez.

## Additional official KVKK topic pages

The manifest also includes these publicly available pages. They are indexed as guidance sections, not statutory articles:

- Aydınlatma: <https://www.kvkk.gov.tr/Icerik/2033/Aydinlatma-Yukumlulugu->
- Veri güvenliği: <https://www.kvkk.gov.tr/Icerik/2040/Veri-Guvenligine-Iliskin-Yukumlulukler>
- Özel nitelikli kişisel veriler: <https://www.kvkk.gov.tr/Icerik/2051/Ozel-Nitelikli-Kisisel-Veriler>
- Yurt dışına aktarım: <https://www.kvkk.gov.tr/Icerik/2053/Yurtdisina-Aktarim>
- İlgili kişinin hakları: <https://www.kvkk.gov.tr/Icerik/2036/Ilgili-Kisinin-Haklari>
- İlgili kişi başvurularının cevaplanması: <https://www.kvkk.gov.tr/Icerik/2046/Ilgili-Kisiler-Tarafindan-Yapilan-Basvurularin-Cevaplanmasi-Yukumlulugu>
- Açık rıza: <https://www.kvkk.gov.tr/Icerik/2037/Acik-Riza-Alirken-Dikkat-Edilecek-Hususlar>

Additional pages appear in the [official KVKK guides index](https://www.kvkk.gov.tr/Icerik/2030/rehberler).
