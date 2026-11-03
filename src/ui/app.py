"""FastAPI RAG endpoint'ine bağlanan Gradio mevzuat araştırma arayüzü."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

APP_CSS = """
:root {
    --ink: #172a3a;
    --muted: #647586;
    --line: #e1e9ee;
    --teal: #087e78;
    --teal-soft: #e8f5f3;
}
body, .gradio-container {
    background: #f4f7f8 !important;
    color: var(--ink) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
.gradio-container {
    max-width: 1160px !important;
    padding: 12px 22px 28px !important;
}
#topbar {
    align-items: center;
    border-bottom: 1px solid #e2e9ed;
    height: 54px !important;
    margin: 0 0 17px;
    min-height: 54px !important;
    padding: 0 0 9px;
}
#topbar > div { min-height: 0 !important; }
#topbar .prose p { margin: 0 !important; }
#brand-mark {
    align-items: center;
    background: #0c716d;
    border-radius: 10px;
    color: white;
    display: inline-flex;
    font-size: 15px;
    font-weight: 800;
    height: 34px;
    justify-content: center;
    letter-spacing: -.04em;
    margin-right: 11px;
    width: 34px;
}
#brand-copy { color: #20384a; font-size: 14px; font-weight: 750; }
#brand-copy span {
    color: #70808d;
    display: block;
    font-size: 10px;
    font-weight: 500;
    margin-top: 1px;
}
.surface {
    background: white !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 28px rgba(27, 55, 70, .045) !important;
}
#question-card { gap: 0 !important; padding: 13px 16px 11px !important; }
#question-card #question-label,
#question-card #question-hint,
#question-card #examples-label { margin: 0 !important; padding: 0 !important; }
#question-card #question-label .prose p,
#question-card #question-hint .prose p,
#question-card #examples-label .prose p { margin: 0 !important; }
#question-label { color: #243b4b; font-size: 13px; font-weight: 750; }
#question-hint { color: #788793; font-size: 11px; margin-top: 4px !important; }
#question-input textarea {
    background: #fbfcfd !important;
    border: 1px solid #dce5e9 !important;
    border-radius: 13px !important;
    box-shadow: none !important;
    color: #20384a !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    min-height: 66px !important;
    padding: 10px 13px !important;
}
#question-input textarea:focus {
    border-color: #6cb5ad !important;
    box-shadow: 0 0 0 3px #e5f4f1 !important;
}
#submit-row { align-items: center; margin-top: 6px; }
#ask-button button {
    background: #087e78 !important;
    border: 0 !important;
    border-radius: 11px !important;
    box-shadow: 0 5px 12px rgba(8, 126, 120, .18) !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    min-height: 40px !important;
    padding: 0 21px !important;
}
#ask-button button:hover { background: #066a65 !important; }
#query-note { color: #82909a; font-size: 10px; padding: 0 4px; }
#examples { margin: 9px 0 0; }
#examples-label { color: #73818c; font-size: 10px; font-weight: 700; margin: 0 0 5px !important; }
.example-button button {
    background: #f7f9fa !important;
    border: 1px solid #e1e8ec !important;
    border-radius: 999px !important;
    color: #4b6170 !important;
    font-size: 11px !important;
    min-height: 28px !important;
    padding: 0 10px !important;
}
.example-button button:hover {
    background: var(--teal-soft) !important;
    border-color: #c9e5e0 !important;
    color: #176d67 !important;
}
#results-row { align-items: stretch; margin-top: 17px; }
.result-card { height: 100%; min-height: 258px; padding: 19px 20px !important; }
#answer-heading, #sources-heading { align-items: center; margin-bottom: 14px; }
.section-icon {
    align-items: center;
    background: #eaf5f3;
    border-radius: 9px;
    color: #14736d;
    display: inline-flex;
    font-size: 13px;
    font-weight: 800;
    height: 30px;
    justify-content: center;
    margin-right: 9px;
    width: 30px;
}
.section-title { color: #243b4b; font-size: 14px; font-weight: 750; }
.section-subtitle { color: #82909a; font-size: 11px; margin-top: 2px; }
#answer-panel { color: #334b5c; font-size: 14px; line-height: 1.8; min-height: 180px; }
#answer-panel p { margin-top: 0; }
#answer-panel code { background: #f0f4f5; border-radius: 4px; color: #2c5263; }
#loading-panel { text-align: center; }
.loading-state {
    align-items: center;
    animation: loading-appear .25s ease-out both;
    background: linear-gradient(145deg, #f4fbfa, #fbfdfd);
    border: 1px solid #e3f0ee;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 2px 0 8px;
    min-height: 190px;
    padding: 18px;
}
.loader-orbit {
    animation: loader-spin 1.15s linear infinite;
    border: 3px solid #d8eeeb;
    border-radius: 50%;
    height: 72px;
    margin-bottom: 15px;
    position: relative;
    width: 72px;
}
.loader-orbit::before {
    background: #087e78;
    border: 4px solid #ffffff;
    border-radius: 50%;
    box-shadow: 0 2px 9px rgba(8, 126, 120, .25);
    content: "";
    height: 14px;
    left: -5px;
    position: absolute;
    top: 18px;
    width: 14px;
}
.loader-core {
    align-items: center;
    animation: loader-pulse 1.4s ease-in-out infinite;
    background: #e6f4f2;
    border-radius: 50%;
    color: #087e78;
    display: flex;
    font-size: 21px;
    font-weight: 800;
    height: 48px;
    justify-content: center;
    left: 9px;
    position: absolute;
    top: 9px;
    width: 48px;
}
.loading-title { color: #294353; font-size: 14px; font-weight: 750; }
.loading-caption { color: #80909a; font-size: 11px; margin-top: 5px; }
@keyframes loader-spin { to { transform: rotate(360deg); } }
@keyframes loader-pulse { 50% { transform: scale(.88); opacity: .72; } }
@keyframes loading-appear {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
#sources-panel { color: #435a69; font-size: 12px; line-height: 1.65; }
#sources-panel ul { list-style: none; margin: 0; padding: 0; }
#sources-panel li {
    background: #f8fafb;
    border: 1px solid #e6ecef;
    border-radius: 11px;
    margin: 0 0 9px;
    padding: 11px 12px;
}
#sources-panel a { color: #087e78; font-size: 11px; font-weight: 700; text-decoration: none; }
#sources-panel a:hover { text-decoration: underline; }
#disclaimer {
    color: #87939c;
    font-size: 11px;
    line-height: 1.6;
    margin: 17px auto 0;
    max-width: 740px;
    text-align: center;
}
footer { display: none !important; }
@media (max-width: 760px) {
    .gradio-container { padding: 14px 13px 24px !important; }
    #topbar { margin-bottom: 13px; }
    #question-card, .result-card { padding: 12px !important; }
    #results-row { gap: 12px; }
    #query-note { padding-top: 7px; }
}
@media (prefers-reduced-motion: reduce) {
    .loader-orbit, .loader-core, .loading-state {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
    }
}
"""


def _plain_markdown(value: str) -> str:
    """Mevzuat metadata'sını Markdown metni içinde güvenli ve düz gösterir."""
    for marker in ("\\", "`", "*", "_", "[", "]", "<", ">", "#", "|"):
        value = value.replace(marker, f"\\{marker}")
    return value


def ask_api(question: str, *, api_url: str, timeout: float = 180.0) -> tuple[str, str]:
    """API cevabını metin ve doğrulanabilir kaynak bağlantıları olarak döndürür."""
    if not question.strip():
        return "Lütfen bir soru yazın.", ""
    request = Request(
        api_url,
        data=json.dumps({"question": question.strip()}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return f"API hatası ({error.code}): {detail}", ""
    except (TimeoutError, URLError) as error:
        return f"API bağlantısı kurulamadı: {error}", ""

    answer = str(result.get("answer", "Yanıt alınamadı."))
    answer = answer.split("\n\nKaynaklar:", maxsplit=1)[0]
    citations = result.get("citations", [])
    if not citations:
        if result.get("verification_status") == "failed":
            return (
                answer,
                "Otomatik kaynak kontrolü yanıtı destekleyemedi; kaynak kartları gösterilmiyor.",
            )
        return answer, "İlgili bağlam bulunamadığı için bu yanıta dayanak gösterilemiyor."

    verification_status = result.get("verification_status", "not_checked")
    if verification_status == "passed":
        sources = [
            "_Otomatik kaynak kontrolü: yanıt, gösterilen metinlerle uyumlu bulundu. "
            "Bu model tabanlı kontrol hatasızlık garantisi değildir._",
            "",
        ]
    else:
        sources = ["_Bu yanıt için otomatik kaynak destek kontrolü yapılmadı._", ""]
    for citation in citations:
        paragraph = (
            f", fıkra {_plain_markdown(str(citation['paragraph_no']))}"
            if citation.get("paragraph_no")
            else ""
        )
        label = _plain_markdown(str(citation.get("label", "Kaynak")))
        law = _plain_markdown(str(citation.get("law_name", "Mevzuat kaynağı")))
        article = _plain_markdown(str(citation.get("article_no", "?")))
        reference = "Bölüm" if citation.get("unit_type") == "guidance" else "Madde"
        source = f"**{label} · {law}**  \n{reference} {article}{paragraph}"
        excerpt = " ".join(str(citation.get("excerpt", "")).split())
        if excerpt:
            source += f"\n\n> _{_plain_markdown(excerpt)}_"
        url = str(citation.get("source_url", ""))
        if urlsplit(url).scheme in {"http", "https"} and urlsplit(url).netloc:
            sources.append(f"- {source}  \n  [Resmî kaynağı aç ↗]({url})")
        else:
            sources.append(f"- {source}")
    return answer, "\n".join(sources)


def build_demo():
    """Responsive, kaynak görünürlüğü odaklı Gradio arayüzünü oluşturur."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Arayüz için `pip install -e '.[web]'` çalıştırın.") from error

    api_url = os.getenv("KVKK_API_URL", "http://127.0.0.1:8001/ask")

    def respond(question_text: str) -> Iterator[tuple[str, str, str]]:
        """Bekleme durumunu hemen, sonra yanıtı ve dayanaklarını gösterir."""
        if not question_text.strip():
            yield (
                "Bir mevzuat sorusu yazın; yanıtınız burada görünecek.",
                "Soruyla ilişkili dayanak maddeler burada listelenir.",
                "",
            )
            return
        loading_html = (
            '<div class="loading-state" role="status" aria-live="polite">'
            '<div class="loader-orbit"><div class="loader-core">§</div></div>'
            '<div class="loading-title">Mevzuat taranıyor</div>'
            '<div class="loading-caption">Sorunuz, ilgili kanun ve maddelerle eşleştiriliyor…</div>'
            "</div>"
        )
        yield "", "Dayanak maddeler aranıyor…", loading_html
        answer_text, source_text = ask_api(question_text, api_url=api_url)
        yield answer_text, source_text, ""

    examples = (
        "İlgili kişinin KVKK kapsamındaki hakları nelerdir?",
        "Kişisel veriler yurt dışına hangi şartlarda aktarılabilir?",
        "Veri sorumlusu başvurulara kaç gün içinde yanıt vermelidir?",
    )

    with gr.Blocks(title="KVKK RAG · Mevzuat Araştırma") as demo:
        with gr.Row(elem_id="topbar"):
            gr.HTML(
                '<div id="brand-mark">K</div>'
                '<div id="brand-copy">KVKK RAG<span>MEVZUAT ARAŞTIRMA</span></div>'
            )
            gr.Markdown("Kamuya açık kaynaklarla çalışır", elem_id="query-note")

        with gr.Column(elem_id="question-card", elem_classes=["surface"]):
            gr.Markdown("Sorunuzu sorun", elem_id="question-label")
            gr.Markdown(
                "Doğal dille yazın; mümkünse konuyu ve yapmak istediğiniz işlemi belirtin.",
                elem_id="question-hint",
            )
            question = gr.Textbox(
                placeholder=(
                    "Örn. Sağlık verisini yurt dışına aktarmak isteyen bir şirket "
                    "hangi koşulları sağlamalı?"
                ),
                lines=2,
                max_length=2000,
                show_label=False,
                elem_id="question-input",
            )
            with gr.Row(elem_id="submit-row"):
                submit = gr.Button("Yanıtı getir  →", variant="primary", elem_id="ask-button")
                gr.Markdown(
                    "Kaynaklar yanıttan ayrı gösterilir ve bağlantı olarak açılabilir.",
                    elem_id="query-note",
                )
            with gr.Column(elem_id="examples"):
                gr.Markdown("ÖRNEK SORULAR", elem_id="examples-label")
                with gr.Row():
                    example_buttons = [
                        gr.Button(label, size="sm", elem_classes=["example-button"])
                        for label in examples
                    ]

        with gr.Row(elem_id="results-row"):
            with gr.Column(scale=7, min_width=320, elem_classes=["surface", "result-card"]):
                with gr.Row(elem_id="answer-heading"):
                    gr.HTML(
                        '<span class="section-icon">✦</span>'
                        '<div><div class="section-title">Yanıt</div>'
                        '<div class="section-subtitle">Getirilen mevzuat bağlamına göre</div></div>'
                    )
                loading = gr.HTML("", elem_id="loading-panel")
                answer = gr.Markdown(
                    "Sorunuzu gönderdiğinizde yanıt burada görünecek.",
                    elem_id="answer-panel",
                )
            with gr.Column(scale=5, min_width=280, elem_classes=["surface", "result-card"]):
                with gr.Row(elem_id="sources-heading"):
                    gr.HTML(
                        '<span class="section-icon">§</span>'
                        '<div><div class="section-title">Dayanaklar</div>'
                        '<div class="section-subtitle">İlgili kanun ve maddeler</div></div>'
                    )
                sources = gr.Markdown(
                    "Yanıtın kullandığı kaynaklar ve madde bağlantıları burada listelenir.",
                    elem_id="sources-panel",
                )

        gr.Markdown(
            "Yanıtları ilgili mevzuat metninden kontrol edin. "
            "Bu araç bilgilendirme ve araştırma amaçlıdır; hukuki tavsiye değildir.",
            elem_id="disclaimer",
        )
        submit.click(respond, inputs=question, outputs=[answer, sources, loading])
        question.submit(respond, inputs=question, outputs=[answer, sources, loading])
        for button, example in zip(example_buttons, examples, strict=True):
            button.click(lambda text=example: text, outputs=question)
    return demo


def main() -> None:
    """Gradio sunucusunu yerel geliştirme veya container portunda başlatır."""
    import gradio as gr

    demo = build_demo()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="slate"),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
