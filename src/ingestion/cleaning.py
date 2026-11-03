"""HTML'den görünen metni çıkarıp mevzuat metnini normalize eder."""

from __future__ import annotations

import re
from codecs import lookup
from html.parser import HTMLParser


class _VisibleTextParser(HTMLParser):
    """Script ve stil içeriğini atlayıp HTML görünür metnini toplar."""

    _BLOCK_TAGS = {"br", "div", "p", "li", "tr", "td", "th", "h1", "h2", "h3"}

    def __init__(self, content_class: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._content_class = content_class
        self._capturing = content_class is None
        self._content_found = content_class is None
        self._content_tag: str | None = None
        self._content_depth = 0
        self._finished = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if self._finished:
            return
        if not self._capturing:
            classes = next((value for key, value in attrs if key.lower() == "class"), "") or ""
            if self._content_class not in classes.split():
                return
            self._capturing = True
            self._content_found = True
            self._content_tag = lowered
            self._content_depth = 1
            self.parts.append("\n")
            return
        if self._content_tag and lowered == self._content_tag:
            self._content_depth += 1
        if lowered in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif self._skip_depth == 0 and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if not self._capturing or self._finished:
            return
        if lowered in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")
        if self._content_tag and lowered == self._content_tag:
            self._content_depth -= 1
            if self._content_depth == 0:
                self._capturing = False
                self._finished = True

    def handle_data(self, data: str) -> None:
        if self._capturing and self._skip_depth == 0:
            self.parts.append(data)


_CHARSET_RE = re.compile(
    rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]+)", re.IGNORECASE
)


def decode_html(payload: bytes) -> str:
    """Decode HTML using its BOM/meta charset, with strict UTF-8/Windows-1254 fallbacks."""
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig")
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")

    header = payload[:8192]
    match = _CHARSET_RE.search(header)
    if match:
        charset = match.group(1).decode("ascii")
        try:
            lookup(charset)
        except LookupError as error:
            raise ValueError(f"HTML kaynağında bilinmeyen charset: {charset}") from error
        try:
            return payload.decode(charset, errors="strict")
        except UnicodeDecodeError:
            if charset.casefold().replace("_", "-") not in {
                "windows-1254",
                "cp1254",
                "iso-8859-9",
            }:
                raise
            # Some Turkish government pages declare Windows-1254 while
            # containing a mixture of UTF-8 and legacy single-byte characters.
            mixed = payload.decode("utf-8", errors="surrogateescape")
            decoded: list[str] = []
            for character in mixed:
                codepoint = ord(character)
                if not 0xDC80 <= codepoint <= 0xDCFF:
                    decoded.append(character)
                    continue
                byte = bytes([codepoint - 0xDC00])
                codec = "windows-1252" if 0x80 <= byte[0] <= 0x9F else "windows-1254"
                decoded.append(byte.decode(codec, errors="replace"))
            return "".join(decoded)

    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return payload.decode("windows-1254", errors="strict")


def normalize_text(text: str) -> str:
    """Boşlukları düzenler, satır sonlarını korur ve boş satır tekrarlarını azaltır."""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def html_to_text(html: str, *, content_class: str | None = None) -> str:
    """HTML belgesinden script/stil dışındaki görünür metni çıkarır."""
    parser = _VisibleTextParser(content_class)
    parser.feed(html)
    parser.close()
    if content_class is not None and not parser._content_found:
        parser = _VisibleTextParser()
        parser.feed(html)
        parser.close()
    return normalize_text("".join(parser.parts))
