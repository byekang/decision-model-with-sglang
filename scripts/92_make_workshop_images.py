"""Workshop: render the synthetic image fixtures (PIL), in Korean and English.

Three screenshots a banking customer would realistically upload to support chat,
produced per language with a filename prefix:
  [KR]_receipt_dispute.png / [EN]_receipt_dispute.png : card-approval alert (large FX amount)
  [KR]_phishing_sms.png   / [EN]_phishing_sms.png     : classic smishing text
  [KR]_legit_sms.png      / [EN]_legit_sms.png        : legitimate approval notice

All content is synthetic (fictional bank/merchant/numbers).
Output: data/workshop/*.png
Usage: python3 scripts/92_make_workshop_images.py   (system python has PIL)
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_DIR, log

PHASE = "workshop_images"
OUT_DIR = DATA_DIR / "workshop"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def fonts(size):
    return (ImageFont.truetype(FONT_PATH, size),
            ImageFont.truetype(FONT_BOLD if Path(FONT_BOLD).exists() else FONT_PATH, size))


def phone_frame(w, h, title):
    img = Image.new("RGB", (w, h), "#F2F3F5")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 64], fill="#2C3E50")
    f, fb = fonts(26)
    d.text((20, 16), title, font=fb, fill="white")
    return img, d


def bubble(d, img, xy, lines, size=24, fill="#FFFFFF", color="#1A1A1A", pad=18, lh=1.45):
    f, _ = fonts(size)
    wmax = max(d.textlength(t, font=f) for t in lines)
    x, y = xy
    hline = int(size * lh)
    box = [x, y, x + wmax + pad * 2, y + hline * len(lines) + pad * 2]
    d.rounded_rectangle(box, radius=16, fill=fill, outline="#DDDDDD")
    for i, t in enumerate(lines):
        d.text((x + pad, y + pad + i * hline), t, font=f, fill=color)
    return box[3]


RECEIPT = {
    "ko": {
        "title": "KB한빛카드 승인 알림", "kind": "해외 승인 (ONLINE)",
        "merchant": "LUXE GADGETS LTD (GB)",
        "amount": "USD 1,284.00  (약 1,742,000원)",
        "rows": [("승인일시", "2026-10-27 03:41 (KST)"),
                 ("카드", "체크카드 (끝자리 7712)"),
                 ("승인번호", "88213457"),
                 ("가맹점 유형", "전자상거래 / 해외"),
                 ("할부", "일시불"),
                 ("누적 사용액(당월)", "2,311,400원")],
        "footer": "본인 결제가 아닌 경우 즉시 카드사로 문의하세요.",
    },
    "en": {
        "title": "HANBIT CARD Approval Alert", "kind": "OVERSEAS APPROVAL (ONLINE)",
        "merchant": "LUXE GADGETS LTD (GB)",
        "amount": "USD 1,284.00",
        "rows": [("Approved at", "2026-10-27 03:41 (KST)"),
                 ("Card", "Debit card (ending 7712)"),
                 ("Approval no.", "88213457"),
                 ("Merchant type", "E-commerce / Overseas"),
                 ("Installments", "None (single charge)"),
                 ("Month-to-date spend", "USD 1,710.40")],
        "footer": "Not your payment? Contact the card issuer immediately.",
    },
}

PHISHING = {
    "ko": {
        "frame": "문자 메시지", "sender": "발신번호 010-2847-XXXX (국제발신)",
        "lines": ["[Web발신] (광고)KB한빛은행",
                  "고객님의 계좌가 금일 02:00부로",
                  "[지급정지] 처리되었습니다.",
                  "",
                  "본인확인 후 정지 해제가 가능합니다.",
                  "아래 주소에서 보안카드 번호 전체를",
                  "입력해 주세요.",
                  "",
                  "▶ http://kb-hanbit-safe.xyz/unlock",
                  "",
                  "미처리 시 전 계좌 출금이 제한됩니다."],
    },
    "en": {
        "frame": "Messages", "sender": "From +44 7911 0XXXXX (INTERNATIONAL)",
        "lines": ["[HANBIT BANK] Security Notice",
                  "Your account was [SUSPENDED]",
                  "today at 02:00 AM.",
                  "",
                  "Verify your identity to lift the",
                  "suspension. Enter ALL of your",
                  "security card numbers at:",
                  "",
                  "> http://hanbit-safe.xyz/unlock",
                  "",
                  "Withdrawals from all accounts will",
                  "be blocked if not completed."],
    },
}

LEGIT = {
    "ko": {
        "frame": "문자 메시지", "sender": "발신번호 1588-XXXX",
        "lines": ["[KB한빛카드]", "10/27 14:02", "체크카드(7712) 승인",
                  "45,000원 일시불", "TechMart 강남점", "누적 2,356,400원"],
    },
    "en": {
        "frame": "Messages", "sender": "From 1588-XXXX",
        "lines": ["[HANBIT CARD]", "10/27 14:02", "Debit card (7712) approved",
                  "USD 34.60", "TechMart Gangnam", "MTD total USD 1,745.00"],
    },
}


def make_receipt(lang):
    c = RECEIPT[lang]
    img = Image.new("RGB", (560, 640), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f18, _ = fonts(18)
    f22, b22 = fonts(22)
    _, b26 = fonts(26)
    _, b34 = fonts(34)
    d.rectangle([0, 0, 560, 70], fill="#1B4F72")
    d.text((24, 18), c["title"], font=b26, fill="white")
    d.text((24, 95), c["kind"], font=b22, fill="#C0392B")
    d.text((24, 140), c["merchant"], font=b34, fill="#1A1A1A")
    d.text((24, 195), c["amount"], font=b34, fill="#C0392B")
    y = 265
    for k, v in c["rows"]:
        d.text((24, y), k, font=f22, fill="#777777")
        d.text((250, y), v, font=f22, fill="#1A1A1A")
        y += 44
    d.line([24, y + 8, 536, y + 8], fill="#DDDDDD", width=2)
    d.text((24, y + 24), c["footer"], font=f18, fill="#777777")
    return img


def make_phishing(lang):
    c = PHISHING[lang]
    img, d = phone_frame(560, 560, c["frame"])
    f, _ = fonts(20)
    d.text((20, 84), c["sender"], font=f, fill="#888888")
    bubble(d, img, (20, 120), c["lines"], size=23)
    return img


def make_legit(lang):
    c = LEGIT[lang]
    img, d = phone_frame(560, 360, c["frame"])
    f, _ = fonts(20)
    d.text((20, 84), c["sender"], font=f, fill="#888888")
    bubble(d, img, (20, 120), c["lines"], size=23)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = {"ko": "[KR]_", "en": "[EN]_"}
    for lang in ("ko", "en"):
        for name, fn in (("receipt_dispute", make_receipt),
                         ("phishing_sms", make_phishing),
                         ("legit_sms", make_legit)):
            p = OUT_DIR / f"{prefix[lang]}{name}.png"
            fn(lang).save(p)
            log(PHASE, f"wrote {p}")


if __name__ == "__main__":
    main()
