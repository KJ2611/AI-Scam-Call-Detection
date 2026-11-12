"""Transparent, rule-based suspicious-pattern indicators.

These are NOT the machine-learning model. They are hand-written phrase patterns, grouped
into social-engineering categories, that the app shows *separately* from model evidence.
A matched phrase is a reason to look closer, never proof of fraud on its own.

Each category has a severity:
* HIGH   — requests that legitimate organisations essentially never make on a call
           (OTP/PIN/password, remote-access apps, moving money to a "safe" account).
* MEDIUM — common manipulation tactics that also occur in legitimate calls
           (urgency, authority claims, prizes, verification language).

"Safe signals" (e.g. "we will never ask for your OTP") are detected too, and suppress
an overlapping credential match, so that a genuine bank warning is not flagged as a request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HIGH, MEDIUM = "HIGH", "MEDIUM"


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    severity: str
    why: str
    advice: str
    patterns: tuple[str, ...]
    hmm_words: tuple[str, ...] = field(default_factory=tuple)


CATEGORIES: tuple[Category, ...] = (
    Category(
        "credential_request", "OTP / PIN / password request", HIGH,
        "The caller asks for a one-time password, PIN, CVV, password or login details. Banks and payment apps do not ask for these over a call.",
        "Never read out an OTP, PIN, CVV or password to anyone — hang up and call the official number yourself.",
        (r"\botp\b", r"one[\s-]?time[\s-]?pass(?:word|code)", r"\bpin\b(?:\s+(?:number|code))?", r"\bcvv\b", r"\bcvc\b",
         r"pass(?:word|code)", r"verification code", r"security code", r"(?:the )?code (?:we|i) (?:just )?sent",
         r"login (?:details|credentials|id)", r"net ?banking (?:id|password|details)", r"card (?:number|details)",
         r"expiry date", r"\bmpin\b", r"\bupi pin\b", r"digit code"),
        ("otp", "pin", "cvv", "password", "passcode", "code", "mpin", "credentials", "login", "expiry"),
    ),
    Category(
        "remote_access", "Remote-access / app installation", HIGH,
        "The caller asks you to install an app or share your screen. Remote-access tools let a scammer control your phone or computer.",
        "Do not install apps (AnyDesk, TeamViewer, QuickSupport…) or share your screen at a caller's request.",
        (r"any ?desk", r"team ?viewer", r"quick ?support", r"rust ?desk", r"screen ?shar(?:e|ing)", r"remote (?:access|control|desktop)",
         r"(?:install|download) (?:the |this |an |our )?(?:app|application|software)", r"\bapk\b"),
        ("anydesk", "teamviewer", "quicksupport", "install", "download", "apk", "remote", "screen"),
    ),
    Category(
        "payment_request", "Payment / money-transfer request", HIGH,
        "The caller asks you to pay, transfer, deposit or buy something (fees, gift cards, crypto, a 'safe account').",
        "Do not pay 'processing', 'release' or 'verification' fees, and never move money to a 'safe account' because a caller told you to.",
        (r"transfer (?:the |your |some )?(?:money|amount|funds|balance)", r"(?:processing|registration|release|clearance|customs|handling|verification|service|advance) (?:fee|charge)s?",
         r"gift ?cards?", r"(?:itunes|google play|amazon|steam) cards?", r"\bwire\b(?: transfer)?", r"safe (?:account|wallet)",
         r"(?:send|pay|deposit|remit) (?:the |a |us |me )?(?:money|amount|payment|fee|deposit|rs|rupees|\$|dollars)", r"bitcoin|crypto(?:currency)?|usdt",
         r"\b(?:paytm|phonepe|gpay|google ?pay|bhim|paypal|venmo|zelle|cash ?app)\b", r"\bupi\b", r"\bpay (?:now|immediately|today|first|the)\b", r"refundable deposit", r"scan (?:the |this )?qr"),
        ("transfer", "pay", "payment", "deposit", "fee", "fees", "remit", "wire", "giftcard", "bitcoin", "crypto", "upi", "__payapp__", "__money__", "funds", "refund", "remittance"),
    ),
    Category(
        "personal_info_request", "Personal / identity information request", MEDIUM,
        "The caller asks for identity documents or personal details (Aadhaar, PAN, SSN, date of birth, account number).",
        "Share identity details only through official channels you contacted yourself.",
        (r"aadha?a?r", r"\bpan (?:card|number)\b", r"social security(?: number)?", r"\bssn\b", r"date of birth", r"\bdob\b",
         r"(?:bank )?account (?:number|details)", r"id (?:number|card)", r"identity (?:card|number|information|details)", r"personal (?:information|details|data)",
         r"mother'?s maiden name", r"passport (?:number|details)", r"full name and address"),
        ("aadhaar", "aadhar", "pan", "ssn", "birth", "identity", "id", "passport", "address", "personal", "details", "information"),
    ),
    Category(
        "account_verification", "Account problem / verification claim", MEDIUM,
        "The caller says your account, card or SIM has a problem and must be 'verified' or 'updated' (KYC, suspicious activity, suspension).",
        "If you are told your account has a problem, end the call and check in the official app or at the branch.",
        (r"\bkyc\b", r"verify (?:your )?(?:account|identity|details|information)", r"(?:account|card|sim|number) (?:will be |has been |is )?(?:blocked|suspended|frozen|deactivated|closed|locked)",
         r"(?:unusual|suspicious|abnormal|unauthori[sz]ed) (?:activity|transactions?|login|access)", r"update (?:your )?(?:kyc|details|account|information)",
         r"re-?activate", r"security (?:check|alert|verification)"),
        ("kyc", "verify", "verification", "blocked", "suspended", "frozen", "unusual", "suspicious", "abnormal", "unauthorized", "update", "reactivate", "authenticate"),
    ),
    Category(
        "urgency_threat", "Urgency, pressure or threats", MEDIUM,
        "The caller creates time pressure or threatens consequences (arrest, legal action, penalties, account closure).",
        "Real authorities do not demand instant action over the phone. Take time, hang up and verify independently.",
        (r"\bimmediately\b", r"\burgent(?:ly)?\b", r"right now", r"within (?:the next )?(?:__num__|\d+|one|two|twenty[- ]four) (?:hours?|minutes?)", r"today itself", r"last chance",
         r"\barrest(?:ed)?\b", r"\bwarrant\b", r"legal action", r"police (?:case|complaint|station)", r"\bfir\b", r"money laundering", r"digital arrest",
         r"\bpenalt(?:y|ies)\b", r"\bjail\b", r"court (?:case|order|summons)", r"as soon as possible", r"otherwise (?:your|you)", r"do it now"),
        ("immediately", "urgent", "urgently", "now", "today", "arrest", "warrant", "police", "legal", "penalty", "jail", "court", "laundering", "deadline", "hurry", "otherwise", "soon"),
    ),
    Category(
        "impersonation_authority", "Authority or brand impersonation", MEDIUM,
        "The caller claims to be from a bank, government agency, police, courier, telecom or tech company.",
        "Caller ID and official-sounding titles can be faked. Call back using the number on the official website or your card.",
        (r"calling (?:you )?from (?:the )?(?:\w+ ){0,3}(?:bank|police|customs|department|income tax|government|ministry|court|courier|telecom|microsoft|amazon|apple|support)",
         r"\b(?:rbi|cbi|trai|sebi|irs|ssa|fbi|ncb|customs|income tax department)\b", r"(?:bank|police|customs|cyber ?crime|tax) (?:officer|official|department|manager)",
         r"customer (?:service|care|support)(?: (?:center|centre|executive|representative))?", r"technical support|tech support", r"on behalf of", r"\bofficer\b",
         r"\b(?:sbi|hdfc|icici|axis|kotak|paytm|amazon|flipkart|fedex|dhl) (?:bank|customer care|support|team)"),
        ("bank", "police", "officer", "customs", "department", "government", "official", "service", "customer", "support", "agency", "bureau", "manager", "representative", "investigator"),
    ),
    Category(
        "reward_lure", "Prize, reward or investment lure", MEDIUM,
        "The caller offers a prize, lottery win, cashback, refund or unusually high investment returns.",
        "You cannot win a lottery you did not enter. Unrealistic returns or prizes that need a fee are classic lures.",
        (r"congratulations", r"you (?:have )?won", r"\bwinner\b", r"lottery|jackpot|lucky draw|sweepstakes", r"\bprize\b", r"cash ?back", r"\breward\b",
         r"guaranteed (?:returns?|profit)", r"high returns?", r"double your money", r"investment (?:opportunity|plan|scheme)", r"free (?:gift|vacation|trip|iphone)",
         r"(?:you have been|you are) selected", r"\brefund\b"),
        ("congratulations", "won", "winner", "lottery", "jackpot", "prize", "reward", "cashback", "returns", "profit", "investment", "bonus", "selected", "free", "lucky"),
    ),
    Category(
        "secrecy_isolation", "Secrecy or isolation", MEDIUM,
        "The caller tells you to keep the call secret, not to tell family or the bank, or not to hang up.",
        "Requests for secrecy are a red flag. Talk to someone you trust before acting.",
        (r"(?:don'?t|do not) (?:tell|inform|share (?:this )?with) (?:anyone|anybody|your family|the bank)", r"keep (?:this|it) (?:confidential|secret|between us)",
         r"(?:don'?t|do not) (?:disconnect|hang up|cut the call)", r"stay on the (?:line|call)", r"confidential (?:matter|case|investigation)"),
        ("confidential", "secret", "secrecy", "disconnect", "hang"),
    ),
    Category(
        "suspicious_link", "Link or call-back number", MEDIUM,
        "The conversation contains a link or asks you to click/visit/call a number.",
        "Do not open links or call numbers given in an unsolicited call or message; type the official address yourself.",
        (r"https?://\S+", r"www\.\S+", r"\b(?:bit\.ly|tinyurl|goo\.gl|t\.co)/\S*", r"click (?:on )?(?:the |this )?link", r"__url__", r"visit (?:the |this )?(?:link|website|site)"),
        ("__url__", "link", "click", "website", "__phone__"),
    ),
)

_SECRET = r"(?:your |the )?(?:otp|pin|password|cvv|card details|one[\s-]?time password)"
_SECRET_LIST = rf"{_SECRET}(?:\s*(?:,|or|and|/)\s*{_SECRET})*"
SAFE_SIGNALS = (
    rf"(?:will|would|do|does) never (?:ask|call) (?:you )?(?:for )?{_SECRET_LIST}",
    rf"(?:never|do not|don'?t) (?:share|disclose|tell anyone) {_SECRET_LIST}(?: with anyone)?",
    r"visit (?:your )?(?:nearest|local) branch",
    r"call the (?:number|helpline) on the back of your card",
    r"through (?:the )?official (?:app|website)",
    r"no (?:payment|fee|charges?) (?:is )?(?:required|needed|involved)",
    r"there is no cost involved",
)

CAT_BY_KEY = {c.key: c for c in CATEGORIES}
_COMPILED = [(c, [re.compile(p, re.I) for p in c.patterns]) for c in CATEGORIES]
_SAFE = [re.compile(p, re.I) for p in SAFE_SIGNALS]


def detect_indicators(text: str) -> dict:
    """Return matched indicator categories, phrases and safe signals for a raw transcript."""
    text = str(text or "")
    safe_spans, safe_phrases = [], []
    for rx in _SAFE:
        for m in rx.finditer(text):
            safe_spans.append(m.span())
            safe_phrases.append(m.group(0))

    def overlaps_safe(span):
        return any(s < span[1] and span[0] < e for s, e in safe_spans)

    found = []
    for cat, rxs in _COMPILED:
        phrases, spans = [], []
        for rx in rxs:
            for m in rx.finditer(text):
                if cat.key in ("credential_request", "payment_request") and overlaps_safe(m.span()):
                    continue
                phrases.append(m.group(0).strip())
                spans.append(m.span())
        if phrases:
            uniq = list(dict.fromkeys(p.lower() for p in phrases))
            found.append({
                "key": cat.key, "label": cat.label, "severity": cat.severity,
                "why": cat.why, "advice": cat.advice,
                "phrases": uniq[:8], "count": len(phrases), "spans": spans,
            })
    found.sort(key=lambda d: (d["severity"] != HIGH, -d["count"]))
    return {
        "indicators": found,
        "n_high": sum(d["severity"] == HIGH for d in found),
        "n_medium": sum(d["severity"] == MEDIUM for d in found),
        "safe_signals": list(dict.fromkeys(p.lower() for p in safe_phrases)),
    }


# ---------------------------------------------------------------- HMM symbols
HMM_SYMBOLS = ["NEUTRAL", "AUTHORITY", "ACCOUNT", "URGENCY", "REWARD", "PERSONAL_INFO", "CREDENTIAL", "PAYMENT", "LINK", "SECRECY", "REMOTE"]
_SYMBOL_OF_CATEGORY = {
    "impersonation_authority": "AUTHORITY", "account_verification": "ACCOUNT", "urgency_threat": "URGENCY",
    "reward_lure": "REWARD", "personal_info_request": "PERSONAL_INFO", "credential_request": "CREDENTIAL",
    "payment_request": "PAYMENT", "suspicious_link": "LINK", "secrecy_isolation": "SECRECY", "remote_access": "REMOTE",
}
# priority when a window contains several categories (most specific first)
_PRIORITY = ["CREDENTIAL", "REMOTE", "PAYMENT", "SECRECY", "PERSONAL_INFO", "LINK", "URGENCY", "ACCOUNT", "REWARD", "AUTHORITY"]
_WORD2SYM: dict[str, str] = {}
for _c in CATEGORIES:
    for _w in _c.hmm_words:
        _WORD2SYM.setdefault(_w, _SYMBOL_OF_CATEGORY[_c.key])


def text_to_symbols(cleaned_text: str, window: int = 12, max_len: int = 200) -> list[int]:
    """Convert CLEANED text into a sequence of stage symbols (one per `window` tokens).

    Uses fixed-size token windows instead of utterance boundaries so that the sequence does
    not depend on the data-source formatting (audit §5).
    """
    toks = cleaned_text.split()
    seq = []
    for i in range(0, len(toks), window):
        chunk = toks[i:i + window]
        syms = {_WORD2SYM[t] for t in chunk if t in _WORD2SYM}
        sym = next((s for s in _PRIORITY if s in syms), "NEUTRAL")
        seq.append(HMM_SYMBOLS.index(sym))
        if len(seq) >= max_len:
            break
    return seq
