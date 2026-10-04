import html
import os
import random
import re
from collections import Counter

import nltk
import pandas as pd
import pyiwn
import stanza
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# ============================================================
# NLTK RESOURCES
# ============================================================

for _pkg in (
    "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
):
    nltk.download(_pkg, quiet=True)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="LyricLens", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# STYLING
# ============================================================

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Sora:wght@600;700;800&family=Noto+Sans+Devanagari:wght@400;600&display=swap');
:root { --ink:#1b1740; --muted:#6b6890; --line:#e7e3ff; --violet:#6d4aff; --pink:#ff5d9e; --amber:#ffb43a; --mint:#19c3a0; }

.stApp {
    background:
        radial-gradient(900px 480px at 6% -6%, #e8e0ff 0%, transparent 60%),
        radial-gradient(760px 460px at 100% 0%, #ffe2ee 0%, transparent 55%),
        #f8f7ff;
}
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {
    font-family: 'DM Sans', 'Noto Sans Devanagari', 'Nirmala UI', sans-serif; color: var(--ink);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; padding: 34px 38px 30px; background: #1b1740; color: #fff; margin-bottom: 18px; }
.hero:before { content: ""; position: absolute; width: 460px; height: 460px; right: -110px; top: -190px;
    background: radial-gradient(circle, #6d4aff 0%, transparent 65%); }
.hero:after { content: ""; position: absolute; width: 380px; height: 380px; left: -120px; bottom: -230px;
    background: radial-gradient(circle, rgba(255,93,158,.75) 0%, transparent 65%); }
.hero > * { position: relative; z-index: 1; }
.hero h1 { font-family: 'Sora', sans-serif; font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; margin: 0; padding: 0; color: #fff; }
.hero p { color: #d3ccff; max-width: 540px; margin: 8px 0 0; font-size: 1.02rem; }
.steps { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
.step { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2); padding: 6px 14px; border-radius: 999px; font-size: .85rem; color: #fff; }
.step b { color: #ffb43a; margin-right: 6px; }
.eq { position: absolute !important; right: 38px; bottom: 30px; display: flex; gap: 5px; align-items: flex-end; height: 40px; }
.eq i { width: 6px; height: 10px; border-radius: 4px; background: linear-gradient(#ff5d9e, #ffb43a); animation: eq 1.3s ease-in-out infinite; }
.eq i:nth-child(2) { animation-delay: .15s; } .eq i:nth-child(3) { animation-delay: .3s; }
.eq i:nth-child(4) { animation-delay: .45s; } .eq i:nth-child(5) { animation-delay: .6s; }
@keyframes eq { 0%, 100% { height: 8px; } 50% { height: 38px; } }
@media (prefers-reduced-motion: reduce) { .eq i { animation: none; height: 22px; } }

/* ---------- cards & text ---------- */
[class*="st-key-card-"] { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 18px 20px;
    box-shadow: 0 10px 28px rgba(70,45,170,.07); }
.h { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: .88rem; margin: 0 0 10px; }
.tile { --acc: var(--violet); background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px 12px; box-shadow: 0 8px 22px rgba(70,45,170,.06); }
.tile:before { content: ""; display: block; width: 26px; height: 4px; border-radius: 4px; background: var(--acc); margin-bottom: 9px; }
.tile .k { font-size: .78rem; color: var(--muted); font-weight: 600; }
.tile .v { font-family: 'Sora', sans-serif; font-size: 1.55rem; font-weight: 700; line-height: 1.25; color: var(--ink); overflow-wrap: anywhere; }
.tile .s { font-size: .78rem; color: var(--muted); margin-top: 2px; }
.tile.mint { --acc: #19c3a0; } .tile.pink { --acc: #f0457f; } .tile.amber { --acc: #ffb43a; } .tile.slate { --acc: #9a97b8; }
.live { background: #f3f0ff; border: 1px dashed #cfc6ff; border-radius: 12px; padding: 8px 12px; font-size: .88rem; margin-top: 10px; }
.tip { color: var(--muted); font-size: .85rem; margin-top: 10px; line-height: 1.5; }
.feat { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 18px; height: 100%; box-shadow: 0 8px 22px rgba(70,45,170,.06); }
.feat .t { font-family: 'Sora', sans-serif; font-weight: 700; margin-bottom: 6px; }
.feat .d { color: var(--muted); font-size: .9rem; line-height: 1.55; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { background: #f1edff; border: 1px solid #ddd5ff; border-radius: 9px; padding: 3px 10px; font-size: .88rem; }
.chip.good { background: #e6f8f3; border-color: #bfeadf; } .chip.bad { background: #ffeaf1; border-color: #ffd0e0; }
.textbox { background: #faf9ff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; line-height: 1.75; }

.pill { display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: .92rem; color: #fff; }
.pill.pos { background: #19c3a0; } .pill.neg { background: #f0457f; } .pill.neu { background: #8b88a8; }
.meter { margin: 16px 4px 10px; }
.meter .track { height: 10px; border-radius: 999px; position: relative; background: linear-gradient(90deg, #f0457f, #dcd9ee 50%, #19c3a0); }
.meter .mark { position: absolute; top: -6px; width: 22px; height: 22px; border-radius: 50%; background: #fff; border: 4px solid var(--ink); transform: translateX(-50%); }
.meter .ends { display: flex; justify-content: space-between; font-size: .75rem; color: var(--muted); margin-top: 8px; }

.poem { border-radius: 22px; padding: 28px 32px; color: #fff; line-height: 2; font-size: 1.22rem; font-weight: 500;
    background: linear-gradient(135deg, #6d4aff 0%, #b44cff 52%, #ff5d9e 100%); box-shadow: 0 16px 36px rgba(109,74,255,.32); }
.poem .tag { font-size: .78rem; opacity: .85; font-weight: 600; margin-bottom: 8px; }

/* ---------- widgets ---------- */
.stTextArea textarea { background: #fff; border-radius: 16px; border: 1.5px solid #d9d2ff; font-size: 1rem; padding: 14px; }
.stTextArea textarea:focus { border-color: var(--violet); box-shadow: 0 0 0 4px rgba(109,74,255,.15); }
.stButton > button, .stDownloadButton > button { width: 100%; border-radius: 13px; font-weight: 700; padding: .55rem 1rem; border: 1px solid #d9d2ff; background: #fff; color: var(--ink); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--violet); color: var(--violet); }
.stButton > button[kind="primary"] { background: linear-gradient(120deg, #6d4aff, #b44cff); border: none; color: #fff; box-shadow: 0 8px 20px rgba(109,74,255,.35); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #fff; }
.stTabs [data-baseweb="tab-list"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 5px; gap: 4px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 8px 16px; font-weight: 600; height: auto; }
.stTabs [data-baseweb="tab"] p { color: inherit; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, #6d4aff, #b44cff); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }
@media (max-width: 760px) { .hero { padding: 24px; } .hero h1 { font-size: 1.9rem; } .eq { display: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# UI HELPERS
# ============================================================

def card(name):
    """A keyed container styled as a card (plain container on old Streamlit)."""
    try:
        return st.container(key=f"card-{name}")
    except TypeError:
        return st.container()


def heading(title, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="h">{title}</div>{sub_html}', unsafe_allow_html=True)


def tile(label, value, sub="", tone=""):
    sub_html = f'<div class="s">{html.escape(str(sub))}</div>' if sub else ""
    return (
        f'<div class="tile {tone}"><div class="k">{html.escape(str(label))}</div>'
        f'<div class="v">{html.escape(str(value))}</div>{sub_html}</div>'
    )


def chips(words, limit=300, tone=""):
    if not words:
        st.caption("Nothing to show.")
        return
    shown = "".join(f'<span class="chip {tone}">{html.escape(w)}</span>' for w in words[:limit])
    extra = f'<span class="chip">+{len(words) - limit} more</span>' if len(words) > limit else ""
    st.markdown(f'<div class="chips">{shown}{extra}</div>', unsafe_allow_html=True)


def textbox(text):
    st.markdown(f'<div class="textbox">{html.escape(text)}</div>', unsafe_allow_html=True)


def pill(label):
    tone = {"Positive": "pos", "Negative": "neg"}.get(label, "neu")
    icon = {"Positive": "😊", "Negative": "😔"}.get(label, "😐")
    return f'<span class="pill {tone}">{label} {icon}</span>'


def meter(score):
    pct = max(0, min(100, (score + 1) / 2 * 100))
    return (
        f'<div class="meter"><div class="track"><div class="mark" style="left:{pct:.0f}%"></div></div>'
        '<div class="ends"><span>Negative</span><span>Neutral</span><span>Positive</span></div></div>'
    )


def show_df(data, height=None):
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    try:
        st.dataframe(df, width="stretch", hide_index=True, height=height)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def show_image(img):
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            st.image(img, **kwargs)
            return
        except Exception:
            continue


# ============================================================
# HINDI RESOURCES
# ============================================================

HINDI_STOPWORDS = {
    "और", "या", "लेकिन", "मगर", "तो", "भी", "ही",
    "का", "के", "की", "को", "से", "में", "पर", "ने",
    "है", "हैं", "था", "थी", "थे", "हो", "हूँ",
    "मैं", "हम", "हमारा", "हमारी", "हमारे",
    "तुम", "तुम्हारा", "तुम्हारी", "तुम्हारे",
    "आप", "आपका", "आपकी", "आपके",
    "वह", "वो", "यह", "ये", "इस", "उस",
    "इसे", "उसे", "इन", "उन",
    "एक", "दो",
    "जो", "जिस", "जिसे", "जिसका", "जिसकी",
    "जब", "जहाँ", "जहां", "कहाँ", "कहां",
    "क्यों", "कैसे",
    "नहीं", "न", "मत", "हाँ",
    "अपने", "अपना", "अपनी",
    "कुछ", "कोई", "सब", "सभी",
    "तक", "लिए", "बाद", "पहले",
    "अब", "फिर", "बहुत",
    "जैसे", "वैसे",
    "साथ", "द्वारा",
}

HINDI_POSITIVE_WORDS = {
    "अच्छा", "अच्छी", "अच्छे", "खुश", "खुशी", "प्यार", "प्रेम",
    "सुंदर", "अद्भुत", "शानदार", "बेहतरीन", "जीत", "सफल", "सफलता",
    "आनंद", "मुस्कान", "उम्मीद", "दोस्ती", "खूबसूरत", "खुशनुमा",
    "हंसी", "हँसी", "सुकून", "विश्वास", "आज़ादी", "आजादी", "जश्न",
    "जीतना", "पसंद", "पसंदीदा", "रोमांच", "गर्व",
}

HINDI_NEGATIVE_WORDS = {
    "बुरा", "बुरी", "बुरे", "दुख", "दुःख", "दर्द", "नफरत", "घृणा",
    "गुस्सा", "अकेला", "अकेली", "अकेलापन", "हार", "हारना", "असफल",
    "असफलता", "रोना", "आंसू", "आँसू", "डर", "भय", "निराशा",
    "निराश", "उदास", "उदासी", "तकलीफ", "तन्हाई", "टूट", "टूटा",
    "टूटी", "मौत", "मरना", "झूठ", "धोखा", "नुकसान", "दुश्मन",
}


def find_devanagari_font():
    candidates = [
        "NotoSansDevanagari-Regular.ttf",
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",
        r"C:\Windows\Fonts\NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


@st.cache_resource
def get_hindi_nlp():
    return stanza.Pipeline(lang="hi", processors="tokenize,pos,lemma", verbose=False)


# ============================================================
# TEXT PROCESSING
# ============================================================

def detect_language(text):
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "Hindi" if devanagari > latin else "English"


def clean_text(text, language):
    # Remove section labels such as [Verse], [Chorus]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    if language == "Hindi":
        text = re.sub(r"[^\u0900-\u097F\sA-Za-z0-9']", " ", text)
    else:
        text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_hindi(text):
    return re.findall(r"[\u0900-\u097F]+", text)


def remove_hindi_stopwords(tokens):
    return [w for w in tokens if w not in HINDI_STOPWORDS]


def create_wordcloud(words, language):
    frequency = Counter(words)
    if not frequency:
        return None
    palette = ["#6d4aff", "#ff5d9e", "#ffb43a", "#19c3a0", "#2a2566", "#b44cff"]
    options = dict(
        width=1200, height=560, background_color=None, mode="RGBA", max_words=90,
        collocations=False, margin=6, prefer_horizontal=0.9,
        color_func=lambda *a, **k: random.choice(palette),
    )
    if language == "Hindi":
        font_path = find_devanagari_font()
        if font_path:
            options["font_path"] = font_path
    return WordCloud(**options).generate_from_frequencies(frequency)


def get_ngrams(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def ngram_frame(tokens, n):
    counts = get_ngrams(tokens, n)
    if not counts:
        return None
    return pd.DataFrame(
        [(" ".join(k), v) for k, v in counts.most_common(10)],
        columns=["Phrase", "Count"],
    )


@st.cache_resource(show_spinner="Loading Hindi WordNet...")
def get_indowordnet():
    return pyiwn.IndoWordNet(lang=pyiwn.Language.HINDI)


def _get(obj, names, default=None):
    """Read an attribute or method from a pyiwn object, whichever exists."""
    for name in names:
        attr = getattr(obj, name, None)
        if attr is None:
            continue
        try:
            value = attr() if callable(attr) else attr
        except Exception:
            continue
        if value:
            return value
    return default


def _synset_info(synset):
    gloss = str(_get(synset, ["gloss", "definition"], "") or "")
    examples = _get(synset, ["examples", "example"], []) or []
    if isinstance(examples, str):
        examples = [examples]
    lemmas = _get(synset, ["lemma_names", "words"], []) or []
    return gloss, [str(e) for e in examples], [str(w) for w in lemmas]


def hindi_wordnet_analysis(tokens):
    """Hindi lexical lookup using IndoWordNet through pyiwn."""
    data = []
    try:
        iwn = get_indowordnet()
    except Exception as e:
        return {"error": str(e), "data": []}

    for word in sorted(set(tokens)):
        if len(word) < 2:
            continue
        try:
            synsets = iwn.synsets(word)
        except Exception:
            continue
        for synset in synsets[:3]:
            gloss, _, lemmas = _synset_info(synset)
            data.append({
                "Word": word,
                "Meaning": gloss or "Meaning available in IndoWordNet",
                "Synonyms": ", ".join(sorted(set(l for l in lemmas if l != word))[:8]),
                "Synset": str(synset),
            })
    return {"error": None, "data": data}


def hindi_wsd(tokens, max_words=60):
    """Simplified Lesk for Hindi: pick the IndoWordNet sense whose gloss,
    examples and synonyms overlap most with the surrounding words."""
    try:
        iwn = get_indowordnet()
    except Exception as e:
        return {"error": str(e), "data": []}

    data, seen = [], set()
    for i, word in enumerate(tokens):
        if word in seen or len(word) < 2:
            continue
        seen.add(word)
        if len(seen) > max_words:
            break
        try:
            synsets = iwn.synsets(word)
        except Exception:
            continue
        if not synsets:
            continue

        context = set(tokens[max(0, i - 5): i] + tokens[i + 1: i + 6]) - {word}
        best, best_score = synsets[0], -1
        for synset in synsets:
            gloss, examples, lemmas = _synset_info(synset)
            signature = set(tokenize_hindi(" ".join([gloss] + examples + lemmas)))
            signature -= HINDI_STOPWORDS
            score = len(context & signature)
            if score > best_score:
                best, best_score = synset, score

        gloss, _, _ = _synset_info(best)
        data.append({
            "Word": word,
            "Context": " ".join(tokens[max(0, i - 4): i + 5]),
            "Meaning": gloss or "Meaning available in IndoWordNet",
            "Senses found": len(synsets),
            "Context score": max(best_score, 0),
            "Method": "Context overlap" if best_score > 0 else "Most common sense",
            "Synset": str(best),
        })
    return {"error": None, "data": data}


def english_wordnet_analysis(tokens):
    data = []
    for word in sorted(set(tokens)):
        if len(word) < 2:
            continue
        synsets = wordnet.synsets(word)
        if not synsets:
            continue
        synonyms, antonyms = set(), set()
        for syn in synsets:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
                for ant in lemma.antonyms():
                    antonyms.add(ant.name())
        data.append({
            "Word": word,
            "Meaning": synsets[0].definition(),
            "Synonyms": ", ".join(sorted(synonyms)[:5]),
            "Antonyms": ", ".join(sorted(antonyms)[:5]),
        })
    return data


def english_wsd(tokens):
    data = []
    for i, word in enumerate(tokens):
        if not word.isalpha() or len(word) < 3:
            continue
        context = tokens[max(0, i - 5): min(len(tokens), i + 6)]
        sense = lesk(context, word)
        if sense:
            data.append({"Word": word, "Meaning": sense.definition(), "Synset": sense.name()})
    return data


def hindi_sentiment_analysis(text):
    """Fast lexicon-based Hindi sentiment analysis."""
    tokens = tokenize_hindi(clean_text(text, "Hindi"))
    pos_words = [w for w in tokens if w in HINDI_POSITIVE_WORDS]
    neg_words = [w for w in tokens if w in HINDI_NEGATIVE_WORDS]
    positive, negative = len(pos_words), len(neg_words)
    total = positive + negative
    score = 0.0 if total == 0 else (positive - negative) / total

    if score >= 0.15:
        overall = "Positive"
    elif score <= -0.15:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "overall": overall, "positive": positive, "negative": negative, "score": score,
        "matched_positive": sorted(set(pos_words)),
        "matched_negative": sorted(set(neg_words)),
    }


# ============================================================
# POEM GENERATION
# ============================================================

POEM_METHODS = ["Theme poem", "Remix lines"]
POEM_HELP = {
    "Theme poem": "Writes brand-new, grammatical lines from your song's own nouns, verbs and adjectives, in rhyming couplets with a closing line that matches its mood.",
    "Remix lines": "Rebuilds the song from its own real lines, pairing lines that rhyme. Every line is original lyric text.",
}
POEM_TAG = {
    "Theme poem": "New poem built from your song's words and mood",
    "Remix lines": "Remixed from lines of your song",
}


def extract_lines(lyrics):
    """Lyric lines without [Verse] labels or punctuation, duplicates removed."""
    language = detect_language(lyrics)
    text = re.sub(r"\[[^\]]*\]", " ", lyrics)
    lines, seen = [], set()
    for raw in text.splitlines():
        line = clean_text(raw.replace("।", " ").replace("॥", " "), language)
        if len(line.split()) < 2 or line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    return lines


def rhyme_key(line, language):
    last = re.sub(r"[^\w\u0900-\u097F]", "", line.split()[-1].lower())
    return last[-2:] if len(last) >= 2 else last


def to_stanzas(lines, size=4):
    return "\n\n".join("\n".join(lines[i:i + size]) for i in range(0, len(lines), size))


def remix_poem(lines, n_lines=8, language="English"):
    """Re-arrange real lyric lines into new stanzas, pairing rhyming endings."""
    if not lines:
        return None
    groups = {}
    for line in lines:
        groups.setdefault(rhyme_key(line, language), []).append(line)

    couplets, leftovers = [], []
    for group in groups.values():
        random.shuffle(group)
        while len(group) >= 2:
            couplets.append((group.pop(), group.pop()))
        leftovers.extend(group)
    random.shuffle(couplets)
    random.shuffle(leftovers)

    n = min(n_lines, len(lines))
    chosen = []
    for a, b in couplets:
        if len(chosen) + 2 <= n:
            chosen += [a, b]
    for line in leftovers:
        if len(chosen) < n:
            chosen.append(line)
    return to_stanzas(chosen[:n])



CLOSERS = {
    "English": {
        "Positive": ["And the morning answers back", "So we keep the light alive", "Because tomorrow feels like home"],
        "Negative": ["Still, the dawn will find us here", "And we carry on, unbroken", "Even this ache will learn to fade"],
        "Neutral": ["And the story carries on", "Quiet, steady, ever on", "So we walk, and let it be"],
    },
    "Hindi": {
        "Positive": ["और सुबह फिर मुस्कुराई है", "चलो, यही तो ज़िंदगी है", "हर खुशी अब अपने पास है"],
        "Negative": ["फिर भी चलना है, रुकना नहीं है", "हर दर्द की भी एक सुबह है", "रात कितनी भी लंबी हो, उम्मीद बाकी है"],
        "Neutral": ["बस यही तो ज़िंदगी है", "कहानी अभी बाकी है", "चलते रहना ही ज़िंदगी है"],
    },
}

EN_DEFAULTS = {
    "noun": ["night", "heart", "light", "dream", "road", "sky", "rain", "song"],
    "adj": ["bright", "quiet", "golden", "restless", "gentle", "wild"],
    "verb": ["rise", "dance", "shine", "stay", "wander", "sing", "burn"],
}
EN_BAD_NOUNS = {"tonight", "today", "tomorrow", "yesterday", "thing", "something"}
EN_SKIP_VERBS = {"be", "have", "do", "let", "get", "make", "go", "say", "see", "know", "use", "try"}
# (pattern, slot that ends the line, used for rhyming)
EN_TEMPLATES = [
    ("In the {adj} {noun}, we {verb}", "verb"),
    ("Every {noun} learns to {verb}", "verb"),
    ("I still {verb} for the {adj} {noun}", "noun"),
    ("The {noun} will {verb} when the {adj} {noun2} is near", "noun2"),
    ("Hold on to the {adj} {noun}, and {verb}", "verb"),
    ("Let the {noun} {verb} through the {adj} {noun2}", "noun2"),
    ("We {verb} where the {noun} meets the {noun2}", "noun2"),
    ("Even the {adj} {noun} can {verb}", "verb"),
    ("Somewhere a {adj} {noun} begins to {verb}", "verb"),
    ("Close your eyes and {verb} with the {noun}", "noun"),
    ("Nothing is as {adj} as the {noun}", "noun"),
    ("Through every {noun}, we {verb} again", "verb"),
    ("Sing for the {noun}, sing for the {noun2}", "noun2"),
    ("Under the {adj} {noun}, hearts {verb}", "verb"),
]


def english_pools(tagged):
    lem = WordNetLemmatizer()
    stop = set(stopwords.words("english"))
    pools = {"noun": [], "adj": [], "verb": []}
    for word, tag in tagged:
        w = word.lower()
        if not w.isalpha() or len(w) < 3 or w in stop:
            continue
        if tag in ("NN", "NNS") and w not in EN_BAD_NOUNS:
            pools["noun"].append(lem.lemmatize(w, "n"))
        elif tag.startswith("JJ"):
            pools["adj"].append(w)
        elif tag.startswith("VB"):
            v = lem.lemmatize(w, "v")
            if v not in EN_SKIP_VERBS and len(v) >= 3:
                pools["verb"].append(v)
    for key, defaults in EN_DEFAULTS.items():
        pools[key] = sorted(set(pools[key]))
        if len(pools[key]) < 4:
            pools[key] += [d for d in defaults if d not in pools[key]]
    return pools


def _pick(pool, used, rhyme=None):
    options = [w for w in pool if w not in used] or list(pool)
    if rhyme:
        match = [w for w in options if w != rhyme and w[-2:] == rhyme[-2:]]
        if match:
            return random.choice(match)
    return random.choice(options)


def _line_en(pools, used, rhyme_with=None):
    templates = EN_TEMPLATES[:]
    random.shuffle(templates)
    chosen = templates[0]
    if rhyme_with:
        for t in templates:
            pool = pools[t[1].rstrip("2")]
            if any(w != rhyme_with and w[-2:] == rhyme_with[-2:] for w in pool):
                chosen = t
                break
    pattern, end_slot = chosen
    values, end_word = {}, ""
    for slot in re.findall(r"\{(\w+)\}", pattern):
        word = _pick(pools[slot.rstrip("2")], used | set(values.values()),
                     rhyme_with if slot == end_slot else None)
        values[slot] = word
        if slot == end_slot:
            end_word = word
    used.update(values.values())
    line = pattern.format(**values)
    line = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", line)
    return line[0].upper() + line[1:], end_word


def theme_poem_en(tagged, mood, n_lines=8):
    pools = english_pools(tagged)
    body = n_lines - 1 if n_lines >= 6 else n_lines
    lines, used = [], set()
    while len(lines) < body:
        line, end = _line_en(pools, used)
        lines.append(line)
        if len(lines) < body:
            lines.append(_line_en(pools, used, rhyme_with=end)[0])
    if n_lines >= 6:
        lines.append(random.choice(CLOSERS["English"].get(mood, CLOSERS["English"]["Neutral"])))
    return to_stanzas(lines)


HI_DEFAULTS = {
    "noun": ["रात", "दिल", "राह", "सपना", "याद", "सुबह", "आसमान"],
    "verb": ["गा", "चल", "जी", "हँस", "मुस्करा"],
}
HI_SKIP_VERBS = {"है", "हो", "था", "रह", "सक", "लग", "पड़", "चाह"}
# Two rhyme families: lines ending in "है" and lines ending in "जाएंगे".
HI_TEMPLATES = {
    "hai": [
        "आज फिर {n1} के साथ {v}ने का मन है",
        "इस {n1} में भी एक {n2} की झलक है",
        "{n1} के बिना भी {n2} का साथ है",
        "हर {n1} में छुपी एक {n2} की कहानी है",
        "{n1} हो या {n2}, सबमें बसा प्यार है",
        "{n1} की बातें, {n2} की यादें, यही तो मेरे पास है",
    ],
    "ge": [
        "{n1} की राह में हम {v}ते जाएंगे",
        "{n1} के साथ हम {v}ना सीख जाएंगे",
        "हर {n1} को भूलकर हम {n2} में खो जाएंगे",
        "{n1} और {n2} के गीत हम {v}ते जाएंगे",
    ],
}


def hindi_pools(rows):
    nouns, verbs = set(), set()
    for row in rows:
        lemma = (row.get("Lemma") or row.get("Word") or "").strip()
        if not re.fullmatch(r"[\u0900-\u097F]+", lemma) or lemma in HINDI_STOPWORDS:
            continue
        if row.get("POS") == "NOUN" and len(lemma) >= 2:
            nouns.add(lemma)
        elif row.get("POS") == "VERB":
            root = lemma[:-2] if lemma.endswith("ना") and len(lemma) > 3 else lemma
            if len(root) >= 2 and root not in HI_SKIP_VERBS:
                verbs.add(root)
    pools = {"noun": sorted(nouns), "verb": sorted(verbs)}
    for key, defaults in HI_DEFAULTS.items():
        if len(pools[key]) < 3:
            pools[key] += [d for d in defaults if d not in pools[key]]
    return pools


def theme_poem_hi(rows, mood, n_lines=8):
    pools = hindi_pools(rows)
    body = n_lines - 1 if n_lines >= 6 else n_lines
    lines, used, used_templates = [], set(), set()
    while len(lines) < body:
        family = random.choice(list(HI_TEMPLATES))
        for _ in range(2):  # a rhyming couplet = two lines from the same family
            if len(lines) >= body:
                break
            options = [t for t in HI_TEMPLATES[family] if t not in used_templates] or HI_TEMPLATES[family]
            template = random.choice(options)
            used_templates.add(template)
            n1 = _pick(pools["noun"], used)
            used.add(n1)
            n2 = _pick([w for w in pools["noun"] if w != n1] or pools["noun"], used)
            used.add(n2)
            verb = _pick(pools["verb"], used)
            used.add(verb)
            lines.append(template.format(n1=n1, n2=n2, v=verb))
    if n_lines >= 6:
        lines.append(random.choice(CLOSERS["Hindi"].get(mood, CLOSERS["Hindi"]["Neutral"])))
    return to_stanzas(lines)


# ============================================================
# ANALYSIS PIPELINE (runs once per click, results kept in session_state)
# ============================================================

def run_analysis(lyrics, log=lambda m: None):
    language = detect_language(lyrics)
    log(f"Detected **{language}** lyrics")
    clean = clean_text(lyrics, language)

    log("Cleaning and tokenizing…")
    if language == "Hindi":
        tokens = tokenize_hindi(clean)
        filtered = remove_hindi_stopwords(tokens)
        processed = filtered
    else:
        tokens = nltk.word_tokenize(clean)
        eng_stop = set(stopwords.words("english"))
        filtered = [w.lower() for w in tokens if w.lower() not in eng_stop]
        lemmatizer = WordNetLemmatizer()
        processed = [lemmatizer.lemmatize(w) for w in filtered]

    r = dict(
        language=language, clean=clean, tokens=tokens, filtered=filtered, processed=processed,
        freq=Counter(processed), wc=None, wc_error=None,
        font_missing=language == "Hindi" and not find_devanagari_font(),
        pos_rows=[], pos_dist={}, pos_error=None,
        wn_rows=[], wn_error=None, wsd_rows=[], wsd_error=None,
    )

    r["lines"] = extract_lines(lyrics)

    log("Building word cloud…")
    try:
        wc = create_wordcloud(processed, language)
        r["wc"] = wc.to_array() if wc else None
    except Exception as e:
        r["wc_error"] = str(e)

    log("Tagging parts of speech…")
    try:
        if language == "English":
            r["tagged"] = pos_tag(tokens)
            tags = pos_tag(processed)
            r["pos_rows"] = [{"Word": w, "POS": t} for w, t in tags]
            r["pos_dist"] = dict(Counter(t for _, t in tags))
        else:
            dist = Counter()
            for sentence in get_hindi_nlp()(clean).sentences:
                for w in sentence.words:
                    r["pos_rows"].append({
                        "Word": w.text, "POS": w.upos,
                        "Lemma": w.lemma or w.text, "Features": w.feats or "",
                    })
                    dist[w.upos] += 1
            r["pos_dist"] = dict(dist)
    except Exception as e:
        r["pos_error"] = str(e)

    log("Looking up meanings and word senses…")
    try:
        if language == "English":
            r["wn_rows"] = english_wordnet_analysis(processed)
            r["wsd_rows"] = english_wsd(tokens)
        else:
            wn = hindi_wordnet_analysis(processed)
            r["wn_rows"], r["wn_error"] = wn["data"], wn["error"]
            wsd = hindi_wsd(processed)
            r["wsd_rows"], r["wsd_error"] = wsd["data"], wsd["error"]
    except Exception as e:
        r["wn_error"] = r["wsd_error"] = str(e)

    log("Scoring sentiment…")
    if language == "English":
        v = SentimentIntensityAnalyzer().polarity_scores(clean)
        label = "Positive" if v["compound"] >= 0.05 else "Negative" if v["compound"] <= -0.05 else "Neutral"
        r["sentiment"] = dict(
            label=label, score=v["compound"], note=f"VADER compound score {v['compound']:.2f}",
            parts=[("Positive", f"{v['pos']:.2f}", "mint"), ("Neutral", f"{v['neu']:.2f}", "slate"),
                   ("Negative", f"{v['neg']:.2f}", "pink")],
            pos_words=[], neg_words=[],
        )
    else:
        h = hindi_sentiment_analysis(clean)
        r["sentiment"] = dict(
            label=h["overall"], score=h["score"], note=f"Lexicon score {h['score']:.2f}",
            parts=[("Positive words", h["positive"], "mint"), ("Negative words", h["negative"], "pink")],
            pos_words=h["matched_positive"], neg_words=h["matched_negative"],
        )
    return r


# ============================================================
# STATE + CALLBACKS
# ============================================================

SAMPLES = {
    "English": """[Verse 1]
Walking through the city lights tonight
Holding on to hope, we're burning bright
Every broken dream becomes a song
Together we are strong

[Chorus]
Dance with me under the open sky
Let the music lift us high
Love is the rhythm, love is the way
We'll sing until the break of day""",
    "Hindi": """[Verse 1]
रात की खामोशी में तेरा नाम पुकारा है
दिल की हर धड़कन में बस तेरा सहारा है
आँखों में उम्मीद है, होंठों पे मुस्कान है
तेरे साथ हर सफर लगता आसान है

[Chorus]
प्यार की इस राह में हम चलते जाएंगे
खुशियों के गीत हम मिलकर गाएंगे
दर्द कितना भी हो, हम न घबराएंगे
सुकून की सुबह हम फिर से पाएंगे""",
}

st.session_state.setdefault("lyrics_input", "")


def use_sample(name):
    st.session_state["lyrics_input"] = SAMPLES[name]


def clear_all():
    st.session_state["lyrics_input"] = ""
    st.session_state.pop("result", None)
    st.session_state.pop("poem", None)


def new_poem():
    """Build a poem with the chosen method."""
    r = st.session_state.get("result")
    if not r:
        return
    method = st.session_state.get("poem_method", POEM_METHODS[0])
    n = st.session_state.get("poem_lines", 8)
    mood = r["sentiment"]["label"]
    st.session_state["poem_error"] = None
    try:
        if method == "Remix lines":
            poem = remix_poem(r["lines"], n, r["language"])
        elif r["language"] == "English":
            poem = theme_poem_en(r.get("tagged", []), mood, n)
        else:
            poem = theme_poem_hi(r["pos_rows"], mood, n)
        st.session_state["poem"] = poem
        st.session_state["poem_tag"] = POEM_TAG[method]
    except Exception as e:
        st.session_state["poem_error"] = str(e)


# ============================================================
# HERO + INPUT
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎵 Lyricbstrom</h1>
        <p>Paste a song in English or Hindi. Get its words, meaning and mood, then remix it into a new lyrics for your next song.</p>
        <div class="steps">
            <span class="step"><b>1</b>Paste lyrics</span>
            <span class="step"><b>2</b>Analyze</span>
            <span class="step"><b>3</b>Explore &amp; generate a poem</span>
        </div>
        <div class="eq"><i></i><i></i><i></i><i></i><i></i></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([5, 2], gap="large")

with left:
    with card("input"):
        heading("Your lyrics", "Section labels like [Verse] and [Chorus] are removed automatically.")
        lyrics = st.text_area(
            "Lyrics", key="lyrics_input", height=250, label_visibility="collapsed",
            placeholder="Paste your English or Hindi song lyrics here…",
        )
        analyze = st.button("✨ Analyze lyrics", type="primary")

with right:
    with card("quick"):
        heading("Quick start", "No lyrics handy? Try a sample.")
        st.button("🇬🇧 English sample", on_click=use_sample, args=("English",))
        st.button("🇮🇳 Hindi sample", on_click=use_sample, args=("Hindi",))
        st.button("🗑️ Clear", on_click=clear_all)
        text_now = st.session_state.get("lyrics_input", "")
        if text_now.strip():
            n_words = len(re.findall(r"[\u0900-\u097F]+|[A-Za-z']+", text_now))
            st.markdown(
                f'<div class="live"><b>{detect_language(text_now)}</b> detected · {n_words} words</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="tip">Tip: Hindi works best with Devanagari script, not romanized text.</div>',
            unsafe_allow_html=True,
        )

if analyze:
    if not lyrics.strip():
        st.warning("Paste some lyrics first, or try a sample.")
    else:
        with st.status("Analyzing your lyrics…", expanded=True) as status:
            st.session_state["result"] = run_analysis(lyrics, log=st.write)
            st.session_state.pop("poem", None)
            new_poem()
            status.update(label="Analysis complete", state="complete", expanded=False)

# ============================================================
# RESULTS
# ============================================================

r = st.session_state.get("result")
st.write("")

if not r:
    cols = st.columns(3, gap="medium")
    features = [
        ("🧹 Clean & tokenize", "Removes section labels, splits the text into words and drops stop words in English or Hindi."),
        ("🧠 Understand", "Parts of speech, WordNet meanings and word-sense disambiguation for each word."),
        ("🎤 Create", "A mood meter, a word cloud and a poem remixed from your own vocabulary."),
    ]
    for col, (title, desc) in zip(cols, features):
        col.markdown(
            f'<div class="feat"><div class="t">{title}</div><div class="d">{desc}</div></div>',
            unsafe_allow_html=True,
        )
else:
    total, unique = len(r["processed"]), len(set(r["processed"]))
    top = r["freq"].most_common(1)
    tiles = st.columns(4, gap="medium")
    tile_data = [
        ("Language", r["language"], "auto-detected", ""),
        ("Meaningful words", total, "after stop-word removal", "pink"),
        ("Unique words", unique, f"{unique / total:.0%} variety" if total else "", "mint"),
        ("Top word", top[0][0] if top else "–", f"used {top[0][1]}×" if top else "", "amber"),
    ]
    for col, (k, v, s, t) in zip(tiles, tile_data):
        col.markdown(tile(k, v, s, t), unsafe_allow_html=True)
    st.write("")

    t_over, t_poem, t_words, t_ling, t_pipe = st.tabs(
        ["✨ Overview", "🎤 Poem Studio", "📊 Words & phrases", "📖 Linguistics", "🧹 Text pipeline"]
    )

    # ---------------- Overview ----------------
    with t_over:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            with card("cloud"):
                heading("Word cloud", "Bigger words appear more often in your lyrics.")
                if r["wc_error"]:
                    st.error(f"Word cloud failed: {r['wc_error']}")
                elif r["wc"] is not None:
                    show_image(r["wc"])
                else:
                    st.info("No meaningful words available for the word cloud.")
                if r["font_missing"]:
                    st.warning("For Hindi word clouds, place NotoSansDevanagari-Regular.ttf beside app.py.")
        with c2:
            with card("mood"):
                s = r["sentiment"]
                heading("Mood", s["note"])
                st.markdown(pill(s["label"]) + meter(s["score"]), unsafe_allow_html=True)
                part_cols = st.columns(len(s["parts"]))
                for col, (k, v, t) in zip(part_cols, s["parts"]):
                    col.markdown(tile(k, v, "", t), unsafe_allow_html=True)
                if s["pos_words"]:
                    st.write("")
                    st.caption("Positive cues")
                    chips(s["pos_words"], tone="good")
                if s["neg_words"]:
                    st.caption("Negative cues")
                    chips(s["neg_words"], tone="bad")
        st.caption("👉 Open **Poem Studio** to generate a new poem from this song's vocabulary.")

    # ---------------- Poem Studio ----------------
    with t_poem:
        p1, p2 = st.columns([2, 3], gap="large")
        with p1:
            with card("poemctl"):
                heading("Poem controls", "Choose how the poem is made.")
                st.radio("Method", POEM_METHODS, key="poem_method", on_change=new_poem)
                st.slider("Lines", 4, 16, 8, 2, key="poem_lines", on_change=new_poem)
                st.caption(POEM_HELP[st.session_state.get("poem_method", POEM_METHODS[0])])
                gen_clicked = st.button("🎲 Generate new poem", type="primary")
        if gen_clicked:
            new_poem()
        with p2:
            poem = st.session_state.get("poem")
            if st.session_state.get("poem_error"):
                st.error(st.session_state["poem_error"])
            if poem:
                tag = st.session_state.get("poem_tag", "")
                st.markdown(
                    f'<div class="poem"><div class="tag">{html.escape(tag)}</div>'
                    + html.escape(poem).replace("\n", "<br>") + "</div>",
                    unsafe_allow_html=True,
                )
                st.write("")
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button("⬇️ Download as .txt", poem, file_name="lyriclens_poem.txt")
                with d2:
                    with st.popover("📋 Copy text"):
                        st.code(poem, language=None)
            elif not st.session_state.get("poem_error"):
                st.warning("Not enough material to build a poem. Try longer lyrics.")

    # ---------------- Words & phrases ----------------
    with t_words:
        with card("topwords"):
            heading("Most frequent words")
            top_n = st.slider("How many words", 5, 30, 15, key="top_n")
            if r["freq"]:
                freq_df = pd.DataFrame(r["freq"].most_common(top_n), columns=["Word", "Count"])
                st.bar_chart(freq_df.set_index("Word"), color="#6d4aff", height=300)
        g1, g2 = st.columns(2, gap="large")
        for col, n, name, key in ((g1, 2, "Top bigrams", "bi"), (g2, 3, "Top trigrams", "tri")):
            with col:
                with card(key):
                    heading(name, f"Most common {n}-word phrases.")
                    frame = ngram_frame(r["processed"], n)
                    if frame is not None:
                        show_df(frame, height=320)
                    else:
                        st.info(f"Not enough words for {name.lower()}.")

    # ---------------- Linguistics ----------------
    with t_ling:
        sub_pos, sub_wn, sub_wsd = st.tabs(["Parts of speech", "Word meanings", "Word senses"])
        with sub_pos:
            if r["pos_error"]:
                st.error(f"POS tagging failed: {r['pos_error']}")
            elif r["pos_rows"]:
                a, b = st.columns([3, 2], gap="large")
                with a:
                    show_df(r["pos_rows"], height=340)
                with b:
                    st.bar_chart(pd.Series(r["pos_dist"], name="Count"), color="#ff5d9e", height=340)
                if r["language"] == "Hindi":
                    st.caption("Hindi tags come from the pretrained Stanza Hindi model.")
            else:
                st.info("No POS results.")
        with sub_wn:
            if r["wn_error"]:
                st.error("WordNet could not be loaded: " + r["wn_error"])
            elif r["wn_rows"]:
                show_df(r["wn_rows"], height=380)
                if r["language"] == "Hindi":
                    st.caption("Hindi meanings come from IndoWordNet through pyiwn.")
            else:
                st.info("No WordNet entries were found.")
        with sub_wsd:
            if r["wsd_error"]:
                st.error("Word-sense disambiguation failed: " + r["wsd_error"])
            elif r["wsd_rows"]:
                show_df(r["wsd_rows"], height=380)
                st.caption(
                    "Each word's sense is picked by comparing its neighbours with sense definitions (Lesk method)."
                )
            else:
                st.info("No word senses could be identified.")

    # ---------------- Text pipeline ----------------
    with t_pipe:
        with card("clean"):
            heading("Cleaned lyrics")
            textbox(r["clean"])
        q1, q2 = st.columns(2, gap="large")
        with q1:
            with card("tokens"):
                heading(f"Tokens ({len(r['tokens'])})")
                chips(r["tokens"])
        with q2:
            with card("stops"):
                heading(f"After stop-word removal ({len(r['filtered'])})")
                chips(r["filtered"])
        with card("processed"):
            heading(f"Processed words ({len(r['processed'])})", "These feed the word cloud, phrases and poem.")
            chips(r["processed"])

st.write("")
st.caption("LyricLens · NLP lyrics analyzer")