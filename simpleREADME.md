# 🎵 LyricBstorm

**LyricBstorm** is a Python-based NLP project that analyzes song lyrics and generates a new short lyrical poem using the vocabulary and patterns found in the input.

You can enter **English or Hindi or even Marathi lyrics (devnagri script)** and explore different NLP techniques through a simple Streamlit interface.

## ✨ Features

- 📝 Lyrics text input
- 🔤 Tokenization & text cleaning
- 🚫 Stop-word removal
- 🌱 Lemmatization
- 🏷️ POS (Part-of-Speech) tagging
- 📖 WordNet / IndoWordNet lexical analysis
- 🧠 Word Sense Disambiguation (WSD)
- ☁️ Word Cloud generation
- 🔢 N-Gram analysis
- 📊 Word frequency & vocabulary analysis
- ❤️ Sentiment analysis
- 🎵 New lyric/poem generation
- 🇬🇧 English + 🇮🇳 Hindi support

## 🔄 How It Works

```text
Lyrics Input
     ↓
Text Preprocessing
     ↓
Tokenization + Stopword Removal
     ↓
Lemmatization
     ↓
POS Tagging
     ↓
WordNet / WSD
     ↓
Vocabulary + N-Gram Analysis
     ↓
Sentiment Analysis
     ↓
New Lyrics Generation
```

## 🛠️ Technologies

- **Python**
- **Streamlit** – Web interface
- **NLTK** – NLP processing
- **WordNet / IndoWordNet** – Word meanings & relations
- **Stanza** – Hindi NLP
- **WordCloud** – Visualization
- **VADER** – English sentiment analysis
- **Pandas / NumPy / Matplotlib** – Data processing & visualization

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/parth-bidave/LyricLens
cd LyricLens
```

Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install streamlit nltk wordcloud matplotlib pandas numpy vaderSentiment stanza pyiwn
```

Download the Hindi NLP model:

```bash
python -c "import stanza; stanza.download('hi')"
```

## ▶️ Run

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## 🎵 Usage

1. Enter or paste song lyrics.
2. Select the language if required.
3. Click **Analyze**.
4. Explore the NLP results.
5. View vocabulary, Word Cloud, POS, N-Grams, sentiment and lexical information.
6. Generate a new short lyrical poem based on the input.

## 🧠 Generation Method

The current lyric generator uses a **simple probabilistic word-chain approach** based on relationships between consecutive words in the input lyrics.

It is an **NLP/academic project**, not an LLM or music-generation model.

## ⚠️ Note

Installation may take 15-20 min

## 🎓 Project Purpose

LyricBstorm demonstrates how multiple NLP techniques can be combined into one practical application for **lyrics analysis and creative text generation**.

> *Feed it lyrics. Let NLP make a little lyrical chaos.* 🎵😄
