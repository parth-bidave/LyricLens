# 🎵 LyricBstorm

### NLP-Based Song Lyrics Analyzer & Generator

> **Give it lyrics. Let NLP do the overthinking. 🎶🧠**

LyricBstorm is a fun **Natural Language Processing (NLP)** project built with Python and Streamlit.

The idea started from a simple college NLP project question:

> **“Can we take an existing song and use NLP techniques to understand it and generate a completely new lyrical piece from its language patterns?”**

So... I decided to find out.

And yes, things got unnecessarily interesting. 😂

---

## 🚀 What is LyricBstorm?

LyricBstorm takes **English or Hindi song lyrics** as input and applies multiple NLP techniques to analyze the text.

After analyzing the lyrics, it can generate a **new short lyrical poem of up to 14 lines**, using vocabulary and word relationships learned from the provided lyrics.

It is **not an AI music generator** and it does not generate audio.

It is primarily an **NLP learning/project system** that demonstrates how different NLP techniques can be combined into one application.

---

## ✨ What Can It Do?

You paste your lyrics → LyricBstorm starts doing NLP things to them.

### 🧹 Text Processing

The system performs basic preprocessing such as:

- Text cleaning
- Tokenization
- Stop-word removal
- Lemmatization
- Language detection

It currently supports:

- 🇬🇧 English
- 🇮🇳 Hindi

---

### 🏷️ POS Tagging

LyricBstorm identifies the grammatical role of words using **Part-of-Speech (POS) tagging**.

For example:

```text
love → noun/verb
beautiful → adjective
run → verb
```

This helps the system understand the linguistic structure of the lyrics.

---

### 📖 WordNet Analysis

For English lyrics, the project uses **WordNet** to explore:

- Word meanings
- Synonyms
- Antonyms
- Synsets

Hindi lexical analysis is supported through **IndoWordNet** using `pyiwn`.

Because apparently knowing that a word exists wasn't enough. We had to ask what it means too. 😭

---

### 🧠 Word Sense Disambiguation

Words can have multiple meanings depending on their context.

LyricBstorm attempts to identify the appropriate meaning of a word based on its surrounding context.

For example:

```text
bank
```

could refer to:

```text
🏦 financial institution
```

or

```text
🌊 river bank
```

Context matters.

The project uses **NLTK's WSD functionality for English** and **IndoWordNet-based processing for Hindi**.

---

### ☁️ Word Cloud

The application generates a visual word cloud from the lyrics.

This makes it easy to see which words dominate the song.

Basically:

> “What words does this song obsess over?”

Now we have data to prove it. 😂

---

### 🔢 N-Grams

LyricBstorm generates:

- Unigrams
- Bigrams
- Trigrams

This helps identify common sequences and patterns of words appearing together.

For example:

```text
I love
love you
you forever
```

These patterns can also contribute to the lyric-generation process.

---

### 📊 Vocabulary Analysis

The system calculates information such as:

- Total meaningful words
- Unique vocabulary
- Most frequent words
- Word frequencies
- POS distribution

This gives a basic vocabulary profile of the song.

---

### ❤️ Sentiment Analysis

LyricBstorm analyzes the emotional tone of the lyrics.

For English, it uses **VADER Sentiment Analysis**.

Hindi lyrics use a lightweight **Hindi sentiment lexicon**.

The system classifies the overall tone as:

```text
😊 Positive
😐 Neutral
😔 Negative
```

So yes, you can finally prove that the song you've been listening to for 3 hours is, in fact, depressing.

---

## 🎵 Lyric Generation

And now the fun part.

After processing the lyrics, LyricBstorm uses the words and patterns from the input to generate a **new short lyrical poem**.

The current generator uses a simple **word-chain / probabilistic approach**.

Conceptually:

```text
Input Lyrics
      ↓
Tokenization
      ↓
Cleaning
      ↓
Stop-word Removal
      ↓
Vocabulary Extraction
      ↓
Word Relationships
      ↓
N-Gram / Word Chain Patterns
      ↓
Generated Lyrics
```

The generated output is intended as an **NLP-generated lyrical experiment**, not as a replacement for a professional songwriter.

Sometimes it produces something surprisingly good.

Sometimes it sounds like three poets arguing inside a washing machine.

That's part of the experiment. 😂

---

# 🧠 How The Project Works

The overall pipeline looks like this:

```text
             SONG LYRICS
                  │
                  ▼
          Language Detection
                  │
                  ▼
           Text Cleaning
                  │
                  ▼
            Tokenization
                  │
                  ▼
         Stop-word Removal
                  │
                  ▼
           Lemmatization
                  │
                  ▼
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   POS Tagging          WordNet
        │                   │
        │                   ▼
        │                 WSD
        │                   │
        └─────────┬─────────┘
                  ▼
          Vocabulary Analysis
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Word Cloud          Sentiment
        │                   │
        └─────────┬─────────┘
                  ▼
          Pattern Analysis
                  │
                  ▼
          Lyrics Generation
                  │
                  ▼
             NEW LYRICS 🎵
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web interface |
| NLTK | NLP processing |
| WordNet | English lexical analysis |
| IndoWordNet / pyiwn | Hindi lexical analysis |
| Stanza | Hindi linguistic processing |
| WordCloud | Word-cloud visualization |
| Pandas | Data handling |
| Matplotlib | Visualization |
| VADER | English sentiment analysis |

---

# 📂 Project Structure

A simple version of the project looks like:

```text
LyricBstorm/
│
├── app.py
├── README.md
├── requirements.txt
│
└── venv/
```

> `venv/` is your local virtual environment and should **not** be uploaded to GitHub.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/LyricBstorm.git
cd LyricBstorm
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

You should see something similar to:

```text
(venv) C:\...\LyricBstorm>
```

---

## 3. Install dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you are setting it up manually:

```bash
pip install streamlit nltk wordcloud matplotlib pandas numpy
pip install vaderSentiment stanza pyiwn
```

---

## 4. Download Stanza Hindi Model

For Hindi processing:

```bash
python -c "import stanza; stanza.download('hi')"
```

This downloads the Hindi language model required by Stanza.

---

## 5. Run the application

Start Streamlit:

```bash
streamlit run app.py
```

Streamlit will provide a local address, usually something like:

```text
http://localhost:8501
```

Open that address in your browser.

And congratulations.

You have successfully turned your terminal into a tiny music laboratory. 🎵🧪

---

# 🎯 How To Use It

### Step 1

Start the application:

```bash
streamlit run app.py
```

### Step 2

Paste a song's lyrics into the text box.

You can use:

- English lyrics
- Hindi lyrics

### Step 3

Click:

```text
🔍 Analyze Lyrics
```

### Step 4

Explore the generated results:

```text
🧹 Text Processing
🏷️ POS Tagging
📖 WordNet
🧠 WSD
☁️ Word Cloud
🔢 N-Grams
📊 Vocabulary Profile
❤️ Sentiment Analysis
🎵 Generated Poem
```

---

# 🧪 Example

Input:

```text
Paste your song lyrics here...
```

LyricBstorm processes them and produces information such as:

```text
Language: Hindi

Meaningful Words: 42
Unique Vocabulary: 28

Sentiment: Positive

Top Words:
love
heart
dream
life
...
```

Then it generates a new lyrical piece based on the vocabulary and word relationships found in the input.

---

# 🎓 Why This Is an NLP Project

LyricBstorm combines several fundamental NLP concepts into a single application:

- Tokenization
- Stop-word removal
- Lemmatization
- POS tagging
- Lexical databases
- Word Sense Disambiguation
- N-Gram analysis
- Frequency analysis
- Sentiment analysis
- Text generation

Instead of demonstrating each NLP technique as an isolated experiment, the project connects them into one complete workflow.

That makes it useful as a practical demonstration of how NLP techniques can work together on real-world text.

---

# ⚠️ Important Note About Generated Lyrics

LyricBstorm is an **educational NLP project**.

The generated lyrics are created from patterns and vocabulary found in the input text.

The system does **not guarantee grammatically perfect, meaningful, or professionally written lyrics**.

The quality of generation depends heavily on the amount and quality of text provided.

More text generally gives the model more patterns to work with.

So if you paste:

```text
I love you
```

and expect Shakespeare...

please lower your expectations. 😂

---

# 📚 Learning Purpose

This project was built primarily to understand how NLP works in practice.

It demonstrates the journey from:

```text
Raw Text
   ↓
Preprocessing
   ↓
Linguistic Analysis
   ↓
Semantic Analysis
   ↓
Statistical Patterns
   ↓
Text Generation
```

It is especially useful for students learning the fundamentals of Natural Language Processing.

---

# 🚧 Current Limitations

This project is intentionally built using relatively lightweight NLP techniques.

Some limitations include:

- Generated lyrics may sometimes be repetitive.
- Word-chain generation does not have deep semantic understanding.
- Hindi NLP resources are more limited than English resources.
- Sentiment analysis is lexicon-based for Hindi.
- WSD can be imperfect for ambiguous or poetic language.
- The system does not generate music or vocals.
- The generated text should be considered experimental rather than production-quality songwriting.

---

# 🔮 Possible Future Improvements

Some interesting directions for future versions:

- Transformer-based lyric generation
- Better multilingual sentiment analysis
- Improved Hindi WSD
- Better semantic similarity
- Topic/theme extraction
- Rhyme detection
- Meter/rhythm analysis
- Genre classification
- More advanced language models
- Song structure detection
- Originality / similarity analysis
- Export generated lyrics as `.txt` or `.pdf`

Basically...

Version 1 is the little NLP project.

Version 10 is where we start worrying about why it has written 47 songs about heartbreak. 💀

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you find a bug or have an idea:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test them.
5. Open a Pull Request.

Or open an issue and tell me what you broke.

I'll probably appreciate it. 😭

---

# 📜 Disclaimer

Please use lyrics that you have the right to use.

Do not upload or redistribute copyrighted lyrics without permission.

LyricBstorm is an educational NLP project and is not intended to reproduce or distribute copyrighted song lyrics.

---

# ❤️ Built For Learning

LyricBstorm started as a college NLP project idea and turned into a small experiment in combining multiple NLP techniques into one application.

If you try it, experiment with different songs, languages, and writing styles.

Some inputs will produce surprisingly interesting results.

Some will produce absolute lyrical chaos.

Both are data.

**Have fun with it. 🎵🧠**

---

## ⭐ If You Like It

If you found the project interesting, consider giving the repository a ⭐ on GitHub.

It costs exactly **₹0** and makes the developer suspiciously happy. 😄
