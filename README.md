
# Module 3 – SupportAI

A simple AI-powered customer support system built using FAQ search, LLM responses, TF-IDF matching, and conversational support.

## Tasks

### Task 1 – FAQ Search

* Stores FAQs with categories, questions, answers, and keywords.
* Supports keyword-based FAQ search.
* Provides FAQ lookup by ID and category.
* Returns relevant FAQ results for user queries.

**File:** `task1.py`

### Task 2 – LLM Integration

* Integrates OpenRouter API for AI-generated responses.
* Uses the matched FAQ as the knowledge source.
* Generates clear responses to user questions.
* Uses `.env` for the `OPENROUTER_API_KEY`.

**File:** `task2.py`

### Task 3 – Semantic FAQ Matching

* Uses **TF-IDF** and **cosine similarity** for FAQ matching.
* Implements hybrid search using keyword search + TF-IDF.
* Provides confidence scores for matches.
* Supports configurable confidence thresholds.

**File:** `task3.py`

### Task 4 – Support Agent

* Integrates Tasks 1–3 into a conversational support agent.
* Maintains conversation history.
* Generates FAQ-based AI responses.
* Provides confidence scores and FAQ IDs.
* Supports human escalation with mock ticket IDs.
* Supports `history`, `reset`, `escalate`, and `quit` commands.
* Suggests escalation after repeated low-confidence queries.

**File:** `task4.py`

## Project Structure

```text
Module-3/
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── requirements.txt
├── .env
└── README.md
```

## Technologies

* Python
* OpenRouter API
* Scikit-learn
* TF-IDF
* Cosine Similarity
* python-dotenv

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Run:

```bash
python task1.py
python task2.py
python task3.py
python task4.py
```
=======

