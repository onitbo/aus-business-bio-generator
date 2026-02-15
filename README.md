# Australian Business Bio Generator

Generates high-quality plain-text business descriptions for Australian businesses using Google Places data and Anthropic's Claude API.

## How It Works

1. **Research** — Fetches business data from Google Places API and the business website
2. **Draft** — Claude generates a plain-text bio based on research
3. **Review** — Claude reviews the draft for quality, accuracy, and formatting
4. **Revise** — If needed, iterates with feedback until confidence threshold is met

## Setup

```bash
cp .env.example .env
# Edit .env with your API keys
pip install anthropic httpx python-dotenv
```

## Usage

```bash
python3 src/batch_runner.py
```

Edit `BUSINESSES` in `src/batch_runner.py` to change target businesses.

## Output

- `outputs/<slug>.md` — Plain text business description
- `outputs/<slug>_description_meta.json` — Metadata including confidence score, token usage, sources

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Anthropic model to use | `claude-opus-4-6` |
| `MODEL_TEMPERATURE` | Generation temperature | `0.7` |
| `MAX_ITERATIONS` | Max review/revise cycles | `3` |
| `TARGET_CONFIDENCE` | Minimum confidence to pass | `0.8` |
