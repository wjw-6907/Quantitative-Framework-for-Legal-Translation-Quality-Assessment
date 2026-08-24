# Legal Translation Quality Assessment

A research prototype for quantitatively evaluating Chinese–English legal translations.

法律翻译质量评估量化模型：从术语、逻辑关系和法律效力三个维度，对中英法律文本进行可解释的辅助评分。

## What it evaluates

- **Terminology consistency** — matches source terms against a weighted bilingual legal glossary.
- **Logical consistency** — compares legal relations such as condition, obligation, prohibition, causality, time, and choice.
- **Legal-effect equivalence** — uses an LLM to extract normative and performative verbs, then compares their strength.
- **Batch and desktop workflows** — supports spreadsheet-based evaluation and a small Tkinter review interface.

> [!NOTE]
> This is an experimental research implementation, not a certified legal translation or legal-advice system. Scores should be reviewed by qualified humans.

## Repository contents

| Path | Purpose |
| --- | --- |
| `test.py` | Core scoring functions and an optional batch runner |
| `show.py` | Tkinter interface for reviewing spreadsheet rows |
| `gen_terms.py` | Converts `legal_terms.xlsx` into the runtime glossary |
| `terms.txt` | Generated bilingual term dictionary |
| `bigmodel_test.py` | Minimal DashScope integration experiment |
| `legal_terms.xlsx` | Source glossary and weights |

## Requirements

- Python 3.10+
- A DashScope API key for legal-effect extraction
- spaCy English and Chinese small models

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

cp .env.example .env
export DASHSCOPE_API_KEY="your-key"
```

Generate the glossary when `legal_terms.xlsx` changes:

```bash
python gen_terms.py
```

Use the scorer from Python:

```python
from test import all_score

score, has_terms, has_logic, has_legal_effect = all_score(
    "卖方必须按时交货。",
    "The seller shall deliver the goods on time.",
)
print(score)
```

The batch runner at the bottom of `test.py` expects a workbook named `测试语料1000条.xlsx` with `中文` and `英文` columns. The desktop interface in `show.py` similarly expects a local workbook; update its `file_path` before running.

## Configuration and security

The project reads `DASHSCOPE_API_KEY` from the environment. Never commit real API keys or local `.env` files. If a key has ever been committed to a public repository, revoke it in the provider console even after removing it from the current code, because it may remain in Git history.

## Known limitations

- The model is heuristic and requires domain-expert validation.
- Some prototype helpers and dataset paths are retained for reproducibility.
- LLM responses can vary and may fail schema parsing.
- The included glossary may not cover every jurisdiction or legal domain.

## Roadmap

- Add a stable command-line interface and configurable input paths.
- Introduce unit tests for each scoring dimension.
- Validate weights and thresholds on a documented benchmark.
- Separate experimental scripts from the reusable scoring package.

## License

No license has been selected yet. Until one is added, the repository remains viewable but is not automatically licensed for reuse.
