# Grammar-constrained decoding

Option 1 from the final project. A legal signup line looks like:

```
name=maya grade=10 club=robotics
```

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 demo.py
```

`demo.ipynb` runs the same comparison and saves the printed configuration.
`results.json` stores every line and the summary counts.
`report.pdf` is the write-up.

The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB).
The full run is 50 sentences in four conditions.

## Data

- 25 standard sentences, all phrased like "Maya is in grade 10 and wants the robotics club."
- 25 held-out sentences with different word order and extra details that should be ignored.
- The 1-shot example is not in either split: "Alex is in grade 7 and wants the music club."

## Conditions

All four runs use `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding (`do_sample=False`), `max_new_tokens=32`, and seed 0.

Inside one prompt, the grammar mask is the only change.
The 0-shot prompt and the 1-shot prompt differ by that one example.

## Semantic accuracy from `demo.ipynb`

Correct means the line parses and the name, grade, and club match.

| Prompt | Decoding | Standard | Held-out | Overall | Avg seconds |
| --- | --- | --- | --- | --- | --- |
| 0-shot | baseline | 0/25 | 0/25 | 0/50 | 0.8 |
| 0-shot | constrained | 16/25 | 12/25 | 28/50 | 1.3 |
| 1-shot | baseline | 22/25 | 11/25 | 33/50 | 0.6 |
| 1-shot | constrained | 22/25 | 13/25 | 35/50 | 1.5 |

`force_finish` ran 0 times. Every constrained line parsed. The 1-shot standard outputs are the same with the mask on or off.
