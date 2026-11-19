# Grammar-constrained decoding demo

Option 1 from the final project. The language is a signup line:

```
name=maya grade=10 club=robotics
```

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 demo.py
```

`python3 demo.py 1` runs the first sentence only.

The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB).
After that, ten sentences takes about half a minute on a CPU.

## What is held fixed

The baseline is the same model, the same prompts, and the same greedy
decoding. The only difference is that the grammar mask is turned off.

## The number

Semantic accuracy: does the line have the right name, grade, and club?
Ten sentences. A second number is average generation time.
