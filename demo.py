"""Grammar-constrained decoding for signup lines.

A signup line looks like: name=maya grade=10 club=robotics

Baseline and constrained runs use the same model, the same sentences,
and the same greedy decoding. The only difference is the grammar mask.

The number this demo reports is semantic accuracy: of 10 sentences,
how many lines name the right student, grade, and club.
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from grammar import fields, force_finish, is_complete, scan

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
SEED = 0
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789= ")

SYSTEM = (
    "You write exactly one signup line and nothing else. "
    "The form is name=<word> grade=<number> club=<word>. "
    "Use lowercase letters only. "
    "The club is one word and does not include the word club. "
    "No quotes and no extra words."
)

# sentence, name, grade, club
PROBLEMS = [
    ("Maya is in grade 10 and wants the robotics club.", "maya", 10, "robotics"),
    ("Sam is in grade 11 and wants the chess club.", "sam", 11, "chess"),
    ("Ana is in grade 9 and wants the art club.", "ana", 9, "art"),
    ("Leo is in grade 12 and wants the soccer club.", "leo", 12, "soccer"),
    ("Nina is in grade 10 and wants the band club.", "nina", 10, "band"),
    ("Omar is in grade 11 and wants the drama club.", "omar", 11, "drama"),
    ("Priya is in grade 9 and wants the math club.", "priya", 9, "math"),
    ("Chris is in grade 12 and wants the debate club.", "chris", 12, "debate"),
    ("Elena is in grade 10 and wants the dance club.", "elena", 10, "dance"),
    ("Jordan is in grade 11 and wants the coding club.", "jordan", 11, "coding"),
]


def format_prompt(sentence):
    user = f"Write the signup line for this sentence: {sentence}"
    return (
        "<|im_start|>system\n"
        f"{SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def gold_line(name, grade, club):
    return f"name={name} grade={grade} club={club}"


def legal_pieces(tokenizer):
    """Tokens whose text can appear inside a signup line."""
    pieces = []
    ids = list(range(len(tokenizer)))
    for start in range(0, len(ids), 2048):
        batch = ids[start : start + 2048]
        texts = tokenizer.batch_decode([[i] for i in batch], skip_special_tokens=True)
        for token_id, text in zip(batch, texts):
            if text and all(ch in ALLOWED for ch in text):
                pieces.append((token_id, text))

    prefixes = [
        "",
        "name=",
        "name=maya",
        "name=maya ",
        "name=maya grade=",
        "name=maya grade=10",
        "name=maya grade=10 club=",
        "name=maya grade=10 club=r",
    ]
    stable = []
    for token_id, text in pieces:
        good = True
        for prefix in prefixes:
            base_ids = tokenizer.encode(prefix, add_special_tokens=False)
            before = tokenizer.decode(base_ids, skip_special_tokens=True)
            after = tokenizer.decode(base_ids + [token_id], skip_special_tokens=True)
            if after != before + text:
                good = False
                break
        if good:
            stable.append((token_id, text))
    return stable


class GrammarMask(LogitsProcessor):
    """Cross out any next token that would break the signup grammar."""

    def __init__(self, prompt_length, pieces, eos_ids):
        self.prompt_length = prompt_length
        self.pieces = pieces
        self.piece_of = dict(pieces)
        self.eos_ids = eos_ids

    def __call__(self, input_ids, scores):
        generated = input_ids[0, self.prompt_length :].tolist()
        text = "".join(self.piece_of.get(token_id, "") for token_id in generated)
        original = scores[0].clone()
        scores[0, :] = float("-inf")
        if scan(text) != "bad":
            for token_id, piece in self.pieces:
                if scan(text + piece) != "bad":
                    scores[0, token_id] = original[token_id]
        if is_complete(text):
            for eos_id in self.eos_ids:
                scores[0, eos_id] = original[eos_id]
        if not torch.isfinite(scores[0]).any():
            for token_id, piece in self.pieces:
                if scan(text + piece) != "bad":
                    scores[0, token_id] = 0.0
                    break
        return scores


def category(text, name, grade, club):
    parsed = fields(text)
    if parsed is None:
        return "invalid"
    got_name, got_grade, got_club = parsed
    if got_name == name and got_grade == grade and got_club == club:
        return "correct"
    return "wrong_meaning"


def generate_line(model, tokenizer, prompt_ids, pieces, eos_ids, constrained):
    processors = LogitsProcessorList()
    if constrained:
        processors.append(GrammarMask(prompt_ids.shape[1], pieces, eos_ids))
    torch.manual_seed(SEED)
    started = time.perf_counter()
    output = model.generate(
        prompt_ids,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        logits_processor=processors,
    )
    elapsed = time.perf_counter() - started
    new_ids = output[0, prompt_ids.shape[1] :].tolist()
    while new_ids and new_ids[-1] in eos_ids:
        new_ids.pop()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    forced = False
    if constrained:
        piece_of = dict(pieces)
        text = "".join(piece_of.get(token_id, "") for token_id in new_ids)
        if text != decoded:
            print(f"  note: grammar text {text!r} vs decoded {decoded!r}")
        if not is_complete(text):
            text = force_finish(text)
            forced = True
    else:
        text = decoded
    return text, elapsed, forced


def main():
    limit = len(PROBLEMS)
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    problems = PROBLEMS[:limit]

    for sentence, name, grade, club in problems:
        parsed = fields(gold_line(name, grade, club))
        if parsed != (name, grade, club):
            raise SystemExit(f"bad problem: {sentence}")

    print("Grammar-constrained decoding demo")
    print(f"model: {MODEL_NAME}")
    print("decoding: greedy (do_sample=False)")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"seed: {SEED}")
    print(f"problems: {len(problems)}")
    print("held fixed: model, prompts, greedy decoding, max_new_tokens, seed")
    print("only difference: grammar mask on or off")
    print("loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    pieces = legal_pieces(tokenizer)
    singles = {text: token_id for token_id, text in pieces if len(text) == 1}
    missing = ALLOWED - set(singles)
    if missing:
        raise SystemExit(f"tokenizer is missing characters: {sorted(missing)}")
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    print(f"grammar tokens: {len(pieces)}")
    print()

    rows = []
    for sentence, name, grade, club in problems:
        prompt = format_prompt(sentence)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        baseline, base_time, _base_forced = generate_line(
            model, tokenizer, prompt_ids, pieces, eos_ids, constrained=False
        )
        constrained, con_time, forced = generate_line(
            model, tokenizer, prompt_ids, pieces, eos_ids, constrained=True
        )
        base_cat = category(baseline, name, grade, club)
        con_cat = category(constrained, name, grade, club)
        rows.append((sentence, baseline, base_cat, base_time, constrained, con_cat, con_time, forced))
        print(f"sentence: {sentence}")
        print(f"  gold: {gold_line(name, grade, club)}")
        print(f"  baseline:    {baseline!r}  {base_cat}  {base_time:.1f}s")
        note = "  (grammar finished it)" if forced else ""
        print(f"  constrained: {constrained!r}  {con_cat}  {con_time:.1f}s{note}")
        print()

    def count(label, column):
        return sum(1 for row in rows if row[column] == label)

    print("Summary")
    print(f"  baseline    correct {count('correct', 2)}/{len(rows)}")
    print(f"              wrong_meaning {count('wrong_meaning', 2)}/{len(rows)}")
    print(f"              invalid {count('invalid', 2)}/{len(rows)}")
    print(f"  constrained correct {count('correct', 5)}/{len(rows)}")
    print(f"              wrong_meaning {count('wrong_meaning', 5)}/{len(rows)}")
    print(f"              invalid {count('invalid', 5)}/{len(rows)}")
    base_avg = sum(row[3] for row in rows) / len(rows)
    con_avg = sum(row[6] for row in rows) / len(rows)
    print(f"  average time baseline: {base_avg:.1f}s")
    print(f"  average time constrained: {con_avg:.1f}s")
    print()
    print(f"NUMBER: constrained semantic accuracy {count('correct', 5)}/{len(rows)}")
    print(f"NUMBER: baseline semantic accuracy {count('correct', 2)}/{len(rows)}")


if __name__ == "__main__":
    main()
