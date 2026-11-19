"""Grammar-constrained decoding for signup lines.

A legal line looks like: name=maya grade=10 club=robotics

Four runs share the model, greedy decoding, max_new_tokens, and seed.
Two prompts (0-shot and 1-shot) cross two decoders (baseline and constrained).
Inside one prompt, the grammar mask is the only difference.
"""

import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from grammar import fields, force_finish, is_complete, scan

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
SEED = 0
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789= ")
RESULTS_PATH = "results.json"

SYSTEM = (
    "You write exactly one signup line and nothing else. "
    "The form is name=<word> grade=<number> club=<word>. "
    "Use lowercase letters only. "
    "The club is one word and does not include the word club. "
    "No quotes and no extra words."
)

# One worked example. It is not one of the evaluation sentences.
EXAMPLE_SENTENCE = "Alex is in grade 7 and wants the music club."
EXAMPLE_NAME = "alex"
EXAMPLE_GRADE = 7
EXAMPLE_CLUB = "music"

# Standard sentences all use one phrasing. Held-out sentences do not.
# tuple: split, sentence, name, grade, club
PROBLEMS = [
    ("standard", "Maya is in grade 10 and wants the robotics club.", "maya", 10, "robotics"),
    ("standard", "Sam is in grade 11 and wants the chess club.", "sam", 11, "chess"),
    ("standard", "Ana is in grade 9 and wants the art club.", "ana", 9, "art"),
    ("standard", "Leo is in grade 12 and wants the soccer club.", "leo", 12, "soccer"),
    ("standard", "Nina is in grade 10 and wants the band club.", "nina", 10, "band"),
    ("standard", "Omar is in grade 11 and wants the drama club.", "omar", 11, "drama"),
    ("standard", "Priya is in grade 9 and wants the math club.", "priya", 9, "math"),
    ("standard", "Chris is in grade 12 and wants the debate club.", "chris", 12, "debate"),
    ("standard", "Elena is in grade 10 and wants the dance club.", "elena", 10, "dance"),
    ("standard", "Jordan is in grade 11 and wants the coding club.", "jordan", 11, "coding"),
    ("standard", "Liam is in grade 9 and wants the baseball club.", "liam", 9, "baseball"),
    ("standard", "Sofia is in grade 10 and wants the choir club.", "sofia", 10, "choir"),
    ("standard", "Noah is in grade 11 and wants the tennis club.", "noah", 11, "tennis"),
    ("standard", "Ava is in grade 12 and wants the yearbook club.", "ava", 12, "yearbook"),
    ("standard", "Ethan is in grade 10 and wants the swimming club.", "ethan", 10, "swimming"),
    ("standard", "Mia is in grade 9 and wants the photography club.", "mia", 9, "photography"),
    ("standard", "Lucas is in grade 11 and wants the volleyball club.", "lucas", 11, "volleyball"),
    ("standard", "Emma is in grade 12 and wants the theater club.", "emma", 12, "theater"),
    ("standard", "Jack is in grade 9 and wants the hockey club.", "jack", 9, "hockey"),
    ("standard", "Zoe is in grade 10 and wants the gardening club.", "zoe", 10, "gardening"),
    ("standard", "Henry is in grade 11 and wants the fencing club.", "henry", 11, "fencing"),
    ("standard", "Lily is in grade 12 and wants the archery club.", "lily", 12, "archery"),
    ("standard", "Owen is in grade 10 and wants the knitting club.", "owen", 10, "knitting"),
    ("standard", "Chloe is in grade 9 and wants the sailing club.", "chloe", 9, "sailing"),
    ("standard", "Ryan is in grade 11 and wants the manga club.", "ryan", 11, "manga"),
    ("held_out", "The robotics club is the one Amir wants, and Amir is in grade 10.", "amir", 10, "robotics"),
    ("held_out", "For a grade 12 student named Brooke, the chosen club is sailing.", "brooke", 12, "sailing"),
    ("held_out", "Kenji is in 9th and he is trying to join the manga club.", "kenji", 9, "manga"),
    ("held_out", "Claire is in grade 11. Her homeroom is 112 and her friend is Pete, but the club she wants is writing.", "claire", 11, "writing"),
    ("held_out", "Please sign up Hector for the archery club. He is in grade 10 this school year, and his friend Luis is not part of the signup.", "hector", 10, "archery"),
    ("held_out", "Student Gina, grade 8, club cooking, homeroom 3.", "gina", 8, "cooking"),
    ("held_out", "The student named Felix wants baking, and he is in the 12th grade.", "felix", 12, "baking"),
    ("held_out", "Do not put Tara in the band club. Tara is in grade 9 and wants the chess club.", "tara", 9, "chess"),
    ("held_out", "Grade 10 student Pablo has been asking about the fishing club all year.", "pablo", 10, "fishing"),
    ("held_out", "It is not the art club. Quinn in grade 11 wants the science club.", "quinn", 11, "science"),
    ("held_out", "Sign Willa up for the hiking club. Her grade is 12. She also plays piano, which is not a club here.", "willa", 12, "hiking"),
    ("held_out", "Before lunch, the office should add Caleb, who is in grade 10, to the boxing club rather than the club his friend Adam chose.", "caleb", 10, "boxing"),
    ("held_out", "Journalism is the club. The student is Dana, and the grade is 9.", "dana", 9, "journalism"),
    ("held_out", "Add Sasha to the pottery club. She is in grade 11. Her homeroom is 18.", "sasha", 11, "pottery"),
    ("held_out", "This year's signup: Ivan wants the cycling club. Ivan's grade is 10, and his friend Nora is already signed up for something else.", "ivan", 10, "cycling"),
    ("held_out", "Put the grade first: 11. Then the name Helen. Then the club debate.", "helen", 11, "debate"),
    ("held_out", "Omar's brother Theo, who is in grade 9, is the one who wants the soccer club, not Omar.", "theo", 9, "soccer"),
    ("held_out", "A note from a grade 12 homeroom says Uma wants knitting, and her friend Chloe should be ignored.", "uma", 12, "knitting"),
    ("held_out", "The only club that matters for Reid, a grade 10 student whose school year started in August, is fencing.", "reid", 10, "fencing"),
    ("held_out", "Can the form list Lucia under grade 11 for the choir club? Her homeroom number is 220 and is not a grade.", "lucia", 11, "choir"),
    ("held_out", "Victor is in grade 8 and the slip says wrestling, not swimming.", "victor", 8, "wrestling"),
    ("held_out", "Name comes last here. Grade 9, photography club, student Yara.", "yara", 9, "photography"),
    ("held_out", "Despite his friend Max joining band, Elliot in grade 12 asked for the gaming club.", "elliot", 12, "gaming"),
    ("held_out", "Office slip: student Nadia, club running, grade 10, school year 2026, homeroom 14.", "nadia", 10, "running"),
    ("held_out", "The dance club is full, so put Wesley, who is in grade 11, in the drama club instead.", "wesley", 11, "drama"),
]

STANDARD_RE = re.compile(
    r"^[A-Z][a-z]+ is in grade [0-9]+ and wants the [a-z]+ club\.$"
)
PROMPT_NAMES = ("0-shot", "1-shot")
DECODE_NAMES = ("baseline", "constrained")
SPLIT_NAMES = ("standard", "held_out", "overall")


def gold_line(name, grade, club):
    return f"name={name} grade={grade} club={club}"


def user_text(sentence):
    return f"Write the signup line for this sentence: {sentence}"


def format_prompt(sentence, shots):
    """Build the chat prompt. shots is 0 or 1. Both decoders use this string."""
    if shots not in (0, 1):
        raise ValueError(shots)
    parts = [
        "<|im_start|>system\n",
        SYSTEM,
        "<|im_end|>\n",
    ]
    if shots == 1:
        parts.extend(
            [
                "<|im_start|>user\n",
                user_text(EXAMPLE_SENTENCE),
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
                gold_line(EXAMPLE_NAME, EXAMPLE_GRADE, EXAMPLE_CLUB),
                "<|im_end|>\n",
            ]
        )
    parts.extend(
        [
            "<|im_start|>user\n",
            user_text(sentence),
            "<|im_end|>\n",
            "<|im_start|>assistant\n",
        ]
    )
    return "".join(parts)


def check_problems():
    sentences = []
    counts = {"standard": 0, "held_out": 0}
    example_line = gold_line(EXAMPLE_NAME, EXAMPLE_GRADE, EXAMPLE_CLUB)
    if fields(example_line) != (EXAMPLE_NAME, EXAMPLE_GRADE, EXAMPLE_CLUB):
        raise SystemExit("bad worked example")
    if not STANDARD_RE.fullmatch(EXAMPLE_SENTENCE):
        raise SystemExit("worked example should use the standard phrasing")
    for split, sentence, name, grade, club in PROBLEMS:
        if split not in ("standard", "held_out"):
            raise SystemExit(f"bad split: {sentence}")
        if sentence in sentences or sentence == EXAMPLE_SENTENCE:
            raise SystemExit(f"repeated sentence: {sentence}")
        sentences.append(sentence)
        parsed = fields(gold_line(name, grade, club))
        if parsed != (name, grade, club):
            raise SystemExit(f"bad gold line: {sentence}")
        if not re.fullmatch(r"[a-z]+", name) or not re.fullmatch(r"[a-z]+", club):
            raise SystemExit(f"name and club must be one lowercase word: {sentence}")
        if not isinstance(grade, int):
            raise SystemExit(f"grade must be an int: {sentence}")
        is_standard = STANDARD_RE.fullmatch(sentence) is not None
        if split == "standard" and not is_standard:
            raise SystemExit(f"standard sentence has the wrong shape: {sentence}")
        if split == "held_out" and is_standard:
            raise SystemExit(f"held-out sentence uses the standard shape: {sentence}")
        counts[split] += 1
    return counts


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


def summarize(rows, prompt_name, decode_name, split):
    chosen = []
    for row in rows:
        if row["prompt"] != prompt_name or row["decoding"] != decode_name:
            continue
        if split != "overall" and row["split"] != split:
            continue
        chosen.append(row)
    total = len(chosen)
    correct = sum(row["category"] == "correct" for row in chosen)
    wrong = sum(row["category"] == "wrong_meaning" for row in chosen)
    invalid = sum(row["category"] == "invalid" for row in chosen)
    forced = sum(row["force_finish"] for row in chosen)
    seconds = [row["seconds"] for row in chosen]
    total_seconds = sum(seconds)
    average = total_seconds / total if total else 0.0
    valid = correct + wrong
    return {
        "prompt": prompt_name,
        "decoding": decode_name,
        "split": split,
        "n": total,
        "correct": correct,
        "wrong_meaning": wrong,
        "invalid": invalid,
        "syntactic_validity": valid / total if total else 0.0,
        "correct_rate": correct / total if total else 0.0,
        "wrong_meaning_rate": wrong / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "force_finish": forced,
        "average_seconds": average,
        "total_seconds": total_seconds,
    }


def _frac(count, total):
    return f"{count}/{total}"


def print_summary(summary_rows):
    print(
        f"{'prompt':<8} {'decoding':<12} {'split':<10} {'n':>3} "
        f"{'valid':>6} {'correct':>8} {'wrong':>8} {'invalid':>8} "
        f"{'forced':>6} {'avg_s':>6} {'total_s':>7}"
    )
    for row in summary_rows:
        total = row["n"]
        print(
            f"{row['prompt']:<8} {row['decoding']:<12} {row['split']:<10} {total:>3} "
            f"{row['syntactic_validity']:>6.0%} {_frac(row['correct'], total):>8} "
            f"{_frac(row['wrong_meaning'], total):>8} {_frac(row['invalid'], total):>8} "
            f"{row['force_finish']:>6} {row['average_seconds']:>6.1f} {row['total_seconds']:>7.1f}"
        )


def print_failures(rows):
    print()
    print("Failure examples")
    for prompt_name in PROMPT_NAMES:
        for decode_name in DECODE_NAMES:
            group = [
                row
                for row in rows
                if row["prompt"] == prompt_name and row["decoding"] == decode_name
            ]
            wrong = [row for row in group if row["category"] == "wrong_meaning"]
            invalid = [row for row in group if row["category"] == "invalid"]
            print(
                f"\n{prompt_name} {decode_name}: "
                f"{len(wrong)} wrong_meaning, {len(invalid)} invalid"
            )
            shown = wrong[:6] + invalid[:3]
            if not shown:
                print("  none")
                continue
            for row in shown:
                print(f"  [{row['split']}] {row['category']}")
                print(f"    sentence: {row['sentence']}")
                print(f"    gold: {row['gold']}")
                print(f"    output: {row['output']!r}")
            hidden = len(wrong) + len(invalid) - len(shown)
            if hidden > 0:
                print(f"  ... {hidden} more failures in results.json")


def main(limit=None):
    counts = check_problems()
    problems = PROBLEMS if limit is None else PROBLEMS[:limit]

    print("Grammar-constrained decoding")
    print(f"model: {MODEL_NAME}")
    print("dtype: float32")
    print("decoding: greedy (do_sample=False)")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"seed: {SEED}")
    print(f"standard sentences: {counts['standard']}")
    print(f"held_out sentences: {counts['held_out']}")
    print(f"running: {len(problems)}")
    print(f"worked example: {EXAMPLE_SENTENCE}")
    print(f"worked line: {gold_line(EXAMPLE_NAME, EXAMPLE_GRADE, EXAMPLE_CLUB)}")
    print("Inside one prompt, held fixed: model, prompt text, greedy decoding, max_new_tokens, seed")
    print("Only difference inside one prompt: grammar mask on or off")
    print("0-shot vs 1-shot changes the prompt by one example and nothing else")
    print("loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    pieces = legal_pieces(tokenizer)
    singles = {text for _token_id, text in pieces if len(text) == 1}
    missing = ALLOWED - singles
    if missing:
        raise SystemExit(f"tokenizer is missing characters: {sorted(missing)}")
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    print(f"grammar tokens: {len(pieces)}")
    print()

    rows = []
    for index, (split, sentence, name, grade, club) in enumerate(problems, start=1):
        gold = gold_line(name, grade, club)
        print(f"[{index}/{len(problems)}] {split}: {sentence}")
        print(f"  gold: {gold}")
        for shots, prompt_name in ((0, "0-shot"), (1, "1-shot")):
            prompt = format_prompt(sentence, shots)
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            for constrained, decode_name in ((False, "baseline"), (True, "constrained")):
                text, elapsed, forced = generate_line(
                    model, tokenizer, prompt_ids, pieces, eos_ids, constrained
                )
                label = category(text, name, grade, club)
                row = {
                    "split": split,
                    "sentence": sentence,
                    "name": name,
                    "grade": grade,
                    "club": club,
                    "gold": gold,
                    "prompt": prompt_name,
                    "shots": shots,
                    "decoding": decode_name,
                    "output": text,
                    "category": label,
                    "seconds": round(elapsed, 4),
                    "force_finish": forced,
                }
                rows.append(row)
                note = " force_finish" if forced else ""
                print(f"  {prompt_name:<6} {decode_name:<12} {text!r}  {label}  {elapsed:.1f}s{note}")
        print()

    summary_rows = []
    for prompt_name in PROMPT_NAMES:
        for decode_name in DECODE_NAMES:
            for split in SPLIT_NAMES:
                summary_rows.append(summarize(rows, prompt_name, decode_name, split))

    print("Summary")
    print_summary(summary_rows)
    print_failures(rows)

    if limit is None:
        payload = {
            "config": {
                "model": MODEL_NAME,
                "dtype": "float32",
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "seed": SEED,
                "example_sentence": EXAMPLE_SENTENCE,
                "example_line": gold_line(EXAMPLE_NAME, EXAMPLE_GRADE, EXAMPLE_CLUB),
                "standard": counts["standard"],
                "held_out": counts["held_out"],
            },
            "rows": rows,
            "summary": summary_rows,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print()
        print(f"wrote {RESULTS_PATH}")
    return summary_rows


if __name__ == "__main__":
    chosen = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(chosen)
