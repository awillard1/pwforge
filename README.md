# PWForge – Password Candidate Generator

High-performance, multi-mode wordlist generator designed for **Hashcat** and **John the Ripper**.  
Streams to STDOUT for live cracking or writes compressed/split files for large runs.

> This README reflects all options implemented in `pwforge.py`, including **neural** generation, hybrid/combo modes, entropy filtering, chunking, splitting, and estimation helpers.

---

## 🔧 Requirements

### Core (non-neural modes)
- **Python 3.8+** (tested on Linux, WSL, macOS, Windows)
- Standard library only (no extra pip deps required for non-neural modes)

### Neural generation (`--mode neural`)
- **PyTorch** (CPU or CUDA build)  
  Install from the official selector: https://pytorch.org/get-started/locally/
- PyTorch brings in `numpy` as a dependency; you don’t need to `pip install numpy` separately unless you want to.

### Neural training (`finetune_neural.py`)
- **PyTorch**
- **tqdm** for progress bars

Install:

```bash
# Linux/macOS
python3 -m pip install torch tqdm

# Windows
python -m pip install torch tqdm
# (or python3 depending on your environment)
```

### Quick check

```bash
python3 --version
python3 -c "import sys; print('python ok')"

# Neural users:
python3 -c "import torch, platform; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 🚀 Quick Start

### Stream directly to Hashcat (no disk I/O)

```bash
# Random passwords (pw mode)
python3 pwforge.py --mode pw --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm

# Markov (requires training corpus)
python3 pwforge.py --mode markov   --markov-train rockyou.txt --markov-order 3   --count 2000000   | hashcat -a 0 -m 0 hashes.md5

# PCFG (requires training corpus)
python3 pwforge.py --mode pcfg   --pcfg-train rockyou.txt   --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

### Stream to John the Ripper (stdin)

```bash
python3 pwforge.py --mode pw --count 1000000   | john --stdin --format=nt hashes/*
```

> John’s `--fork` does **not** work with stdin. For multi-core JtR, generate a file:

```bash
python3 pwforge.py   --mode walk   --count 2000000   --out walks.txt --no-stdout

john --wordlist=walks.txt --format=nt --fork=16 hashes/*
```

---

## 💻 Neural Mode (`--mode neural`)

Neural generation uses a character-level LSTM checkpoint (`.pt`) and samples candidates, clamped to your `--min` / `--max` length.

```bash
# 1M neural candidates to hashcat
python3 pwforge.py   --mode neural   --model finetuned_model.pt   --batch-size 512   --max-gen-len 32   --min 8 --max 20   --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

Example file output:

```bash
python3 pwforge.py   --mode neural   --model ../neural/master_lstm.pt   --count 1000000   --min 8 --max 24   --chunk 10000   --out ../candidates.txt   --no-stdout --append
```

**Behavior:**

- PWForge auto-selects **CUDA** if available, else CPU:
  - Prints `[i] Neural mode → cuda` or `→ cpu` to `stderr`.
- If PyTorch is **not available**, `--mode neural` prints:
  - `[!] PyTorch not available – neural mode disabled`
  - and returns **no candidates**.
- If `--model` is missing or invalid:
  - Prints `[!] --model path.pt required. Train with finetune_neural.py`
  - Returns no candidates.

**Generation details:**

- Starts each sample from a BOS token and repeatedly:
  - Runs the LSTM forward,
  - Softmaxes the last timestep,
  - Samples characters via your RNG (`--seed` affects this).
- Stops when:
  - Reaches `--max-gen-len`, or
  - Samples EOS.
- If a sample is shorter than `--min`, it’s padded with random printable characters, then truncated to `--max`.
- Final candidates are filtered by length and optional entropy/dict filters, then written via the usual output flow.

**Neural hyperparameters (generation-side):**

- `--batch-size N` (default: 512)  
- `--max-gen-len N` (default: 32)
- `--embed-dim N` (default: 128)
- `--hidden-dim N` (default: 384)
- `--num-layers N` (default: 2)
- `--dropout FLOAT` (default: 0.3)

> These must match the hyperparameters used when training the checkpoint (see **Fine-Tuning a Neural Model**).

---

## 🧠 Other Advanced Modes

### Hybrid (`--mode hybrid`)

Hybrid applies simple **Hashcat-style rules** to dictionary words and can optionally append a mask-style pattern.

```bash
python3 pwforge.py   --mode hybrid   --dict words.txt   --rules rules.txt   --mask "?l?l?d"   --count 2000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

**Mask syntax in hybrid:**

Each character in `--mask` is treated literally **except** for:

- `?l` → random lowercase letter
- `?u` → random uppercase letter
- `?d` → random digit
- `?s` → random symbol from `--symbols-file` (or default set)
- `?y` → random year from `--years-file` (or default range 1990-2035)
- `?w` → random word from `--dict`

The generated mask part is randomly placed **before or after** the rule-modified word.

### Combo (`--mode combo`)

Weighted mixture of multiple generators in the *same* stream.

```bash
# 60% walk, 30% prince, 10% pcfg
python3 pwforge.py   --mode combo   --combo "walk:0.6,prince:0.3,pcfg:0.1"   --dict words_demo.txt   --pcfg-train rockyou.txt   --count 3000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

- `--combo "mode:weight,mode:weight,..."`  
  Weights are normalized internally.
- Valid mode names here are the same as primary modes (`pw`, `walk`, `prince`, `markov`, `pcfg`, `mask`, `passphrase`, `numeric`, `syllable`, `mobile-walk`, `hybrid`, `neural`), but:
  - Some modes need extra inputs (`--dict`, `--markov-train`, `--pcfg-train`, `--model`, etc.).
- For each candidate, PWForge:
  - Randomly picks a mode according to the normalized weights,
  - Generates a single candidate from that mode,
  - Repeats until `--count` is reached.

---

## 🔍 Entropy & Dictionary Filtering

```bash
# Skip low-entropy strings and drop anything in denylist.txt
python3 pwforge.py   --mode pw   --min-entropy 2.2   --no-dict denylist.txt   --count 2000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

- `--min-entropy <H>`  
  - Compute Shannon entropy over characters.  
  - Keep only candidates with entropy ≥ H.  
  - `0.0` (default) = disabled.
- `--no-dict path.txt`  
  - Load a list of forbidden strings (case-insensitive).  
  - Drop any candidate whose lowercase form is present.

Filtering is applied **after generation** but **before output** and meter updates.

---

## 💾 Output (Plain / Gzip / Split)

### Basic file vs. stdout

- If **no** `--out`:
  - All candidates are printed to **stdout**.
- If `--out PATH`:
  - Candidates are written to `PATH` (or split variants),
  - And *also* echoed to stdout **unless** `--no-stdout` is set.
- `--append` makes file output use append mode if the file already exists.

```bash
# Plain file
python3 pwforge.py   --mode pw   --count 5000000   --out pw.txt --no-stdout

# Gzip (recommended for large runs)
python3 pwforge.py   --mode walk   --count 5000000   --out walks.txt.gz   --gz --no-stdout
```

> Note: gzip is auto-detected from the filename extension `.gz`/`.gzip`.  
> `--gz` is mainly used together with `--split` (see below).

### Split into N parts (`--split`)

`--split N` divides **each batch of generated lines** into N shards and writes them to separate files.

```bash
# Split into 8 parts
python3 pwforge.py   --mode pw   --count 80000000   --split 8   --out pw.txt.gz   --gz --no-stdout
# => pw00000000.txt.gz ... pw00000007.txt.gz
```

- Shard file naming:
  - Uses zero-padded numeric suffixes based on `N`:
    - `pw00000000.txt.gz`, `pw00000001.txt.gz`, …
- `--append` works with split:
  - Running again with `--append` will **append** to the existing shard files.

> Splitting is **per call**: each run of PWForge partitions its internal batch into `N` shards. If you want real parallelism, you can run PWForge multiple times in parallel (see “Parallel without GNU parallel” below).

### WSL/Linux I/O tip

Avoid `/mnt/c/...` for heavy I/O under WSL; use `/home` or `/dev/shm`:

```bash
python3 pwforge.py   --mode pw   --count 10000000   --out /dev/shm/pw.txt   --no-stdout
```

---

## 🧭 Modes & Behavior Summary

These are the **implemented** modes in `pwforge.py`:

- `pw` – Random passwords
  - `--charset`, `--exclude-ambiguous`, `--require`
- `walk` – Keyboard adjacency walks
  - `--keymap`, `--keymap-file`, `--walk-allow-shift`, `--starts-file`, `--window`, `--relax-backtrack`, `--upper`, `--suffix-digits`
- `both` – Combine `pw` + `walk`
  - The requested `--count` is split:
    - `ceil(count/2)` for `pw`, `floor(count/2)` for `walk`.
  - There is **no `--both-policy`** flag; split behavior is fixed.
- `mask` – Word + year + symbol pattern
  - `--dict`, `--years-file`, `--symbols-file`, `--upper-first`
- `passphrase` – Multi-word passphrases
  - `--dict`, `--words`, `--sep`, `--upper-first`
- `numeric` – Digits only
- `syllable` – Pronounceable pseudo-words
  - `--template`, `--upper-first`
- `prince` – Word combination enumeration
  - `--dict`, `--prince-min/max`, `--sep`, `--bias-terms`, `--bias-factor`, `--prince-suffix-digits`, `--prince-symbol`
- `markov` – Character Markov model
  - `--markov-train`, `--markov-order`
- `pcfg` – Probabilistic Context-Free Grammar
  - `--pcfg-train`, optional `--pcfg-model` (advanced, see below)
- `mobile-walk` – Phone keypad walks
- `hybrid` – Rules + optional mask over dictionary
  - `--dict`, `--rules`, `--mask`, `--years-file`, `--symbols-file`
- `combo` – Weighted mixture of multiple modes
  - `--combo "mode:weight,mode:weight,..."` plus relevant per-mode options
- `neural` – Character LSTM checkpoint
  - `--model`, `--batch-size`, `--max-gen-len`, `--embed-dim`, `--hidden-dim`, `--num-layers`, `--dropout`

---

## 🔌 Piping & Files: Practical Recipes

### Stream to Hashcat

```bash
# Pure random passwords
python3 pwforge.py --mode pw --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm

# Keyboard walks, with custom starts
python3 pwforge.py   --mode walk   --count 2000000   --starts-file starts.txt   | hashcat -a 0 -m 1000 hashes.ntlm

# Markov over rockyou
python3 pwforge.py   --mode markov   --markov-train rockyou.txt --markov-order 3   --count 2000000   | hashcat -a 0 -m 0 hashes.md5

# PCFG over rockyou
python3 pwforge.py   --mode pcfg   --pcfg-train rockyou.txt   --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm

# Neural
python3 pwforge.py   --mode neural   --model finetuned_model.pt   --count 1000000   | hashcat -a 0 -m 1000 hashes.ntlm
```

### Stream to John

```bash
python3 pwforge.py   --mode prince   --dict words_demo.txt   --count 1000000   | john --stdin --format=nt hashes/*
```

### File then crack

```bash
python3 pwforge.py   --mode prince   --dict words_demo.txt   --count 1000000   --out prince.txt --no-stdout

hashcat -a 0 -m 1000 hashes.ntlm prince.txt
```

### Parallel without GNU parallel

Use `xargs -P` to run multiple processes in parallel, each with its own output file:

```bash
printf "%s
" pw walk markov pcfg   | xargs -P4 -I{} sh -c '
      python3 pwforge.py         --mode "{}"         --count 200000         --out "{}".txt         --no-stdout
    '
```

---

## 🧪 Deterministic & Chunked Runs

### Determinism (`--seed`)

- `--seed N` drives the internal RNG used for:
  - Password lengths, character choices, walks, Markov/PCFG sampling, neural sampling (after seeding PyTorch).
- Same Python version + same seed + same options → same output sequence.

```bash
python3 pwforge.py   --mode pw   --seed 1337   --count 1000000   --out stable.txt --no-stdout --append
```

### Chunking (`--chunk`)

- Generation happens in batches of size `--chunk` (default 100k).
- Larger chunks:
  - Fewer I/O calls, potentially faster,
  - But more memory usage.

```bash
python3 pwforge.py   --mode pw   --chunk 500000   --count 5000000   --out pw.txt --no-stdout --append
```

---

## 📊 Metering, Dry-Run & Estimation

### Progress meter (`--meter`)

- `--meter N` (default `100000`):
  - Every N **produced** lines, PWForge prints to `stderr`:
    - `[meter] 1,000,000 lines, 5,000,000 lps`
- `--meter 0` disables the meter.

### Dry-run (`--dry-run N`)

- Overrides `--count` with `N` **without changing anything else**.
- Useful for quick tests:

```bash
# Equivalent to --count 10000
python3 pwforge.py --mode pw --count 100000000 --dry-run 10000
```

### Estimate-only (`--estimate-only`)

- For `markov` or `pcfg`, PWForge may need to train/load a model.
- With `--estimate-only`, it:
  - Prepares the models if needed,
  - Prints a simple JSON object and exits, without generating candidates:

```bash
python3 pwforge.py   --mode markov   --markov-train rockyou.txt   --min 8 --max 16   --count 1000000   --estimate-only
```

Outputs something like:

```json
{"mode": "markov", "count": 1000000, "avg_len": 12.0}
```

This is a basic helper – it doesn’t compute a full keyspace size, just the configured `avg_len` and `count`.

---

## 🛠️ How PWForge Works (Architecture & Flow)

### 1) Argument parsing

`pwforge.py`:

- Builds an `ArgumentParser` with all options listed in **Full CLI** below.
- Determines the **effective count**:
  - If `--dry-run N` is set, uses `N`,
  - Else uses `--count`.
- Builds a **target map** for modes:

```text
if mode == "both":
    pw    gets ceil(count/2)
    walk  gets floor(count/2)
else:
    the selected mode gets count
```

### 2) Pre-loading resources

Before generation, PWForge:

- Builds the **charset** for `pw`:
  - From `--charset`, and/or default printable classes,
  - Applies `--exclude-ambiguous` (e.g., drop 0/O, 1/l, etc.) if enabled.
- Loads:
  - `--starts-file` (for `walk`),
  - `--dict` (for many modes),
  - `--years-file` (or defaults 1990-2035),
  - `--symbols-file` (or defaults `!@#$%^&*?_+-=`),
  - `--bias-terms` for `prince`,
  - `--rules` for `hybrid`,
  - `--no-dict` into a set for filtering.
- Builds keyboard adjacency graphs:
  - `--keymap` (`qwerty`, `qwertz`, `azerty`) or `--keymap-file` JSON map,
  - Adds symbol graph if `--walk-allow-shift` is used.
- Trains or loads:
  - **Markov** model (if `--markov-train` and mode is `markov` or `estimate-only`),
  - **PCFG** model:
    - From `--pcfg-model` (JSON) if given, else
    - Trains from `--pcfg-train` when mode is `pcfg` or `estimate-only`.

### 3) Chunked generation loop

For each mode with a non-zero target:

1. Compute remaining `count` for that mode.
2. While remaining > 0:
   - `chunk = min(CHUNK, remaining)` where `CHUNK = max(1, --chunk or 100000)`.
   - Call the appropriate generator (`gen_pw`, `gen_walk`, `gen_markov`, `gen_neural`, etc.).
   - Optionally apply:
     - `--min-entropy`,
     - `--no-dict`.
   - Call `write_output(args, lines)` which:
     - Either writes whole `lines` to one file/stdout, or
     - Splits into `--split` shards and writes to multiple files.
   - Update counters and `meter_printer`.

### 4) Neural generation internals

When `--mode neural`:

- Imports PyTorch safely; if not available, prints a warning and disables neural.
- Loads `CharLSTM` with your hyperparameters.
- Loads `state_dict` from `--model`.
- Seeds PyTorch’s RNG based on PWForge’s RNG.
- Generates passwords in batches (`--batch-size`) up to `--max-gen-len`, then clamps to `--min`/`--max`, and feeds them into the standard output/filters.

### 5) Markov & PCFG

- **Markov**:
  - Builds an n-gram character model (`--markov-order`) from `--markov-train`.
  - Samples characters according to conditional probabilities of the previous `order` chars.
- **PCFG**:
  - Tokenizes training lines into patterns combining:
    - Character classes (digits, lower, upper, symbols),
    - Dictionary words & lengths.
  - Learns pattern frequencies, then samples:
    - A pattern (structure),
    - Concrete tokens (words/digits/symbols) according to the model.

Both feed candidates into the same entropy/dict filters and output pipeline.

---

## 🧪 Fine-Tuning a Neural Model (Training)

Use `finetune_neural.py` to create `model.pt` from any `.txt` or `.gz` corpus.

### Basic training

```bash
# Train on rockyou.txt (or any leak/corpus you are authorized to use)
python3 finetune_neural.py   rockyou.txt   --output finetuned_model.pt   --epochs 10   --batch-size 512   --max-len 32
```

### Training with an existing checkpoint

```bash
python3 finetune_neural.py   rockyou.txt   --pretrained base_model.pt   --output finetuned_model.pt   --epochs 5   --batch-size 512   --max-len 32   --lr 0.0005
```

### Quick smoke test

```bash
python3 finetune_neural.py   rockyou.txt   --output finetuned_model.pt   --epochs 2   --test
```

**CLI options for `finetune_neural.py`:**

- Positional:
  - `input` – Path to corpus (`.txt` or `.gz`).
- General:
  - `--output PATH` – Output checkpoint file (default: `finetuned_model.pt`).
  - `--pretrained PATH` – Start from an existing `.pt` model.
- Optimization:
  - `--epochs INT` (default: 10)
  - `--batch-size INT` (default: 512)
  - `--lr FLOAT` (default: 0.001)
- Architecture (must match generation time):
  - `--hidden-dim INT` (default: 384)
  - `--embed-dim INT` (default: 128)
  - `--num-layers INT` (default: 2)
  - `--dropout FLOAT` (default: 0.3)
  - `--max-len INT` (default: 32; training max sequence length)
- Testing:
  - `--test` – After training, sample a few passwords and print them.

Internally:

- Splits data ~95/5 into train/validation.
- Builds a character vocabulary (printable ASCII range).
- Trains a character-level LSTM with cross-entropy.
- Saves the **best** checkpoint by validation loss to `--output`.

---

## 📚 Full CLI (by category)

Below is the CLI exactly as implemented in `pwforge.py`.

### General

```text
--mode {pw,walk,both,mask,passphrase,numeric,syllable,prince,
        markov,pcfg,mobile-walk,hybrid,combo,neural}
--count INT               # total candidates to generate (default: 100000)
--min INT                 # minimum length (default: 8)
--max INT                 # maximum length (default: 16)
--seed INT                # RNG seed for deterministic runs
```

### Core password generation (pw mode)

```text
--charset STR             # custom charset; if omitted, uses defaults
--exclude-ambiguous       # drop ambiguous chars (e.g., 0/O, 1/l, etc.)
--require "classes"       # e.g. "upper,lower,digit,symbol"
                          # enforce required character classes
```

### Walk & Mobile-walk

```text
--starts-file PATH        # file with starting keys (one per line)
--window INT              # backtrack window (default: 2)
--relax-backtrack         # allow revisiting when no non-recent neighbors
--upper                   # capitalize first character (walk only)
--suffix-digits INT       # digits to append at end (walk only)
--keymap {qwerty,qwertz,azerty}   # keyboard layout (walk only; default: qwerty)
--keymap-file PATH        # JSON custom adjacency map (walk only)
--walk-allow-shift        # include shifted symbol layer in graph (walk only)
```

- `mobile-walk` uses a fixed phone keypad graph; it respects `--window` and `--relax-backtrack`, but not keyboard-specific flags.

### Dictionary / pattern-based modes

Used across `mask`, `passphrase`, `prince`, `hybrid`, some PCFG helpers:

```text
--dict PATH               # base wordlist
--years-file PATH         # list of years; defaults 1990-2035 if missing
--symbols-file PATH       # list of symbols; default "!@#$%^&*?_+-="
--bias-terms PATH         # extra bias terms for prince
--bias-factor FLOAT       # strength of bias (0..5 scaled; default: 2.0)

--words INT               # passphrase word count (default: 3)
--sep STR                 # separator between words/tokens
--upper-first             # capitalize first word/token (mask/passphrase/syllable)

--template STR            # syllable template (default: "CVC")
                          # C = consonant, V = vowel, other chars = literal

--prince-min INT          # min tokens per prince candidate (default: 2)
--prince-max INT          # max tokens per prince candidate (default: 3)
--prince-suffix-digits INT# digits to append (default: 0)
--prince-symbol STR       # symbol to append (default: "")
```

### Hybrid / Combo

```text
--rules PATH              # Hashcat-style rules for hybrid
--mask STR                # hybrid mask (uses ?l ?u ?d ?s ?y ?w tokens)
--combo "mode:weight,..." # combo mode specification
```

### Neural

```text
--model PATH              # required checkpoint path (.pt)
--batch-size INT          # batch size for sampling (default: 512)
--max-gen-len INT         # max sequence length during sampling (default: 32)
--embed-dim INT           # embedding size (default: 128)
--hidden-dim INT          # LSTM hidden size (default: 384)
--num-layers INT          # LSTM layers (default: 2)
--dropout FLOAT           # dropout probability (default: 0.3)
```

### Filtering

```text
--min-entropy FLOAT       # minimum Shannon entropy (default: 0.0 = off)
--no-dict PATH            # drop candidates found in this (case-insensitive) file
```

### Output & Performance

```text
--out PATH                # output file path
--append                  # append to existing output file(s)
--split INT               # number of shards per batch (default: 1)
--gz                      # (mainly used with split; gzip auto-detected by .gz)
--no-stdout               # suppress printing candidates to stdout

--dry-run INT             # override count for this run only
--meter INT               # progress meter interval (default: 100000; 0 = off)
--estimate-only           # print JSON estimate and exit (no candidates)
--chunk INT               # batch size per mode (default: 100000)
```

### Models

```text
--markov-train PATH       # training corpus for Markov mode
--markov-order INT        # Markov order (default: 3; 1..5 recommended)

--pcfg-train PATH         # training corpus for PCFG mode
--pcfg-model PATH         # advanced: precomputed PCFG model JSON
                          # (hidden from --help via argparse.SUPPRESS)
```

---

## 🧠 Tips

- On WSL, use Linux paths (`/home`, `/dev/shm`) instead of `/mnt/c` for heavy output.
- Use `--seed` for reproducible experiments; combine with `--chunk` and `--split` for predictable file layouts.
- If you need multi-core performance, run multiple independent PWForge processes with different `--out` or `--split` settings and use `xargs -P` or GNU parallel.
- For John, prefer file-based workflows + `--fork` for scalability; stdin is single process.

---

## 📄 License

```text
MIT License © 2025 Adam Willard
```

**For authorized penetration testing and research use only.**
