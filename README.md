
# PWForge – Password Candidate Generator

High‑performance, multi‑mode wordlist generator designed for **Hashcat** and **John the Ripper**.  
Streams to STDOUT for live cracking or writes compressed/split files for large runs.

> This README reflects all options implemented in `pwforge.py`, including **neural** generation, **hybrid/combo** modes, entropy filtering, chunking, and multi‑process sharding.

---

## 🔧 Requirements

- **Python 3.8+** (tested on Linux, WSL, macOS)
- Optional for `--mode neural`: **PyTorch** (CPU or CUDA build). Install from https://pytorch.org/get-started/locally/
- Optional for large gzip merges: `gzip`, `zcat` available on your OS
- `pip install torch tqdm`
- `pip install numpy`

On windows you may need to use
- `python3 -m pip install torch`
- `python3 -m pip install tqdm`
- `python3 -m pip install numpy`

### Quick check
```bash
python3 --version
python3 -c "import sys; print('ok')"
# Neural users:
python3 -c "import torch, platform; print('torch', torch.__version__,'cuda', torch.cuda.is_available())"
```

---

## 🚀 Quick Start

### Stream directly to Hashcat (no disk I/O)
```bash
# Random passwords
python3 pwforge.py --mode pw --count 1_000_000 | hashcat -a 0 -m 1000 hashes.ntlm

# Markov (requires training corpus)
python3 pwforge.py --mode markov --markov-train rockyou.txt --count 2_000_000 | hashcat -a 0 -m 0 hashes.md5

# PCFG (requires training corpus)
python3 pwforge.py --mode pcfg --pcfg-train rockyou.txt --count 1_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

### Stream to John the Ripper (stdin)
```bash
python3 pwforge.py --mode pw --count 1_000_000 | john --stdin --format=nt hashes/*
```
> John’s `--fork` does **not** work with stdin. For multi‑core JtR use a file:
```bash
python3 pwforge.py --mode walk --count 2_000_000 --out walks.txt --no-stdout
john --wordlist=walks.txt --format=nt --fork=16 hashes/*
```

---

## 💻 Neural Mode (`--mode neural`)

Neural generation uses a character‑level LSTM checkpoint (`.pt`) and samples candidates up to your length bounds.

```bash
# 2M neural candidates to hashcat
python3 pwforge.py --mode neural --model finetuned_model.pt   --batch-size 512 --max-gen-len 32   --min 8 --max 20 --count 2_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

- PWForge auto‑selects **CUDA** if available, else CPU.
- The checkpoint is trained with `finetune_neural.py` (see the **How PWForge Works** section below). Hyperparameters must match your checkpoint.

**File output with true multi‑process sharding**
```bash
python3 pwforge.py --mode neural --model finetuned_model.pt   --count 20_000_000 --mp-workers 8   --out neural.txt.gz --gz --no-stdout
# => neural_w00.txt.gz ... neural_w07.txt.gz
```

---

## 🧠 Other Advanced Modes

### Hybrid (`--mode hybrid`)
Apply a subset of **Hashcat‑style rules** to a dictionary, optionally combining with a light mask.
```bash
python3 pwforge.py --mode hybrid   --dict words_demo.txt --rules rules.txt   --mask "?d?d"   --count 2_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

### Combo (`--mode combo`)
Weighted mixture of multiple generators in one stream.
```bash
# 60% walk, 30% prince, 10% pcfg
python3 pwforge.py --mode combo   --combo "walk:0.6,prince:0.3,pcfg:0.1"   --dict words_demo.txt --pcfg-train rockyou.txt   --count 3_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

---

## 🔍 Entropy & Dictionary Filtering

```bash
# Skip low‑entropy strings and drop anything in denylist.txt
python3 pwforge.py --mode pw --min-entropy 2.2 --no-dict denylist.txt   --count 2_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

- `--min-entropy <H>`: keep only candidates with Shannon entropy ≥ H  
- `--no-dict path.txt`: drop any candidate (case‑insensitive) present in the file

---

## 💾 Output (Plain / Gzip / Split)

```bash
# Plain file
python3 pwforge.py --mode pw --count 5_000_000 --out pw.txt --no-stdout

# Gzip (recommended for huge runs)
python3 pwforge.py --mode walk --count 5_000_000 --out walks.txt.gz --gz --no-stdout

# Split into 8 parts (mutually exclusive with --mp-workers)
python3 pwforge.py --mode pw --count 80_000_000 --split 8 --out pw.txt --gz --no-stdout
# => pw_00000000.txt.gz ... pw_00000007.txt.gz
```

**WSL/Linux tip:** avoid `/mnt/c/...` for heavy I/O; use `/home` or `/dev/shm`:
```bash
python3 pwforge.py --mode pw --count 10_000_000 --out /dev/shm/pw.txt --no-stdout
```

---

## ⚡ True Multi‑Process Sharding (`--mp-workers`)

`--mp-workers N` launches **N child processes**. Each writes a shard (e.g., `file_w00.txt[.gz]`).  
**Requires** `--out` **and** `--no-stdout`.

```bash
python3 pwforge.py --mode markov   --markov-train rockyou.txt   --count 80_000_000 --mp-workers 8   --out markov.txt.gz --gz --no-stdout --seed 42
# => markov_w00.txt.gz ... markov_w07.txt.gz
```

> `--mp-workers` and `--split` are **mutually exclusive**. Use one strategy.

---

## 🧭 Modes & Examples

- `pw` – random passwords (`--charset`, `--exclude-ambiguous`, `--require`)
- `walk` – keyboard adjacency (QWERTY/QWERTZ/AZERTY or `--keymap-file`, `--starts-file`, `--suffix-digits`)
- `both` – combine `pw` + `walk` (`--both-policy split|each`)
- `mask` – word+year+symbol (`--dict`, `--years-file`, `--symbols-file`, `--upper-first`)
- `passphrase` – multi‑word phrases (`--dict`, `--words`, `--sep`, `--upper-first`)
- `numeric` – digits only
- `syllable` – pronounceable using templates (`--template`, `--upper-first`)
- `prince` – word combination enumeration (`--prince-min/max`, `--sep`, `--bias-terms`, `--prince-suffix-digits`, `--prince-symbol`)
- `markov` – trained corpus (`--markov-train`, `--markov-order 1..5`)
- `pcfg` – trained corpus (`--pcfg-train`)
- `mobile-walk` – phone keypad walks
- `hybrid` – rules + mask over a dictionary (`--rules`, `--mask`)
- `combo` – weighted mixture (`--combo "mode:w,mode:w,..."`)
- `neural` – LSTM checkpoint (`--model`, `--batch-size`, `--max-gen-len`, `--embed-dim`, `--hidden-dim`, `--num-layers`, `--dropout`)

---

## 🔌 Piping & Files: Practical Recipes

### Stream to Hashcat
```bash
python3 pwforge.py --mode pw --count 1_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
python3 pwforge.py --mode walk --count 2_000_000 --starts-file starts.json | hashcat -a 0 -m 1000 hashes.ntlm
python3 pwforge.py --mode markov --markov-train rockyou.txt --count 2_000_000 | hashcat -a 0 -m 0 hashes.md5
python3 pwforge.py --mode pcfg --pcfg-train rockyou.txt --count 1_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
python3 pwforge.py --mode neural --model finetuned_model.pt --count 1_000_000 | hashcat -a 0 -m 1000 hashes.ntlm
```

### Stream to John
```bash
python3 pwforge.py --mode prince --dict words_demo.txt --count 1_000_000 | john --stdin --format=nt hashes/*
```

### File then crack
```bash
python3 pwforge.py --mode prince --dict words_demo.txt --count 1_000_000 --out prince.txt --no-stdout
hashcat -a 0 -m 1000 hashes.ntlm prince.txt
```

### Parallel without GNU parallel
```bash
printf "%s\n" pw walk markov pcfg | xargs -P4 -I{} sh -c 'python3 pwforge.py --mode "{}" --count 200000 --out "{}".txt --no-stdout'
```

---

## 🧪 Deterministic & Chunked Runs

- `--seed N` gives identical output on any machine.
- `--chunk N` controls internal batch size (default 100k). Increase for faster throughput on high‑RAM systems.

```bash
python3 pwforge.py --mode pw --seed 1337 --count 1_000_000 --out stable.txt --no-stdout
python3 pwforge.py --mode pw --chunk 500_000 --count 5_000_000 --out pw.txt --no-stdout
```

---

## 🛠️ How PWForge Works (Architecture & Flow)

### 1) Argument parsing
`pwforge.py` parses the CLI and builds a **target map** per mode. If `--mode both` and `--both-policy split`, the count is split between `pw` and `walk`. In `each`, both get the full count.

### 2) Chunked generation loop
Generation is performed in **batches** (`--chunk`, default 100k) per mode:
- Each generator (`pw`, `walk`, `markov`, `pcfg`, `neural`, etc.) is called with `count=chunk`.
- Candidates can be filtered (`--min-entropy`, `--no-dict`).
- Output is written via `--out` (optionally `--gz`/`--split`) and/or streamed to STDOUT (if `--no-stdout` not set).
- The loop repeats until the mode’s target count is met.

### 3) True multi‑process sharding
If `--mp-workers N` is set **with** `--out` and `--no-stdout`:
- The parent process splits `--count` across workers.
- It spawns **N child processes** with the same options plus `--mp-child`, each writing a **shard** (`*_wNN.txt[.gz]`).
- The parent waits for all children and exits.

> This design avoids file/pipe contention and scales across CPU cores.

### 4) Neural generation internals
- On `--mode neural`, PWForge lazily imports **torch**, loads your `--model` checkpoint into an internal character LSTM, and **samples** up to `--max-gen-len` characters per password.
- Samples are then **length‑clamped** to `--min/--max` and passed through the same output/filters as other modes.

### 5) Markov & PCFG
- **Markov** builds a character n‑gram model (`--markov-order`) from `--markov-train`, then samples.
- **PCFG** tokenizes the training corpus into patterns (words/digits/symbols), learns pattern frequencies, and samples patterns + lexicons to emit candidates.

---

## 🧪 Fine‑Tuning a Neural Model (training)

Use the provided `finetune_neural.py` script to create `model.pt` from any `.txt` or `.gz` corpus.

```bash
# Train on rockyou.txt (or any leak/corpus you own and may use)
python3 finetune_neural.py rockyou.txt --output finetuned_model.pt --epochs 10 --batch-size 512 --max-len 32

# Verify a few samples after training
python3 finetune_neural.py rockyou.txt --output finetuned_model.pt --epochs 2 --test
```

Model hyperparameters (must match at generation time when you pass `--embed-dim`, `--hidden-dim`, etc.):
- `--embed-dim 128`
- `--hidden-dim 384`
- `--num-layers 2`
- `--dropout 0.3`
- `--max-len 32` (training max length)

The training script:
- Splits data 95/5 train/val, creates a small **character vocabulary** (ASCII 33‑126), and trains an **LSTM** with cross‑entropy.
- Saves the **best** checkpoint by validation loss to `--output`.
- Optional `--test` step generates 10 samples using the trained model.

---

## 📚 Full CLI (by category)

### General
```
--mode {pw,walk,both,mask,passphrase,numeric,syllable,prince,markov,pcfg,mobile-walk,hybrid,combo,neural}
--count N
--min N
--max N
--seed N
--charset STR              (pw)
--exclude-ambiguous
--require "upper,lower,digit,symbol"
```

### Walks
```
--starts-file PATH
--window N                 (default 2)
--relax-backtrack
--upper
--suffix-digits N
--keymap {qwerty,qwertz,azerty}
--keymap-file PATH
--walk-allow-shift
--both-policy {split,each}
```

### Mask / Passphrase / Prince
```
--dict PATH
--years-file PATH
--symbols-file PATH
--bias-terms PATH
--bias-factor FLOAT        (default 2.0)
--words N                  (passphrase)
--sep STR
--upper-first
--template STR             (syllable)
--prince-min N             (default 2)
--prince-max N             (default 3)
--prince-suffix-digits N
--prince-symbol STR
```

### Hybrid / Combo
```
--rules PATH               (hybrid)
--mask STR                 (hybrid)
--combo "mode:w,mode:w"    (combo)
```

### Neural
```
--model PATH               (required)
--batch-size N             (default 512)
--max-gen-len N            (default 32)
--embed-dim N              (default 128)
--hidden-dim N             (default 384)
--num-layers N             (default 2)
--dropout FLOAT            (default 0.3)
```

### Filtering
```
--min-entropy FLOAT        (default 0.0)
--no-dict PATH             (drop candidates present in file)
```

### Output & Performance
```
--out PATH
--append
--split N
--gz
--no-stdout
--dry-run N
--meter N                  (default 100000; 0 disables)
--estimate-only
--chunk N                  (default 100000)
--mp-workers N             (requires --out and --no-stdout)
```

### Models
```
--markov-train PATH
--markov-order N           (1..5; default 3)
--pcfg-train PATH
```

---

## 🧠 Tips

- Prefer `/home` or `/dev/shm` over `/mnt/c` on WSL for high throughput.
- Use `--mp-workers` for real speed‑up with file output (each shard writes independently).
- For John multi‑core, write a file and use `--fork` instead of piping.

---

## 📄 License

```
MIT License © 2025 Adam Willard
```
**For authorized penetration testing and research use only.**
