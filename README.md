

# 🔥 PWForge
### Advanced Multi-Mode Password Generator for Hashcat & John the Ripper

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-stable-success.svg)]()
[![PRs](https://img.shields.io/badge/PRs-welcome-orange.svg)]()

PWForge is a **powerful, modular password and wordlist generator** built for professional penetration testers,
red-team operators, and password-cracking researchers.  
It unifies multiple generation techniques — **Markov, PCFG, Mask, Keyboard Walks, Passphrases, Numeric, Syllable, PRINCE, and Mobile Walks** — into a single, scriptable CLI tool.

---

## ✨ Key Features

- **Multi-mode generation:** `pw`, `walk`, `mask`, `passphrase`, `numeric`, `syllable`, `prince`, `markov`, `pcfg`, and `mobile-walk`
- **Smart sampling:** Markov and PCFG models trained from breach data or internal corpora
- **Keyboard & mobile walks:** QWERTY, QWERTZ, AZERTY, and keypad path simulation
- **Rules & biasing:** Built-in bias terms, Bloom deduplication, leet profiles, and Hashcat rule hand-off
- **Performance:** Chunked workers, gzip output, resumable generation, deterministic seeds
- **Compatibility:** Output streams directly to Hashcat or John the Ripper
- **Output options:** `--out`, `--split`, `--gz`, `--no-stdout`, `--estimate-only`, and more

---

## 🚀 Quick Start

### Clone the Repository
```bash
git clone https://github.com/awillard1/pwforge.git
cd pwforge
```

### Run Examples
```bash
# Random password list
python3 pwforge.py --mode pw --min 8 --max 16 --count 500000 --out pwlist.txt.gz --gz

# Keyboard walks
python3 pwforge.py --mode walk --min 6 --max 10 --count 300000 --starts-file starts.json --upper --suffix-digits 2 --out walks.txt.gz --gz

# Markov mode (train from corpus)
python3 pwforge.py --mode markov --markov-train words_demo.txt --markov-order 3 --count 100000 --out markov.txt

# PCFG mode
python3 pwforge.py --mode pcfg --pcfg-train words_demo.txt --count 100000 --out pcfg.txt

# Estimate keyspace only
python3 pwforge.py --mode pw --min 10 --max 16 --estimate-only
```

---

## 🧩 Modes Overview
| Mode | Description |
|------|--------------|
| **pw** | Random password generator with customizable charset |
| **walk** | QWERTY/QWERTZ/AZERTY keyboard walks |
| **mask** | Template-driven “Word+Year+Symbol” patterns |
| **passphrase** | Multi-word Diceware-style passphrases |
| **numeric** | Dates, PINs, and numeric patterns |
| **syllable** | Pronounceable random syllables |
| **prince** | Multi-word chained combinations (PRINCE-like) |
| **markov** | Character-level Markov model learned from a corpus |
| **pcfg** | Probabilistic Context-Free Grammar learned from corpora |
| **mobile-walk** | Smartphone keypad walk sequences |

---

## 🛠 Installation

Python 3.8+ required.

```bash
python3 pwforge.py --help
```

---

## 🧰 Integrations

**Hashcat:**
```bash
python3 pwforge.py --mode both --count 1000000 | hashcat -a 0 -m 1000 hashes.txt
```

**John the Ripper:**
```bash
python3 pwforge.py --mode pw --count 500000 | john --stdin --format=nt hashes/*
```

---

## ⚙️ Development & Contribution

Contributions are welcome!  
If you add new generation techniques, please:
1. Add tests in `/tests`.
2. Document your new mode in `README.md`.
3. Update the argument parser and mode table.

---

## 🧾 License

MIT License © 2025  
Created by Adam Willard & Contributors  
For educational and professional security research only.

---

### 🔑 Keywords
`password generator` · `hashcat` · `john the ripper` · `markov` · `pcfg` · `mask` · `passphrase` · `keyboard walk` · `pentest` · `wordlist` · `security research`


# Unified Password Candidate Generator

This toolkit emits password candidates directly to **stdout** (for piping into Hashcat/JtR) or to **files**. It merges multiple generators:

- **pw**: random passwords (with optional class requirements)
- **walk**: keyboard walks (QWERTY/QWERTZ/AZERTY; optional symbol row)
- **both**: `pw` + `walk` (split or each)
- **mask**: template patterns (e.g., `WordYYYY!`, `Word99!`)
- **passphrase**: multi-word passphrases
- **numeric**: dates & PIN shapes
- **syllable**: pronounceable C/V templates

Also includes: uniqueness controls (set/Bloom), dedupe against existing lists, deterministic runs (`--seed`), resume, file output/split/gzip, stats meter, and a **JtR helper**.

---

## Files

- `pwforge.py` — main generator script
- `starts.json` — weighted keyboard-walk starts
- `words_demo.txt` — small demo wordlist (mask/passphrase)
- `eff_demo.txt` — small demo passphrase wordlist
- `years_demo.txt` — demo years
- `symbols_demo.txt` — demo symbols

---

## Quick Start

```bash
# Random passwords to Hashcat
python3 pwforge.py --mode pw --min 10 --max 16 --count 10000   --require upper,lower,digit,symbol | hashcat -a 0 -m 1000 hashes.txt

# Keyboard walks (QWERTY) with tweaks
python3 pwforge.py --mode walk --min 6 --max 10 --count 50000   --starts-file starts.json --upper --suffix-digits 2 | hashcat -a 0 -m 1000 hashes.txt

# Both (split 50/50) and interleaved
python3 pwforge.py --mode both --count 100000 --interleave   --starts-file starts.json --upper --suffix-digits 2

# Save to gzipped list and split into 8 files
python3 pwforge.py --mode mask --count 200000 --mask-set corp   --dict words_demo.txt --years-file years_demo.txt --symbols-file symbols_demo.txt   --out corp_list.txt --gz --split 8
```

---

## CLI Reference

### Global
- `--mode pw|walk|both|mask|passphrase|numeric|syllable`
- `--count N` — number of lines (**for `both` see policy**)
- `--seed N` — deterministic output (reproducible)
- `--resume session.json` — save/load RNG state, produced count
- `--dry-run N` — preview N candidates per active mode and exit
- `--meter N` — stats to stderr every N lines (default 100k; 0=off)

### Uniqueness / Dedupe
- `--unique` — per-type uniqueness (pw vs walk)
- `--unique-global` — across all outputs
- `--bloom MB` — Bloom filter to reduce memory usage
- `--dedupe-file path` — do not emit if present in file (supports .gz)

### Output
- `--out path` — write to a file (or use stdout only)
- `--split N` — when writing out, split into N files (suffix _00..)
- `--gz` — gzip output file(s)
- `--no-stdout` — suppress stdout (useful with `--out`)

### Mode: `pw` (random)
- `--min, --max` — length range
- `--charset "..."` — base charset
- `--exclude-ambiguous` — drop `O0l1I|` etc.
- `--require upper,lower,digit,symbol` — enforce classes

### Mode: `walk` (keyboard walks)
- `--min, --max` — length range
- `--starts-file starts.json` — weighted starts
- `--window N` — loop-avoidance window (default 3)
- `--relax-backtrack` — allow backtracking if stuck
- `--upper` — random capitalization (~30% per letter)
- `--leet-profile off|minimal|common|aggressive` — set leet rate
- `--suffix-digits N` — append up to N digits
- `--keymap qwerty|qwertz|azerty` — layout
- `--walk-allow-shift` — include symbol row adjacency

### Mode: `both` (pw + walk)
- `--both-policy split|each`
  - `split` (default): split `--count` 50/50 (odd goes to pw)
  - `each` : generate `--count` of **each** (≈ 2× lines)
- `--interleave` — alternate pw/walk outputs

### Mode: `mask` (templates)
- `--mask-set common|corp|ni`
- `--dict words.txt` — base words (defaults to demo if missing)
- `--years-file years.txt` — defaults to demo years
- `--symbols-file symbols.txt` — defaults to demo symbols
- `--emit-base` — emit only the base (preview)

### Mode: `passphrase`
- `--dict eff.txt` — wordlist (defaults to `eff_demo.txt` internally)
- `--words N` — number of words (default 3, min 2)
- `--sep "-"` — separator
- `--upper-first` — capitalize first word
- `--emit-base` — just the phrase (no extra mutations)

### Mode: `numeric`
Generates common patterns: `YYYY`, `MMDD`, `DDMM`, `YYMMDD`, `####`, `######`.

### Mode: `syllable`
- `--template CVCVCV` — C/V template
- `--upper` — capitalize first letter
- `--suffix-digits N` — append up to N digits

### JtR Helper
If you need John the Ripper with `--fork`, stdin cannot be shared. Use:
```
python3 pwforge.py --mode both --count 200000 --out list.txt
john --wordlist=list.txt --format=nt --fork=8 hashes/*
```
Or use the built-in helper to **print** a suggested command (does not execute John):
```
python3 pwforge.py --mode both --count 200000 --out list.txt --jtr-fork 8 --jtr-format nt --jtr-hashes "hashes/*"
```

---

## Notes & Tips

- Use `--seed` + `--resume` for long, reproducible runs and recovery.
- For gigantic lists with uniqueness: prefer `--bloom` to reduce memory.
- For multi-host: generate once with `--split N`, copy to nodes, and use JtR `--node=N/M` or run Hashcat with `--skip/--limit` per shard.
- Keyboard walk graphs for QWERTZ/AZERTY are **simplified** demos; drop in exact keymaps if needed.
- Security note: generator uses a deterministic PRNG for reproducibility; for secret generation, swap to `secrets`.

---

## Demo Datasets
Small demo files are included for immediate use. Swap in your real corp/target lists when ready.


---

## New: PRINCE-like mode

Combine 2..N tokens/terms (from `--dict` and/or `--add-terms`) into chained candidates.

**Example**
```bash
python3 pwforge.py --mode prince --add-terms words_demo.txt       --prince-min 2 --prince-max 3 --sep "" --upper-first --prince-suffix-digits 2       --count 50000 --out prince.txt
```

## Improved keymaps

- `--keymap qwerty|qwertz|azerty` now applies approximate adjacency transforms
- `--walk-allow-shift` includes the shifted symbol row adjacency (`!@#$...+`)



---

## Org terms bias

Use `--bias-terms file.txt` to up-weight organization-specific tokens in `mask`, `passphrase`, and `prince` modes.
- File format: one token per line, optionally followed by a weight. Examples:
  ```
  okta 4
  azure 3
  company
  ```
- Control intensity with `--bias-factor` (default 2.0). The effective weight is `weight * bias-factor`.

**Example (PRINCE with bias):**
```bash
python3 pwforge.py --mode prince --dict words_demo.txt --bias-terms bias_terms_demo.txt       --bias-factor 2.5 --prince-min 2 --prince-max 3 --sep "" --upper-first       --count 100000 --out prince_bias.txt
```

## Hashcat rules hand-off (helper)

If you want to generate **base candidates** and then let Hashcat apply a rules file, use the helper options.
The script will write (or reuse) `--out` and print a suggested command (it does **not** run Hashcat).

```bash
python3 pwforge.py --mode mask --dict words_demo.txt --years-file years_demo.txt --symbols-file symbols_demo.txt       --emit-base --count 500000 --out base.txt       --hc-attack 0 --hc-mode 1000 --hc-hashes "hashes.txt" --hc-rules /path/to/best64.rule
```

It will print a line like:
```
[hashcat] Suggested command:
hashcat -a 0 -m 1000 "hashes.txt" --rules-file "/path/to/best64.rule" "base.txt"
```


---

## 🔧 How to Generate a Wordlist File

You can generate password candidates directly to **stdout**, or save them to a file using `--out`.

### 📝 Basic File Output

To generate a simple list of passwords:

```bash
python3 pwforge.py --mode pw --min 8 --max 16 --count 100000 --out pwlist.txt
```

This creates a file named **pwlist.txt** in the current directory.

### 🗜️ Compressing Output

You can automatically compress the file with gzip using `--gz`:

```bash
python3 pwforge.py --mode pw --count 500000 --gz --out pwlist.txt
```

This produces `pwlist.txt.gz`.

### ✂️ Splitting into Multiple Files

Split output into N files for parallel cracking (good for multi-GPU or cluster use):

```bash
python3 pwforge.py --mode walk --count 2000000 --split 4 --out walks.txt
```

This will produce:
```
walks_00.txt
walks_01.txt
walks_02.txt
walks_03.txt
```

You can combine this with `--gz` to compress each file.

### 🚫 Suppressing Console Output

If you only want to write to file(s) and not print anything to screen:

```bash
python3 pwforge.py --mode mask --count 100000 --out masklist.txt --no-stdout
```

### 📦 Combined Example with All Output Options

```bash
python3 pwforge.py --mode prince --dict words_demo.txt --bias-terms bias_terms_demo.txt   --prince-min 2 --prince-max 3 --sep "" --upper-first   --count 1000000 --out prince_combo.txt --split 8 --gz --no-stdout
```

This will create **8 compressed files**:  
`prince_combo_00.txt.gz` → `prince_combo_07.txt.gz`

### ⚙️ Piping Directly into Hashcat or John the Ripper

If you prefer to **stream output** instead of writing files:

#### Hashcat
```bash
python3 pwforge.py --mode pw --count 500000 | hashcat -a 0 -m 1000 hashes.txt
```

#### John the Ripper
```bash
python3 pwforge.py --mode pw --count 500000 | john --stdin --format=nt hashes/*
```

> **Note:** John’s `--fork` cannot be used when reading from `stdin`. Use `--split` to create multiple files for each fork.

### 🎯 Reproducible and Resumable Generation

Use `--seed` to make output deterministic (same list every time):

```bash
python3 pwforge.py --mode walk --count 100000 --seed 1337 --out repeatable.txt
```

To resume a long run after interruption:

```bash
python3 pwforge.py --mode mask --count 1000000 --resume session.pkl --out masklist.txt
```


---

## Phase 2 Features

### Markov mode
Train on a corpus and generate character-level candidates that "look like" it.
```bash
python3 pwforge.py --mode markov --markov-train words_demo.txt --markov-order 3       --min 8 --max 14 --count 200000 --out markov.txt
```

### PCFG mode
Learn common templates from a corpus (e.g., `CapWord+YYYY+!`) and sample from them.
```bash
python3 pwforge.py --mode pcfg --pcfg-train words_demo.txt       --min 8 --max 16 --count 200000 --out pcfg.txt
```

### Mobile keypad walks
```bash
python3 pwforge.py --mode mobile-walk --min 6 --max 10 --count 150000 --out mobile.txt
```

### Estimator
Get a rough sense of keyspace/plan and exit:
```bash
python3 pwforge.py --mode pw --min 10 --max 16 --estimate-only
```

### Workers (simple chunking)
Split generation into N chunks (lightweight, memory-safe). Combine with `--split` or run multiple invocations in parallel for scale.
```bash
python3 pwforge.py --mode both --count 1000000 --workers 8 --out both.txt --gz --split 8 --no-stdout
```
