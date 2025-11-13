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
- **Performance:** Chunked and multi-process workers, gzip output, resumable generation, deterministic seeds
- **Compatibility:** Output streams directly to Hashcat or John the Ripper
- **Output options:** `--out`, `--split`, `--gz`, `--no-stdout`, `--estimate-only`, and more

---

## 🚀 Quick Start

```bash
git clone https://github.com/awillard1/pwforge.git
cd pwforge

# Random password list
python3 pwforge.py --mode pw --min 8 --max 16 --count 500000 --out pwlist.txt.gz --gz

# Keyboard walks
python3 pwforge.py --mode walk --min 6 --max 10 --count 300000 --starts-file starts.json --upper --suffix-digits 2 --out walks.txt.gz --gz

# Markov mode (train from corpus)
python3 pwforge.py --mode markov --markov-train words_demo.txt --markov-order 3 --count 100000 --out markov.txt

# PCFG mode
python3 pwforge.py --mode pcfg --pcfg-train words_demo.txt --count 100000 --out pcfg.txt
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

## ⚙️ Using PWForge with Hashcat

PWForge integrates directly with **Hashcat** for dynamic password cracking workflows.
You can stream wordlists to Hashcat in real time (stdin) or pre-generate structured lists for large jobs.

### 1) Direct streaming to Hashcat
```bash
# Randoms into NTLM
python3 pwforge.py --mode pw --count 500000 | hashcat -a 0 -m 1000 hashes.txt

# Keyboard walks
python3 pwforge.py --mode walk --min 6 --max 10 --count 250000 | hashcat -a 0 -m 1800 hashes.txt
```

### 2) File-based runs (recommended for long jobs)
```bash
python3 pwforge.py --mode both --count 1000000 --out candidates.txt --no-stdout
hashcat -a 0 -m 1000 hashes.txt candidates.txt
# or compressed
python3 pwforge.py --mode pw --count 5000000 --gz --out pwlist.txt --no-stdout
hashcat -a 0 -m 1000 hashes.txt pwlist.txt.gz
```

### 3) Split lists for multi-GPU/multi-node cracking
```bash
python3 pwforge.py --mode pw --count 20000000 --split 8 --gz --out pwlist.txt --no-stdout
# -> pwlist_00.txt.gz ... pwlist_07.txt.gz
```

### 4) Multi-process generation (true parallel)
```bash
python3 pwforge.py --mode walk --count 10000000 --mp-workers 8 --chunk 250000 --out walklist.txt --gz --no-stdout
# -> walklist_w00.txt.gz ... walklist_w07.txt.gz
```

### 5) With Hashcat rules (base candidates → mutations)
```bash
python3 pwforge.py --mode mask --dict words_demo.txt --years-file years_demo.txt   --symbols-file symbols_demo.txt --emit-base --count 100000 | hashcat -a 0 -m 0 hashes.txt --rules-file /opt/hashcat/rules/best64.rule
```

### 6) Hybrid attacks
```bash
# Append digits (left hybrid)
python3 pwforge.py --mode pw --count 1000000 | hashcat -a 6 -m 1000 hashes.txt ?d?d?d

# Prepend years (right hybrid)
python3 pwforge.py --mode pw --count 1000000 | hashcat -a 7 -m 1000 hashes.txt 20?d?d
```

### 7) Markov/PCFG streamed samples
```bash
python3 pwforge.py --mode markov --markov-train words_demo.txt --count 500000 | hashcat -a 0 -m 1000 hashes.txt
python3 pwforge.py --mode pcfg   --pcfg-train   words_demo.txt --count 500000 | hashcat -a 0 -m 0    hashes.txt
```

### 8) Resume & determinism
```bash
python3 pwforge.py --mode pw --count 1000000 --seed 1337 --out stable.txt --no-stdout
# later
hashcat -a 0 -m 1000 hashes.txt stable.txt --restore
```

---

## ⚙️ CLI Reference (Updated)

### Global
- `--mode pw|walk|both|mask|passphrase|numeric|syllable|prince|markov|pcfg|mobile-walk`
- `--count N` — number of lines (**for `both` see policy**)
- `--min, --max` — length range (where applicable)
- `--seed N` — deterministic output (reproducible)
- `--resume session.json` — save/load RNG state, produced count
- `--dry-run N` — preview N candidates per active mode and exit
- `--meter N` — stats to stderr every N lines (default 100k; 0=off)
- `--chunk N` — generate in batches of N lines (default: 100000); reduces memory use for large runs
- `--workers N` — chunked workers (single process)
- `--mp-workers N` — launch N parallel subprocesses for true multi-process generation (requires `--out` and `--no-stdout`)

### Output
- `--out path` — write to a file (or use stdout only)
- `--split N` — split output into N files (suffix `_00..`)
- `--gz` — gzip output file(s)
- `--no-stdout` — suppress stdout (useful with `--out`)

### Markov/PCFG
- `--markov-train path` — corpus for Markov training
- `--markov-order N` — Markov order (default 3; 1..5 recommended)
- `--pcfg-train path` — corpus for PCFG training

---

## 🧾 License

MIT License © 2025  
Created by Adam Willard & Contributors  
For educational and professional security research only.
