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

> **Note:** John’s `--fork` cannot be used when reading from `stdin`. Use `--split` to create multiple files for each fork.

---

## ⚙️ CLI Reference (Updated)

### Global
- `--mode pw|walk|both|mask|passphrase|numeric|syllable|prince|markov|pcfg|mobile-walk`
- `--count N` — number of lines (**for `both` see policy**)
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

---

## 🧮 Multi-Process Workers (True Parallel Mode)

PWForge now supports **true multi-process sharding** via `--mp-workers`.
Each worker spawns a child process that writes its own shard (`_w00`, `_w01`, etc.).

```bash
python3 pwforge.py --mode pw --count 10000000 --mp-workers 8 --out pwlist.txt --gz --no-stdout
```

This command launches **8 subprocesses**, each producing 1/8th of the total count.

Combine it with `--chunk` to control per-batch size and memory footprint:

```bash
python3 pwforge.py --mode walk --count 5000000 --mp-workers 4 --chunk 250000 --out walks.txt --gz --no-stdout
```

> **Notes**
> - Requires both `--out` and `--no-stdout`.
> - Worker files are named like `pwlist_w00.txt.gz`, `pwlist_w01.txt.gz`, etc.
> - Provides near-linear CPU scaling on multi-core systems.

---

## 🧾 License

MIT License © 2025  
Created by Adam Willard & Contributors  
For educational and professional security research only.

---

### 🔑 Keywords
`password generator` · `hashcat` · `john the ripper` · `markov` · `pcfg` · `mask` · `passphrase` · `keyboard walk` · `pentest` · `wordlist` · `security research`
