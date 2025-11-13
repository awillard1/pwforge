
# PWForge – Professional Password Candidate Generator

High‑performance, multi‑mode wordlist generator designed for **Hashcat** and **John the Ripper**.  
Streams to STDOUT for live cracking or writes compressed/split files for large runs.

---

## ✅ What this README covers (explicit & complete)

This README documents **every CLI option** implemented in `pwforge.py` and shows **concrete examples** for:
- Piping into **Hashcat** and **John the Ripper**
- Writing **plain** / **gzip** files (with **split** support)
- **Parallel** multi‑process generation (`--mp-workers`), and how it differs from `--split`
- Deterministic runs (`--seed`) and chunked batching (`--chunk`)
- Mode‑specific flags (walks, Markov, PCFG, prince, etc.)

> Source of truth for options: the uploaded `pwforge.py`. See the **CLI Options** section below for a 1:1 mapping.

---

## 🔧 Installation

```bash
git clone https://github.com/awillard1/pwforge.git
cd pwforge
python3 pwforge.py --help
```

> Requires **Python 3.8+**

---

## 🚀 Quick Start

### Stream directly to Hashcat (no disk I/O)
```bash
# Random passwords
python3 pwforge.py --mode pw --count 1000000 | hashcat -a 0 -m 1000 hashes.ntlm

# Markov (requires training corpus)
python3 pwforge.py --mode markov --markov-train rockyou.txt --count 2000000 | hashcat -a 0 -m 0 hashes.raw-md5

# PCFG (requires training corpus)
python3 pwforge.py --mode pcfg --pcfg-train rockyou.txt --count 1000000 | hashcat -a 0 -m 1000 hashes.ntlm
```

### Stream directly to John the Ripper (stdin)
```bash
python3 pwforge.py --mode pw --count 1000000 | john --stdin --format=nt hashes/*
```
> **Note:** John’s `--fork` does **not** work with stdin. For multi‑core JtR, write a file and use `--wordlist`:

```bash
python3 pwforge.py --mode walk --count 2000000 --out walks.txt --no-stdout
john --wordlist=walks.txt --format=nt --fork=16 hashes/*
```

---

## 💾 Writing Files (Plain / Gzip / Split)

```bash
# Plain file
python3 pwforge.py --mode pw --count 5000000 --out pw.txt --no-stdout

# Gzip file (recommended for huge runs)
python3 pwforge.py --mode walk --count 5000000 --out walks.txt.gz --gz --no-stdout

# Split into 8 parts (mutually exclusive with --mp-workers)
python3 pwforge.py --mode pw --count 80000000 --split 8 --out pw.txt --gz --no-stdout
# => pw_00000000.txt.gz ... pw_00000007.txt.gz
```

### Fast paths on WSL/Linux
Avoid `/mnt/c/...` for heavy I/O; prefer Linux FS or tmpfs:
```bash
python3 pwforge.py --mode pw --count 10000000 --out /dev/shm/pw.txt --no-stdout
```

---

## ⚡ Multi‑Process Generation (`--mp-workers`)

`--mp-workers N` launches **N child processes**. Each writes its own shard (e.g., `file_w00.txt[.gz]`).  
**Requirements:** `--out` **and** `--no-stdout`.

```bash
# 8 parallel subprocesses; gzip output; deterministic seed
python3 pwforge.py --mode markov   --markov-train rockyou.txt   --count 80000000   --mp-workers 8   --out markov.txt.gz --gz --no-stdout --seed 42
# => markov_w00.txt.gz ... markov_w07.txt.gz
```

> `--mp-workers` and `--split` are **mutually exclusive** (use one strategy).

---

## 🧭 Modes & Key Examples

### 1) `--mode pw` (random passwords)
General purpose random with class policy and custom charset.

```bash
# 1M random passwords 8..16 chars; require upper/lower/digit/symbol
python3 pwforge.py --mode pw --min 8 --max 16 --count 1000000   --require upper,lower,digit,symbol | hashcat -a 0 -m 1000 hashes.ntlm
```

**Useful flags**: `--charset`, `--exclude-ambiguous`, `--require`, `--seed`

---

### 2) `--mode walk` (keyboard walks)
QWERTY/QWERTZ/AZERTY adjacency walks; optional custom graph and starts list.

```bash
# QWERTZ with custom starts and suffix digits appended
python3 pwforge.py --mode walk --keymap qwertz   --starts-file starts.json --suffix-digits 2   --min 6 --max 10 --count 2000000 | hashcat -a 0 -m 1000 hashes.ntlm
```

**Useful flags**:  
`--keymap qwerty|qwertz|azerty`, `--keymap-file graph.json`, `--walk-allow-shift`,  
`--starts-file`, `--window`, `--relax-backtrack`, `--upper`, `--suffix-digits`

---

### 3) `--mode both` (combine `pw` + `walk`)
```bash
# Split policy: half pw, half walk (total == count)
python3 pwforge.py --mode both --both-policy split --count 1000000 | hashcat -a 0 -m 1000 hashes.ntlm

# Each policy: generate full count for BOTH (2x total emitted)
python3 pwforge.py --mode both --both-policy each --count 1000000 --out both.txt --no-stdout
```

**Flag**: `--both-policy split|each`

---

### 4) `--mode mask` (word + year + symbol)
Lightweight patterning (good for corporate).

```bash
python3 pwforge.py --mode mask   --dict words_demo.txt --years-file years_demo.txt --symbols-file symbols_demo.txt   --count 2000000 | hashcat -a 0 -m 1000 hashes.ntlm
```

**Flags**: `--mask-set`, `--dict`, `--years-file`, `--symbols-file`, `--upper-first`, `--emit-base`

---

### 5) `--mode passphrase`
```bash
python3 pwforge.py --mode passphrase   --dict eff_demo.txt --words 4 --sep "-" --upper-first   --count 1000000 --out passphrases.txt --no-stdout
```

**Flags**: `--dict`, `--words`, `--sep`, `--upper-first`

---

### 6) `--mode numeric`
```bash
python3 pwforge.py --mode numeric --min 6 --max 8 --count 1000000 | hashcat -a 0 -m 0 hashes.raw-md5
```

---

### 7) `--mode syllable`
```bash
python3 pwforge.py --mode syllable --template CVCVC --count 1000000 | hashcat -a 0 -m 1000 hashes.ntlm
```

**Flags**: `--template`, `--upper-first`

---

### 8) `--mode prince` (composition of words)
```bash
python3 pwforge.py --mode prince --dict words_demo.txt   --bias-terms bias_terms_demo.txt --bias-factor 2.0   --prince-min 2 --prince-max 3 --sep ""   --prince-suffix-digits 2 --prince-symbol "!"   --count 2000000 | hashcat -a 0 -m 1000 hashes.ntlm
```

**Flags**: `--dict`, `--bias-terms`, `--bias-factor`, `--prince-min`, `--prince-max`, `--sep`, `--prince-suffix-digits`, `--prince-symbol`

---

### 9) `--mode markov` (requires training)
```bash
python3 pwforge.py --mode markov --markov-train rockyou.txt --markov-order 3   --count 5000000 --out markov.txt.gz --gz --no-stdout
```
**Flags**: `--markov-train`, `--markov-order` (1..5)

---

### 10) `--mode pcfg` (requires training)
```bash
python3 pwforge.py --mode pcfg --pcfg-train rockyou.txt   --count 5000000 | hashcat -a 0 -m 1000 hashes.ntlm
```
**Flags**: `--pcfg-train`

---

## 🧩 CLI Options (exhaustive)

### General
```
--mode {pw,walk,both,mask,passphrase,numeric,syllable,prince,markov,pcfg,mobile-walk}
--count N
--min N
--max N
--seed N
--resume PATH              (reserved)
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
--leet-profile {off,minimal,common,aggressive}
--suffix-digits N
--keymap {qwerty,qwertz,azerty}
--keymap-file PATH
--walk-allow-shift
--both-policy {split,each}
--interleave               (present in CLI; no-op in generator)
```

### Mask / Passphrase / Prince
```
--mask-set {common,corp,ni}
--dict PATH
--years-file PATH
--symbols-file PATH
--emit-base
--bias-terms PATH
--bias-factor FLOAT        (default 2.0)
--words N                  (passphrase)
--sep STR
--upper-first
--template STR             (syllable)
--add-terms PATH           (present in CLI; no-op in generator)
--prince-min N             (default 2)
--prince-max N             (default 3)
--prince-suffix-digits N
--prince-symbol STR
```

### Uniqueness (reserved / not active)
```
--unique
--unique-global
--bloom MB
--dedupe-file PATH
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
--workers N                (single-process chunking)
--mp-workers N             (spawn N child processes; requires --out and --no-stdout)
--chunk N                  (batch size per inner loop; default 100000)
```

### Models
```
--markov-train PATH
--markov-order N           (1..5; default 3)
--pcfg-train PATH
```

### Cracker passthroughs (accepted, not used by generator)
```
--jtr-fork N
--jtr-format STR
--jtr-hashes PATH
--hc-attack N
--hc-mode N
--hc-hashes PATH
--hc-rules PATH
```

---

## 🧪 Deterministic & Chunked Runs

```bash
# Deterministic
python3 pwforge.py --mode pw --seed 1337 --count 1000000 --out stable.txt --no-stdout

# Larger chunks (better throughput on big RAM boxes)
python3 pwforge.py --mode pw --chunk 500000 --count 5000000 --out pw.txt --no-stdout
```

---

## 🧵 Parallel Without GNU parallel (pure Bash)

```bash
# Four modes in parallel
printf "%s\n" pw walk markov pcfg | xargs -P4 -I{} sh -c 'python3 pwforge.py --mode "{}" --count 200000 --out "{}".txt --no-stdout'

# Shard one mode and merge
for i in 0 1 2 3; do
  python3 pwforge.py --mode pw --count 1000000 --out "pw_$i.txt" --no-stdout &
done
wait
cat pw_*.txt > pw_all.txt
```

---

## 🧠 Tips

- For **WSL**, prefer generating into Linux FS (e.g., `/home`, `/dev/shm`) instead of `/mnt/c`.
- `--mp-workers` is the preferred scaling method when writing to files (each shard writes independently).
- For **JtR multi‑core**, do not pipe — **write a file** and run with `--fork`.

---

## 📄 License

```
MIT License © 2025 Adam Willard
```
**For authorized penetration testing and research use only.**
