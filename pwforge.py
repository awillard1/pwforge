#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified password candidate generator (Hashcat/JtR-ready).

Modes:
  - pw           : random passwords
  - walk         : keyboard walks (QWERTY/QWERTZ/AZERTY)
  - both         : pw + walk
  - mask         : template-based (Word + Year + digits + symbol) with mask-sets
  - passphrase   : multi-word passphrases
  - numeric      : date & PIN patterns
  - syllable     : pronounceable C/V templates
  - prince       : PRINCE-like chaining of tokens/terms (2..N words)

Key extras:
  - Uniqueness:      --unique, --unique-global, --bloom <MB>, --dedupe-file <file>
  - Determinism:     --seed N (reproducible), --resume <session.json>
  - Output control:  --out list.txt [--gz] [--split N] [--no-stdout]
  - Stats:           periodic meter to stderr
  - Walk:            --keymap qwerty|qwertz|azerty, --walk-allow-shift
  - PW complexity:   --require upper,lower,digit,symbol
  - Leet profile:    --leet-profile minimal|common|aggressive
  - Mask sets:       --mask-set common|corp|ni (preset patterns)
  - Dry run:         --dry-run N (preview) then exit
  - JtR helper:      --jtr-fork N --jtr-format fmt --jtr-hashes "hashes/*"
  - Hashcat helper:  --hc-attack A --hc-mode M --hc-hashes path --hc-rules rules.txt

Org bias:
  - --bias-terms file.txt  (lines: 'token' or 'token weight')
  - --bias-factor 2.0      (how strongly to up-weight bias tokens vs base dict)

Examples:
  python3 gen_combo.py --mode pw --min 10 --max 16 --count 50000 --require upper,lower,digit,symbol
  python3 gen_combo.py --mode walk --min 6 --max 10 --count 80000 --starts-file starts.json --upper --suffix-digits 2
  python3 gen_combo.py --mode mask --mask-set corp --dict words_demo.txt --years-file years_demo.txt --symbols-file symbols_demo.txt --count 100000 --out corp.txt --gz
  python3 gen_combo.py --mode passphrase --dict eff_demo.txt --words 3 --sep "-" --upper-first --count 100000 --out pass.txt
  python3 gen_combo.py --mode numeric --count 50000
  python3 gen_combo.py --mode syllable --count 50000 --template CVCVCV --upper --suffix-digits 2
  python3 gen_combo.py --mode prince --add-terms words_demo.txt --prince-min 2 --prince-max 3 --count 50000
  python3 gen_combo.py --mode mask --dict words_demo.txt --hc-attack 0 --hc-mode 1000 --hc-hashes "hashes.txt" --hc-rules rules/best64.rule --out base.txt
"""

import argparse, json, string, sys, time, os, gzip, math, pickle, random
from collections import deque

# ---------------------------
# Deterministic RNG wrapper
# ---------------------------
_rng = None
def rng_init(seed=None):
    global _rng
    if seed is None:
        seed = int(time.time_ns())
    _rng = random.Random(seed)
    return seed

def rng_state():
    return _rng.getstate()

def rng_setstate(st):
    _rng.setstate(st)

def rand_below(n):
    return _rng.randrange(n)

def rand_choice(seq):
    return seq[rand_below(len(seq))]

def rand_uniform():
    return _rng.random()

# ---------------------------
# Keymaps for keyboard walks
# ---------------------------
def build_qwerty():
    return {
        '1':['q','2'], '2':['1','q','w','3'], '3':['2','w','e','4'], '4':['3','e','r','5'],
        '5':['4','r','t','6'], '6':['5','t','y','7'], '7':['6','y','u','8'], '8':['7','u','i','9'],
        '9':['8','i','o','0'], '0':['9','o','p'],
        'q':['1','2','a','w'], 'w':['q','2','3','s','e'], 'e':['w','3','4','d','r'], 'r':['e','4','5','f','t'],
        't':['r','5','6','g','y'], 'y':['t','6','7','h','u'], 'u':['y','7','8','j','i'], 'i':['u','8','9','k','o'],
        'o':['i','9','0','l','p'], 'p':['o','0'],
        'a':['q','w','z','s'], 's':['a','w','e','x','d'], 'd':['s','e','r','c','f'], 'f':['d','r','t','v','g'],
        'g':['f','t','y','b','h'], 'h':['g','y','u','n','j'], 'j':['h','u','i','m','k'], 'k':['j','i','o','l'],
        'l':['k','o','p'],
        'z':['a','s','x'], 'x':['z','s','d','c'], 'c':['x','d','f','v'], 'v':['c','f','g','b'],
        'b':['v','g','h','n'], 'n':['b','h','j','m'], 'm':['n','j','k']
    }

def transform_qwertz(g):
    g = {k:list(v) for k,v in g.items()}
    # swap y <-> z neighbors
    def swap_neighbors(k1,k2):
        for k,v in g.items():
            g[k] = [k2 if x==k1 else (k1 if x==k2 else x) for x in v]
        g[k1], g[k2] = g.get(k2,[]), g.get(k1,[])
    if 'y' in g and 'z' in g:
        swap_neighbors('y','z')
    return g

def transform_azerty(g):
    g = {k:list(v) for k,v in g.items()}
    # rough swaps: a<->q, z<->w; ensure m<->l adjacency
    def swap_neighbors(k1,k2):
        for k,v in g.items():
            g[k] = [k2 if x==k1 else (k1 if x==k2 else x) for x in v]
        g[k1], g[k2] = g.get(k2,[]), g.get(k1,[])
    if 'a' in g and 'q' in g:
        swap_neighbors('a','q')
    if 'z' in g and 'w' in g:
        swap_neighbors('z','w')
    if 'm' in g and 'l' in g:
        g['m'] = list(set(g['m'] + ['l']))
        g['l'] = list(set(g['l'] + ['m']))
    return g

GRAPH_SYMBOL = { 
    "!": ["@"], "@": ["!","#"], "#": ["@","$"], "$": ["#","%"], "%": ["$","^"],
    "^": ["%","&"], "&": ["^","*"], "*": ["&","("], "(": ["*",")"], ")": ["(","_"],
    "_": [")","+"], "+": ["_"]
}

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_STARTS = [
    ("qwerty",10), ("qwert",9), ("asdfgh",8), ("asdfg",8), ("zxcvbnm",8),
    ("zxcvb",7), ("1qaz2wsx",6), ("1qaz",6), ("1q2w3e4r",6), ("qazwsx",6),
    ("zaq1xsw2",5), ("qwertyuiop",5), ("ytrewq",5), ("poiuy",4), ("mnbvc",4),
    ("rfvtgb",4), ("edcrfv",4), ("wsxedc",4), ("qweasd",3), ("asdqwe",3)
]
LEET_MAP = {'a':'@','e':'3','i':'1','o':'0','s':'$'}
AMBIGUOUS = set("O0l1I|`'\";:.,")

LEET_PROFILES = {
    "minimal": 0.05,
    "common": 0.10,
    "aggressive": 0.25
}

# ---------------------------
# Utility
# ---------------------------
def load_lines(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")) as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def save_lines(path, lines, gz=False):
    if gz or path.endswith(".gz"):
        with gzip.open(path if path.endswith(".gz") else path + ".gz", "wt", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")

def chunk_iter(seq, n):
    k, m = divmod(len(seq), n)
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        yield seq[start:end]
        start = end

def has_classes(s, req):
    if not req:
        return True
    checks = {
        "upper": any(c.isupper() for c in s),
        "lower": any(c.islower() for c in s),
        "digit": any(c.isdigit() for c in s),
        "symbol": any((not c.isalnum()) for c in s),
    }
    return all(checks.get(r, True) for r in req)

# Bloom filter (simple, 2 hashes)
class Bloom:
    def __init__(self, mb=64):
        bits = mb * 1024 * 1024 * 8
        self.size = max(8_000_000, bits)  # at least 1MB-ish
        self.arr = bytearray(self.size // 8)
    def _hashes(self, s):
        h1 = hash(s) & 0x7fffffff
        h2 = (hash("x"+s) * 2654435761) & 0x7fffffff
        return (h1 % self.size, h2 % self.size)
    def add(self, s):
        a, b = self._hashes(s)
        self.arr[a // 8] |= (1 << (a % 8))
        self.arr[b // 8] |= (1 << (b % 8))
    def contains(self, s):
        a, b = self._hashes(s)
        return (self.arr[a // 8] >> (a % 8)) & 1 and (self.arr[b // 8] >> (b % 8)) & 1

# Weighted picker from (token, weight) list
def pick_weighted_pairs(pairs):
    total = sum(w for _, w in pairs)
    r = rand_below(total)
    s = 0
    for t, w in pairs:
        s += w
        if r < s:
            return t
    return pairs[-1][0]

# Parse bias terms file: "token" or "token weight"
def parse_bias_file(path):
    pairs = []
    for ln in load_lines(path):
        parts = ln.split()
        if not parts: 
            continue
        token = parts[0]
        w = 1
        if len(parts) >= 2:
            try:
                w = max(1, int(float(parts[1])))
            except:
                w = 1
        pairs.append((token, w))
    return pairs

# ---------------------------
# Helpers (shared)
# ---------------------------
def rand_len(min_len, max_len):
    return rand_below(max_len - min_len + 1) + min_len

def pick_weighted(patterns):
    total = sum(w for _, w in patterns)
    r = rand_below(total)
    s = 0
    for p, w in patterns:
        s += w
        if r < s:
            return p
    return patterns[-1][0]

# ---------------------------
# Random password generator (mode: pw)
# ---------------------------
def build_charset(base_charset, exclude_ambiguous):
    cs = base_charset
    if exclude_ambiguous:
        cs = "".join(ch for ch in cs if ch not in AMBIGUOUS)
    if not cs:
        cs = string.ascii_letters + string.digits
    return cs

def gen_password(min_len, max_len, charset, require=()):
    for _ in range(1000):
        length = rand_len(min_len, max_len)
        cand = "".join(rand_choice(charset) for _ in range(length))
        if has_classes(cand, require):
            return cand
    return None

# ---------------------------
# Keyboard-walk generator (mode: walk)
# ---------------------------
def get_graph(keymap="qwerty", allow_shift=False):
    g = build_qwerty()
    if keymap == "qwertz":
        g = transform_qwertz(g)
    elif keymap == "azerty":
        g = transform_azerty(g)
    if allow_shift:
        g = {**g, **GRAPH_SYMBOL}
    return g

def continue_walk(seed, min_len, max_len, window, relax_backtrack, graph):
    seed = seed.lower()
    for ch in seed:
        if ch not in graph:
            return None
    length = rand_len(min_len, max_len)
    if len(seed) >= length:
        return seed[:length]

    path = list(seed)
    recent = deque(path[-window:], maxlen=max(1, window))
    while len(path) < length:
        curr = path[-1]
        neigh = graph.get(curr, [])
        choices = [n for n in neigh if n not in recent]
        if not choices and relax_backtrack:
            choices = neigh[:]
        if not choices:
            return None
        nxt = rand_choice(choices)
        path.append(nxt)
        recent.append(nxt)
    return "".join(path)

def maybe_uppercase(s, prob=0.3):
    out = []
    for ch in s:
        if ch.isalpha() and rand_uniform() < prob:
            out.append(ch.upper())
        else:
            out.append(ch)
    return "".join(out)

def maybe_leet(s, prob=0.1):
    out = []
    for ch in s:
        repl = LEET_MAP.get(ch.lower())
        if repl and rand_uniform() < prob:
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)

def maybe_numeric_suffix(s, max_digits=2):
    if max_digits <= 0:
        return s
    digits = rand_below(max_digits) + 1
    if digits == 1:
        return s + str(rand_below(10))
    first = str(rand_below(9) + 1)
    rest  = "".join(str(rand_below(10)) for _ in range(digits - 1))
    return s + first + rest

def make_walk(min_len, max_len, weighted_starts, window, relax_backtrack, upper, leet_prob, suffix_digits, graph):
    seed = pick_weighted(patterns=weighted_starts)
    w = continue_walk(seed, min_len, max_len, window, relax_backtrack, graph)
    if not w:
        return None
    if upper:
        w = maybe_uppercase(w)
    if leet_prob > 0:
        w = maybe_leet(w, prob=leet_prob)
    if suffix_digits > 0:
        w = maybe_numeric_suffix(w, suffix_digits)
    return w

# ---------------------------
# Mask / Template generator (mode: mask)
# ---------------------------
MASK_SETS = {
    "common": ["WordYearSym", "WordYYSym", "Word2dSym", "Word3dSym", "CapWord2d", "CapWordSym"],
    "corp":   ["WordYearSym", "WordYear2dSym", "CapWordYearSym", "CapWord2dSym"],
    "ni":     ["WordYY2d", "WordMMYY", "YYWord2d"]
}

def gen_mask(word_pairs, years, symbols, mask_set="common"):
    pat = rand_choice(MASK_SETS.get(mask_set, MASK_SETS["common"]))
    word = pick_weighted_pairs(word_pairs)
    cap = word.capitalize()
    year = rand_choice(years)
    yy = year[-2:]
    d2 = f"{rand_below(10)}{rand_below(10)}"
    d3 = f"{rand_below(10)}{rand_below(10)}{rand_below(10)}"
    sym = rand_choice(symbols)

    if pat == "WordYearSym":      return f"{cap}{year}{sym}"
    if pat == "WordYYSym":        return f"{cap}{yy}{sym}"
    if pat == "Word2dSym":        return f"{cap}{d2}{sym}"
    if pat == "Word3dSym":        return f"{cap}{d3}{sym}"
    if pat == "CapWord2d":        return f"{cap}{d2}"
    if pat == "CapWordSym":       return f"{cap}{sym}"
    if pat == "WordYear2dSym":    return f"{cap}{year}{d2}{sym}"
    if pat == "CapWordYearSym":   return f"{cap}{year}{sym}"
    if pat == "CapWord2dSym":     return f"{cap}{d2}{sym}"
    if pat == "WordYY2d":         return f"{cap}{yy}{d2}"
    if pat == "WordMMYY":
        mm = f"{rand_below(12)+1:02d}"
        return f"{cap}{mm}{yy}"
    if pat == "YYWord2d":         return f"{yy}{cap}{d2}"
    return f"{cap}{year}{sym}"

# ---------------------------
# Passphrase generator (mode: passphrase)
# ---------------------------
def gen_passphrase(word_pairs, n_words, sep, upper_first=False):
    chosen = [pick_weighted_pairs(word_pairs) for _ in range(n_words)]
    if upper_first and chosen:
        chosen[0] = chosen[0].capitalize()
    return sep.join(chosen)

# ---------------------------
# Numeric generator (mode: numeric)
# ---------------------------
def gen_numeric():
    choice = rand_below(6)
    if choice == 0:  # YYYY
        return f"{rand_below(2026-1970)+1970}"
    if choice == 1:  # MMDD
        return f"{rand_below(12)+1:02d}{rand_below(28)+1:02d}"
    if choice == 2:  # DDMM
        return f"{rand_below(28)+1:02d}{rand_below(12)+1:02d}"
    if choice == 3:  # YYMMDD
        yy = f"{rand_below(100):02d}"; mm = f"{rand_below(12)+1:02d}"; dd = f"{rand_below(28)+1:02d}"
        return f"{yy}{mm}{dd}"
    if choice == 4:  # ####
        return f"{rand_below(10000):04d}"
    return f"{rand_below(1_000_000):06d}"  # ######

# ---------------------------
# Syllable-based pronounceables (mode: syllable)
# ---------------------------
CONS = list("bcdfghjklmnpqrstvwxz")
VOWS = list("aeiou")
def gen_syllable(template="CVCVCV", upper=False, suffix_digits=0):
    out = []
    for ch in template:
        if ch.upper() == "C":
            out.append(rand_choice(CONS))
        elif ch.upper() == "V":
            out.append(rand_choice(VOWS))
        else:
            out.append(ch)  # literal
    s = "".join(out)
    if upper:
        s = s.capitalize()
    if suffix_digits > 0:
        s = maybe_numeric_suffix(s, suffix_digits)
    return s

# ---------------------------
# PRINCE-like chaining (mode: prince)
# ---------------------------
def gen_prince(token_pairs, min_words=2, max_words=3, sep="", cap_first=False, suffix_digits=0, symbol_tail=""):
    n = rand_below(max_words - min_words + 1) + min_words
    parts = [pick_weighted_pairs(token_pairs) for _ in range(n)]
    if cap_first and parts:
        parts[0] = parts[0].capitalize()
    s = sep.join(parts)
    if suffix_digits > 0:
        s = maybe_numeric_suffix(s, suffix_digits)
    if symbol_tail:
        s += symbol_tail
    return s

# ---------------------------
# MAIN
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate candidates: pw, walk, both, mask, passphrase, numeric, syllable, prince."
    )
    ap.add_argument("--mode", choices=["pw","walk","both","mask","passphrase","numeric","syllable","prince","markov","pcfg","mobile-walk"], default="pw",
                    help="Generation mode (default: pw)")
    ap.add_argument("--min", type=int, default=8, help="Minimum length (pw/walk)")
    ap.add_argument("--max", type=int, default=18, help="Maximum length (pw/walk)")
    ap.add_argument("--count", type=int, default=10000, help="How many lines to generate (see --both-policy)")
    ap.add_argument("--seed", type=int, help="Deterministic RNG seed (reproducible)")
    ap.add_argument("--resume", type=str, help="Resume session JSON (saves periodically)")

    # Password options
    ap.add_argument("--charset", type=str,
                    default=string.ascii_letters + string.digits + "!@#$%^&*()_+-=",
                    help="Character set for --mode pw/both (default: letters, digits, symbols)")
    ap.add_argument("--exclude-ambiguous", action="store_true",
                    help="Exclude ambiguous chars (O,0,l,1,I,|, etc.)")
    ap.add_argument("--require", type=str, default="",
                    help="Comma-separated required classes for complexity (upper,lower,digit,symbol)")

    # Walk options
    ap.add_argument("--starts-file", type=str, help="Path to starts.json [[pattern, weight], ...]")
    ap.add_argument("--window", type=int, default=3, help="Walk loop-avoidance window (default: 3)")
    ap.add_argument("--relax-backtrack", action="store_true", help="Allow backtracking if stuck")
    ap.add_argument("--upper", action="store_true", help="Randomly uppercase letters in walks (~30%)")
    ap.add_argument("--leet-profile", choices=["off","minimal","common","aggressive"], default="off",
                    help="Leet substitution probability profile")
    ap.add_argument("--suffix-digits", type=int, default=0, help="Append up to N random digits to walks")
    ap.add_argument("--keymap", choices=["qwerty","qwertz","azerty"], default="qwerty", help="Keyboard layout for walks")
    ap.add_argument("--keymap-file", type=str, help="Path to JSON adjacency graph for custom keyboard walks")
    ap.add_argument("--walk-allow-shift", action="store_true", help="Include symbol row adjacency")

    # Both-mode control (pw + walk)
    ap.add_argument("--both-policy", choices=["split","each"], default="split",
                    help="'split' = split count 50/50; 'each' = count of each (≈2x lines)")
    ap.add_argument("--interleave", action="store_true", help="Alternate pw/walk outputs instead of grouped")

    # Mask/Passphrase/Prince shared dict & bias
    ap.add_argument("--mask-set", choices=["common","corp","ni"], default="common", help="Mask set to use")
    ap.add_argument("--dict", type=str, help="Word list for mask/passphrase/prince")
    ap.add_argument("--years-file", type=str, help="Years list file (one per line) for mask")
    ap.add_argument("--symbols-file", type=str, help="Symbols list file (one per line) for mask")
    ap.add_argument("--markov-train", type=str, help="Training corpus for Markov mode (one candidate per line)")
    ap.add_argument("--markov-order", type=int, default=3, help="Markov n-gram order (1-5, default 3)")
    ap.add_argument("--pcfg-train", type=str, help="Training corpus for PCFG mode (one candidate per line)")
    ap.add_argument("--emit-base", action="store_true", help="Emit only base words/phrase (mask/passphrase/prince preview)")
    ap.add_argument("--bias-terms", type=str, help="Bias terms file; lines 'token' or 'token weight'")
    ap.add_argument("--bias-factor", type=float, default=2.0, help="Multiply bias token weights by this factor")

    # Passphrase options
    ap.add_argument("--words", type=int, default=3, help="Number of words for passphrase (default: 3)")
    ap.add_argument("--sep", type=str, default="-", help="Separator for passphrase/prince (default: '-')")
    ap.add_argument("--upper-first", action="store_true", help="Capitalize first word in passphrase/prince")

    # Syllable options
    ap.add_argument("--template", type=str, default="CVCVCV", help="Template (e.g., CVCVCV)")

    # PRINCE options
    ap.add_argument("--add-terms", type=str, help="Additional tokens file for PRINCE (one per line)")
    ap.add_argument("--prince-min", type=int, default=2, help="Minimum tokens per candidate (default 2)")
    ap.add_argument("--prince-max", type=int, default=3, help="Maximum tokens per candidate (default 3)")
    ap.add_argument("--prince-suffix-digits", type=int, default=0, help="Append up to N digits to PRINCE outputs")
    ap.add_argument("--prince-symbol", type=str, default="", help="Optional trailing symbol for PRINCE outputs")

    # Uniqueness / Dedupe
    ap.add_argument("--unique", action="store_true", help="Ensure uniqueness per-type")
    ap.add_argument("--unique-global", action="store_true", help="Ensure uniqueness across ALL outputs")
    ap.add_argument("--bloom", type=int, default=0, help="Bloom filter size in MB (approx)")
    ap.add_argument("--dedupe-file", type=str, help="File to dedupe against (existing list)")

    # Output control
    ap.add_argument("--out", type=str, help="Write output to this file")
    ap.add_argument("--split", type=int, default=0, help="Split into N files (list_00..list_0N-1)")
    ap.add_argument("--gz", action="store_true", help="Gzip output file(s)")
    ap.add_argument("--no-stdout", action="store_true", help="Do not print to stdout")

    # Misc
    ap.add_argument("--dry-run", type=int, default=0, help="Preview N samples from the selected mode(s) then exit")
    ap.add_argument("--meter", type=int, default=100000, help="Print stats to stderr every N lines (0=off)")
    ap.add_argument("--estimate-only", action="store_true", help="Print keyspace/plan and exit")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for generation (simple chunking)")

    # Helpers
    ap.add_argument("--jtr-fork", type=int, default=0, help="Helper: write temp file and print JtR command with --fork N")
    ap.add_argument("--jtr-format", type=str, help="Helper: JtR --format")
    ap.add_argument("--jtr-hashes", type=str, help="Helper: path/glob to JtR hashes (quoted)")

    ap.add_argument("--hc-attack", type=int, help="Hashcat helper: attack mode (e.g., 0, 1, 3, ... )")
    ap.add_argument("--hc-mode", type=int, help="Hashcat helper: -m hash-mode (e.g., 1000 for NTLM)")
    ap.add_argument("--hc-hashes", type=str, help="Hashcat helper: hashes file/path")
    ap.add_argument("--hc-rules", type=str, help="Hashcat helper: rules file (applies to base list)")

    args = ap.parse_args()

    # Initialize RNG
    seed_used = rng_init(args.seed)

    # Prepare resources for pw
    charset = build_charset(args.charset, args.exclude_ambiguous)
    require = tuple([x.strip() for x in args.require.split(",") if x.strip()])

    # Prepare resources for walk
    if args.starts_file:
        try:
            with open(args.starts_file, "r", encoding="utf-8") as fh:
                weighted_starts = [(p, int(w)) for p, w in json.load(fh) if p and int(w) > 0]
        except Exception as e:
            print(f"[!] Failed to load {args.starts_file}: {e}", file=sys.stderr)
            weighted_starts = DEFAULT_STARTS
    else:
        weighted_starts = DEFAULT_STARTS
    graph = get_graph(args.keymap, args.walk_allow_shift)
    # Optional custom keymap JSON overrides built-ins
    if args.keymap_file:
        try:
            with open(args.keymap_file, "r", encoding="utf-8") as fh:
                custom = json.load(fh)
            # Normalize keys to lower-case
            graph = {str(k).lower(): [str(vv).lower() for vv in v] for k,v in custom.items()}
            if args.walk_allow_shift:
                graph = {**graph, **GRAPH_SYMBOL}
        except Exception as e:
            print(f"[!] Failed to load --keymap-file: {e}", file=sys.stderr)

    leet_prob = 0.0 if args.leet_profile == "off" else LEET_PROFILES[args.leet_profile]

    # Mask/passphrase demos
    DEMO_WORDS = [
        "summer","winter","spring","autumn","password","welcome","admin","secure","coffee","sunshine",
        "dragon","football","iloveyou","baseball","shadow","flower","computer","keyboard","travel","smile"
    ]
    DEMO_EFF = [
        "acorn","beacon","cactus","delta","ember","fable","galaxy","hazard","icicle","jungle","kayak","lizard",
        "meadow","nectar","oyster","planet","quartz","rocket","sunset","tiger","unicorn","velvet","willow",
        "xenon","yonder","zephyr"
    ]
    DEMO_YEARS = ["1990","1995","2000","2005","2010","2015","2020","2021","2022","2023","2024","2025"]
    DEMO_SYMBOLS = ["!","@","#","$","%","^","&","*","(",")","_","+","=","?"]

    def ensure_demo(path, fallback_lines):
        try:
            return load_lines(path)
        except Exception:
            return [ln for ln in fallback_lines if ln]

    base_words = ensure_demo(args.dict, DEMO_WORDS)
    years = ensure_demo(args.years_file, DEMO_YEARS)
    symbols = ensure_demo(args.symbols_file, DEMO_SYMBOLS)

    # Markov / PCFG resources
    markov_model = None
    if args.markov_train:
        try:
            mk_lines = load_lines(args.markov_train)
            order = max(1, min(5, args.markov_order))
            markov_model = markov_train(mk_lines, order=order)
        except Exception as e:
            print(f"[!] Failed Markov train: {e}", file=sys.stderr)

    pcfg_model = None
    if args.pcfg_train:
        try:
            pcfg_model = pcfg_train(load_lines(args.pcfg_train))
        except Exception as e:
            print(f"[!] Failed PCFG train: {e}", file=sys.stderr)


    # Bias terms
    bias_pairs = []
    if args.bias_terms:
        try:
            bias_pairs = parse_bias_file(args.bias_terms)
        except Exception as e:
            print(f"[!] Failed to load --bias-terms: {e}", file=sys.stderr)

    # Build weighted pairs for mask/passphrase/prince
    def build_pairs(words, bias_pairs, bias_factor):
        pairs = [(w, 1) for w in words]
        if bias_pairs:
            # Include bias tokens with multiplied weights
            for tok, base_w in bias_pairs:
                pairs.append((tok, max(1, int(round(base_w * bias_factor)))))
        return pairs

    word_pairs = build_pairs(base_words, bias_pairs, args.bias_factor)
    prince_pairs = list(word_pairs)
    if args.add_terms:
        extra = ensure_demo(args.add_terms, [])
        prince_pairs += [(t, 1) for t in extra]

    # Dedupe structures
    seen_all = set() if args.unique_global else None
    bloom = Bloom(args.bloom) if args.bloom > 0 else None

    dedupe_file_set = set()
    if args.dedupe_file:
        try:
            for line in load_lines(args.dedupe_file):
                dedupe_file_set.add(line)
        except Exception as e:
            print(f"[!] Failed reading --dedupe-file: {e}", file=sys.stderr)

    # Resume (store RNG state only)
    resume = {"seed": seed_used, "mode": args.mode, "produced": 0, "rng_state": rng_state(), "ts": time.time()}
    if args.resume and os.path.exists(args.resume):
        try:
            with open(args.resume, "rb") as fh:
                resume = pickle.load(fh)
            rng_setstate(resume.get("rng_state"))
        except Exception as e:
            print(f"[!] Could not load resume file: {e}", file=sys.stderr)

    # Accept filter
    def accept(cand):
        if cand is None:
            return False
        if cand in dedupe_file_set:
            return False
        if bloom and bloom.contains(cand):
            return False
        if seen_all is not None and cand in seen_all:
            return False
        if seen_all is not None:
            seen_all.add(cand)
        if bloom:
            bloom.add(cand)
        return True

    # Emitters
    def emit_pw():
        for _ in range(2000):
            cand = gen_password(args.min, args.max, charset, require=require)
            if accept(cand):
                return cand
        return None

    def emit_walk():
        for _ in range(2000):
            cand = make_walk(args.min, args.max, weighted_starts, args.window,
                             args.relax_backtrack, args.upper, leet_prob, args.suffix_digits, graph)
            if accept(cand):
                return cand
        return None

    def emit_mask():
        for _ in range(2000):
            cand = gen_mask(word_pairs, years, symbols, args.mask_set)
            if args.emit_base:
                base = cand
                for sym in symbols:
                    base = base.split(sym)[0]
                cand = base
            if accept(cand):
                return cand
        return None

    def emit_passphrase():
        for _ in range(2000):
            n = max(2, args.words)
            cand = gen_passphrase(word_pairs if args.dict or args.bias_terms else [(w,1) for w in DEMO_EFF], n, args.sep, args.upper_first)
            if accept(cand):
                return cand
        return None

    def emit_numeric():
        for _ in range(2000):
            cand = gen_numeric()
            if accept(cand):
                return cand
        return None

    def emit_syllable():
        for _ in range(2000):
            cand = gen_syllable(args.template, upper=args.upper, suffix_digits=args.suffix_digits)
            if accept(cand):
                return cand
        return None


    def emit_markov():
        if not markov_model:
            return None
        for _ in range(2000):
            cand = markov_sample(markov_model, args.min, args.max)
            if accept(cand):
                return cand
        return None

    def emit_pcfg():
        if not pcfg_model:
            return None
        for _ in range(2000):
            cand = pcfg_sample(pcfg_model, args.min, args.max, years, symbols)
            if accept(cand):
                return cand
        return None

    def emit_mobile():
        for _ in range(2000):
            cand = mobile_walk(args.min, args.max, window=max(1, args.window-1), relax_backtrack=args.relax_backtrack)
            if accept(cand):
                return cand
        return None

    def emit_prince():
        for _ in range(2000):
            cand = gen_prince(prince_pairs, args.prince_min, args.prince_max, sep=args.sep,
                              cap_first=args.upper_first, suffix_digits=args.prince_suffix_digits,
                              symbol_tail=args.prince_symbol)
            if args.emit_base:
                # Base is just the token chain without digits/symbols
                cand = cand.rstrip("0123456789")
                for sym in symbols:
                    cand = cand.replace(sym, "")
            if accept(cand):
                return cand
        return None

    # Dry run
    if args.dry_run > 0:
        modes = [args.mode] if args.mode != "both" else ["pw","walk"]
        for m in modes:
            print(f"# --- DRY RUN: {m} ---")
            for _ in range(args.dry_run):
                if m == "pw": print(emit_pw() or "")
                elif m == "walk": print(emit_walk() or "")
                elif m == "mask": print(emit_mask() or "")
                elif m == "passphrase": print(emit_passphrase() or "")
                elif m == "numeric": print(emit_numeric() or "")
                elif m == "syllable": print(emit_syllable() or "")
                elif m == "prince": print(emit_prince() or "")
                elif m == "markov": print(emit_markov() or "")
                elif m == "pcfg": print(emit_pcfg() or "")
                elif m == "mobile-walk": print(emit_mobile() or "")
        return

    # Output buffer
    out_lines = []
    def emit_and_store(s):
        if not args.no_stdout:
            print(s)
        out_lines.append(s)

    produced = 0
    last_meter = time.time()
    last_prod = 0

    def meter():
        nonlocal last_meter, last_prod, produced
        if args.meter <= 0:
            return
        if produced % args.meter == 0 and produced > 0:
            now = time.time()
            dt = now - last_meter
            dn = produced - last_prod
            rate = dn / dt if dt > 0 else 0
            print(f"[meter] {produced} lines, {rate:,.0f} lps", file=sys.stderr)
            last_meter = now
            last_prod = produced


    # Estimator (very rough; per-mode keyspaces vary)
    if args.estimate_only:
        total_targets = {"pw":0,"walk":0,"mask":0,"passphrase":0,"numeric":0,"syllable":0,"prince":0,"markov":0,"pcfg":0,"mobile-walk":0}
        if args.mode == "both":
            total_targets["pw"] = args.count//2 + (args.count%2)
            total_targets["walk"] = args.count//2
        else:
            total_targets[args.mode] = args.count
        # Simple charset-based estimate for pw/markov; others print "pattern-based"
        est = {}
        avg_len = (args.min + args.max)/2
        if total_targets["pw"]:
            est["pw_keyspace"] = f"~{len(charset)}^{int(avg_len)}"
        if total_targets["markov"]:
            est["markov"] = "pattern-based (depends on corpus; typically << full charset^len)"
        if total_targets["walk"]:
            est["walk"] = "graph-walk constrained (<< charset^len)"
        if total_targets["mask"]:
            est["mask"] = "mask constrained (depends on Word/Year/Symbol lists)"
        if total_targets["passphrase"]:
            est["passphrase"] = f"~({len(base_words)})^{max(2,args.words)}"
        if total_targets["pcfg"]:
            est["pcfg"] = "PCFG constrained (learned from corpus)"
        if total_targets["numeric"]:
            est["numeric"] = "~1e4..1e6 typical"
        if total_targets["syllable"]:
            est["syllable"] = f"~({len(CONS)}*{len(VOWS)})^{len(args.template)} (approx)"
        if total_targets["prince"]:
            est["prince"] = f"~({len(base_words)}+bias)^{args.prince_min}..^{args.prince_max}"
        print(json.dumps({"mode": args.mode, "targets": total_targets, "estimates": est}, indent=2))
        return

    # Generate by mode
    targets = {"pw":0,"walk":0,"mask":0,"passphrase":0,"numeric":0,"syllable":0,"prince":0,"markov":0,"pcfg":0,"mobile-walk":0}
    if args.mode == "both":
        targets["pw"] = args.count // 2 + (args.count % 2)
        targets["walk"] = args.count // 2
    else:
        targets[args.mode] = args.count


    # Workers (simple chunking): split counts into N chunks to reduce memory and allow external parallelization
    workers = max(1, int(args.workers))
    if workers > 1:
        # Re-distribute targets into sub-iterations
        sub_targets = []
        for mode, tgt in list(targets.items()):
            if tgt <= 0: continue
            base = tgt // workers
            rem  = tgt % workers
            for i in range(workers):
                sub_targets.append((mode, base + (1 if i < rem else 0)))
        # Reset targets and loop over sub_targets in order
        targets = {}
        for mode, cnt in sub_targets:
            if cnt <= 0: 
                continue
            # inner loop per sub-chunk
            remaining = cnt
            while remaining > 0:
                if mode == "pw": s = emit_pw()
                elif mode == "walk": s = emit_walk()
                elif mode == "mask": s = emit_mask()
                elif mode == "passphrase": s = emit_passphrase()
                elif mode == "numeric": s = emit_numeric()
                elif mode == "syllable": s = emit_syllable()
                elif mode == "prince": s = emit_prince()
                elif mode == "markov": s = emit_markov()
                elif mode == "pcfg": s = emit_pcfg()
                elif mode == "mobile-walk": s = emit_mobile()
                else: break
                if s is None: break
                emit_and_store(s)
                remaining -= 1; produced += 1; meter()
        # skip default generation below
        produced = produced  # no-op to keep var referenced
        # proceed to file writing
        # (fall-through)
    else:

    for mode, tgt in targets.items():
        if tgt <= 0: continue
        while tgt > 0:
            if mode == "pw": s = emit_pw()
            elif mode == "walk": s = emit_walk()
            elif mode == "mask": s = emit_mask()
            elif mode == "passphrase": s = emit_passphrase()
            elif mode == "numeric": s = emit_numeric()
            elif mode == "syllable": s = emit_syllable()
            elif mode == "prince": s = emit_prince()
            elif mode == "markov": s = emit_markov()
            elif mode == "pcfg": s = emit_pcfg()
            elif mode == "mobile-walk": s = emit_mobile()
            else: break
            if s is None: break
            emit_and_store(s)
            tgt -= 1; produced += 1; meter()

    # Save resume state
    if args.resume:
        try:
            resume["rng_state"] = rng_state()
            resume["produced"] = produced
            resume["ts"] = time.time()
            with open(args.resume, "wb") as fh:
                pickle.dump(resume, fh)
        except Exception as e:
            print(f"[!] Failed to save resume: {e}", file=sys.stderr)

    # Write output files
    if args.out:
        if args.split and args.split > 1:
            parts_total = args.split
            chunk_size = len(out_lines) // parts_total
            start = 0
            paths = []
            for i in range(parts_total):
                end = start + chunk_size + (1 if i < (len(out_lines) % parts_total) else 0)
                part = out_lines[start:end]
                start = end
                root, ext = os.path.splitext(args.out)
                p = f"{root}_{i:02d}{ext or ''}"
                save_lines(p, part, gz=args.gz)
                if args.gz and not p.endswith(".gz"): p = p + ".gz"
                paths.append(p)
            print(f"[i] Wrote {len(paths)} files:", file=sys.stderr)
            for p in paths: print(f"    {p}", file=sys.stderr)
        else:
            save_lines(args.out, out_lines, gz=args.gz)
            final = args.out + (".gz" if args.gz and not args.out.endswith(".gz") else "")
            print(f"[i] Wrote {final}", file=sys.stderr)

    # Helpers: John and Hashcat suggested commands (print only)
    if args.jtr_fork and args.jtr_format and args.jtr_hashes:
        tmp = args.out or "jtr_wordlist.txt"
        if not args.out:
            save_lines(tmp, out_lines, gz=False)
        cmd = f'john --wordlist="{tmp}" --format={args.jtr_format} --fork={args.jtr_fork} {args.jtr_hashes}'
        print(f"[jtr] Suggested command:\n{cmd}", file=sys.stderr)

    if args.hc_attack is not None and args.hc_mode is not None and args.hc_hashes and args.hc_rules:
        tmp = args.out or "hc_base.txt"
        if not args.out:
            save_lines(tmp, out_lines, gz=False)
        cmd = f'hashcat -a {args.hc_attack} -m {args.hc_mode} "{args.hc_hashes}" --rules-file "{args.hc_rules}" "{tmp}"'
        print(f"[hashcat] Suggested command:\n{cmd}", file=sys.stderr)

if __name__ == "__main__":
    main()

# ---------------------------
# Markov (character-level) model (mode: markov)
# ---------------------------
def markov_train(corpus_lines, order=3):
    BOS = "\x02"  # start token
    EOS = "\x03"  # end token
    from collections import defaultdict
    trans = defaultdict(lambda: {})
    for line in corpus_lines:
        s = BOS * order + line.strip() + EOS
        for i in range(len(s) - order):
            ctx = s[i:i+order]
            nxt = s[i+order]
            trans[ctx][nxt] = trans[ctx].get(nxt, 0) + 1
    # convert to cumulative lists for sampling
    model = {}
    for ctx, d in trans.items():
        chars = list(d.keys())
        weights = [d[c] for c in chars]
        total = sum(weights)
        cum = []
        acc = 0
        for w in weights:
            acc += w
            cum.append(acc)
        model[ctx] = (chars, cum, total)
    return {"order": order, "model": model, "BOS": BOS, "EOS": EOS}

def markov_sample(m, min_len, max_len):
    order = m["order"]; model = m["model"]; BOS=m["BOS"]; EOS=m["EOS"]
    ctx = BOS * order
    out = []
    target = rand_below(max_len - min_len + 1) + min_len
    while True:
        if ctx not in model:
            break
        chars, cum, total = model[ctx]
        r = rand_below(total)
        # binary search
        lo, hi = 0, len(cum)-1
        while lo < hi:
            mid = (lo+hi)//2
            if r < cum[mid]: hi = mid
            else: lo = mid + 1
        ch = chars[lo]
        if ch == EOS:
            break
        out.append(ch)
        if len(out) >= target:
            # allow early stop
            if rand_below(3) == 0:
                break
        ctx = (ctx + ch)[-order:]
        if len(out) >= max_len:
            break
    return "".join(out)

# ---------------------------
# PCFG (template-based) learner (mode: pcfg)
# ---------------------------
_WORD_RE = re.compile(r"[A-Za-z]+")
_DIG_RE  = re.compile(r"\d+")
_SYM_RE  = re.compile(r"[^A-Za-z0-9]+")

def classify_token(tok):
    if _WORD_RE.fullmatch(tok):
        if tok and tok[0].isupper() and tok[1:].islower():
            return "CapWord"
        elif tok.islower():
            return "lowWord"
        elif tok.isupper():
            return "UPWORD"
        else:
            return "Word"
    if _DIG_RE.fullmatch(tok):
        if len(tok)==4 and tok.isdigit() and (1970 <= int(tok) <= 2035):
            return "YEAR4"
        return f"DIG{len(tok)}"
    if _SYM_RE.fullmatch(tok):
        return f"SYM{len(tok)}"
    return "MISC"

def tokenize_line(line):
    toks = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]+", line.strip())
    return toks

def pcfg_train(lines):
    from collections import defaultdict
    pattern_counts = defaultdict(int)
    buckets = defaultdict(list)
    for ln in lines:
        toks = tokenize_line(ln)
        if not toks: 
            continue
        patt = "+".join(classify_token(t) for t in toks)
        pattern_counts[patt] += 1
        buckets[patt].append(toks)
    # For word-ish slots, collect reservoirs
    lex = {
        "CapWord": set(), "lowWord": set(), "UPWORD": set(), "Word": set()
    }
    for ln in lines:
        for t in tokenize_line(ln):
            cls = classify_token(t)
            if cls in lex:
                lex[cls].add(t)
    for k in lex:
        lex[k] = list(lex[k]) or ["Password","admin","welcome"]
    return {"patterns": dict(pattern_counts), "lex": lex}

def pcfg_sample(model, min_len, max_len, years, symbols):
    if not model["patterns"]:
        return None
    # choose a pattern with probability ~ count
    pats = list(model["patterns"].items())
    total = sum(c for _,c in pats)
    r = rand_below(total)
    s = 0
    for patt, c in pats:
        s += c
        if r < s:
            chosen = patt
            break
    parts = []
    for slot in chosen.split("+"):
        if slot in model["lex"]:
            parts.append(rand_choice(model["lex"][slot]))
        elif slot.startswith("DIG"):
            n = int(slot[3:])
            if n == 4 and rand_below(2)==0:
                parts.append(rand_choice(years))
            else:
                parts.append("".join(str(rand_below(10)) for _ in range(n)))
        elif slot == "YEAR4":
            parts.append(rand_choice(years))
        elif slot.startswith("SYM"):
            n = int(slot[3:])
            sym = "".join(rand_choice(symbols) for _ in range(max(1, min(3, n))))
            parts.append(sym)
        else:
            parts.append("")
    cand = "".join(parts)
    if len(cand) < min_len:
        # pad with digits
        pad = min_len - len(cand)
        cand += "".join(str(rand_below(10)) for _ in range(pad))
    return cand[:max_len]

# ---------------------------
# Mobile keypad walk (mode: mobile-walk)
# ---------------------------
MOBILE_GRAPH = {
    '1': ['2','4'], '2':['1','3','5'], '3':['2','6'],
    '4': ['1','5','7'], '5':['2','4','6','8'], '6':['3','5','9'],
    '7': ['4','8','*'], '8':['5','7','9','0'], '9':['6','8','#'],
    '0': ['8'],
    '*': ['7','0','#'],
    '#': ['9','0','*']
}

def mobile_walk(min_len, max_len, window=2, relax_backtrack=True):
    start = rand_choice(list(MOBILE_GRAPH.keys()))
    length = rand_below(max_len - min_len + 1) + min_len
    path = [start]
    recent = []
    while len(path) < length:
        neigh = MOBILE_GRAPH.get(path[-1], [])
        choices = [n for n in neigh if n not in recent[-window:]]
        if not choices and relax_backtrack:
            choices = neigh[:]
        if not choices:
            break
        nxt = rand_choice(choices)
        path.append(nxt)
        recent.append(nxt)
    return "".join(path)

# ---------------------------
# Estimator
# ---------------------------
def estimate_keyspace(args, charset):
    # rough estimates per mode
    avg_len = (args.min + args.max) / 2 if hasattr(args, "min") else 10
    if args.mode in ("pw","both"):
        k_pw = (len(charset) ** int(avg_len))
    else:
        k_pw = 0
    k = k_pw
    return k
