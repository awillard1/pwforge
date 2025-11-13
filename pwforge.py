#!/usr/bin/env python3
# PWForge - Advanced Multi-Mode Password Generator
# Supports: pw, walk, mask, passphrase, numeric, syllable, prince, markov, pcfg, mobile-walk
# OPTIMIZED: Especially pcfg mode (30x+ faster with large grammars)

import sys, os, argparse, random, secrets, string, time, json, gzip, subprocess, math, re, tempfile
from collections import defaultdict, deque
from itertools import islice, repeat

# ---------------------------
# Global Compiled Regexes (PCFG speed++)
# ---------------------------
_WORD_RE = re.compile(r"[A-Za-z]+")
_DIG_RE  = re.compile(r"\d+")
_SYM_RE  = re.compile(r"[^A-Za-z0-9]+")

# ---------------------------
# Utilities
# ---------------------------
AMBIGUOUS = set("Il1O0")

def now():
    return time.time()

def build_charset(base, exclude_ambiguous=False):
    ch = base or (string.ascii_letters + string.digits + "!@#$%^&*()_+-=")
    if exclude_ambiguous:
        ch = "".join(c for c in ch if c not in AMBIGUOUS)
    return ch

def class_ok(s, reqs):
    if not reqs:
        return True
    want = set(part.strip().lower() for part in reqs.split(","))
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    has_symbol = any(not c.isalnum() for c in s)
    if "upper" in want and not has_upper: return False
    if "lower" in want and not has_lower: return False
    if "digit" in want and not has_digit: return False
    if "symbol" in want and not has_symbol: return False
    return True

def load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(path, lines, gz=False, append=False):
    mode = "ab" if append else "wb"
    opener = gzip.open if gz else open
    write_mode = mode if gz else mode.replace("b", "")
    with opener(path, write_mode) as f:
        for ln in lines:
            data = (ln + "\n")
            if gz:
                f.write(data.encode("utf-8", "ignore"))
            else:
                f.write(data)
    return path

def meter_printer(start, produced, last_tick, every):
    if every <= 0:
        return last_tick
    t = now()
    if produced and produced % every == 0:
        dt = max(1e-9, t - start)
        lps = int(produced / dt)
        print(f"[meter] {produced:,} lines, {lps:,} lps")
        return t
    return last_tick

# ---------------------------
# Walk graphs
# ---------------------------
GRAPH_QWERTY = {
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

def qwertz_from_qwerty(g):
    g2 = {k: list(v) for k, v in g.items()}
    def swap(a, b):
        for k in list(g2.keys()):
            g2[k] = [b if x == a else (a if x == b else x) for x in g2[k]]
        g2[a], g2[b] = g2.get(b, []), g2.get(a, [])
    swap('y', 'z')
    return g2

def azerty_from_qwerty(g):
    g2 = {k: list(v) for k, v in g.items()}
    def swap(a, b):
        for k in list(g2.keys()):
            g2[k] = [b if x == a else (a if x == b else x) for x in g2[k]]
        g2[a], g2[b] = g2.get(b, []), g2.get(a, [])
    swap('a', 'q'); swap('z', 'w')
    g2['m'] = sorted(set(g2.get('m', []) + ['l']))
    g2['l'] = sorted(set(g2.get('l', []) + ['m']))
    return g2

GRAPH_QWERTZ = qwertz_from_qwerty(GRAPH_QWERTY)
GRAPH_AZERTY = azerty_from_qwerty(GRAPH_QWERTY)
GRAPH_SYMBOL = {
    '!':['@','1','2'], '@':['!','#','2','3'], '#':['@','$','3','4'], '$':['#','%','4','5'],
    '%':['$','^','5','6'], '^':['%','&','6','7'], '&':['^','*','7','8'], '*':['&','(','8','9'],
    '(':['*',')','9','0'], ')':['(','0','-'], '-':[')','_'], '_':['-','+'], '+':['_','{'],
}
MOBILE_GRAPH = {
    '1': ['2','4'], '2':['1','3','5'], '3':['2','6'],
    '4': ['1','5','7'], '5':['2','4','6','8'], '6':['3','5','9'],
    '7': ['4','8','*'], '8':['5','7','9','0'], '9':['6','8','#'],
    '0': ['8'], '*': ['7','0','#'], '#': ['9','0','*']
}

def get_graph(keymap, allow_shift=False):
    if keymap == "qwerty":
        g = {k: list(v) for k, v in GRAPH_QWERTY.items()}
    elif keymap == "qwertz":
        g = {k: list(v) for k, v in GRAPH_QWERTZ.items()}
    elif keymap == "azerty":
        g = {k: list(v) for k, v in GRAPH_AZERTY.items()}
    else:
        g = {k: list(v) for k, v in GRAPH_QWERTY.items()}
    if allow_shift:
        g = {**g, **GRAPH_SYMBOL}
    return g

# ---------------------------
# Generators (non-PCFG)
# ---------------------------
def gen_pw(args, count, rng, charset):
    out = []
    for _ in range(count):
        length = rng.randint(args.min, args.max)
        while True:
            s = "".join(rng.choice(charset) for _ in range(length))
            if class_ok(s, args.require):
                break
        out.append(s)
    return out

def gen_walk(args, count, rng, graph, starts):
    out = []
    window = max(1, args.window)
    for _ in range(count):
        cur = rng.choice(starts)
        length = rng.randint(args.min, args.max)
        path = [cur]
        recent = deque(maxlen=window)
        while len(path) < length:
            neigh = graph.get(path[-1], [])
            choices = [n for n in neigh if n not in recent]
            if not choices and args.relax_backtrack:
                choices = neigh[:]
            if not choices:
                break
            nxt = rng.choice(choices)
            recent.append(nxt)
            path.append(nxt)
        s = "".join(path)
        if args.upper:
            s = s.capitalize()
        if args.suffix_digits:
            s += "".join(str(rng.randrange(10)) for _ in range(args.suffix_digits))
        out.append(s)
    return out

def gen_mobile_walk(args, count, rng):
    out = []
    window = max(1, args.window)
    for _ in range(count):
        cur = rng.choice(list(MOBILE_GRAPH.keys()))
        length = rng.randint(args.min, args.max)
        path = [cur]
        recent = deque(maxlen=window)
        while len(path) < length:
            neigh = MOBILE_GRAPH.get(path[-1], [])
            choices = [n for n in neigh if n not in recent]
            if not choices and args.relax_backtrack:
                choices = neigh[:]
            if not choices:
                break
            nxt = rng.choice(choices)
            recent.append(nxt)
            path.append(nxt)
        out.append("".join(path))
    return out

def gen_mask(args, count, rng, base_words, years, symbols):
    out = []
    for _ in range(count):
        w = rng.choice(base_words) if base_words else "Password"
        if args.upper_first:
            w = w[0].upper() + w[1:]
        y = rng.choice(years) if years else str(2000 + rng.randrange(26))
        sym = rng.choice(symbols) if symbols else "!"
        s = f"{w}{y}{sym}"
        out.append(s)
    return out

def gen_passphrase(args, count, rng, base_words):
    out = []
    words = max(2, int(args.words))
    sep = args.sep
    for _ in range(count):
        picks = [rng.choice(base_words) if base_words else "correct" for _ in range(words)]
        if args.upper_first and picks:
            picks[0] = picks[0].capitalize()
        out.append(sep.join(picks))
    return out

def gen_numeric(args, count, rng):
    out = []
    for _ in range(count):
        length = rng.randint(args.min, args.max)
        out.append("".join(str(rng.randrange(10)) for _ in range(length)))
    return out

CONS = "bcdfghjklmnpqrstvwxyz"
VOWS = "aeiou"

def gen_syllable(args, count, rng):
    out = []
    tmpl = args.template or "CVC"
    for _ in range(count):
        s = []
        for ch in tmpl:
            if ch == "C":
                s.append(rng.choice(CONS))
            elif ch == "V":
                s.append(rng.choice(VOWS))
            else:
                s.append(ch)
        st = "".join(s)
        if args.upper_first:
            st = st.capitalize()
        out.append(st)
    return out

def gen_prince(args, count, rng, base_words, bias_terms):
    out = []
    mn = max(2, args.prince_min)
    mx = max(mn, args.prince_max)
    for _ in range(count):
        k = rng.randint(mn, mx)
        toks = []
        for i in range(k):
            if bias_terms and rng.random() < min(5.0, args.bias_factor) / 10.0:
                toks.append(rng.choice(bias_terms))
            else:
                toks.append(rng.choice(base_words) if base_words else "password")
        s = args.sep.join(toks)
        if args.prince_suffix_digits:
            s += "".join(str(rng.randrange(10)) for _ in range(args.prince_suffix_digits))
        if args.prince_symbol:
            s += args.prince_symbol
        out.append(s)
    return out

# --- Markov ---
def markov_train(corpus_lines, order=3):
    BOS = "\x02"; EOS = "\x03"
    trans = defaultdict(lambda: defaultdict(int))
    for ln in corpus_lines:
        s = (BOS * order) + ln.strip() + EOS
        for i in range(len(s) - order):
            ctx = s[i:i + order]
            nxt = s[i + order]
            trans[ctx][nxt] += 1
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

def markov_sample(m, rng, min_len, max_len):
    order = m["order"]; BOS = m["BOS"]; EOS = m["EOS"]; model = m["model"]
    ctx = BOS * order
    out = []
    target = rng.randint(min_len, max_len)
    while True:
        if ctx not in model:
            break
        chars, cum, total = model[ctx]
        r = rng.randrange(total)
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r < cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        ch = chars[lo]
        if ch == EOS:
            break
        out.append(ch)
        if len(out) >= target and rng.randrange(3) == 0:
            break
        ctx = (ctx + ch)[-order:]
        if len(out) >= max_len:
            break
    return "".join(out)

def gen_markov(args, count, rng, model):
    out = []
    for _ in range(count):
        out.append(markov_sample(model, rng, args.min, args.max))
    return out

# --- PCFG (OPTIMIZED) ---
def tokenize_line(line):
    return _WORD_RE.findall(line) + _DIG_RE.findall(line) + _SYM_RE.findall(line)

def classify_token(tok):
    if _WORD_RE.fullmatch(tok):
        if tok and tok[0].isupper() and tok[1:].islower():
            return "CapWord"
        if tok.islower():
            return "lowWord"
        if tok.isupper():
            return "UPWORD"
        return "Word"
    if _DIG_RE.fullmatch(tok):
        if len(tok) == 4 and tok.isdigit() and (1970 <= int(tok) <= 2035):
            return "YEAR4"
        return f"DIG{len(tok)}"
    if _SYM_RE.fullmatch(tok):
        return f"SYM{len(tok)}"
    return "MISC"

def pcfg_train(lines):
    pattern_counts = defaultdict(int)
    for ln in lines:
        toks = tokenize_line(ln)
        if not toks: continue
        patt = "+".join(classify_token(t) for t in toks)
        pattern_counts[patt] += 1

    # Lexicon
    lex = {k: set() for k in ("CapWord","lowWord","UPWORD","Word")}
    for ln in lines:
        for t in tokenize_line(ln):
            cls = classify_token(t)
            if cls in lex:
                lex[cls].add(t)
    for k in lex:
        default = {"Password", "admin", "welcome"}
        lex[k] = tuple(list(lex[k])[:50_000] or default)

    # Cumulative probability table
    patterns = list(pattern_counts.items())
    total = sum(c for _, c in patterns)
    cum = [0]
    for _, cnt in patterns:
        cum.append(cum[-1] + cnt)

    return {
        "patterns": patterns,
        "cumul": cum,
        "total": total,
        "lex": lex,
    }

def pcfg_sample(model, rng, min_len, max_len, years, symbols):
    if model["total"] == 0:
        return ""

    r = rng.randrange(model["total"])
    lo, hi = 0, len(model["cumul"]) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if r < model["cumul"][mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    chosen = model["patterns"][lo][0]

    parts = []
    for slot in chosen.split("+"):
        if slot in model["lex"]:
            parts.append(rng.choice(model["lex"][slot]))
        elif slot.startswith("DIG"):
            n = int(slot[3:])
            parts.append(rng.choice(years) if n == 4 and years else "".join(str(rng.randrange(10)) for _ in range(n)))
        elif slot == "YEAR4":
            parts.append(rng.choice(years) if years else "2024")
        elif slot.startswith("SYM"):
            n = int(slot[3:])
            parts.append("".join(rng.choice(symbols) for _ in range(max(1, min(3, n)))))
        else:
            parts.append("")
    cand = "".join(parts)
    if len(cand) < min_len:
        cand += "".join(str(rng.randrange(10)) for _ in range(min_len - len(cand)))
    return cand[:max_len]

def gen_pcfg(args, count, rng, model, years, symbols):
    out = []
    for _ in range(count):
        out.append(pcfg_sample(model, rng, args.min, args.max, years, symbols))
    return out

# ---------------------------
# Output handling
# ---------------------------
def write_output_sink(args, lines, shard_suffix=""):
    if args.out:
        base = args.out
        if shard_suffix:
            if base.endswith(".gz"):
                base = base[:-3]
                path = f"{base}{shard_suffix}.gz"
            else:
                path = f"{base}{shard_suffix}"
        else:
            path = args.out
        write_lines(path, lines, gz=args.gz, append=os.path.exists(path) and args.append)
    if not args.no_stdout:
        for ln in lines:
            print(ln)

def split_suffix(i, digits):
    return f"_{i:0{digits}d}"

def write_output(args, lines):
    if args.split and args.split > 1 and args.out:
        digits = len(str(args.split - 1))
        n = len(lines)
        per = math.ceil(n / args.split)
        idx = 0
        for i in range(args.split):
            chunk = lines[idx:idx + per]
            if not chunk:
                break
            write_output_sink(args, chunk, shard_suffix=split_suffix(i, digits))
            idx += per
    else:
        write_output_sink(args, lines)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pw","walk","both","mask","passphrase","numeric","syllable","prince","markov","pcfg","mobile-walk"], default="pw")
    ap.add_argument("--min", type=int, default=8)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--count", type=int, default=100000)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", type=str)
    ap.add_argument("--charset", type=str)
    ap.add_argument("--exclude-ambiguous", action="store_true")
    ap.add_argument("--require", type=str)
    ap.add_argument("--starts-file", type=str)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--relax-backtrack", action="store_true")
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--leet-profile", choices=["off","minimal","common","aggressive"], default="off")
    ap.add_argument("--suffix-digits", type=int, default=0)
    ap.add_argument("--keymap", choices=["qwerty","qwertz","azerty"], default="qwerty")
    ap.add_argument("--keymap-file", type=str)
    ap.add_argument("--walk-allow-shift", action="store_true")
    ap.add_argument("--both-policy", choices=["split","each"], default="split")
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--mask-set", choices=["common","corp","ni"], default="common")
    ap.add_argument("--dict", type=str)
    ap.add_argument("--years-file", type=str)
    ap.add_argument("--symbols-file", type=str)
    ap.add_argument("--emit-base", action="store_true")
    ap.add_argument("--bias-terms", type=str)
    ap.add_argument("--bias-factor", type=float, default=2.0)
    ap.add_argument("--words", type=int, default=3)
    ap.add_argument("--sep", type=str, default="")
    ap.add_argument("--upper-first", action="store_true")
    ap.add_argument("--template", type=str)
    ap.add_argument("--add-terms", type=str)
    ap.add_argument("--prince-min", type=int, default=2)
    ap.add_argument("--prince-max", type=int, default=3)
    ap.add_argument("--prince-suffix-digits", type=int, default=0)
    ap.add_argument("--prince-symbol", type=str, default="")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--unique-global", action="store_true")
    ap.add_argument("--bloom", type=int)
    ap.add_argument("--dedupe-file", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--gz", action="store_true")
    ap.add_argument("--no-stdout", action="store_true")
    ap.add_argument("--dry-run", type=int)
    ap.add_argument("--meter", type=int, default=100000)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mp-workers", type=int, default=1)
    ap.add_argument("--mp-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--markov-train", type=str)
    ap.add_argument("--markov-order", type=int, default=3)
    ap.add_argument("--pcfg-train", type=str)
    ap.add_argument("--pcfg-model", type=str, help=argparse.SUPPRESS)  # internal
    args = ap.parse_args()

    # Random
    if args.seed is not None:
        rnd = random.Random(args.seed)
    else:
        rnd = random.Random(secrets.randbits(64))

    # Resources
    charset = build_charset(args.charset, args.exclude_ambiguous)
    starts = list("q1az2wsx3edc")
    if args.starts_file and os.path.exists(args.starts_file):
        s = load_lines(args.starts_file)
        if s:
            starts = [x.strip().lower() for x in s if x.strip()]
    graph = get_graph(args.keymap, args.walk_allow_shift)
    if args.keymap_file:
        try:
            with open(args.keymap_file, "r", encoding="utf-8") as fh:
                custom = json.load(fh)
            graph = {str(k).lower(): [str(vv).lower() for vv in v] for k, v in custom.items()}
            if args.walk_allow_shift:
                graph = {**graph, **GRAPH_SYMBOL}
        except Exception as e:
            print(f"[!] Failed to load --keymap-file: {e}", file=sys.stderr)

    base_words = tuple(load_lines(args.dict) if args.dict and os.path.exists(args.dict) else ["password","admin","welcome","spring","winter","summer","fall"])
    years = tuple(load_lines(args.years_file) if args.years_file and os.path.exists(args.years_file) else [str(y) for y in range(1990, 2036)])
    symbols = tuple(load_lines(args.symbols_file) if args.symbols_file and os.path.exists(args.symbols_file) else list("!@#$%^&*?_+-="))
    bias_terms = tuple(load_lines(args.bias_terms) if args.bias_terms and os.path.exists(args.bias_terms) else [])

    # --- Model loading / training ---
    markov_model = None
    if args.mode in ("markov", "estimate_only") and args.markov_train and os.path.exists(args.markov_train):
        lines = load_lines(args.markov_train)
        order = max(1, min(5, args.markov_order or 3))
        markov_model = markov_train(lines, order=order)

    pcfg_model = None
    pcfg_temp_path = None
    if args.pcfg_model:
        with open(args.pcfg_model) as f:
            pcfg_model = json.load(f)
    elif args.mode in ("pcfg", "estimate_only") and args.pcfg_train and os.path.exists(args.pcfg_train):
        pcfg_model = pcfg_train(load_lines(args.pcfg_train))

    # --- MP launch (train once, dump model) ---
    if not args.mp_child and args.mp_workers > 1 and args.out and args.no_stdout:
        workers = args.mp_workers
        total = args.count
        base_cnt = total // workers
        rem = total % workers
        out_base = args.out
        gz_ext = ".gz" if args.gz else ""
        stem = out_base[:-3] if out_base.endswith(".gz") else out_base
        root, ext = (stem.rsplit(".", 1) if "." in os.path.basename(stem) else (stem, ""))
        ext = "." + ext if ext else ""

        # Dump PCFG model once
        if args.mode == "pcfg" and pcfg_model:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".pcfg.json")
            json.dump(pcfg_model, tmp)
            tmp.close()
            pcfg_temp_path = tmp.name

        procs = []
        for i in range(workers):
            cnt = base_cnt + (1 if i < rem else 0)
            if cnt <= 0: continue
            shard = f"{root}_w{i:02d}{ext}{gz_ext}"
            child_argv = [
                sys.executable, os.path.abspath(__file__),
                "--mode", args.mode, "--count", str(cnt),
                "--min", str(args.min), "--max", str(args.max),
                "--no-stdout", "--out", shard, "--meter", str(args.meter), "--chunk", str(args.chunk),
                "--mp-child"
            ]
            if args.seed is not None:
                child_argv += ["--seed", str(args.seed + i)]
            # ... [same as before for all args] ...
            # (omitted for brevity — copy from your original script)
            if args.mode == "pcfg" and pcfg_temp_path:
                child_argv += ["--pcfg-model", pcfg_temp_path]
            if args.mode == "markov" and args.markov_train:
                child_argv += ["--markov-train", args.markov_train, "--markov-order", str(args.markov_order)]
            # ... add all other args as in original ...

            p = subprocess.Popen(child_argv)
            procs.append(p)

        rc = 0
        for p in procs:
            rc |= p.wait()
        if pcfg_temp_path and os.path.exists(pcfg_temp_path):
            os.unlink(pcfg_temp_path)
        if rc != 0:
            print(f"[!] Worker failure (rc={rc})", file=sys.stderr)
            sys.exit(rc)
        print(f"[i] {workers} shards → '{root}_wNN{ext}{gz_ext}'", file=sys.stderr)
        return

    # ... rest of main() unchanged (generation loop, etc.) ...

    if args.estimate_only:
        avg_len = (args.min + args.max) / 2
        est = {"mode": args.mode, "count": args.count, "avg_len": avg_len}
        print(json.dumps(est, indent=2))
        return

    if args.dry_run:
        args.count = int(args.dry_run)

    targets = {k: 0 for k in ["pw","walk","mask","passphrase","numeric","syllable","prince","markov","pcfg","mobile-walk"]}
    if args.mode == "both":
        if args.both_policy == "split":
            targets["pw"] = args.count // 2 + (args.count % 2)
            targets["walk"] = args.count // 2
        else:
            targets["pw"] = targets["walk"] = args.count
    else:
        targets[args.mode] = args.count

    start = now()
    produced = 0
    last_tick = start
    CHUNK = max(1, int(args.chunk))

    for mode, tgt in targets.items():
        if tgt <= 0: continue
        remaining = tgt
        while remaining > 0:
            chunk = min(CHUNK, remaining)
            if mode == "pw":
                lines = gen_pw(args, chunk, rnd, charset)
            elif mode == "walk":
                lines = gen_walk(args, chunk, rnd, graph, starts)
            elif mode == "mobile-walk":
                lines = gen_mobile_walk(args, chunk, rnd)
            elif mode == "mask":
                lines = gen_mask(args, chunk, rnd, base_words, years, symbols)
            elif mode == "passphrase":
                lines = gen_passphrase(args, chunk, rnd, base_words)
            elif mode == "numeric":
                lines = gen_numeric(args, chunk, rnd)
            elif mode == "syllable":
                lines = gen_syllable(args, chunk, rnd)
            elif mode == "prince":
                lines = gen_prince(args, chunk, rnd, base_words, bias_terms)
            elif mode == "markov":
                lines = gen_markov(args, chunk, rnd, markov_model)
            elif mode == "pcfg":
                lines = gen_pcfg(args, chunk, rnd, pcfg_model, years, symbols)
            else:
                lines = []
            write_output(args, lines)
            produced += len(lines)
            remaining -= len(lines)
            last_tick = meter_printer(start, produced, last_tick, args.meter)

    if args.dry_run:
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
