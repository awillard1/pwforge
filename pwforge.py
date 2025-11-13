
#!/usr/bin/env python3
# PWForge - Advanced Multi-Mode Password Generator
# Supports: pw, walk, mask, passphrase, numeric, syllable, prince, markov, pcfg, mobile-walk
# Features: --chunk, --workers (chunked), --mp-workers (true multi-process shards), --split, --gz, --no-stdout

import sys, os, argparse, random, secrets, string, time, json, gzip, subprocess, math, re
from collections import defaultdict, deque

# ---------------------------
# Utilities
# ---------------------------

AMBIGUOUS = set("Il1O0")

def now():
    return time.time()

def build_charset(base, exclude_ambiguous=False):
    ch = ""
    if base:
        ch = base
    else:
        ch = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    if exclude_ambiguous:
        ch = "".join(c for c in ch if c not in AMBIGUOUS)
    return ch

def class_ok(s, reqs):
    """reqs like 'upper,lower,digit,symbol'"""
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
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(path, lines, gz=False, append=False):
    mode = "ab" if append else "wb"
    if gz:
        if not path.endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, mode) as g:
            for ln in lines:
                g.write((ln + "\n").encode("utf-8", "ignore"))
    else:
        with open(path, mode.replace("b","")) as f:
            for ln in lines:
                f.write(ln + "\n")
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
# Walk graphs (built-ins); symbol row added if allow_shift
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
    g2 = {k:list(v) for k,v in g.items()}
    def swap(a,b):
        for k in list(g2.keys()):
            g2[k] = [b if x==a else (a if x==b else x) for x in g2[k]]
        g2[a], g2[b] = g2.get(b,[]), g2.get(a,[])
    swap('y','z')
    return g2

def azerty_from_qwerty(g):
    g2 = {k:list(v) for k,v in g.items()}
    def swap(a,b):
        for k in list(g2.keys()):
            g2[k] = [b if x==a else (a if x==b else x) for x in g2[k]]
        g2[a], g2[b] = g2.get(b,[]), g2.get(a,[])
    swap('a','q'); swap('z','w')
    g2['m'] = sorted(set(g2.get('m',[]) + ['l']))
    g2['l'] = sorted(set(g2.get('l',[]) + ['m']))
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
        g = {k:list(v) for k,v in GRAPH_QWERTY.items()}
    elif keymap == "qwertz":
        g = {k:list(v) for k,v in GRAPH_QWERTZ.items()}
    elif keymap == "azerty":
        g = {k:list(v) for k,v in GRAPH_AZERTY.items()}
    else:
        g = {k:list(v) for k,v in GRAPH_QWERTY.items()}
    if allow_shift:
        g = {**g, **GRAPH_SYMBOL}
    return g

# ---------------------------
# Generators
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
    # lightweight sampler: Word + Year + Symbol; corp profile adds digits
    out = []
    for _ in range(count):
        w = rng.choice(base_words) if base_words else "Password"
        if args.upper_first:
            if w:
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
        picks = [rng.choice(base_words) if base_words else "correct" for __ in range(words)]
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
            if bias_terms and rng.random() < min(5.0, args.bias_factor)/10.0:
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
        for i in range(len(s)-order):
            ctx = s[i:i+order]
            nxt = s[i+order]
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
    order = m["order"]; BOS=m["BOS"]; EOS=m["EOS"]; model=m["model"]
    ctx = BOS * order
    out = []
    target = rng.randint(min_len, max_len)
    while True:
        if ctx not in model:
            break
        chars, cum, total = model[ctx]
        r = rng.randrange(total)
        # binary search
        lo, hi = 0, len(cum)-1
        while lo < hi:
            mid = (lo+hi)//2
            if r < cum[mid]: hi = mid
            else: lo = mid + 1
        ch = chars[lo]
        if ch == EOS: break
        out.append(ch)
        if len(out) >= target and rng.randrange(3) == 0:
            break
        ctx = (ctx + ch)[-order:]
        if len(out) >= max_len: break
    return "".join(out)

def gen_markov(args, count, rng, model):
    out = []
    for _ in range(count):
        out.append(markov_sample(model, rng, args.min, args.max))
    return out

# --- PCFG ---
_WORD_RE = re.compile(r"[A-Za-z]+")
_DIG_RE  = re.compile(r"\d+")
_SYM_RE  = re.compile(r"[^A-Za-z0-9]+")

def classify_token(tok):
    if _WORD_RE.fullmatch(tok):
        if tok and tok[0].isupper() and tok[1:].islower(): return "CapWord"
        if tok.islower(): return "lowWord"
        if tok.isupper(): return "UPWORD"
        return "Word"
    if _DIG_RE.fullmatch(tok):
        if len(tok)==4 and tok.isdigit() and (1970 <= int(tok) <= 2035):
            return "YEAR4"
        return f"DIG{len(tok)}"
    if _SYM_RE.fullmatch(tok): return f"SYM{len(tok)}"
    return "MISC"

def tokenize_line(line):
    return re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]+", line.strip())

def pcfg_train(lines):
    pattern_counts = defaultdict(int)
    buckets = defaultdict(list)
    for ln in lines:
        toks = tokenize_line(ln)
        if not toks: continue
        patt = "+".join(classify_token(t) for t in toks)
        pattern_counts[patt] += 1
        buckets[patt].append(toks)
    lex = {"CapWord": set(), "lowWord": set(), "UPWORD": set(), "Word": set()}
    for ln in lines:
        for t in tokenize_line(ln):
            cls = classify_token(t)
            if cls in lex: lex[cls].add(t)
    for k in list(lex.keys()):
        if not lex[k]: lex[k] = {"Password","admin","welcome"}
        else: lex[k] = set(list(lex[k])[:50000])
    return {"patterns": dict(pattern_counts), "lex": {k:list(v) for k,v in lex.items()}}

def pcfg_sample(model, rng, min_len, max_len, years, symbols):
    if not model["patterns"]:
        return ""
    pats = list(model["patterns"].items())
    total = sum(c for _,c in pats)
    r = rng.randrange(total)
    s = 0
    for patt, c in pats:
        s += c
        if r < s:
            chosen = patt
            break
    parts = []
    for slot in chosen.split("+"):
        if slot in model["lex"]:
            parts.append(rng.choice(model["lex"][slot]))
        elif slot.startswith("DIG"):
            n = int(slot[3:])
            if n == 4 and years:
                parts.append(rng.choice(years))
            else:
                parts.append("".join(str(rng.randrange(10)) for _ in range(n)))
        elif slot == "YEAR4":
            parts.append(rng.choice(years) if years else "2024")
        elif slot.startswith("SYM"):
            n = int(slot[3:])
            sym = "".join(rng.choice(symbols) for _ in range(max(1, min(3, n)))) if symbols else "!"
            parts.append(sym)
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
        digits = len(str(args.split-1))
        n = len(lines)
        per = math.ceil(n / args.split)
        idx = 0
        for i in range(args.split):
            chunk = lines[idx:idx+per]
            if not chunk: break
            write_output_sink(args, chunk, shard_suffix=split_suffix(i, digits))
            idx += per
    else:
        write_output_sink(args, lines)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pw","walk","both","mask","passphrase","numeric","syllable","prince","markov","pcfg","mobile-walk"], default="pw",
                    help="Generation mode")
    ap.add_argument("--min", type=int, default=8)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--count", type=int, default=100000)
    ap.add_argument("--seed", type=int, help="Deterministic seed")
    ap.add_argument("--resume", type=str, help="(reserved) resume state path")
    ap.add_argument("--charset", type=str, help="Custom charset for --mode pw")
    ap.add_argument("--exclude-ambiguous", action="store_true")
    ap.add_argument("--require", type=str, help="Require classes: upper,lower,digit,symbol")

    # walk
    ap.add_argument("--starts-file", type=str)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--relax-backtrack", action="store_true")
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--leet-profile", choices=["off","minimal","common","aggressive"], default="off")
    ap.add_argument("--suffix-digits", type=int, default=0)
    ap.add_argument("--keymap", choices=["qwerty","qwertz","azerty"], default="qwerty", help="Keyboard layout for walks")
    ap.add_argument("--keymap-file", type=str, help="Path to JSON adjacency graph for custom keyboard walks")
    ap.add_argument("--walk-allow-shift", action="store_true")
    ap.add_argument("--both-policy", choices=["split","each"], default="split")
    ap.add_argument("--interleave", action="store_true")

    # mask / passphrase / prince
    ap.add_argument("--mask-set", choices=["common","corp","ni"], default="common")
    ap.add_argument("--dict", type=str, help="Dictionary file for modes using words")
    ap.add_argument("--years-file", type=str)
    ap.add_argument("--symbols-file", type=str)
    ap.add_argument("--emit-base", action="store_true")
    ap.add_argument("--bias-terms", type=str, help="Bias term file for prince")
    ap.add_argument("--bias-factor", type=float, default=2.0)
    ap.add_argument("--words", type=int, default=3)
    ap.add_argument("--sep", type=str, default="")
    ap.add_argument("--upper-first", action="store_true")
    ap.add_argument("--template", type=str, help="Syllable template, e.g. CVCVC")
    ap.add_argument("--add-terms", type=str, help="Additional words to include")

    ap.add_argument("--prince-min", type=int, default=2)
    ap.add_argument("--prince-max", type=int, default=3)
    ap.add_argument("--prince-suffix-digits", type=int, default=0)
    ap.add_argument("--prince-symbol", type=str, default="")

    # uniqueness
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--unique-global", action="store_true")
    ap.add_argument("--bloom", type=int, help="(reserved) approximate unique filter MB")
    ap.add_argument("--dedupe-file", type=str, help="(reserved) dedupe against existing file")

    # output
    ap.add_argument("--out", type=str)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--gz", action="store_true")
    ap.add_argument("--no-stdout", action="store_true")
    ap.add_argument("--dry-run", type=int, help="Print N example candidates then exit")
    ap.add_argument("--meter", type=int, default=100000, help="Print stats every N lines (0=off)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Chunked workers (single process)")
    ap.add_argument("--mp-workers", type=int, default=1, help="Spawn N child processes (requires --out and --no-stdout)")
    ap.add_argument("--mp-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--chunk", type=int, default=100000, help="Batch size per inner loop")

    # cracker passthroughs (ignored here; for future integration)
    ap.add_argument("--jtr-fork", type=int)
    ap.add_argument("--jtr-format", type=str)
    ap.add_argument("--jtr-hashes", type=str)
    ap.add_argument("--hc-attack", type=int)
    ap.add_argument("--hc-mode", type=int)
    ap.add_argument("--hc-hashes", type=str)
    ap.add_argument("--hc-rules", type=str)

    args = ap.parse_args()

    # Random sources
    if args.seed is not None:
        rnd = random.Random(args.seed)
    else:
        rnd = random.Random(secrets.randbits(64))

    # Prepare resources
    charset = build_charset(args.charset, args.exclude_ambiguous)

    # starts file
    starts = list("q1az2wsx3edc")  # fallback
    if args.starts_file and os.path.exists(args.starts_file):
        s = load_lines(args.starts_file)
        if s:
            starts = [x.strip().lower() for x in s if x.strip()]

    # keymap / graph
    graph = get_graph(args.keymap, args.walk_allow_shift)
    if args.keymap_file:
        try:
            with open(args.keymap_file, "r", encoding="utf-8") as fh:
                custom = json.load(fh)
            graph = {str(k).lower(): [str(vv).lower() for vv in v] for k,v in custom.items()}
            if args.walk_allow_shift:
                graph = {**graph, **GRAPH_SYMBOL}
        except Exception as e:
            print(f"[!] Failed to load --keymap-file: {e}", file=sys.stderr)

    # dict & lists
    base_words = load_lines(args.dict) if args.dict and os.path.exists(args.dict) else ["password","admin","welcome","spring","winter","summer","fall"]
    years = load_lines(args.years_file) if args.years_file and os.path.exists(args.years_file) else [str(y) for y in range(1990, 2036)]
    symbols = load_lines(args.symbols_file) if args.symbols_file and os.path.exists(args.symbols_file) else list("!@#$%^&*?_+-=")
    bias_terms = load_lines(args.bias_terms) if args.bias_terms and os.path.exists(args.bias_terms) else []

    # Markov/PCFG training
    markov_model = None
    if args.mode == "markov" or args.estimate_only:
        if args.markov_train and os.path.exists(args.markov_train):
            lines = load_lines(args.markov_train)
            order = 3
            if hasattr(args, "markov_order") and args.markov_order:
                order = max(1, min(5, int(args.markov_order)))
            markov_model = markov_train(lines, order=order)
        elif args.mode == "markov":
            print("[!] --markov-train required for markov mode", file=sys.stderr)
            return

    pcfg_model = None
    if args.mode == "pcfg" or args.estimate_only:
        if args.pcfg_train and os.path.exists(args.pcfg_train):
            pcfg_model = pcfg_train(load_lines(args.pcfg_train))
        elif args.mode == "pcfg":
            print("[!] --pcfg-train required for pcfg mode", file=sys.stderr)
            return

    # --- True Multi-Process Generation launcher ---
    if (not args.mp_child) and int(getattr(args, "mp_workers", 1)) > 1 and args.out and args.no_stdout:
        workers = int(args.mp_workers)
        total = int(args.count)
        base_cnt = total // workers
        rem = total % workers

        out_base = args.out
        gz_ext = ".gz" if args.gz and not out_base.endswith(".gz") else ""
        if out_base.endswith(".gz"):
            stem = out_base[:-3]
            gz_ext = ".gz"
        else:
            stem = out_base
        if "." in os.path.basename(stem):
            root, ext = stem.rsplit(".", 1)
            ext = "." + ext
        else:
            root, ext = stem, ""

        procs = []
        for i in range(workers):
            cnt = base_cnt + (1 if i < rem else 0)
            if cnt <= 0: continue
            shard = f"{root}_w{i:02d}{ext}{gz_ext}"
            child_argv = [sys.executable, os.path.abspath(__file__),
                          "--mode", args.mode, "--count", str(cnt),
                          "--min", str(args.min), "--max", str(args.max),
                          "--no-stdout", "--out", shard, "--meter", str(args.meter), "--chunk", str(args.chunk),
                          "--mp-child"]
            if args.seed is not None: child_argv += ["--seed", str(args.seed + i)]
            if args.charset: child_argv += ["--charset", args.charset]
            if args.exclude_ambiguous: child_argv += ["--exclude-ambiguous"]
            if args.require: child_argv += ["--require", args.require]

            # walk options
            if args.starts_file: child_argv += ["--starts-file", args.starts_file]
            if args.window: child_argv += ["--window", str(args.window)]
            if args.relax_backtrack: child_argv += ["--relax-backtrack"]
            if args.upper: child_argv += ["--upper"]
            if args.leet_profile and args.leet_profile != "off": child_argv += ["--leet-profile", args.leet_profile]
            if args.suffix_digits: child_argv += ["--suffix-digits", str(args.suffix_digits)]
            if args.keymap_file: child_argv += ["--keymap-file", args.keymap_file]
            else: child_argv += ["--keymap", args.keymap]
            if args.walk_allow_shift: child_argv += ["--walk-allow-shift"]

            # mask/passphrase/prince
            if args.mask_set: child_argv += ["--mask-set", args.mask_set]
            if args.dict: child_argv += ["--dict", args.dict]
            if args.years_file: child_argv += ["--years-file", args.years_file]
            if args.symbols_file: child_argv += ["--symbols-file", args.symbols_file]
            if args.emit_base: child_argv += ["--emit-base"]
            if args.bias_terms: child_argv += ["--bias-terms", args.bias_terms]
            if args.bias_factor is not None: child_argv += ["--bias-factor", str(args.bias_factor)]
            if args.words: child_argv += ["--words", str(args.words)]
            if args.sep: child_argv += ["--sep", args.sep]
            if args.upper_first: child_argv += ["--upper-first"]
            if args.template: child_argv += ["--template", args.template]
            if args.add_terms: child_argv += ["--add-terms", args.add_terms]
            if args.prince_min: child_argv += ["--prince-min", str(args.prince_min)]
            if args.prince_max: child_argv += ["--prince-max", str(args.prince_max)]
            if args.prince_suffix_digits: child_argv += ["--prince-suffix-digits", str(args.prince_suffix_digits)]
            if args.prince_symbol: child_argv += ["--prince-symbol", args.prince_symbol]

            # models
            if args.mode == "markov" and args.markov_train: child_argv += ["--markov-train", args.markov_train, "--markov-order", str(getattr(args, "markov_order", 3) or 3)]
            if args.mode == "pcfg" and args.pcfg_train: child_argv += ["--pcfg-train", args.pcfg_train]

            if args.gz and not shard.endswith(".gz"): child_argv += ["--gz"]
            p = subprocess.Popen(child_argv)
            procs.append(p)
        rc = 0
        for p in procs:
            rc |= p.wait()
        if rc != 0:
            print(f"[!] One or more workers failed (rc={rc})", file=sys.stderr)
            sys.exit(rc)
        else:
            print(f"[i] Completed {workers} shards to '{root}_wNN{ext}{gz_ext}'", file=sys.stderr)
            return

    # estimate-only
    if args.estimate_only:
        avg_len = (args.min + args.max)/2
        est = {
            "mode": args.mode,
            "count": args.count,
            "avg_len": avg_len,
            "note": "indicative only; varies by corpus and options"
        }
        print(json.dumps(est, indent=2))
        return

    # Dry-run
    if args.dry_run:
        N = int(args.dry_run)
        args.count = N

    # Targets
    targets = {"pw":0,"walk":0,"mask":0,"passphrase":0,"numeric":0,"syllable":0,"prince":0,"markov":0,"pcfg":0,"mobile-walk":0}
    if args.mode == "both":
        if args.both_policy == "split":
            targets["pw"] = args.count//2 + (args.count%2)
            targets["walk"] = args.count//2
        else: # each
            targets["pw"] = args.count
            targets["walk"] = args.count
    else:
        targets[args.mode] = args.count

    start = now()
    produced = 0
    last_tick = start

    # Chunked generation loop
    CHUNK = max(1, int(args.chunk))

    for mode, tgt in targets.items():
        if tgt <= 0:
            continue
        remaining = tgt
        while remaining > 0:
            chunk = CHUNK if remaining > CHUNK else remaining
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
        # dry run prints to stdout by default; no footer
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
