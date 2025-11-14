#!/usr/bin/env python3
# PWForge – Ultimate Password Generator (robust + deduped)
# -------------------------------------------------
import sys, os, argparse, random, secrets, string, time, json, gzip, math, re
from collections import defaultdict, deque, Counter
from typing import List, Tuple, Any, Dict, Set, Optional

# -------------------------------------------------
# Safe imports
# -------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    print(f"[!] PyTorch import failed ({e}). Neural mode disabled.", file=sys.stderr)
    TORCH_AVAILABLE = False
    nn = F = None

# -------------------------------------------------
# Regexes
# -------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z]+")
_DIG_RE  = re.compile(r"\d+")
_SYM_RE  = re.compile(r"[^A-Za-z0-9]+")

# -------------------------------------------------
# Utils
# -------------------------------------------------
AMBIGUOUS = set("Il1O0")

def now() -> float:
    return time.time()

def build_charset(base: Optional[str], exclude_ambiguous: bool = False) -> str:
    ch = base or (string.ascii_letters + string.digits + "!@#$%^&*()_+-=")
    if exclude_ambiguous:
        ch = "".join(c for c in ch if c not in AMBIGUOUS)
    return ch

def class_ok(s: str, reqs: Optional[str]) -> bool:
    if not reqs:
        return True
    want = {p.strip().lower() for p in reqs.split(",")}
    has = {
        "upper": any(c.isupper() for c in s),
        "lower": any(c.islower() for c in s),
        "digit": any(c.isdigit() for c in s),
        "symbol": any(not c.isalnum() for c in s)
    }
    return all(has.get(k, True) for k in want)

def load_lines(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    opener = gzip.open if path.lower().endswith(('.gz', '.gzip')) else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"[!] Failed to read {path}: {e}", file=sys.stderr)
        return []

def write_lines(path: str, lines: List[str], gz: bool = False, append: bool = False) -> None:
    mode = "ab" if append else "wb"
    opener = gzip.open if gz else open
    wmode = mode if gz else mode.replace("b", "t")
    try:
        with opener(path, wmode, encoding="utf-8" if not gz else None) as f:
            for ln in lines:
                data = ln + "\n"
                if gz:
                    f.write(data.encode("utf-8"))
                else:
                    f.write(data)
    except Exception as e:
        print(f"[!] Failed to write {path}: {e}", file=sys.stderr)

def meter_printer(start: float, produced: int, last_tick: float, every: int) -> float:
    if every <= 0:
        return last_tick
    t = now()
    if produced and produced % every == 0:
        dt = max(1e-9, t - start)
        lps = int(produced / dt)
        print(f"[meter] {produced:,} lines, {lps:,} lps", file=sys.stderr)
        return t
    return last_tick

def load_dict_set(path: Optional[str]) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    return {ln.strip().lower() for ln in load_lines(path) if ln.strip()}

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    L = len(s)
    return -sum((c/L)*math.log2(c/L) for c in cnt.values())

def apply_entropy_filter(lines: List[str], min_entropy: float = 0.0, dict_set: Optional[Set[str]] = None) -> List[str]:
    if min_entropy <= 0 and not dict_set:
        return lines
    return [l for l in lines
            if (min_entropy <= 0 or shannon_entropy(l) >= min_entropy)
            and (not dict_set or l.lower() not in dict_set)]

# -------------------------------------------------
# Keyboard Graphs
# -------------------------------------------------
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

def get_graph(keymap: str, allow_shift: bool = False) -> Dict[str, List[str]]:
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

# -------------------------------------------------
# Generators (all with dedup support)
# -------------------------------------------------
def gen_pw(args, count: int, rng: random.Random, charset: str, seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        length = rng.randint(args.min, args.max)
        while True:
            s = "".join(rng.choice(charset) for _ in range(length))
            if class_ok(s, args.require):
                break
        if s not in seen:
            seen.add(s)
            out.append(s)
    if attempts >= max_attempts:
        print(f"[!] gen_pw exhausted attempts, produced {len(out)}/{count}", file=sys.stderr)
    return out

def gen_walk(args, count: int, rng: random.Random, graph: Dict[str, List[str]],
             starts: List[str], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    window = max(1, args.window)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
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
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def gen_mobile_walk(args, count: int, rng: random.Random, seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    window = max(1, args.window)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
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
        s = "".join(path)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def gen_mask(args, count: int, rng: random.Random, base_words: Tuple[str, ...], years: Tuple[str, ...], symbols: Tuple[str, ...], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        w = rng.choice(base_words) if base_words else "Password"
        if args.upper_first:
            w = w[0].upper() + w[1:]
        y = rng.choice(years) if years else str(2000 + rng.randrange(26))
        sym = rng.choice(symbols) if symbols else "!"
        s = f"{w}{y}{sym}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def gen_passphrase(args, count: int, rng: random.Random, base_words: Tuple[str, ...], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    words = max(2, int(args.words))
    sep = args.sep
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        picks = [rng.choice(base_words) if base_words else "correct" for _ in range(words)]
        if args.upper_first and picks:
            picks[0] = picks[0].capitalize()
        s = sep.join(picks)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def gen_numeric(args, count: int, rng: random.Random, seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        length = rng.randint(args.min, args.max)
        s = "".join(str(rng.randrange(10)) for _ in range(length))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

CONS = "bcdfghjklmnpqrstvwxyz"
VOWS = "aeiou"

def gen_syllable(args, count: int, rng: random.Random, seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    tmpl = args.template or "CVC"
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        s = []
        for ch in tmpl:
            if ch == "C": s.append(rng.choice(CONS))
            elif ch == "V": s.append(rng.choice(VOWS))
            else: s.append(ch)
        st = "".join(s)
        if args.upper_first: st = st.capitalize()
        if st not in seen:
            seen.add(st)
            out.append(st)
    return out

def gen_prince(args, count: int, rng: random.Random, base_words: Tuple[str, ...], bias_terms: Tuple[str, ...], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    mn = max(2, args.prince_min)
    mx = max(mn, args.prince_max)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
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
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def markov_train(corpus_lines: List[str], order: int = 3) -> Dict[str, Any]:
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

def markov_sample(m: Dict[str, Any], rng: random.Random, min_len: int, max_len: int) -> str:
    order = m["order"]; BOS = m["BOS"]; EOS = m["EOS"]; model = m["model"]
    ctx = BOS * order
    out = []
    target = rng.randint(min_len, max_len)
    while True:
        if ctx not in model: break
        chars, cum, total = model[ctx]
        r = rng.randrange(total)
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r < cum[mid]: hi = mid
            else: lo = mid + 1
        ch = chars[lo]
        if ch == EOS: break
        out.append(ch)
        if len(out) >= target and rng.randrange(3) == 0: break
        ctx = (ctx + ch)[-order:]
        if len(out) >= max_len: break
    return "".join(out)

def gen_markov(args, count: int, rng: random.Random, model: Optional[Dict[str, Any]], seen: Set[str]) -> List[str]:
    if not model:
        return []
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        s = markov_sample(model, rng, args.min, args.max)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def tokenize_line(line: str) -> List[str]:
    return _WORD_RE.findall(line) + _DIG_RE.findall(line) + _SYM_RE.findall(line)

def classify_token(tok: str) -> str:
    if _WORD_RE.fullmatch(tok):
        if tok and tok[0].isupper() and tok[1:].islower(): return "CapWord"
        if tok.islower(): return "lowWord"
        if tok.isupper(): return "UPWORD"
        return "Word"
    if _DIG_RE.fullmatch(tok):
        if len(tok)==4 and tok.isdigit() and 1970<=int(tok)<=2035: return "YEAR4"
        return f"DIG{len(tok)}"
    if _SYM_RE.fullmatch(tok):
        return f"SYM{len(tok)}"
    return "MISC"

def pcfg_train(lines: List[str]) -> Dict[str, Any]:
    pattern_counts = defaultdict(int)
    for ln in lines:
        toks = tokenize_line(ln)
        if not toks: continue
        patt = "+".join(classify_token(t) for t in toks)
        pattern_counts[patt] += 1
    lex = {k: set() for k in ("CapWord","lowWord","UPWORD","Word")}
    for ln in lines:
        for t in tokenize_line(ln):
            cls = classify_token(t)
            if cls in lex: lex[cls].add(t)
    for k in lex:
        default = {"Password","admin","welcome"}
        lex[k] = tuple(list(lex[k])[:50_000] or default)
    patterns = list(pattern_counts.items())
    total = sum(c for _,c in patterns)
    cum = [0]
    for _,cnt in patterns: cum.append(cum[-1] + cnt)
    return {"patterns":patterns, "cumul":cum, "total":total, "lex":lex}

def pcfg_sample(model: Dict[str, Any], rng: random.Random, min_len: int, max_len: int, years: Tuple[str, ...], symbols: Tuple[str, ...]) -> str:
    if model["total"] == 0: return ""
    r = rng.randrange(model["total"])
    lo, hi = 0, len(model["cumul"])-2
    while lo < hi:
        mid = (lo+hi)//2
        if r < model["cumul"][mid+1]: hi = mid
        else: lo = mid+1
    chosen = model["patterns"][lo][0]
    parts = []
    for slot in chosen.split("+"):
        if slot in model["lex"]:
            parts.append(rng.choice(model["lex"][slot]))
        elif slot.startswith("DIG"):
            n = int(slot[3:])
            parts.append(rng.choice(years) if n==4 and years else "".join(str(rng.randrange(10)) for _ in range(n)))
        elif slot == "YEAR4":
            parts.append(rng.choice(years) if years else "2024")
        elif slot.startswith("SYM"):
            n = int(slot[3:])
            parts.append("".join(rng.choice(symbols) for _ in range(max(1,min(3,n)))))
        else:
            parts.append("")
    cand = "".join(parts)
    if len(cand) < min_len:
        cand += "".join(str(rng.randrange(10)) for _ in range(min_len-len(cand)))
    return cand[:max_len]

def gen_pcfg(args, count: int, rng: random.Random, model: Optional[Dict[str, Any]], years: Tuple[str, ...], symbols: Tuple[str, ...], seen: Set[str]) -> List[str]:
    if not model:
        return []
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        s = pcfg_sample(model, rng, args.min, args.max, years, symbols)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def apply_rule(word: str, rule: str) -> str:
    if not rule: return word
    out = list(word)
    i = 0
    while i < len(rule):
        cmd = rule[i]
        if cmd == ':': i += 1; continue
        elif cmd == 'l': out = [c.lower() for c in out]
        elif cmd == 'u': out = [c.upper() for c in out]
        elif cmd == 'c':
            if out: out[0] = out[0].upper(); out[1:] = [c.lower() for c in out[1:]]
        elif cmd == 't': out = [c.swapcase() for c in out]
        elif cmd == 'r': out = out[::-1]
        elif cmd == 'd': out = out + out
        elif cmd == 'p':
            n = 2
            if i+1 < len(rule) and rule[i+1].isdigit():
                n = int(rule[i+1]); i += 1
            out = out * n
        elif cmd == '$':
            j = i + 1
            while j < len(rule) and rule[j] not in 'luctrdp{}:': j += 1
            out.extend(list(rule[i+1:j])); i = j - 1
        elif cmd == '^':
            j = i + 1
            while j < len(rule) and rule[j] not in 'luctrdp{}:': j += 1
            out = list(rule[i+1:j]) + out; i = j - 1
        elif cmd == 's' and i+2 < len(rule):
            old, new = rule[i+1], rule[i+2]
            out = [new if c == old else c for c in out]; i += 2
            continue
        i += 1
    return ''.join(out)

def gen_hybrid(args, count: int, rng: random.Random, base_words: Tuple[str, ...], rules: List[str], mask: Optional[str], years: Tuple[str, ...], symbols: Tuple[str, ...], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        word = rng.choice(base_words)
        if rules:
            word = apply_rule(word, rng.choice(rules))
        if mask:
            parts = []
            for c in mask:
                if c == '?l': parts.append(rng.choice(string.ascii_lowercase))
                elif c == '?u': parts.append(rng.choice(string.ascii_uppercase))
                elif c == '?d': parts.append(rng.choice(string.digits))
                elif c == '?s': parts.append(rng.choice(symbols))
                elif c == '?y': parts.append(rng.choice(years))
                elif c == '?w': parts.append(rng.choice(base_words))
                else: parts.append(c)
            mstr = ''.join(parts)
            cand = word + mstr if rng.random() < 0.5 else mstr + word
        else:
            cand = word
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out

def parse_combo(spec: Optional[str]) -> List[Tuple[str, float]]:
    if not spec: return []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    modes, weights = [], []
    for p in parts:
        if ":" not in p: continue
        m, w = p.split(":", 1)
        try:
            ww = float(w)
            if ww > 0: modes.append(m); weights.append(ww)
        except: continue
    total = sum(weights)
    if total == 0: return []
    return list(zip(modes, [w/total for w in weights]))

def gen_combo(args, count: int, rng: random.Random, mode_weights: List[Tuple[str, float]], gen_map: Dict[str, Any], base_words: Tuple[str, ...], years: Tuple[str, ...], symbols: Tuple[str, ...], rules: List[str], mask: Optional[str], graph: Dict[str, List[str]], starts: List[str], model: Optional[Dict[str, Any]], bias_terms: Tuple[str, ...], seen: Set[str]) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = count * 10
    mode_names = [m for m, _ in mode_weights]
    probs = [w for _, w in mode_weights]
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        mode = rng.choices(mode_names, probs)[0]
        gen = gen_map.get(mode)
        if gen:
            if mode == "pcfg":
                cand = gen(args, 1, rng, model, years, symbols, seen)[0]
            elif mode == "walk":
                cand = gen(args, 1, rng, graph, starts, seen)[0]
            elif mode == "prince":
                cand = gen(args, 1, rng, base_words, bias_terms, seen)[0]
            elif mode == "mask":
                cand = gen(args, 1, rng, base_words, years, symbols, seen)[0]
            elif mode == "passphrase":
                cand = gen(args, 1, rng, base_words, seen)[0]
            elif mode == "hybrid":
                cand = gen(args, 1, rng, base_words, rules, mask, years, symbols, seen)[0]
            else:
                cand = gen(args, 1, rng, seen)[0]
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out

# -------------------------------------------------
# Neural
# -------------------------------------------------
PRINTABLE = ''.join(chr(i) for i in range(33,127))
CHAR_TO_IDX = {c:i for i,c in enumerate(PRINTABLE)}
IDX_TO_CHAR = {i:c for c,i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(PRINTABLE)
BOS_IDX = CHAR_TO_IDX['!']
EOS_IDX = CHAR_TO_IDX.get('\n', 0)

if TORCH_AVAILABLE and nn is not None:
    class CharLSTM(nn.Module):
        def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=384, layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            return self.fc(out), hidden

    @torch.no_grad()
    def gen_neural(args, count: int, rng: random.Random, model_path: Optional[str],
                   device: Optional[torch.device], seen: Set[str]) -> List[str]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[i] Neural mode → {device}", file=sys.stderr)
        if not model_path or not os.path.exists(model_path):
            print("[!] --model path.pt required. Train with finetune_neural.py", file=sys.stderr)
            return []
        model = CharLSTM(
            embed_dim=getattr(args, 'embed_dim', 128),
            hidden_dim=getattr(args, 'hidden_dim', 384),
            layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.3)
        ).to(device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[!] Failed to load model: {e}", file=sys.stderr)
            return []
        model.eval()
        torch.manual_seed(rng.getrandbits(64))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(rng.getrandbits(64))
        batch = min(getattr(args, 'batch_size', 512), count)
        out = []
        max_gen = getattr(args, "max_gen_len", 32) or 32
        while len(out) < count:
            need = min(batch, count - len(out))
            seqs = [torch.tensor([[BOS_IDX]], dtype=torch.long).to(device) for _ in range(need)]
            hiddens = [(None, None)] * need
            for _ in range(max_gen):
                seq_batch = torch.cat(seqs, dim=0)
                hidden_input = None if hiddens[0][0] is None else tuple(torch.cat(tensors, dim=1) for tensors in zip(*hiddens))
                logits, new_hiddens = model(seq_batch, hidden_input)
                probs = F.softmax(logits[:, -1, :], dim=-1).cpu().numpy()
                nxt = []
                for p in probs:
                    nxt.append(rng.choices(range(len(p)), weights=p, k=1)[0])
                nxt = torch.tensor(nxt, dtype=torch.long).unsqueeze(1).to(device)
                for i in range(need):
                    seqs[i] = torch.cat([seqs[i], nxt[i:i+1]], dim=1)
                    h0_i = new_hiddens[0][:, i:i+1, :].contiguous()
                    c0_i = new_hiddens[1][:, i:i+1, :].contiguous()
                    hiddens[i] = (h0_i, c0_i)
                if (nxt == EOS_IDX).any():
                    break
            for i in range(need):
                pw = ''.join(IDX_TO_CHAR.get(t.item(), '') for t in seqs[i][0, 1:])
                if len(pw) < args.min:
                    pw += ''.join(rng.choice(PRINTABLE) for _ in range(args.min - len(pw)))
                pw = pw[:args.max]
                if len(pw) >= args.min and pw not in seen:
                    seen.add(pw)
                    out.append(pw)
                if len(out) >= count:
                    break
        return out[:count]
else:
    def gen_neural(*args, **kwargs) -> List[str]:
        print("[!] PyTorch not available – neural mode disabled", file=sys.stderr)
        return []

# -------------------------------------------------
# Bloom (optional)
# -------------------------------------------------
try:
    from bloom_filter import BloomFilter
    BLOOM_AVAILABLE = True
except Exception:
    BLOOM_AVAILABLE = False

def load_or_create_bloom(path: Optional[str], capacity: int = 10_000_000) -> Any:
    if not BLOOM_AVAILABLE:
        return None
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return BloomFilter.load(f)
        except Exception as e:
            print(f"[!] Bloom load failed ({e}), falling back to set", file=sys.stderr)
    bf = BloomFilter(max_elements=capacity, error_rate=0.001)
    if path:
        try:
            with open(path, "wb") as f:
                bf.dump(f)
        except Exception:
            pass
    return bf

def bloom_add(bf: Any, item: str) -> bool:
    if bf is None:
        return False
    bf.add(item)
    return True

# -------------------------------------------------
# Output
# -------------------------------------------------
def write_output_sink(args, lines: List[str], shard_suffix: str = "") -> None:
    if not args.out and not args.no_stdout:
        for ln in lines:
            print(ln)
        return

    base = args.out or ""
    is_gz = base.lower().endswith(".gz")
    path = base
    if shard_suffix:
        if is_gz:
            base = base[:-3]
            path = f"{base}{shard_suffix}.gz"
        else:
            path = f"{base}{shard_suffix}"

    append_mode = args.append and os.path.exists(path)
    write_lines(path, lines, gz=is_gz, append=append_mode)

    if not args.no_stdout:
        for ln in lines:
            print(ln)

def split_suffix(i: int, digits: int) -> str:
    return f"_{i:0{digits}d}"

def write_output(args, lines: List[str]) -> None:
    if not lines:
        return
    if args.split and args.split > 1 and args.out:
        digits = len(str(args.split - 1))
        per = math.ceil(len(lines) / args.split)
        idx = 0
        for i in range(args.split):
            chunk = lines[idx:idx + per]
            if not chunk:
                break
            suffix = split_suffix(i, digits)
            write_output_sink(args, chunk, shard_suffix=suffix)
            idx += per
    else:
        write_output_sink(args, lines)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PWForge – Multi-mode password generator (robust + dedup)")
    ap.add_argument("--mode", choices=[
        "pw","walk","both","mask","passphrase","numeric","syllable","prince",
        "markov","pcfg","mobile-walk","hybrid","combo","neural"
    ], default="pw")
    ap.add_argument("--min", type=int, default=8)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--count", type=int, default=100000)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--charset", type=str)
    ap.add_argument("--exclude-ambiguous", action="store_true")
    ap.add_argument("--require", type=str)
    ap.add_argument("--starts-file", type=str)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--relax-backtrack", action="store_true")
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--suffix-digits", type=int, default=0)
    ap.add_argument("--keymap", choices=["qwerty","qwertz","azerty"], default="qwerty")
    ap.add_argument("--keymap-file", type=str)
    ap.add_argument("--walk-allow-shift", action="store_true")
    ap.add_argument("--dict", type=str)
    ap.add_argument("--years-file", type=str)
    ap.add_argument("--symbols-file", type=str)
    ap.add_argument("--bias-terms", type=str)
    ap.add_argument("--bias-factor", type=float, default=2.0)
    ap.add_argument("--words", type=int, default=3)
    ap.add_argument("--sep", type=str, default="")
    ap.add_argument("--upper-first", action="store_true")
    ap.add_argument("--template", type=str)
    ap.add_argument("--prince-min", type=int, default=2)
    ap.add_argument("--prince-max", type=int, default=3)
    ap.add_argument("--prince-suffix-digits", type=int, default=0)
    ap.add_argument("--prince-symbol", type=str, default="")
    ap.add_argument("--rules", type=str)
    ap.add_argument("--mask", type=str)
    ap.add_argument("--combo", type=str)
    ap.add_argument("--min-entropy", type=float, default=0.0)
    ap.add_argument("--no-dict", type=str)
    ap.add_argument("--model", type=str)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-gen-len", type=int, default=32)
    ap.add_argument("--out", type=str)
    ap.add_argument("--append", action="store_true", help="Append to existing output file(s)")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--gz", action="store_true")
    ap.add_argument("--no-stdout", action="store_true")
    ap.add_argument("--dry-run", type=int)
    ap.add_argument("--meter", type=int, default=100000)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--markov-train", type=str)
    ap.add_argument("--markov-order", type=int, default=3)
    ap.add_argument("--pcfg-train", type=str)
    ap.add_argument("--pcfg-model", type=str, help=argparse.SUPPRESS)
    # Neural hyperparams
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dedup-bloom", type=str, help="Persist Bloom filter to this file for huge runs")
    ap.add_argument("--dedup-memory", type=int, default=10_000_000,
                    help="Max passwords kept in memory before falling back to Bloom (0 = unlimited)")
    args = ap.parse_args()

    # Seed & basics
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    rnd = random.Random(seed)
    charset = build_charset(args.charset, args.exclude_ambiguous)

    # Loads
    starts = list("q1az2wsx3edc")
    if args.starts_file:
        starts = [x.lower() for x in load_lines(args.starts_file) if x]

    graph = get_graph(args.keymap, args.walk_allow_shift)
    if args.keymap_file:
        try:
            with open(args.keymap_file) as f:
                custom = json.load(f)
            graph = {str(k).lower(): [str(v).lower() for v in vv] for k, vv in custom.items()}
            if args.walk_allow_shift:
                graph = {**graph, **GRAPH_SYMBOL}
        except Exception as e:
            print(f"[!] keymap-file error: {e}", file=sys.stderr)

    base_words = tuple(load_lines(args.dict) or ["password", "admin", "welcome"])
    years      = tuple(load_lines(args.years_file) or [str(y) for y in range(1990, 2036)])
    symbols    = tuple(load_lines(args.symbols_file) or list("!@#$%^&*?_+-="))
    bias_terms = tuple(load_lines(args.bias_terms) or [])
    rules      = [ln for ln in load_lines(args.rules) if ln and not ln.startswith('#')]

    no_dict_set = load_dict_set(args.no_dict)

    # Models
    markov_model = None
    if args.mode in ("markov", "estimate_only") and args.markov_train:
        corpus = load_lines(args.markov_train)
        if corpus:
            markov_model = markov_train(corpus, order=args.markov_order)
        else:
            print("[!] Markov training corpus empty", file=sys.stderr)

    pcfg_model = None
    if args.pcfg_model:
        try:
            with open(args.pcfg_model) as f:
                pcfg_model = json.load(f)
        except Exception as e:
            print(f"[!] PCFG model load error: {e}", file=sys.stderr)
    elif args.mode in ("pcfg", "estimate_only") and args.pcfg_train:
        corpus = load_lines(args.pcfg_train)
        if corpus:
            pcfg_model = pcfg_train(corpus)
        else:
            print("[!] PCFG training corpus empty", file=sys.stderr)

    # Dedup
    seen: Set[str] = set()
    bloom = None
    if args.dedup_bloom:
        bloom = load_or_create_bloom(args.dedup_bloom)
    mem_limit = args.dedup_memory if args.dedup_memory > 0 else float('inf')

    def add_seen(pw: str) -> bool:
        if pw in seen:
            return False
        if bloom and bloom.__contains__(pw):
            return False
        if len(seen) >= mem_limit and bloom:
            for old in list(seen):
                bloom_add(bloom, old)
            seen.clear()
        seen.add(pw)
        if bloom:
            bloom_add(bloom, pw)
        return True

    # Gen map
    gen_map = {
        "pw": lambda a, c, r: gen_pw(a, c, r, charset, seen),
        "walk": lambda a, c, r: gen_walk(a, c, r, graph, starts, seen),
        "mobile-walk": lambda a, c, r: gen_mobile_walk(a, c, r, seen),
        "mask": lambda a, c, r: gen_mask(a, c, r, base_words, years, symbols, seen),
        "passphrase": lambda a, c, r: gen_passphrase(a, c, r, base_words, seen),
        "numeric": lambda a, c, r: gen_numeric(a, c, r, seen),
        "syllable": lambda a, c, r: gen_syllable(a, c, r, seen),
        "prince": lambda a, c, r: gen_prince(a, c, r, base_words, bias_terms, seen),
        "markov": lambda a, c, r: gen_markov(a, c, r, markov_model, seen),
        "pcfg": lambda a, c, r: gen_pcfg(a, c, r, pcfg_model, years, symbols, seen),
        "hybrid": lambda a, c, r: gen_hybrid(a, c, r, base_words, rules, args.mask, years, symbols, seen),
        "combo": lambda a, c, r: gen_combo(a, c, r, parse_combo(args.combo), gen_map,
                                          base_words=base_words, years=years, symbols=symbols,
                                          rules=rules, mask=args.mask, graph=graph, starts=starts,
                                          model=pcfg_model, bias_terms=bias_terms, seen=seen),
        "neural": lambda a, c, r: gen_neural(a, c, r, args.model,
                                            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                            seen)
    }

    # Early exits
    if args.estimate_only:
        avg = (args.min + args.max) / 2
        print(json.dumps({"mode": args.mode, "count": args.count, "avg_len": avg}))
        return

    if args.dry_run:
        args.count = int(args.dry_run)

    # Targets
    targets: Dict[str, int] = defaultdict(int)
    if args.mode == "both":
        half = args.count // 2
        targets["pw"], targets["walk"] = half + args.count % 2, half
    else:
        targets[args.mode] = args.count

    # Loop
    start = now()
    produced = 0
    last_tick = start
    CHUNK = max(1, int(args.chunk))

    for mode, tgt in targets.items():
        if tgt <= 0: continue
        remaining = tgt
        while remaining > 0:
            chunk = min(CHUNK, remaining)
            try:
                lines = gen_map.get(mode, lambda *x: [])(args, chunk, rnd)
            except Exception as e:
                print(f"[!] Generator {mode} crashed: {e}", file=sys.stderr)
                lines = []

            if args.min_entropy > 0 or args.no_dict:
                lines = apply_entropy_filter(lines, args.min_entropy, no_dict_set)

            final = [pw for pw in lines if add_seen(pw)]
            write_output(args, final)
            produced += len(final)
            remaining -= len(final)
            last_tick = meter_printer(start, produced, last_tick, args.meter)

    if bloom and args.dedup_bloom:
        try:
            with open(args.dedup_bloom, "wb") as f:
                bloom.dump(f)
        except Exception as e:
            print(f"[!] Bloom save failed: {e}", file=sys.stderr)

    print(f"[i] Finished – {produced:,} unique passwords generated.", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[!] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
