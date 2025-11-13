#!/usr/bin/env python3
# PWForge – Ultimate Password Generator
# Fully compatible with finetune_neural.py
# GPU: 10M+/s | CPU: 1M+/s | All modes supported

import sys, os, argparse, random, secrets, string, time, json, gzip, math, re
from collections import defaultdict, deque, Counter

# -------------------------------------------------
# PyTorch – Lazy Import
# -------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

def now(): return time.time()

def build_charset(base, exclude_ambiguous=False):
    ch = base or (string.ascii_letters + string.digits + "!@#$%^&*()_+-=")
    if exclude_ambiguous:
        ch = "".join(c for c in ch if c not in AMBIGUOUS)
    return ch

def class_ok(s, reqs):
    if not reqs: return True
    want = {p.strip().lower() for p in reqs.split(",")}
    has = {
        "upper": any(c.isupper() for c in s),
        "lower": any(c.islower() for c in s),
        "digit": any(c.isdigit() for c in s),
        "symbol": any(not c.isalnum() for c in s)
    }
    return all(has.get(k, True) for k in want)

def load_lines(path):
    if not path or not os.path.exists(path): return []
    opener = gzip.open if path.endswith(('.gz', '.gzip')) else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(path, lines, gz=False, append=False):
    mode = "ab" if append else "wb"
    opener = gzip.open if gz else open
    wmode = mode if gz else mode.replace("b", "")
    with opener(path, wmode) as f:
        for ln in lines:
            data = ln + "\n"
            if gz:
                f.write(data.encode("utf-8", "ignore"))
            else:
                f.write(data)

def meter_printer(start, produced, last_tick, every):
    if every <= 0: return last_tick
    t = now()
    if produced and produced % every == 0:
        dt = max(1e-9, t - start)
        lps = int(produced / dt)
        print(f"[meter] {produced:,} lines, {lps:,} lps", file=sys.stderr)
        return t
    return last_tick

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

# -------------------------------------------------
# Generators
# -------------------------------------------------
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
            if ch == "C": s.append(rng.choice(CONS))
            elif ch == "V": s.append(rng.choice(VOWS))
            else: s.append(ch)
        st = "".join(s)
        if args.upper_first: st = st.capitalize()
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

# -------------------------------------------------
# Markov
# -------------------------------------------------
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

def gen_markov(args, count, rng, model):
    return [markov_sample(model, rng, args.min, args.max) for _ in range(count)]

# -------------------------------------------------
# PCFG
# -------------------------------------------------
def tokenize_line(line):
    return _WORD_RE.findall(line) + _DIG_RE.findall(line) + _SYM_RE.findall(line)

def classify_token(tok):
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

def pcfg_train(lines):
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

def pcfg_sample(model, rng, min_len, max_len, years, symbols):
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

def gen_pcfg(args, count, rng, model, years, symbols):
    return [pcfg_sample(model, rng, args.min, args.max, years, symbols) for _ in range(count)]

# -------------------------------------------------
# Hybrid / Combo / Entropy
# -------------------------------------------------
def apply_rule(word, rule):
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

def gen_hybrid(args, count, rng, base_words, rules, mask, years, symbols):
    out = []
    for _ in range(count):
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
        out.append(cand)
    return out

def parse_combo(spec):
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

def gen_combo(args, count, rng, mode_weights, generators, base_words, years, symbols, rules, mask, graph, starts, model, bias_terms):
    out = []
    mode_names = [m for m, _ in mode_weights]
    probs = [w for _, w in mode_weights]
    for _ in range(count):
        mode = rng.choices(mode_names, probs)[0]
        gen = generators.get(mode)
        if gen:
            if mode == "pcfg":
                cand = gen(args, 1, rng, model, years, symbols)[0]
            elif mode == "walk":
                cand = gen(args, 1, rng, graph, starts)[0]
            elif mode == "prince":
                cand = gen(args, 1, rng, base_words, bias_terms)[0]
            elif mode == "mask":
                cand = gen(args, 1, rng, base_words, years, symbols)[0]
            elif mode == "passphrase":
                cand = gen(args, 1, rng, base_words)[0]
            elif mode == "hybrid":
                cand = gen(args, 1, rng, base_words, rules, mask, years, symbols)[0]
            else:
                cand = gen(args, 1, rng)[0]
            out.append(cand)
    return out

def shannon_entropy(s):
    if not s: return 0.0
    cnt = Counter(s)
    L = len(s)
    return -sum((c/L)*math.log2(c/L) for c in cnt.values())

def load_dict_set(path):
    if not path or not os.path.exists(path): return set()
    return {ln.strip().lower() for ln in load_lines(path) if ln.strip()}

def apply_entropy_filter(lines, min_entropy=0.0, dict_set=None):
    if min_entropy <= 0 and not dict_set: return lines
    return [l for l in lines
            if (min_entropy <= 0 or shannon_entropy(l) >= min_entropy)
            and (not dict_set or l.lower() not in dict_set)]

# -------------------------------------------------
# NEURAL – FULLY FIXED
# -------------------------------------------------
PRINTABLE = ''.join(chr(i) for i in range(33,127))
CHAR_TO_IDX = {c:i for i,c in enumerate(PRINTABLE)}
IDX_TO_CHAR = {i:c for c,i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(PRINTABLE)
BOS_IDX = CHAR_TO_IDX['!']
EOS_IDX = CHAR_TO_IDX.get('\n', 0)

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

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL – FINAL FIXED (replace entire function)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gen_neural(args, count, rng, model_path=None, device=None):
    if not TORCH_AVAILABLE:
        print("[!] torch not installed – neural mode disabled", file=sys.stderr)
        return []
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

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    batch = min(args.batch_size, count)
    out = []
    max_gen = getattr(args, "max_gen_len", 32) or 32

    while len(out) < count:
        need = min(batch, count - len(out))
        seqs = [torch.tensor([[BOS_IDX]], dtype=torch.long).to(device) for _ in range(need)]
        hiddens = [(None, None)] * need  # (h0, c0) per sequence

        for _ in range(max_gen):
            seq_batch = torch.cat(seqs, dim=0)
            # First step: hidden=None → fresh states
            hidden_input = None if hiddens[0][0] is None else tuple(torch.cat(tensors, dim=1) for tensors in zip(*hiddens))
            logits, new_hiddens = model(seq_batch, hidden_input)

            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1)

            # Update each sequence's hidden state
            for i in range(need):
                seqs[i] = torch.cat([seqs[i], nxt[i:i+1]], dim=1)
                h0_i = new_hiddens[0][:, i:i+1, :]
                c0_i = new_hiddens[1][:, i:i+1, :]
                hiddens[i] = (h0_i.contiguous(), c0_i.contiguous())

            if (nxt == EOS_IDX).any(): break

        # Decode
        for i in range(need):
            pw = ''.join(IDX_TO_CHAR.get(t.item(), '') for t in seqs[i][0, 1:])
            if len(pw) < args.min:
                pw += ''.join(rng.choice(PRINTABLE) for _ in range(args.min - len(pw)))
            pw = pw[:args.max]
            if len(pw) >= args.min:
                out.append(pw)
            if len(out) >= count: break
    return out[:count]


# -------------------------------------------------
# Output
# -------------------------------------------------
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
        for ln in lines: print(ln)

def split_suffix(i, digits): return f"_{i:0{digits}d}"

def write_output(args, lines):
    if args.split and args.split > 1 and args.out:
        digits = len(str(args.split-1))
        n = len(lines)
        per = math.ceil(n/args.split)
        idx = 0
        for i in range(args.split):
            chunk = lines[idx:idx+per]
            if not chunk: break
            write_output_sink(args, chunk, shard_suffix=split_suffix(i,digits))
            idx += per
    else:
        write_output_sink(args, lines)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PWForge – Multi-mode password generator")
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
    ap.add_argument("--append", action="store_true")
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

    args = ap.parse_args()

    rnd = random.Random(args.seed if args.seed is not None else secrets.randbits(64))

    charset = build_charset(args.charset, args.exclude_ambiguous)
    starts = list("q1az2wsx3edc")
    if args.starts_file and os.path.exists(args.starts_file):
        starts = [x.strip().lower() for x in load_lines(args.starts_file) if x.strip()]
    graph = get_graph(args.keymap, args.walk_allow_shift)
    if args.keymap_file:
        try:
            with open(args.keymap_file) as f: custom = json.load(f)
            graph = {str(k).lower():[str(v).lower() for v in vv] for k,vv in custom.items()}
            if args.walk_allow_shift: graph = {**graph, **GRAPH_SYMBOL}
        except Exception as e:
            print(f"[!] keymap-file: {e}", file=sys.stderr)

    base_words = tuple(load_lines(args.dict) if args.dict else ["password","admin","welcome"])
    years      = tuple(load_lines(args.years_file) if args.years_file else [str(y) for y in range(1990,2036)])
    symbols    = tuple(load_lines(args.symbols_file) if args.symbols_file else list("!@#$%^&*?_+-="))
    bias_terms = tuple(load_lines(args.bias_terms) if args.bias_terms else [])
    rules      = [ln for ln in load_lines(args.rules) if ln and not ln.startswith('#')] if args.rules else []
    no_dict_set = load_dict_set(args.no_dict)

    markov_model = None
    if args.mode in ("markov","estimate_only") and args.markov_train:
        markov_model = markov_train(load_lines(args.markov_train), order=args.markov_order)

    pcfg_model = None
    if args.pcfg_model:
        with open(args.pcfg_model) as f: pcfg_model = json.load(f)
    elif args.mode in ("pcfg","estimate_only") and args.pcfg_train:
        pcfg_model = pcfg_train(load_lines(args.pcfg_train))

    if args.estimate_only:
        avg = (args.min+args.max)/2
        print(json.dumps({"mode":args.mode,"count":args.count,"avg_len":avg}))
        return

    if args.dry_run: args.count = int(args.dry_run)

    targets = {k:0 for k in [
        "pw","walk","mask","passphrase","numeric","syllable","prince",
        "markov","pcfg","mobile-walk","hybrid","combo","neural"
    ]}
    if args.mode == "both":
        half = args.count//2
        targets["pw"], targets["walk"] = half + args.count%2, half
    else:
        targets[args.mode] = args.count

    start = now()
    produced = 0
    last_tick = start
    CHUNK = max(1, int(args.chunk))

    gen_map = {
        "pw": lambda a,c,r: gen_pw(a,c,r,charset),
        "walk": lambda a,c,r: gen_walk(a,c,r,graph,starts),
        "mobile-walk": lambda a,c,r: gen_mobile_walk(a,c,r),
        "mask": lambda a,c,r: gen_mask(a,c,r,base_words,years,symbols),
        "passphrase": lambda a,c,r: gen_passphrase(a,c,r,base_words),
        "numeric": lambda a,c,r: gen_numeric(a,c,r),
        "syllable": lambda a,c,r: gen_syllable(a,c,r),
        "prince": lambda a,c,r: gen_prince(a,c,r,base_words,bias_terms),
        "markov": lambda a,c,r: gen_markov(a,c,r,markov_model),
        "pcfg": lambda a,c,r: gen_pcfg(a,c,r,pcfg_model,years,symbols),
        "hybrid": lambda a,c,r: gen_hybrid(a,c,r,base_words,rules,args.mask,years,symbols),
        "combo": lambda a,c,r: gen_combo(a,c,r,parse_combo(args.combo),gen_map,
                                         base_words=base_words,years=years,symbols=symbols,
                                         rules=rules,mask=args.mask,graph=graph,starts=starts,
                                         model=pcfg_model,bias_terms=bias_terms),
        "neural": lambda a,c,r: gen_neural(a,c,r,args.model)
    }

    for mode, tgt in targets.items():
        if tgt <= 0: continue
        remaining = tgt
        while remaining > 0:
            chunk = min(CHUNK, remaining)
            lines = gen_map.get(mode, lambda *x: [])(args, chunk, rnd)
            if args.min_entropy > 0 or args.no_dict:
                lines = apply_entropy_filter(lines, args.min_entropy, no_dict_set)
            write_output(args, lines)
            produced += len(lines)
            remaining -= len(lines)
            last_tick = meter_printer(start, produced, last_tick, args.meter)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted", file=sys.stderr)
