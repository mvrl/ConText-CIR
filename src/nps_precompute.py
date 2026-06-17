import argparse
import glob
import json
import os
import pickle
import sqlite3
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Caption sources (dataset-agnostic): each yields caption strings.
def _src_jsonl(path, field, index):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            v = obj.get(field)
            if v is None:
                continue
            if isinstance(v, list):
                if index is None:
                    for x in v:
                        if x:
                            yield str(x)
                    continue
                if len(v) > index:
                    v = v[index]
                else:
                    continue
            if v:
                yield str(v)


def _src_json_list(path, field):
    with open(path) as f:
        data = json.load(f)
    for obj in data:
        v = obj.get(field) if isinstance(obj, dict) else None
        if isinstance(v, list):
            v = v[0] if v else None
        if v:
            yield str(v)


def _src_txt(path):
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def iter_source(spec, field=None, index=None):
    typ, _, path = spec.partition(":")
    if typ == "txt":
        return _src_txt(path)
    if typ == "jsonl":
        return _src_jsonl(path, field=field, index=index)
    if typ == "jsonlist":
        return _src_json_list(path, field=field)
    raise ValueError(f"Unknown source type '{typ}' in spec '{spec}'")


# spaCy noun-chunk extraction.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def _spacy_nps(doc, max_nps):
    seen, out = set(), []
    for ch in doc.noun_chunks:
        t = ch.text.strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_nps:
            break
    return out


def spacy_noun_phrases(caption, max_nps=10):
    if not caption:
        return []
    try:
        return _spacy_nps(_get_nlp()(caption), max_nps)
    except Exception:
        return []



def cmd_extract(args):
    seen = set()
    out = f"{args.prefix}.captions.txt"
    n = 0
    with open(out, "w") as w:
        for cap in iter_source(args.source, args.field, args.index):
            if cap in seen:
                continue
            seen.add(cap)
            w.write(cap.replace("\n", " ") + "\n")
            n += 1
            if n % 1_000_000 == 0:
                log(f"  extracted {n:,} unique captions")
    log(f"DONE extract: {n:,} unique captions -> {out}")


def _load_shard_caps(args):
    caps = []
    with open(f"{args.prefix}.captions.txt") as f:
        for i, line in enumerate(f):
            if i % args.num_shards == args.shard_id:
                caps.append(line.rstrip("\n"))
    return caps


def cmd_parse(args):
    """Parse one shard -> part pkl with spaCy."""
    caps = _load_shard_caps(args)
    log(f"shard {args.shard_id}/{args.num_shards}: {len(caps):,} captions, procs={args.procs}")
    out = {}
    t0 = time.time()
    nlp = _get_nlp()
    for k, (cap, doc) in enumerate(
            zip(caps, nlp.pipe(caps, n_process=args.procs, batch_size=args.batch_size))):
        out[cap] = _spacy_nps(doc, args.max_nps)
        if (k + 1) % 500_000 == 0:
            log(f"  {k+1:,}/{len(caps):,} ({(k+1)/max(time.time()-t0,1):,.0f}/s)")
    part = f"{args.prefix}.part-{args.shard_id:04d}.pkl"
    with open(part, "wb") as w:
        pickle.dump(out, w, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"DONE parse shard {args.shard_id}: {len(out):,} -> {part} in {time.time()-t0:.0f}s")


def cmd_parse_all(args):
    caps = [l.rstrip("\n") for l in open(f"{args.prefix}.captions.txt") if l.strip()]
    log(f"parse-all: {len(caps):,} captions, spaCy n_process={args.procs}, "
        f"batch_size={args.batch_size}")
    nlp = _get_nlp()
    db = f"{args.prefix}.sqlite"
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE np (caption TEXT PRIMARY KEY, nps TEXT)")
    buf, n, t0 = [], 0, time.time()
    for cap, doc in zip(caps, nlp.pipe(caps, n_process=args.procs, batch_size=args.batch_size)):
        buf.append((cap, json.dumps(_spacy_nps(doc, args.max_nps))))
        n += 1
        if len(buf) >= 50_000:
            conn.executemany("INSERT OR REPLACE INTO np(caption, nps) VALUES (?, ?)", buf)
            conn.commit()
            buf = []
        if n % 1_000_000 == 0:
            log(f"  {n:,}/{len(caps):,} ({n/max(time.time()-t0,1):,.0f}/s)")
    if buf:
        conn.executemany("INSERT OR REPLACE INTO np(caption, nps) VALUES (?, ?)", buf)
        conn.commit()
    cnt = conn.execute("SELECT COUNT(*) FROM np").fetchone()[0]
    conn.close()
    log(f"DONE parse-all: {cnt:,} captions -> {db} in {time.time()-t0:.0f}s")


def cmd_merge(args):
    parts = sorted(glob.glob(f"{args.prefix}.part-*.pkl"))
    if not parts:
        raise SystemExit(f"no parts found at {args.prefix}.part-*.pkl")
    db = f"{args.prefix}.sqlite"
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE np (caption TEXT PRIMARY KEY, nps TEXT)")
    total = 0
    for p in parts:
        with open(p, "rb") as f:
            d = pickle.load(f)
        conn.executemany("INSERT OR REPLACE INTO np(caption, nps) VALUES (?, ?)",
                         ((c, json.dumps(nps)) for c, nps in d.items()))
        conn.commit()
        total += len(d)
        log(f"  merged {os.path.basename(p)} ({len(d):,}); running total {total:,}")
    n = conn.execute("SELECT COUNT(*) FROM np").fetchone()[0]
    conn.close()
    log(f"DONE merge: {n:,} unique captions -> {db}  ({len(parts)} shards, {total:,} pre-dedup)")



class NPCache:
    def __init__(self, sqlite_path):
        uri = f"file:{os.path.abspath(sqlite_path)}?mode=ro&immutable=1"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only=1")

    def get(self, caption):
        row = self.conn.execute("SELECT nps FROM np WHERE caption=?", (caption,)).fetchone()
        return json.loads(row[0]) if row is not None else None


def main():
    ap = argparse.ArgumentParser(description="Offline noun-phrase precompute for ConText-CIR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract")
    pe.add_argument("--source", required=True, help="TYPE:PATH (txt|jsonl|jsonlist)")
    pe.add_argument("--field", default=None, help="field name for jsonl/jsonlist sources")
    pe.add_argument("--index", type=int, default=None, help="list index for jsonl list fields")
    pe.add_argument("--prefix", required=True)
    pe.set_defaults(func=cmd_extract)

    pp = sub.add_parser("parse")
    pp.add_argument("--prefix", required=True)
    pp.add_argument("--shard-id", type=int, required=True)
    pp.add_argument("--num-shards", type=int, required=True)
    pp.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pp.add_argument("--max-nps", type=int, default=10)
    pp.add_argument("--batch-size", type=int, default=256, help="spaCy nlp.pipe batch size")
    pp.set_defaults(func=cmd_parse)

    pa = sub.add_parser("parse-all", help="single-job spaCy parse straight to SQLite")
    pa.add_argument("--prefix", required=True)
    pa.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pa.add_argument("--max-nps", type=int, default=10)
    pa.add_argument("--batch-size", type=int, default=512, help="spaCy nlp.pipe batch size")
    pa.set_defaults(func=cmd_parse_all)

    pm = sub.add_parser("merge")
    pm.add_argument("--prefix", required=True)
    pm.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
