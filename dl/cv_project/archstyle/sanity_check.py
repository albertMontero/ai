import os, re, random, shutil
from collections import defaultdict
from PIL import Image

CLASSES = ('modernism', 'romanesque', 'baroque', 'gothic', 'contemporary')
EXTS = ('jpg', 'jpeg', 'png')
PATTERN = re.compile(rf"^[A-Za-z0-9_-]+_\d{{2}}_({'|'.join(CLASSES)})\.({'|'.join(EXTS)})$", re.IGNORECASE)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_image_ok(p):
    try:
        with Image.open(p) as im:
            im.verify();
            return True
    except:
        return False

def collect_valid_files_by_class(root):
    by_class, invalid = defaultdict(list), []
    for d, _, fs in os.walk(root):
        for f in fs:
            fp = os.path.join(d, f)
            m = PATTERN.match(f)
            if not m or not is_image_ok(fp):
                invalid.append(fp)
                continue
            by_class[m.group(1).lower()].append(fp)
    return by_class, invalid

def split_indices(n, pt, pv, seed):
    idx = list(range(n))
    r = random.Random(seed)
    r.shuffle(idxs := idx)
    nt, nv = int(n * pt), int(n * pv)
    return idxs[:nt], idxs[nt:nt + nv], idxs[nt + nv:]

def stratified_split(root, out, train=0.4, validation=0.2, test=0.4, *, seed=17, dry_run=False):
    total = train + validation + test
    if total <= 0: raise ValueError("Invalid split")
    pt, pv, ps = train / total, validation / total, test / total
    by_class, invalid = collect_valid_files_by_class(root)
    splits = ["train", "val"]
    if test > 0: splits.append("test")
    [ensure_dir(os.path.join(out, s, c)) for s in splits for c in CLASSES]
    summary = {s: 0 for s in splits}
    for c in CLASSES:
        fs = by_class.get(c, [])
        if not fs: continue
        ti, vi, si = split_indices(len(fs), pt, pv, seed)
        assigns = [("train", ti), ("val", vi)]
        if test > 0: assigns.append(("test", si))
        for sn, idxs in assigns:
            for i in idxs:
                src, dst = fs[i], os.path.join(out, sn, c, os.path.basename(fs[i]))
                if dry_run:
                    print(f"[DRY RUN] {'COPY'} {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)
                summary[sn] += 1
    print("\n----- Summary -----")
    print(
        f"Root: {root}\nOutput: {out}\nSplits: train={pt:.2f}, val={pv:.2f}" + (f", test={ps:.2f}" if test > 0 else ""))
    for c in CLASSES: print(f"{c:13s}: {len(by_class.get(c, []))}")
    print("\nPlaced:")
    [print(f"{s:12s}: {summary[s]}") for s in splits]
    print(f"\nInvalid: {len(invalid)}\nDone.")
    return {"by_class": by_class, "invalid": invalid, "summary": summary}


if __name__ == "__main__":
    ROOT_DIR = "../data/raw"
    OUTPUT_DIR = "../data/preprocess"

    print(os.getcwd())

    c, inv = collect_valid_files_by_class(ROOT_DIR)
    print(f"Invalid files: {len(inv)}")
    if inv:
        print(inv)
        exit(1)

    stratified_split(ROOT_DIR, OUTPUT_DIR, train=0.4, validation=0.2, test=0.4, seed=17)
    # stratified_split(ROOT_DIR, OUTPUT_DIR, train=0.4, validation=0.2, test=0, seed=17)

