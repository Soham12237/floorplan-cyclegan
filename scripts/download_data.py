"""Fetch a subset of CubiCasa5k straight out of the remote zip via HTTP range requests.

Domain A (blueprint)  : high_quality_architectural  -> N_ARCH random plans
Domain B (rendered)   : colorful                     -> all plans
Only F1_original.png (ground-floor plan image) is downloaded for each plan.
"""
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
N_ARCH = int(sys.argv[1]) if len(sys.argv) > 1 else 800
SEED = 0
THREADS = 3

names = json.load(open(ROOT / "data" / "zip_names.json"))
by_cat = {"high_quality_architectural": [], "colorful": []}
for n in names:
    p = n.split("/")
    if len(p) == 4 and p[3] == "F1_original.png" and p[1] in by_cat:
        by_cat[p[1]].append(n)

rng = random.Random(SEED)
jobs = []
for cat, dom, k in (("colorful", "B", None), ("high_quality_architectural", "A", N_ARCH)):
    files = sorted(by_cat[cat])
    if k is not None:
        files = rng.sample(files, min(k, len(files)))
    jobs += [(dom, f) for f in files]
    (RAW / dom).mkdir(parents=True, exist_ok=True)
print(f"{len(jobs)} files to fetch", flush=True)

local = threading.local()
lock = threading.Lock()
done = 0


def fetch(job):
    global done
    dom, name = job
    out = RAW / dom / f"{name.split('/')[2]}.png"
    if out.exists() and out.stat().st_size > 0:
        return
    for attempt in range(10):
        try:
            if not hasattr(local, "z"):
                local.z = RemoteZip(URL)
            out.write_bytes(local.z.read(name))
            break
        except Exception as e:  # reconnect and retry
            local.__dict__.pop("z", None)
            if "429" in str(e):
                time.sleep(65)  # Zenodo rate limit window
            if attempt == 9:
                print("FAILED", name, e, flush=True)
    with lock:
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(jobs)}", flush=True)


with ThreadPoolExecutor(THREADS) as ex:
    list(ex.map(fetch, jobs))
print("done", flush=True)
