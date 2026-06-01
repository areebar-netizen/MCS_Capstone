import os, glob, pickle, zlib, csv, statistics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EMAIL = 'areebaarashid31@gmail.com'
CSV_GLOB = os.path.join(PROJECT_ROOT, 'dataset', 'our_data', '**', f'*{EMAIL}_*.csv')
cache_dir = os.path.join(PROJECT_ROOT, 'django_cache_temp')

def time_to_seconds(t):
    parts = t.split(':')
    if len(parts) == 2:
        h = int(parts[0])
        m = int(parts[1])
        s = 0
    else:
        h, m, s = (int(p) for p in parts)
    return h*3600 + m*60 + s

# find latest CSV for email
files = sorted(glob.glob(CSV_GLOB, recursive=True))
if not files:
    print('No CSV found for email:', EMAIL)
    raise SystemExit(1)
latest_csv = files[-1]
print('Using CSV:', latest_csv)
# read first and last timestamp
with open(latest_csv, 'r', newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    rows = list(reader)
    if not rows:
        print('CSV empty')
        raise SystemExit(1)
    first_ts = rows[0][0]
    last_ts = rows[-1][0]
    start_sec = time_to_seconds(first_ts)
    end_sec = time_to_seconds(last_ts)
    print('Session time range:', first_ts, '->', last_ts)

# scan cache files for waves with last_updated in range (allow wrap across midnight?)
wave_samples = []
for path in sorted(glob.glob(os.path.join(cache_dir, '*.djcache'))):
    try:
        with open(path, 'rb') as f:
            exp = pickle.load(f)
            raw = f.read()
            try:
                data = pickle.loads(zlib.decompress(raw))
            except Exception:
                # maybe not compressed
                try:
                    data = pickle.loads(raw)
                except Exception:
                    continue
    except Exception:
        continue
    if isinstance(data, dict) and 'waves' in data and isinstance(data.get('last_updated'), str):
        lu = data.get('last_updated')
        try:
            lu_sec = time_to_seconds(lu)
        except Exception:
            continue
        # simple containment: same hour or between start and end
        if start_sec <= end_sec:
            in_range = (start_sec <= lu_sec <= end_sec)
        else:
            # session crossed midnight
            in_range = (lu_sec >= start_sec or lu_sec <= end_sec)
        if in_range:
            wave_samples.append((path, lu, data['waves']))

print('Found', len(wave_samples), 'wave samples in cache within session range')
if not wave_samples:
    # as fallback, collect any wave samples near the start hour
    print('Fallback: collecting any cache waves with same hour as session start')
    start_hour = int(first_ts.split(':')[0])
    for path in sorted(glob.glob(os.path.join(cache_dir, '*.djcache'))):
        try:
            with open(path, 'rb') as f:
                exp = pickle.load(f)
                raw = f.read()
                data = pickle.loads(zlib.decompress(raw))
        except Exception:
            continue
        if isinstance(data, dict) and 'waves' in data and isinstance(data.get('last_updated'), str):
            try:
                hh = int(data['last_updated'].split(':')[0])
            except Exception:
                continue
            if hh == start_hour:
                wave_samples.append((path, data['last_updated'], data['waves']))
    print('Fallback found', len(wave_samples))

# aggregate
bands = ['delta','theta','alpha','beta','gamma']
vals = {b: [] for b in bands}
for p, lu, w in wave_samples:
    for b in bands:
        v = w.get(b)
        try:
            vals[b].append(float(v))
        except Exception:
            pass

averages = {}
for b in bands:
    if vals[b]:
        averages[b] = sum(vals[b]) / len(vals[b])
    else:
        averages[b] = None

print('\nAverages (from cache wave samples):')
for b in bands:
    print(f"{b}: {averages[b]}")

print('\nSample count per band:')
for b in bands:
    print(f"{b}: {len(vals[b])}")

print('\nFirst 10 samples:')
for s in wave_samples[:10]:
    print(s)
