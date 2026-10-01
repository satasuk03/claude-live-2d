#!/usr/bin/env python3
"""OpenRouter /api/v1/images helper (gpt-image-2, ming, seedream, flux...).
usage: img.py MODEL OUT_PREFIX "prompt|@file" [refs...] [--aspect 2:3] [--res 2K] [--size WxH] [--quality high] [--bg transparent] [--n 1]
"""
import sys, os, json, base64, urllib.request, argparse, mimetypes, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
for line in open(ROOT / '.env'):
    if '=' in line:
        k, v = line.strip().split('=', 1); os.environ.setdefault(k, v.strip().strip('"'))
KEY = os.environ['OPENROUTER_API_KEY']
def data_url(p):
    mt = mimetypes.guess_type(p)[0] or 'image/png'
    return f"data:{mt};base64," + base64.b64encode(open(p, 'rb').read()).decode()
ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('out'); ap.add_argument('prompt'); ap.add_argument('images', nargs='*')
ap.add_argument('--aspect'); ap.add_argument('--res'); ap.add_argument('--size'); ap.add_argument('--quality')
ap.add_argument('--bg'); ap.add_argument('--n', type=int); ap.add_argument('--fmt', default='png')
a = ap.parse_args()
prompt = open(a.prompt[1:]).read() if a.prompt.startswith('@') else a.prompt
body = {"model": a.model, "prompt": prompt, "output_format": a.fmt}
if a.images: body["input_references"] = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in a.images]
for k, v in [("aspect_ratio", a.aspect), ("resolution", a.res), ("size", a.size), ("quality", a.quality), ("background", a.bg), ("n", a.n)]:
    if v: body[k] = v
t = time.time()
for attempt in range(3):
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/images", data=json.dumps(body).encode(),
              headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=900).read().decode(), strict=False); break
    except urllib.error.HTTPError as e:
        print('HTTP', e.code, e.read().decode()[:800])
        if attempt == 2 or e.code in (400, 404): sys.exit(1)
        time.sleep(5)
    except Exception as e:
        print('ERR', e)
        if attempt == 2: sys.exit(1)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
data = res.get('data') or []
for i, d in enumerate(data):
    b = base64.b64decode(d['b64_json']); ext = (d.get('media_type') or 'image/png').split('/')[-1].replace('jpeg', 'jpg')
    p = f"{a.out}_{i}.{ext}" if len(data) > 1 else f"{a.out}.{ext}"
    open(p, 'wb').write(b); print('saved', p)
print(f'{time.time()-t:.1f}s usage', res.get('usage'))
if not data: print(json.dumps(res)[:1500])
