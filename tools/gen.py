#!/usr/bin/env python3
"""OpenRouter image generation helper.
usage: gen.py MODEL OUT_PREFIX "prompt" [img1 img2 ...] [--aspect 2:3] [--size 4K] [--n 1]
"""
import sys, os, json, base64, urllib.request, argparse, mimetypes, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
for line in open(ROOT / '.env'):
    if '=' in line:
        k, v = line.strip().split('=', 1)
        os.environ.setdefault(k, v.strip().strip('"'))
KEY = os.environ['OPENROUTER_API_KEY']

def data_url(p):
    mt = mimetypes.guess_type(p)[0] or 'image/png'
    return f"data:{mt};base64," + base64.b64encode(open(p, 'rb').read()).decode()

def run(model, prompt, images, aspect=None, size=None, image_only=False, timeout=600):
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in images]
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "modalities": ["image"] if image_only else ["image", "text"]}
    cfg = {}
    if aspect: cfg["aspect_ratio"] = aspect
    if size: cfg["image_size"] = size
    if cfg: body["image_config"] = cfg
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode(), strict=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('out'); ap.add_argument('prompt')
    ap.add_argument('images', nargs='*')
    ap.add_argument('--aspect'); ap.add_argument('--size')
    ap.add_argument('--image-only', action='store_true')
    a = ap.parse_args()
    prompt = open(a.prompt[1:]).read() if a.prompt.startswith('@') else a.prompt
    t = time.time()
    for attempt in range(3):
        try:
            res = run(a.model, prompt, a.images, a.aspect, a.size, a.image_only); break
        except urllib.error.HTTPError as e:
            print('HTTP', e.code, e.read().decode()[:800]); 
            if attempt == 2: sys.exit(1)
            time.sleep(5)
    if 'choices' not in res:
        print(json.dumps(res)[:2000]); sys.exit(1)
    msg = res['choices'][0]['message']
    imgs = msg.get('images') or []
    if msg.get('content'): print('TEXT:', str(msg['content'])[:500])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    for i, im in enumerate(imgs):
        u = im['image_url']['url']
        b = base64.b64decode(u.split(',', 1)[1])
        ext = 'png' if 'png' in u[:30] else ('webp' if 'webp' in u[:30] else 'jpg')
        p = f"{a.out}_{i}.{ext}" if len(imgs) > 1 else f"{a.out}.{ext}"
        open(p, 'wb').write(b); print('saved', p)
    print(f'{time.time()-t:.1f}s usage', res.get('usage', {}).get('cost'))
    if not imgs: print(json.dumps(res)[:1500])
