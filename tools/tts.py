#!/usr/bin/env python3
"""Generate the preset voice clips with Gemini TTS via OpenRouter.
usage: tts.py [ids...] [--force]       (spec: tools/voice_lines.json -> web/assets/voice/<id>.mp3)
OpenRouter returns raw 24 kHz mono s16 PCM for Gemini TTS; ffmpeg encodes it to mp3.
"""
import sys, os, json, urllib.request, subprocess, pathlib, argparse, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
for line in open(ROOT / '.env'):
    if '=' in line:
        k, v = line.strip().split('=', 1)
        os.environ.setdefault(k, v.strip().strip('"'))
KEY = os.environ['OPENROUTER_API_KEY']
SPEC = ROOT / 'tools' / 'voice_lines.json'
OUT = ROOT / 'web' / 'assets' / 'voice'

def speak(model, voice, text, instructions, timeout=180):
    body = {"model": model, "input": text, "voice": voice, "response_format": "pcm", "instructions": instructions}
    req = urllib.request.Request("https://openrouter.ai/api/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ct = r.headers.get('content-type', '')
        data = r.read()
    if 'pcm' not in ct: raise RuntimeError(f'unexpected response {ct}: {data[:300]!r}')
    rate = int(ct.split('rate=')[1].split(';')[0]) if 'rate=' in ct else 24000
    return data, rate

def encode(pcm, rate, path):
    # trim leading/trailing silence so the animation cues line up with the voice
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', '-',
                    '-af', 'silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,apad=pad_dur=0.08',
                    '-codec:a', 'libmp3lame', '-q:a', '3', str(path)], input=pcm, check=True)
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('ids', nargs='*'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    spec = json.load(open(SPEC)); OUT.mkdir(parents=True, exist_ok=True)
    for ln in spec['lines']:
        if a.ids and ln['id'] not in a.ids: continue
        path = OUT / f"{ln['id']}.mp3"
        if path.exists() and not a.force and not a.ids: print('skip', path.name); continue
        t = time.time()
        for attempt in range(3):
            try: pcm, rate = speak(spec['model'], spec['voice'], ln['text'], spec['persona'] + ' ' + ln['style']); break
            except Exception as e:
                print('error', ln['id'], e)
                if attempt == 2: sys.exit(1)
                time.sleep(4)
        dur = encode(pcm, rate, path)
        print(f"{ln['id']:10s} {dur:5.2f}s  ({time.time() - t:.1f}s)")
