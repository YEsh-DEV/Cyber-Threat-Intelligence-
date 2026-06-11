import json
import random
from pathlib import Path
import sys

sys.path.insert(0, 'Y:/Reserchintern/Experiment2')
from preprocessing import preprocess

events = sorted(preprocess.load_cached_events(), key=lambda x: x.get('global_id', ''))
random.seed(42)
sampled = random.sample(events, 20)

for idx, e in enumerate(sampled):
    narr = e.get('info', '')
    if not narr: narr = e.get('narrative', '')
    gid = e.get("global_id", "")
    print(f'Event {idx+1}: {gid}')
    print(f'Length: {len(narr)}, Words: {len(narr.split())}')
    print(narr[:200])
    print('-'*40)
