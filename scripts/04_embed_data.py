"""Generate Gemini document embeddings. Run with --test for a two-sentence check."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'scripts/output/sentences.json'
OUTPUT_FILE = ROOT / 'scripts/output/sentences_with_embeddings.json'
MODEL = 'gemini-embedding-001'
DIMENSIONS = 1536
BATCH_SIZE = 50
BATCH_INTERVAL = 35  # 50 sentences / 35 seconds is about 86 per minute.
METADATA = {'provider': 'gemini', 'model': MODEL, 'dimensions': DIMENSIONS,
            'task_type': 'RETRIEVAL_DOCUMENT'}


def save_progress(records):
    temporary = OUTPUT_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, OUTPUT_FILE)


def embed_batch(records, api_key):
    # One request per sentence inside the batch gives one vector per sentence.
    payload = {'requests': [
        {'model': f'models/{MODEL}',
         'content': {'parts': [{'text': record['text']}]},
         'taskType': 'RETRIEVAL_DOCUMENT',
         'outputDimensionality': DIMENSIONS}
        for record in records
    ]}
    for attempt in range(6):
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents',
            headers={'x-goog-api-key': api_key}, json=payload, timeout=120)
        if response.status_code == 429:
            try:
                details = response.json().get('error', {}).get('details', [])
                quota_ids = [v.get('quotaId', '').lower()
                             for detail in details for v in detail.get('violations', [])]
            except (ValueError, AttributeError, TypeError):
                quota_ids = []
            if any('perday' in quota or 'daily' in quota for quota in quota_ids):
                raise RuntimeError('Daily Gemini quota reached. Progress is preserved; resume after quota resets.')
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 5:
                delay = 65 if response.status_code == 429 else min(2 ** (attempt + 1), 60)
                print(f'Gemini HTTP {response.status_code}; retrying in {delay}s.')
                time.sleep(delay)
                continue
        if not response.ok:
            # Do not print request headers or raw errors that might contain secrets.
            raise RuntimeError(f'Gemini HTTP {response.status_code}. Check key, model access, billing and quota.')
        items = response.json().get('embeddings', [])
        if len(items) != len(records):
            raise ValueError('Gemini returned an unexpected number of embeddings.')
        result = []
        for record, item in zip(records, items):
            vector = item['values']
            if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
                raise ValueError('Invalid embedding dimensions or values.')
            magnitude = math.sqrt(sum(x * x for x in vector))
            if magnitude == 0:
                raise ValueError('Received a zero vector.')
            # Normalize reduced-dimension embeddings, as Google recommends.
            result.append({**record, 'embedding': [x / magnitude for x in vector]})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Embed two examples without saving output')
    args = parser.parse_args()
    secrets = json.loads((ROOT / 'config.secret.json').read_text(encoding='utf-8-sig'))
    api_key = secrets.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('Add GEMINI_API_KEY to config.secret.json first.')
    if args.test:
        result = embed_batch([{'text': 'Faith can bring comfort during difficult times.'},
                              {'text': 'Prayer can help us find peace.'}], api_key)
        print(f'Success: {len(result)} separate embeddings, {len(result[0]["embedding"])} dimensions each.')
        print('No output files or database rows were changed.')
        return

    records = json.loads(INPUT_FILE.read_text(encoding='utf-8'))
    metadata_file = OUTPUT_FILE.with_suffix('.meta.json')
    existing = []
    if OUTPUT_FILE.exists():
        if not metadata_file.exists() or json.loads(metadata_file.read_text()) != METADATA:
            raise ValueError('Existing embeddings have a different or unknown model. Move the old output file aside before generating Gemini embeddings.')
        existing = json.loads(OUTPUT_FILE.read_text(encoding='utf-8'))
        if len(existing) > len(records) or any(
            {k: v for k, v in saved.items() if k != 'embedding'} != original
            or len(saved.get('embedding', [])) != DIMENSIONS
            for saved, original in zip(existing, records)
        ):
            raise ValueError('Saved embeddings do not match the current sentences. Move old output aside first.')
    else:
        metadata_file.write_text(json.dumps(METADATA, indent=2), encoding='utf-8')

    print(f'Model: {MODEL}; dimensions: {DIMENSIONS}; sentences: {len(records):,}')
    print(f'Resuming after {len(existing):,} saved sentences.')
    stop = len(records)
    print(f'Pacing: up to {BATCH_SIZE} sentences every {BATCH_INTERVAL}s; no per-run sentence cap.')
    print('Waiting 65s initially to let earlier requests leave the minute window.')
    try:
        if len(existing) < stop:
            time.sleep(65)
        for start in tqdm(range(len(existing), stop, BATCH_SIZE), desc='Embedding'):
            request_started = time.monotonic()
            batch = embed_batch(records[start:min(start + BATCH_SIZE, stop)], api_key)
            existing.extend(batch)
            save_progress(existing)
            if len(existing) < stop:
                time.sleep(max(0, BATCH_INTERVAL - (time.monotonic() - request_started)))
    except (Exception, KeyboardInterrupt):
        # Stop at the first failed batch so the saved output remains an exact prefix.
        save_progress(existing)
        print(f'Progress saved: {len(existing):,}/{len(records):,}. Rerun to resume.')
        raise
    print(f'Complete: {len(existing):,} embeddings saved to {OUTPUT_FILE.name}.')
    print('Next learning step: upload these embeddings with scripts/05_update_embeddings.py.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if isinstance(error, (RuntimeError, ValueError, FileNotFoundError)):
            print(f'Error: {error}')
        else:
            print(f'Error: {type(error).__name__}. Progress is preserved; check connectivity and retry.')
        sys.exit(1)

