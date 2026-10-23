"""
Step 5: Import Embeddings to Database
=========================================
Reads scripts/output/sentences_with_embeddings.json and imports
all records (text + embeddings) to Supabase, replacing any
existing data.

After this step, SEMANTIC SEARCH will light up green on your site!

Usage:
    python scripts/05_update_embeddings.py

Input:
    scripts/output/sentences_with_embeddings.json  — from Step 4

Prerequisites:
    - config.public.json with Supabase URL and anon key
    - config.secret.json with Supabase service key
    - Database schema created (Step 1)
    - Embeddings generated (Step 4)
"""

import json
import os
import sys
import time
import uuid

import httpx
from postgrest.exceptions import APIError

from supabase import create_client
from tqdm import tqdm


INPUT_FILE = os.path.join('scripts', 'output', 'sentences_with_embeddings.json')
BATCH_SIZE = 20


def upload_batch(client, batch):
    # A failed response does not prove the server failed to save the rows.
    # Stable IDs plus upsert make repeating the same batch safe.
    for attempt in range(5):
        try:
            client.table('sentence_embeddings').upsert(batch, on_conflict='id').execute()
            return
        except (httpx.TransportError, APIError) as error:
            is_connection_error = isinstance(error, httpx.TransportError) or any(
                marker in str(error).lower()
                for marker in ('bad_record_mac', 'connection', 'timeout'))
            if not is_connection_error or attempt == 4:
                raise
            delay = 2 ** (attempt + 1)
            print(f'\nConnection interrupted; retrying this batch in {delay}s ({attempt + 1}/4).')
            time.sleep(delay)


def load_config():
    with open('config.public.json', 'r') as f:
        public_config = json.load(f)
    with open('config.secret.json', 'r') as f:
        secrets = json.load(f)
    return public_config, secrets


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run scripts/04_embed_data.py first.")
        sys.exit(1)

    # Load embedded records
    print("Loading embeddings data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if not records:
        sys.exit('No embeddings to import; existing database rows were not changed.')
    records = [dict(record, id=str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"conference-rag:{record['talk_id']}:{record['sentence_num']}"
    ))) for record in records]

    # Verify embeddings are present
    with_embeddings = sum(1 for r in records if r.get('embedding'))
    without_embeddings = len(records) - with_embeddings

    print(f"   Loaded {len(records):,} records")
    print(f"   With embeddings:    {with_embeddings:,}")
    if without_embeddings:
        print(f"   Without embeddings: {without_embeddings:,} (will be imported without)")

    # Connect to Supabase
    public_config, secrets = load_config()
    client = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

    # Truncate existing data and re-import everything
    print("\n" + "=" * 60)
    print("Replacing database contents (truncate + re-import)")
    print("=" * 60)

    try:
        result = client.table('sentence_embeddings').select('id', count='exact').limit(1).execute()
        existing_count = result.count or 0
        if existing_count > 0:
            print(f"   Truncating {existing_count:,} existing rows...")
            client.table('sentence_embeddings').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            print("   ✅ Table truncated.")
        else:
            print("   Table is empty — ready for import.")
    except Exception as e:
        print(f"   ⚠️ Could not check existing data: {e}")
        print("   Stopping: existing data could not be safely checked or cleared.")
        sys.exit(1)

    # Batch import (much faster than individual updates!)
    print(f"\n   Importing {len(records):,} records in batches of {BATCH_SIZE}...\n")

    success = 0
    errors = 0

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Importing"):
        batch = records[i:i + BATCH_SIZE]
        try:
            upload_batch(client, batch)
            success += len(batch)
        except Exception as e:
            print(f"\nError at batch {i // BATCH_SIZE}: {e}")
            errors += len(batch)
            print('Stopping at the failed batch; the import is incomplete.')
            break
        time.sleep(0.1)

    print("\nImport incomplete." if errors else "\nAll batches submitted; verifying import...")
    print(f"   Success: {success:,}")
    if errors:
        print(f"   Errors:  {errors:,}")
        print("   Saved embeddings are unchanged. Fix the connection and rerun to replace this partial import.")
        sys.exit(1)

    # Verify
    result = client.table('sentence_embeddings').select('id', count='exact').limit(1).execute()
    total = result.count or 0

    embedded_result = client.table('sentence_embeddings') \
        .select('id', count='exact') \
        .not_.is_('embedding', 'null') \
        .limit(1).execute()
    with_emb = embedded_result.count or 0

    print(f"\n   Total rows:          {total:,}")
    print(f"   Rows with embeddings: {with_emb:,}")
    if total != len(records) or with_emb != with_embeddings:
        print("Import verification failed: database counts do not match the input file.")
        sys.exit(1)

    print("\nEmbedding import verified.")
    print("   Semantic Search also requires a matching embed-question Edge Function.")
    print(f"\nNext: Deploy edge functions to light up 🤖 RAG!")


if __name__ == '__main__':
    main()
