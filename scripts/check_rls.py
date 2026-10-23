"""Compare anonymous read access without loading any secret credentials.

Run from the project folder: python scripts/check_rls.py
"""

import json
from pathlib import Path

from supabase import create_client
from postgrest.exceptions import APIError


project_root = Path(__file__).resolve().parent.parent
with (project_root / "config.public.json").open(encoding="utf-8") as file:
    public = json.load(file)

# No user signs in, so these requests use the anonymous role.
client = create_client(public["SUPABASE_URL"], public["SUPABASE_ANON_KEY"])

for table in ("page_views", "sentence_embeddings"):
    try:
        result = client.table(table).select("id").limit(5).execute()
        print(f"{table}: {len(result.data)} visible rows (maximum 5)")
    except APIError as error:
        print(f"{table}: database error {error.code}: {error.message}")
    except Exception as error:
        print(f"{table}: query failed ({type(error).__name__})")

print("\nExpected: page_views permits anonymous reads; sentence_embeddings shows 0 rows.")
print("An empty result alone cannot distinguish an empty table from RLS filtering.")
