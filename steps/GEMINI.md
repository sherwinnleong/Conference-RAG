# Gemini adaptation: learning checkpoints

Keep `GEMINI_API_KEY` in `config.secret.json` only.

## 1. Test two sentences

Run `python scripts/04_embed_data.py --test`. This makes one Gemini API call,
checks that two separate 1,536-dimensional vectors are returned, and changes no data.

We use `gemini-embedding-001`, `RETRIEVAL_DOCUMENT`, and 1,536 output dimensions.
The existing pgvector schema works with this size. No new Python dependencies are needed.

## 2. Generate the sentence embeddings

Run `python scripts/04_embed_data.py`. Usage is subject to your Google account's
billing and quotas; the original assignment's OpenAI price estimate does not apply.
Progress is saved after every batch. Rerun to resume after an interruption or quota reset.
Model metadata prevents accidentally resuming OpenAI embeddings as Gemini embeddings.
Do not rerun the text import while generating embeddings: it changes sentence identities.

After generation finishes, the existing `05_update_embeddings.py` accepts the same
JSON output format. It replaces the database's sentence rows when run.

## 3. Next guided checkpoint: question Edge Function

The original Step 05 instructions use OpenAI. Instead, create `embed-question` using
`gemini-embedding-001`, 1,536 dimensions, and `RETRIEVAL_QUERY`. Documents and
questions use complementary task types from the same model. Return `{ "embedding": [...] }`.
Keep CORS and manual Supabase user verification before contacting Gemini.
Set `GEMINI_API_KEY` as a Supabase Edge Function secret; the local JSON file does not
automatically upload it. Never add the key to public configuration or browser code.

## 4. Next guided checkpoint: answer Edge Function

Create `generate-answer` with a Gemini text-generation model available to your account,
keeping the assignment's request/response format, authentication, and source citations.
This model generates prose; it is separate from the embedding model.

Reference: https://ai.google.dev/gemini-api/docs/embeddings
