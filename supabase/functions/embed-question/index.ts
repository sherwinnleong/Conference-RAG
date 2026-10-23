import { corsHeaders, jsonResponse } from "../_shared/cors.ts";
import { verifyUser } from "../_shared/auth.ts";

const MODEL = "gemini-embedding-001";
const DIMENSIONS = 1536;

export async function handler(request: Request): Promise<Response> {
  // Browsers send this preflight without a login token, before the POST.
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }
  if (request.method !== "POST") {
    return jsonResponse({ error: "Use POST with a question." }, 405);
  }

  try {
    // Authenticate before spending any Gemini quota.
    if (!await verifyUser(request)) {
      return jsonResponse({ error: "Sign in to ask a question." }, 401);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return jsonResponse({ error: "Request body must be valid JSON." }, 400);
    }
    const question = body?.question;
    if (typeof question !== "string" || !question.trim() || question.length > 4000) {
      return jsonResponse({ error: "question must contain 1–4000 characters." }, 400);
    }

    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) {
      return jsonResponse({ error: "GEMINI_API_KEY is not configured on Supabase." }, 500);
    }

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:embedContent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          model: `models/${MODEL}`,
          content: { parts: [{ text: question.trim() }] },
          // Documents use RETRIEVAL_DOCUMENT; questions use its matching query task.
          taskType: "RETRIEVAL_QUERY",
          outputDimensionality: DIMENSIONS,
        }),
        signal: AbortSignal.timeout(30000),
      },
    );
    if (!response.ok) {
      console.error("Gemini embedding request failed, HTTP", response.status);
      return jsonResponse({
        error: response.status === 429
          ? "Gemini quota reached. Please try again later."
          : "The embedding provider could not process the question.",
      }, response.status === 429 ? 429 : 502);
    }

    const result = await response.json();
    const values = result.embedding?.values;
    if (!Array.isArray(values) || values.length !== DIMENSIONS ||
        !values.every((value: unknown) => typeof value === "number" && Number.isFinite(value))) {
      return jsonResponse({ error: "Provider returned an invalid embedding." }, 502);
    }
    const magnitude = Math.sqrt(values.reduce((sum: number, x: number) => sum + x * x, 0));
    if (!Number.isFinite(magnitude) || magnitude === 0) {
      return jsonResponse({ error: "Provider returned an invalid embedding." }, 502);
    }
    return jsonResponse({ embedding: values.map((x: number) => x / magnitude) });
  } catch {
    // Avoid logging credentials, tokens, or the user's question.
    console.error("embed-question could not complete its upstream request");
    return jsonResponse({ error: "Unable to embed the question. Please try again." }, 503);
  }
}

Deno.serve(handler);
