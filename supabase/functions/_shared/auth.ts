// Ask Supabase Auth to verify the token; decoding a JWT alone is not verification.
export async function verifyUser(request: Request) {
  const authorization = request.headers.get("Authorization");
  if (!authorization || !/^Bearer\s+\S+$/i.test(authorization)) return false;

  const url = Deno.env.get("SUPABASE_URL");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  if (!url || !anonKey) throw new Error("Supabase Auth configuration is missing");

  const response = await fetch(`${url}/auth/v1/user`, {
    headers: { Authorization: authorization, apikey: anonKey },
    signal: AbortSignal.timeout(15000),
  });
  if (response.status === 401 || response.status === 403) return false;
  if (!response.ok) throw new Error("Supabase Auth is unavailable");
  const user = await response.json();
  return typeof user.id === "string" && !user.is_anonymous;
}
