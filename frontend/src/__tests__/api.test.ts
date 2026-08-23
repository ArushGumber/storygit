/**
 * The API client's error handling.
 *
 * This is the one piece of frontend logic that is not obviously correct by inspection, and
 * it is the piece the writer notices when it is wrong: the difference between "try again in
 * 45 seconds", "the server is not running", and "something went wrong" is the difference
 * between a tool that recovers and one that just fails.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api";

/** A minimal stand-in for a Response: the client only reads these four fields. */
function respond(status: number, body: unknown, ok = status < 400): Response {
  return {
    ok,
    status,
    statusText: `HTTP ${status}`,
    json: async () => body,
  } as unknown as Response;
}

/** A Response whose body is not JSON, for the malformed-error case. */
function badBody(status: number): Response {
  return {
    ok: false,
    status,
    statusText: "Bad Gateway",
    json: async () => {
      throw new SyntaxError("not json");
    },
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("error mapping", () => {
  it("carries the backend's typed kind, so the UI can branch on it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => respond(409, { kind: "LockedNodeError", detail: "n_1 is locked" })),
    );
    await expect(api.tree()).rejects.toMatchObject({
      kind: "LockedNodeError",
      status: 409,
      message: "n_1 is locked",
    });
  });

  it("carries retry_after, so a rate limit can say how long", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        respond(429, { kind: "RateLimited", detail: "all keys cooling", retry_after: 45 }),
      ),
    );
    try {
      await api.propose(null, "beat", "go on");
      expect.unreachable("should have thrown");
    } catch (caught) {
      const error = caught as ApiError;
      expect(error.retryAfter).toBe(45);
      expect(error.retryable).toBe(true);
    }
  });

  it("distinguishes an unreachable server from one that answered with an error", async () => {
    // Different problems, different things to tell the writer: start the server, versus
    // wait and try again.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    try {
      await api.health();
      expect.unreachable("should have thrown");
    } catch (caught) {
      const error = caught as ApiError;
      expect(error.kind).toBe("Unreachable");
      expect(error.status).toBe(0);
      expect(error.retryable).toBe(false);
    }
  });

  it("survives an error body that is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => badBody(502)));
    await expect(api.tree()).rejects.toMatchObject({ kind: "HTTP502", status: 502 });
  });

  it("treats 503 as retryable and 422 as not", async () => {
    const cases: Array<[number, boolean]> = [
      [429, true],
      [503, true],
      [422, false],
      [404, false],
      [409, false],
    ];
    for (const [status, retryable] of cases) {
      vi.stubGlobal("fetch", vi.fn(async () => respond(status, { kind: "K", detail: "d" })));
      try {
        await api.tree();
        expect.unreachable("should have thrown");
      } catch (caught) {
        expect((caught as ApiError).retryable).toBe(retryable);
      }
    }
  });
});

describe("request shapes", () => {
  it("sends JSON and posts the proposal id where the backend expects it", async () => {
    const fetchMock = vi.fn(async () => respond(200, { ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await api.accept("p_123");

    const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const [url, init] = call;
    expect(url).toBe("/api/action/accept");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ proposal_id: "p_123" });
    expect((init.headers as Record<string, string>)["content-type"]).toBe("application/json");
  });

  it("passes the branch through on reads that take one", async () => {
    const fetchMock = vi.fn(async () => respond(200, { nodes: [] }));
    vi.stubGlobal("fetch", fetchMock);

    const urls = () =>
      (fetchMock.mock.calls as unknown as Array<[string, RequestInit]>).map(([u]) => u);

    await api.tree("what-if");
    expect(urls()[0]).toBe("/api/tree?branch=what-if");

    await api.tree();
    expect(urls()[1]).toBe("/api/tree");
  });
});

describe("the calls behind the controls added last", () => {
  function mock(body: unknown) {
    const fetchMock = vi.fn(async () => respond(200, body));
    vi.stubGlobal("fetch", fetchMock);
    return () => fetchMock.mock.calls[0] as unknown as [string, RequestInit];
  }

  it("previews a merge before committing one", async () => {
    const first = mock({ clean: true, conflicts: [], summary: [] });
    await api.mergeBranches("main", "what-if", false);
    const [url, init] = first();
    expect(url).toBe("/api/branch/merge");
    expect(JSON.parse(init.body as string)).toEqual({
      ours: "main",
      theirs: "what-if",
      commit: false,
    });
  });

  it("asks for the slow audit explicitly", async () => {
    const first = mock({ flags: [], summary: "" });
    await api.flags(true);
    expect(first()[0]).toBe("/api/flags?audit=true");
  });

  it("merges entities by id, never by name", async () => {
    const first = mock({ ok: true });
    await api.mergeEntities("e_dupe", "e_kael");
    const [url, init] = first();
    expect(url).toBe("/api/entity/merge");
    expect(JSON.parse(init.body as string)).toEqual({
      source_id: "e_dupe",
      target_id: "e_kael",
    });
  });
});
