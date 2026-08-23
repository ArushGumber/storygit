/**
 * The shell: five tabs, in the order the story is told.
 *
 * Problem → Architecture → Live → Gallery → Evaluation. Someone who has never heard of
 * this project should be able to read left to right and understand what was built, how it
 * works, what it does, that it does it, and how well.
 *
 * The tab lives in the URL hash, so a reload keeps the reader where they were and a link
 * to a specific tab works.
 */

import { useEffect, useState } from "react";
import { api, type Health } from "./api";
import { Architecture } from "./tabs/Architecture";
import { Evaluation } from "./tabs/Evaluation";
import { Gallery } from "./tabs/Gallery";
import { Live } from "./tabs/Live";
import { LiveStatic } from "./tabs/LiveStatic";
import { Problem } from "./tabs/Problem";

const TABS = [
  { id: "problem", label: "Problem & Decisions", render: () => <Problem /> },
  { id: "architecture", label: "Architecture", render: () => <Architecture /> },
  // The static build has no Python process behind it, so the tool becomes screenshots
  // plus the commands to run it. Every other tab reads pre-rendered JSON and works
  // unchanged, which is why only this one is swapped.
  {
    id: "live",
    label: "Live",
    render: () => (import.meta.env.VITE_STATIC ? <LiveStatic /> : <Live />),
  },
  { id: "gallery", label: "Gallery", render: () => <Gallery /> },
  { id: "eval", label: "Eval", render: () => <Evaluation /> },
] as const;

type TabId = (typeof TABS)[number]["id"];

function currentHash(): TabId {
  const hash = window.location.hash.replace("#", "");
  return (TABS.find((tab) => tab.id === hash)?.id ?? "problem") as TabId;
}

export function App() {
  const [tab, setTab] = useState<TabId>(currentHash);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    const onHash = () => setTab(currentHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    void api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, [tab]);

  const active = TABS.find((entry) => entry.id === tab) ?? TABS[0];

  return (
    <div className="app">
      <header className="masthead">
        <span className="wordmark">
          storygit<span className="tag">version control for story state</span>
        </span>
        {health && (
          <span className="status">
            {health.branch} · {health.nodes}{" "}
            {health.nodes === 1 ? "node" : "nodes"} · {health.facts}{" "}
            {health.facts === 1 ? "fact" : "facts"}
            {health.openrouter_enabled ? "" : " · metered provider locked"}
          </span>
        )}
        <nav className="tabs">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              aria-selected={entry.id === tab}
              onClick={() => {
                window.location.hash = entry.id;
                setTab(entry.id);
              }}
            >
              {entry.label}
            </button>
          ))}
        </nav>
      </header>
      <main>{active.render()}</main>
    </div>
  );
}
