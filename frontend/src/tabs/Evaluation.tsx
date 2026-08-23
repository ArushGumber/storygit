/**
 * The Eval tab: the numbers, with what each one demonstrates.
 *
 * Every figure carries a caption saying what it shows, because a curve without a claim is
 * decoration. The offline metrics are separated from the live runs, since one is exact and
 * the other depends on what a model produced on the day — and that distinction is the
 * point rather than a footnote.
 */

import { useEffect, useState } from "react";
import { ApiError, api, type EvalPlot } from "../api";

interface RunRow {
  run: string;
  decisions: number;
  acceptance: number;
  acceptance_first_third: number;
  acceptance_last_third: number;
  mean_edit_distance: number;
  weight_recovery: number | null;
  tokens_per_action: number;
  errors: string[];
}

export function Evaluation() {
  const [plots, setPlots] = useState<EvalPlot[]>([]);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const [plotResponse, summaryResponse] = await Promise.all([
          api.evalPlots(),
          api.evalSummary(),
        ]);
        setPlots(plotResponse.plots);
        setSummary(summaryResponse);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    })();
  }, []);

  const available = summary?.available === true;
  const runs = (summary?.runs as RunRow[] | undefined) ?? [];
  const skipped =
    (summary?.skipped as Array<Record<string, string>> | undefined) ?? [];
  const offline = summary?.offline as Record<string, any> | undefined;

  return (
    <div className="prose-page">
      <h1>Evaluation</h1>
      <p className="lede">
        Every claim is either measured here or marked as unmeasured. The
        deterministic tier needs no model calls, so it is exact — a regression
        is a regression rather than a bad sampling day, and it cannot be
        improved by rerunning until the numbers look better.
      </p>

      {error && <div className="error">{error}</div>}
      {!available && !error && (
        <p className="empty">
          No results yet. Run <code>python -m eval.run --config full</code>.
        </p>
      )}

      {offline?.checker_ablation && (
        <>
          <h2>Continuity checker</h2>
          <table>
            <thead>
              <tr>
                <th>configuration</th>
                <th className="num">recall</th>
                <th className="num">precision</th>
                <th className="num">F1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>layer 1 only (deterministic)</td>
                <td className="num">
                  {fmtPct(offline.checker_ablation.layer1_only.recall)}
                </td>
                <td className="num">
                  {fmtPct(offline.checker_ablation.layer1_only.precision)}
                </td>
                <td className="num">
                  {offline.checker_ablation.layer1_only.f1.toFixed(2)}
                </td>
              </tr>
              <tr>
                <td>layer 1 + layer 2 (NLI)</td>
                <td className="num">
                  {fmtPct(offline.checker_ablation.layer1_and_2.recall)}
                </td>
                <td className="num">
                  {fmtPct(offline.checker_ablation.layer1_and_2.precision)}
                </td>
                <td className="num">
                  {offline.checker_ablation.layer1_and_2.f1.toFixed(2)}
                </td>
              </tr>
            </tbody>
          </table>
          <p className="hint">
            {offline.checker?.false_positives_per_beat?.toFixed(2)} false
            positives per beat on a clean story. That number matters more than
            recall — a checker that flags everything has perfect recall and is
            worthless, because the writer learns flags are noise.
          </p>
        </>
      )}

      {offline?.stale_sweep && (
        <>
          <h2>Staleness prediction</h2>
          <table>
            <thead>
              <tr>
                <th>configuration</th>
                <th className="num">precision</th>
                <th className="num">recall</th>
                <th className="num">F1</th>
              </tr>
            </thead>
            <tbody>
              {(offline.stale_sweep.points as Array<any>).map((point) => (
                <tr key={point.label}>
                  <td>{point.label}</td>
                  <td className="num">{point.precision.toFixed(2)}</td>
                  <td className="num">{point.recall.toFixed(2)}</td>
                  <td className="num">{point.f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Declared dependency edges are exact and blind to what nobody
            declared. The prediction stated in advance was that embedding edges
            would trade recall for precision too poorly to use; on this case
            they do better than that, which is recorded as found rather than
            explained away.
          </p>
        </>
      )}

      {offline?.selector_diversity && (
        <>
          <h2>Selector diversity</h2>
          <table>
            <thead>
              <tr>
                <th>selector</th>
                <th className="num">mean pairwise distance</th>
                <th className="num">mean quality</th>
              </tr>
            </thead>
            <tbody>
              {(offline.selector_diversity.points as Array<any>).map(
                (point) => (
                  <tr key={point.label}>
                    <td>{point.label}</td>
                    <td className="num">{point.diversity.toFixed(3)}</td>
                    <td className="num">{point.quality.toFixed(3)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
          <p className="hint">
            Six candidates whose three highest-scoring entries are
            near-paraphrases of each other. The temperature-only baseline takes
            all three.
          </p>
        </>
      )}

      {runs.length > 0 && (
        <>
          <h2>Live runs</h2>
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th className="num">decisions</th>
                <th className="num">acceptance</th>
                <th className="num">first third</th>
                <th className="num">last third</th>
                <th className="num">mean edit</th>
                <th className="num">weight recovery</th>
                <th className="num">tokens/action</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((row) => (
                <tr key={row.run}>
                  <td>{row.run}</td>
                  <td className="num">{row.decisions}</td>
                  <td className="num">{fmtPct(row.acceptance)}</td>
                  <td className="num">{fmtPct(row.acceptance_first_third)}</td>
                  <td className="num">{fmtPct(row.acceptance_last_third)}</td>
                  <td className="num">{row.mean_edit_distance.toFixed(2)}</td>
                  <td className="num">
                    {row.weight_recovery === null
                      ? "—"
                      : row.weight_recovery.toFixed(2)}
                  </td>
                  <td className="num">{Math.round(row.tokens_per_action)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Weight recovery is the sharpest number here: the correlation between
            what the preference head learned and the persona's hidden weights.
            Not “did acceptance rise” — any degenerate system achieves that —
            but “did the machinery recover the taste it was shown”.
          </p>
        </>
      )}

      {skipped.length > 0 && (
        <>
          <h2>Not run</h2>
          <table>
            <thead>
              <tr>
                <th>configuration</th>
                <th>persona</th>
                <th>why</th>
              </tr>
            </thead>
            <tbody>
              {skipped.map((row, index) => (
                <tr key={index}>
                  <td>{row.config}</td>
                  <td>{row.persona}</td>
                  <td>{row.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Listed because a results table that silently omits the runs that did
            not happen is a lie.
          </p>
        </>
      )}

      {plots.length > 0 && (
        <>
          <h2>Figures</h2>
          {plots.map((plot) => (
            <figure className="figure" key={plot.name}>
              <img src={plot.url} alt={plot.name} />
              <figcaption>{plot.caption || plot.name}</figcaption>
            </figure>
          ))}
        </>
      )}

      <h2>What this cannot measure</h2>
      <p>
        Real taste, fatigue, trust, or whether a listener would press next
        episode. A persona is a linear functional over the same thirteen
        features plus noise; it cannot be surprised and cannot change its mind.
        Worse, it is built from the same hypothesis class the preference head is
        fitted on, so the head is correctly specified by construction. That is
        exactly why the claim is <em>recoverability</em> — the estimator is not
        broken — rather than <em>it learns taste</em>.
      </p>
    </div>
  );
}

function fmtPct(value: number | undefined): string {
  return value === undefined ? "—" : `${Math.round(value * 100)}%`;
}
