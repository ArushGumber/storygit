/**
 * The Architecture tab: the diagrams, with a paragraph each.
 *
 * The SVGs are the same files the paper includes, generated from one TikZ style file whose
 * palette is the palette this page uses — so a diagram belongs here rather than merely
 * coexisting with the page.
 */

const DIAGRAMS = [
  {
    file: "state_model.svg",
    title: "The state model",
    caption:
      "Three stores. The plan tree holds structure; the world graph holds truth, with each fact valid over an interval of the beat sequence so a change does not destroy its own history; snapshots hold every version. A beat's produces and consumes lists are the only dependency edges propagation walks, which is what makes the walk exact rather than a guess.",
  },
  {
    file: "routing.svg",
    title: "Model routing",
    caption:
      "Purpose tags decide the provider, so the entire cost policy is six lines in one table. Rotation, caching, budget, and logging are layered around the backends rather than baked into each one. The metered provider is locked by two independent fail-closed checks.",
  },
  {
    file: "checker_layers.svg",
    title: "The continuity checker",
    caption:
      "Three layers, cheapest and most certain first. Layer 1 is deterministic and, by construction, cannot call a model — a test parses its imports to prove it. Layer 2 handles only the free-text residue layer 1 cannot decide. Layer 3 is an opinion and is labelled as one. Each layer's recall is reported separately, which is what turns “it keeps the story consistent” from an assertion into a measurement.",
  },
  {
    file: "selection.svg",
    title: "Candidate selection",
    caption:
      "Six candidates generated under named instructions become three labelled options. The label is the mechanism, not decoration: three named directions cost a decision, three unlabelled paragraphs cost three readings. The other three are kept — the evaluation measures diversity over the whole set, and the preference layer needs them as the negative side of a comparison.",
  },
  {
    file: "preference_loop.svg",
    title: "The preference loop",
    caption:
      "Four learners with different data appetites, so each contributes as soon as it has enough: exemplar retrieval from the first accepted paragraph, the edit-direction vector from the first edit, the ranking head from about ten comparisons. Every one degrades to “no opinion” at zero data rather than to a confident guess.",
  },
  {
    file: "eval_loop.svg",
    title: "The evaluation loop",
    caption:
      "Simulated writers defined over the same feature space the preference head is fitted on, which turns the central question into a measurable one: not “did acceptance rise” but “did the machinery recover the taste it was shown”. The engine is never told the hidden weights.",
  },
];

export function Architecture() {
  return (
    <div className="prose-page">
      <h1>Architecture</h1>
      <p className="lede">
        Five components and an interface. Everything below the interface is a pure Python
        library with no web machinery in it, which is why chunks of it can be tested without
        a network and measured without a server.
      </p>

      {DIAGRAMS.map((diagram) => (
        <figure className="figure" key={diagram.file}>
          <h2>{diagram.title}</h2>
          <img src={`/diagrams/${diagram.file}`} alt={diagram.title} />
          <figcaption>{diagram.caption}</figcaption>
        </figure>
      ))}

      <h2>The packages</h2>
      <table>
        <thead>
          <tr>
            <th>Package</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="mono">domain/</td>
            <td>
              The typed state and the 31 diff operations that are the only way to change any
              of it. Frozen models, a pure apply function, typed errors.
            </td>
          </tr>
          <tr>
            <td className="mono">store/</td>
            <td>
              Content-addressed objects, snapshot manifests, branch pointers, three-way
              merge. Git's data model in one SQLite file.
            </td>
          </tr>
          <tr>
            <td className="mono">graph/</td>
            <td>
              Dependency edges, propagation that marks and never rewrites, and the
              entity-scoped slices that generation prompts consume.
            </td>
          </tr>
          <tr>
            <td className="mono">providers/</td>
            <td>
              One interface over four backends, with key rotation, a read-through cache, a
              budget guard, and a call log. No key is ever logged.
            </td>
          </tr>
          <tr>
            <td className="mono">agents/</td>
            <td>Schema-constrained generation converted into typed diffs, and extraction.</td>
          </tr>
          <tr>
            <td className="mono">selection/</td>
            <td>Named axes, MMR / DPP / top-k behind one signature, and the dial.</td>
          </tr>
          <tr>
            <td className="mono">continuity/</td>
            <td>The three layers, the bible diff, and the periodic audit.</td>
          </tr>
          <tr>
            <td className="mono">preference/</td>
            <td>
              Exemplars, edit mining, the voice model, the ranking head, and the bandit.
            </td>
          </tr>
          <tr>
            <td className="mono">api/</td>
            <td>A thin adapter. No logic lives here.</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
