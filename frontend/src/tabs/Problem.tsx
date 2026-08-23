/**
 * The Problem & Decisions tab.
 *
 * Static content, and deliberately the first tab: someone who has never heard of this
 * project should be able to read left to right — problem, design, tool, proof, numbers —
 * and understand what was built and why before they see a single screen of it.
 *
 * Each decision carries the alternative it rejected. A decision without a rejected
 * alternative is a preference.
 */

function Decision(props: {
  title: string;
  children: React.ReactNode;
  rejected: string;
}) {
  return (
    <div className="decision">
      <h3>{props.title}</h3>
      <div>{props.children}</div>
      <p className="rejected">Rejected: {props.rejected}</p>
    </div>
  );
}

export function Problem() {
  return (
    <div className="prose-page">
      <h1>storygit</h1>
      <p className="lede">
        Version control for story state. An AI-assisted storytelling system that
        develops a one-line premise into a structured serial — Story, Episodes,
        Scenes, Beats, Prose — while keeping the writer in authority over every
        change.
      </p>

      <h2>The problem</h2>
      <p>
        A writer starts with a sentence:{" "}
        <em>
          a powerless orphan discovers an ability that could change the balance
          of power in a world at war.
        </em>{" "}
        They want to end with a serial — coherent across tens of episodes, in
        their voice, with the mechanics that make a listener press{" "}
        <em>next episode</em>. Prompt-to-story fails at this in five reliable
        ways.
      </p>
      <ul>
        <li>
          <strong>State drift.</strong> Facts established early are forgotten or
          contradicted later. The sharpest version is epistemic: a character
          acts on information the story never gave them. A reader forgives an
          inconsistent hair colour; they do not forgive a character who somehow
          already knows the secret.
        </li>
        <li>
          <strong>Broken edit semantics.</strong> Change one scene and either
          nothing downstream responds — the story now contains a contradiction
          nobody flagged — or the plan regenerates and destroys work already
          approved. Both come from the same absence: nothing records what
          depends on what.
        </li>
        <li>
          <strong>Agency erosion.</strong> When the system proposes whole
          drafts, the writer's job collapses into judging them. That is a worse
          job than writing, and it is why a tool can be impressive in a demo and
          unused in a month.
        </li>
        <li>
          <strong>Exploration overwhelm.</strong> More alternatives is not more
          choice. Unlabelled variants cost more to read than they save.
        </li>
        <li>
          <strong>No learning.</strong> Iteration thirty repeats iteration one's
          mistakes.
        </li>
      </ul>
      <p>
        Serialized audio fiction adds a sixth:{" "}
        <strong>no serial mechanics</strong>. Hooks, cliffhangers, recaps, and
        the open threads a listener is waiting to see paid off are the retention
        surface of the product, and a general-purpose generator tracks none of
        them.
      </p>

      <h2>Fabula, and what its study tells us</h2>
      <p>
        Fabula (DeepMind, 2026) is the closest published system: a two-level
        hierarchical plan, then a script, built by prompt-chaining orchestrators
        with structured output, evaluated with 42 writers.
      </p>
      <p>
        Its study's findings matter more than its architecture, and they are
        unusually specific.
      </p>
      <blockquote>
        Writers liked the structure, the scene breaking, and the
        plan-beside-script view. They disliked generic prose and poor handling
        of style and irony. They found it <em>rigid</em>: they could not add a
        character locally, edits rewrote the whole plan, stale downstream scenes
        were not flagged. They asked for locks, insert and delete controls, and
        an “absurdity dial”. The authors' own conclusion is that the system is
        most useful as feedback on the writer's own script, not as a generator.
      </blockquote>
      <p>
        Three of those complaints — edits rewriting the plan, stale scenes going
        unflagged, the inability to add a character locally — are the same
        missing abstraction seen from three angles:{" "}
        <strong>there is no explicit dependency graph</strong>. This builds that
        first and derives the rest from it.
      </p>

      <table>
        <thead>
          <tr>
            <th>Gap</th>
            <th>What storygit does about it</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>No dependency tracking; edits rewrite the plan</td>
            <td>
              Beats declare what they establish and what they rely on;
              propagation <em>marks</em> affected nodes and never rewrites them
            </td>
          </tr>
          <tr>
            <td>No per-character knowledge state</td>
            <td>
              Epistemic edges, checked before a character may act on something
              they were never told
            </td>
          </tr>
          <tr>
            <td>No learning from writer signals</td>
            <td>A preference layer trained on accept, reject, and edit</td>
          </tr>
          <tr>
            <td>No serial mechanics</td>
            <td>
              Episode hooks, cliffhangers, recaps, and an open-thread ledger
            </td>
          </tr>
          <tr>
            <td>Diversity is “do not repeat the last one”</td>
            <td>Axis-conditioned sampling with MMR or DPP selection</td>
          </tr>
        </tbody>
      </table>

      <h2>The thesis</h2>
      <p>
        Treat the story as a <strong>versioned, typed state graph</strong>. Make
        every AI action a <strong>reviewable diff</strong> against it. Make
        dependencies <strong>explicit</strong>, so an edit propagates by{" "}
        <em>marking</em> what it affected and never by rewriting it. Let the
        writer <strong>own the objective</strong>. And{" "}
        <strong>learn their taste</strong> from the signals they are already
        producing.
      </p>
      <p>
        The division of labour follows:{" "}
        <strong>
          the AI handles coherence and mechanics; the human handles taste,
          voice, and retention.
        </strong>
      </p>
      <p>
        Stated as a difference: Fabula is a chain of prompts with a user
        interface. This is a state machine with a learning loop.
      </p>

      <h2>Decisions</h2>

      <Decision
        title="Free-tier keys for all of it, with the metered budget held in reserve"
        rejected="paying for a strong model from day one."
      >
        Every number in this project was produced on free-tier keys — six
        rotated Gemini keys for generation and judging, Groq for extraction, CPU
        models on one laptop for embeddings and NLI.{" "}
        <strong>Total spend across roughly ten million tokens: $0.00.</strong>{" "}
        That is a decision about where risk sits. A metered key from day one
        buys better prose immediately and hides every cost bug until the bill
        arrives; a free tier forces the cost controls to exist before they are
        needed — purpose-tag routing, a read-through cache, key rotation with
        per-model cooldowns, a budget guard that refuses a call it cannot afford
        — and those are the same controls a production system needs. The metered
        budget is consequently unspent, and reserved for the strong-model rerun
        that separates “the system works” from “the model is good”.
      </Decision>

      <Decision
        title="Diffs rather than replacement text"
        rejected="text-level replacement with a visual diff — it looks similar on screen and is far weaker underneath."
      >
        Every AI action is a typed operation list against the current state. A
        text diff cannot say “this introduces a character”, cannot be validated
        against the world graph before it is applied, and cannot tell you what
        it would invalidate. Typed operations do all three, which is what turns
        “review this draft” into “authorize these changes”.
      </Decision>

      <Decision
        title="Declared dependencies rather than inferred ones"
        rejected="inferring dependencies with a model."
      >
        A dependency oracle that is wrong one time in ten makes every mark
        untrustworthy, and the writer stops reading marks — at which point the
        feature is worse than useless, because it also costs money.
        Embedding-similarity edges are implemented, but as a visibly weaker
        “maybe affected” signal and as an ablation to measure.
      </Decision>

      <Decision
        title="A closed predicate vocabulary"
        rejected="open-vocabulary relation extraction."
      >
        Ten predicates plus a free-text escape hatch. Closed predicates make the
        first continuity check a dictionary lookup and an equality test — free,
        instant, and never wrong for a subtle reason. The escape hatch is
        exactly the set of facts that needs the expensive NLI check, which is a
        useful way to have partitioned the problem.
      </Decision>

      <Decision
        title="Marking rather than auto-repair"
        rejected="automatically regenerating stale downstream nodes."
      >
        The most tempting feature in the system, and the one that would destroy
        it. It is precisely what Fabula's writers complained about, and the
        objection is not technical: a system that rewrites approved work without
        asking cannot be trusted with a manuscript.
      </Decision>

      <Decision
        title="Axis-conditioned sampling rather than a similarity penalty"
        rejected="sampling k times and reranking for dissimilarity — Fabula's approach."
      >
        Candidates are generated under named instructions, and the name reaches
        the writer. Three named directions cost a decision; three unlabelled
        paragraphs cost three readings. The reranker is still there — it stops
        two candidates that came out similar anyway — but the axes are what make
        the choice a choice.
      </Decision>

      <Decision
        title="Thirteen interpretable features rather than a learned representation"
        rejected="embedding the candidate and letting the ranking head learn its own features."
      >
        With twenty comparisons that head would memorize rather than generalize,
        and its weights would say nothing a writer could read. An interpretable
        feature vector is what lets the system explain its own ranking back to
        the person it is ranking for.
      </Decision>

      <h2>What the measurements changed</h2>
      <p>
        Two predictions written before the runs turned out wrong, and both
        changed the system rather than the write-up.
      </p>
      <p>
        <strong>
          MMR at the conventional λ = 0.7 was identical to the top-k baseline.
        </strong>{" "}
        Bi-encoder cosine has a high floor, so a nearly constant redundancy
        penalty changes no ordering — the headline selector was reproducing the
        ablation it was meant to beat. Fixed by rescaling similarity within the
        candidate set and moving the default to 0.5, chosen from a sweep. The
        same measurement produced a better argument for DPPs than the one
        originally written: the DPP reaches the diverse answer with{" "}
        <em>no parameter at all</em>, because its kernel is multiplicative.
      </p>
      <p>
        <strong>Embedding dependency edges did better than predicted</strong>,
        recovering an undeclared dependency at no precision cost above a
        threshold of 0.68 — the opposite of the stated prediction.
      </p>
      <p>
        Both are recorded as found, because a system that only reports the
        measurements which agreed with it has not been measured.
      </p>

      <h2>What this cannot show</h2>
      <p>
        The evaluation uses simulated writers, so it measures the machinery and
        not taste. A persona is a linear functional over the same thirteen
        features plus noise: it cannot be surprised, cannot change its mind, and
        cannot tell you the tool is exhausting to use. Whether this helps a
        human write a better serial is not established here, and only a human
        study would establish it.
      </p>
    </div>
  );
}
