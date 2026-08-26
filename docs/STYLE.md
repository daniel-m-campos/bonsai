# bonsai documentation style

Rules for writing and reviewing anything under `docs/`.

## Where things live

The routing table. Every piece of written content has exactly one normative home; everywhere else cites it. This table is the policy, adopted by the footprint decision in [decisions.md](decisions.md); [CLAUDE.md](../CLAUDE.md) points here rather than restating it.

| content | home | kept true by |
|---|---|---|
| conventions and the never-do list | `CLAUDE.md` | loaded every agent session |
| single-file constraints: what breaks if you change this | a comment at the definition site | the diff that breaks it touches the same file |
| cross-file contracts | a test, indexed into [invariants.md](invariants.md) | the test fails when the behavior changes |
| rationale, measurements, rejected alternatives | [decisions.md](decisions.md), cited by number | append-only; corrections are status banners |
| pedagogy | `guide/` and `learn/` | docs CI runs the examples |
| adopter reference | `use/` and `method/`, generated where possible | generators re-render in `docs-check` |
| procedures | `ops/` and `.claude/skills/` | exercised by the rituals that use them |

Two tests route a claim:

- Falsifiable by reading one source file? It is a comment on that file, not a doc.
- Do several sites have to change together when it changes? It is an invariant, and the other sites link to it.

A claim that fails both tests is either rationale (decisions.md) or teaching (guide/, learn/). There is no home for prose descriptions of code structure; the code is that home.

### Writing an invariant

**A contract is worth writing down only if something fails when it breaks.** Prose cannot fail, so the enforced half of `invariants.md` is generated from the tests that prove each contract. Adding one means writing the test, not writing a paragraph:

```cpp
// INVARIANT: levelwise-rejects-constraints
// The levelwise grower rejects monotone and interaction constraints at
// construction rather than silently ignoring them.
TEST_CASE("ObliviousGrower: rejects constraints it cannot honour", "[...][invariant]")
```

The marker names the contract, the comment states it, the assertions hold it true, and `scripts/render_invariants.py` collects all three. Python tests use `# INVARIANT: <id>` above the `def`.

Some contracts genuinely cannot be tested: build flags, packaging matrices, claims about nondeterminism, and decisions. Those are hand-written in `docs/invariants.src.md` and render into a section labelled as asserted, so nobody mistakes a stated promise for a guarded one. That section is also the backlog: a testable claim sitting there is one test away from being enforced.

Never edit `docs/invariants.md`. It is generated, and `docs-check` fails on drift.

## Audience and goal

Four readers, in rough order of arrival volume:

1. **ML practitioners**: know gradient boosting, evaluating whether to switch. Want capability, benchmarks, and API surface. Skim.
2. **Developers**: may not know GBTs deeply, need to integrate the library into a system. Want install, contracts, determinism guarantees, failure modes.
3. **GBT newcomers**: here to learn the algorithm. Want motivation before mechanism, and permission to go slowly.
4. **Agents**: writing most of the code in this repository. Want contracts, operational knowledge, and settled rationale. Arrive by grep, not by navigation.

**The key design fact: these audiences do not want different prose. They want different entry points into the same prose.** All four read faster with short sentences, plain vocabulary, and concrete numbers. What differs is how much scaffolding precedes the technical content, and scaffolding is a structural choice, not a stylistic one.

So: the style rules below are near-uniform across the site. Information architecture does the audience-splitting work.

### What the fourth reader changes

An agent reads differently in three ways, and each one shaped the routing table above.

**It arrives by searching for a claim, not by browsing.** Nav position buys nothing; a greppable statement with the source beside it buys everything. That is why contracts live in one indexed file rather than wherever the topic happens to fit.

**It pays for what it reads and cannot skim.** A person scans a long page in seconds and discards the irrelevant. An agent either loads the page, spending context, or greps it, missing the surrounding argument. Length is a real cost, so a page earns its size or gets cut.

**Stale prose is worse than no prose.** A human reading that a rejected design was rejected, when it shipped, is briefly confused and checks the code. An agent may act on it: write against the wrong model, then discover the error late. Absent documentation costs a lookup; wrong documentation costs the work built on it. This is the strongest argument in the style guide for deleting anything nothing keeps true.

None of this asks for different sentences. It asks that every claim be findable, sized, and provably current, which is what the routing table enforces.

## The rules

### Voice

- **Second person, active, imperative.** "Set `max_depth` to limit tree complexity." Not "the `max_depth` parameter may be configured to constrain tree complexity."
- **Passive voice buys nothing here.** Credibility comes from benchmarks, not register. Use it only when the actor is genuinely irrelevant.
- **First person singular is allowed in Learn and Results pages**, and only where it explains a decision: "I expected X, measured Y, here is what that changed." Never as narration for its own sake.

### Sentences

- **Target 15 to 20 words.** Where precision needs 30, split the qualifier into its own sentence or a note block.
- **One idea per sentence.** If it has four clauses and three embedded links, it is a paragraph pretending to be a sentence.
- **Front-load the subject.** The reader should know what the sentence is about within the first five words.
- **No em-dashes.** Use commas, colons, or parentheses.

### Vocabulary

- **Use precise ML terms without apology**: `gradient`, `leaf-wise growth`, `histogram binning`, `split gain`. Do not simplify away the real vocabulary; newcomers are here to acquire it.
- **Gloss on first occurrence**, inline or via glossary link. One clause is enough: "histogram binning (bucketing feature values so splits are evaluated over bins, not raw values)".
- **Never use a Latinate word where a plain one works.** "We compute", not "we perform computation of."
- **No idiom, no phrasal verbs where a single verb exists, no humor that depends on cultural reference.** A large share of readers are non-native English speakers.

### Claims and evidence

- **Never write a comparative without a number.** Not "significantly faster": "1.8x faster at 16M rows."
- **Never write an unfalsifiable quantifier.** "Small codebase", "the whole API in one read", "clean code" are all unverifiable. Give the LOC count, the API surface count, or cut the claim.
- **Cite tersely and inline.** One clause, one link: "Leaf-wise growth reduces loss faster per node than level-wise ([benchmarks](...))."
- **Concentrate evidence, do not sprinkle it.** Benchmarks live on a benchmarks page with hardware, dataset provenance, competing-library versions, and a reproduction script. Body text links to it.
- **State losses with the same prominence as wins.** This is the strongest credibility move available in technical documentation and it costs nothing with this audience.

### Code

- **Every example complete and copy-pasteable**, imports included.
- **Every example runs in CI.** An example that does not run is worse than no example.
- **Version-pin anything that could drift.**
- **Python is the example language.** CLI examples live only on the CLI reference page.

### Structure

- **Answer "what is this and should I use it" before anything else** on any entry page.
- **Bottom-line-up-front.** Result, then method, then caveats. Never build to a conclusion.
- **Layer so readers can stop early.** Quickstart, then guides, then reference, then concepts, then benchmarks. Each layer complete on its own.
- **Tables for anything with more than two dimensions**: platform support, parameter effects, comparative results. Not prose.
- **Figures for anything spatial**: tree growth strategies, binning, feature interactions.
- **Explain domain assumptions once, in a fixed place**, then link rather than repeat.

### Parameter documentation

Every parameter gets type, default, and **effect on the model**, not just what it is. "Number of bins" is useless; "Number of bins per feature. More bins mean finer splits and slower training; below ~64 accuracy degrades on continuous features" is the actual documentation.

### Failure modes

Document them explicitly and in the open, not in a FAQ. "This overfits on datasets under ~10k rows." "Categorical handling assumes cardinality below ~1000." Stating limits is what distinguishes documentation from marketing.

## Mechanical conventions

Base: **Google Developer Documentation Style Guide**. Fall back to the **Microsoft Writing Style Guide** for anything it does not cover. Project-specific overrides:

- `bonsai`: lowercase everywhere, including at sentence start.
- `XGBoost`, `LightGBM`, `CatBoost`: capitalized in body text, not just in links.
- `hyperparameter`, `dataset`, `runtime`: single words.
- "features", not "columns", except when quoting an API that says columns.
- Proper nouns capitalized: `Linux`, `Python`, `Docker`.

## Review checklist

Before merging any docs change:

- [ ] Any sentence over 25 words? Split it.
- [ ] Any comparative without a number attached?
- [ ] Any unfalsifiable quantifier: "small", "clean", "fast", "simple"?
- [ ] Any em-dash?
- [ ] Any term used before it is glossed?
- [ ] Any idiom or phrasal verb a non-native reader would stumble on?
- [ ] Does every code block run as written?
- [ ] Does the page answer "should I use this" before explaining how?
- [ ] Are the losses as visible as the wins?
