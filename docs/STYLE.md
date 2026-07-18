# bonsai documentation style

Rules for writing and reviewing anything under `docs/`.

## Audience and goal

Three readers, in rough order of arrival volume:

1. **ML practitioners**: know gradient boosting, evaluating whether to switch. Want capability, benchmarks, and API surface. Skim.
2. **Developers**: may not know GBTs deeply, need to integrate the library into a system. Want install, contracts, determinism guarantees, failure modes.
3. **GBT newcomers**: here to learn the algorithm. Want motivation before mechanism, and permission to go slowly.

**The key design fact: these audiences do not want different prose. They want different entry points into the same prose.** All three read faster with short sentences, plain vocabulary, and concrete numbers. What differs is how much scaffolding precedes the technical content, and scaffolding is a structural choice, not a stylistic one.

So: the style rules below are near-uniform across the site. Information architecture does the audience-splitting work.

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
