# Results

Every performance and accuracy claim on this site comes from a committed benchmark run, on named hardware, with matched hyperparameters for every library (rules in the [benchmark protocol](benchmark-protocol.md)). The pages below hold the runs, the rules they follow, and the steps to rerun them.

<div class="grid cards" markdown>

-   **[The results ledger](results.md)**

    ---

    One summary for each division (perf: speed and memory; quality: accuracy; code: the bonsai source itself), then one page per division, plus the hand-kept archive of closed campaigns.

-   **[The benchmark protocol](benchmark-protocol.md)**

    ---

    The rules every published number follows: which benchmarks count as accuracy, speed, or code claims, which datasets and metrics they use, and how timings are taken.

-   **[Reproducing the results](benchmarks.md)**

    ---

    Run the bonsai.bench harness from the installed wheel or a source tree.

</div>

Wins and losses sit on the same pages, and the best cell in each column is bold whoever owns it: where a reference library leads on memory or accuracy, its cell carries the bold, not bonsai's. The raw data behind them is in [`benchmarks/results/`](../../benchmarks/results), and the rulings that cite it are in the [decisions log](../decisions.md).
