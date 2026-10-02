# Historical cluster logs

This directory contains selected historical scheduler output used during
the reproduction.

These logs were important for reconstructing run configurations, job
provenance, sample-size conventions, and discrepancies between recovered
runs and the final validated reproduction.

## Retention policy

Keep:

- logs establishing a reproduced or matched result;
- logs documenting an important discrepancy;
- logs needed to reconstruct an author-associated run;
- representative failed runs that explain a methodological decision;
- unique historical logs whose relevance has not yet been ruled out.

Do not automatically add every new cluster log. New `.out` and `.err`
files are ignored by default and should be force-added only when they have
clear provenance value.

Exact byte-identical duplicate logs may be removed from the current tree
when no tracked file refers specifically to the duplicate filename. The
retained filename and removed duplicate are recorded in
`DEDUPLICATION.tsv`.

Git history is not rewritten, so previously committed logs remain
recoverable from earlier commits.
