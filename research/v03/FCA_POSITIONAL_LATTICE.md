# Positional FCA and cross-context lattice audit

## Scope

This implementation adds a position-aware formal context while preserving
ordinary token-level FCA. It also supplies reference derivation and
closure-mapping diagnostics with explicit metric semantics.

## Positional-window context

For a single sequence with routes \(R(0),\ldots,R(n-1)\), where each token
routes to exactly \(k\) distinct experts, a window of length \(b\) beginning
at \(i\) is an object. The attribute universe is

\[
M^{(b)}=E\times\{0,\ldots,b-1\}.
\]

The attribute index is expert-major:

\[
\operatorname{index}(e,r)=e b+r.
\]

Incidence is defined by

\[
(i,(e,r))\in I^{(b)}
\quad\Longleftrightarrow\quad
e\in R(i+r).
\]

For stride one, the number of objects is \(n-b+1\). Every object has exactly
\(bk\) incident attributes. Sequences must be passed separately so a window
never crosses a sequence boundary.

This differs from a bag-of-experts window context: relative position is part
of the attribute identity. Repeated expert IDs at different offsets are
different formal attributes.

## Galois derivations and closure

The reference functions implement

\[
B'=\{g\in G:\forall m\in B,\;gIm\},\qquad
A'=\{m\in M:\forall g\in A,\;gIm\},
\]

and \(\gamma(B)=B''\). The empty-extent convention is \(A' = M\) for
\(A=\varnothing\).

The tests check extensivity, idempotence, exact agreement between NextClosure
and exhaustive enumeration on small contexts, and object-permutation
invariance of closed intents and supports.

## Closure-induced mapping

For an expert bijection \(\pi:E_s\to E_t\), its positional lift is

\[
\widetilde\pi(e,r)=(\pi(e),r).
\]

For a source intent \(B\), define

\[
\Phi(B)=\gamma_t(\widetilde\pi(B)).
\]

Because set-image mappings and FCA closure are monotone,

\[
B_1\subseteq B_2
\Longrightarrow
\Phi(B_1)\subseteq\Phi(B_2).
\]

Therefore, a metric that claims to measure this exact weak inclusion property
must equal one (provided the same mapping is used and no attributes are
dropped). An empirical value below one indicates that the metric tests a
different relation, uses a partial/non-set-image mapping, or contains an
implementation/denominator issue.

The audit reports separately:

- closure equality: \(\widetilde\pi(B)=\Phi(B)\);
- target support validity of the mapped set;
- membership of its target closure in a supplied target-intent family;
- strict inclusion preservation, which can fail through closure-induced
  collapse;
- order reflection, which is not guaranteed;
- cover preservation within the supplied source and target intent families;
- the fraction of distinct source intents that collapse to identical target
  closures.

When support-filtered intent families are supplied, cover edges are covers
of their induced subposets. They are not automatically covers of the complete
concept lattices. Complete-lattice claims require complete intent enumeration.

## What this module does not claim

The current functions do not prove lattice homomorphism, lattice isomorphism,
or statistical enrichment. They provide explicit, testable quantities for
those investigations. Statistical inference must separately account for
surrogate-chain mixing, the tested statistic, and post-hoc layer selection.

## Empirical positional cross-layer benchmark (reported run)

A positional-intent matching benchmark was run on the Qwen1.5-MoE-A2.7B
WikiText test contexts at 1% support, comparing source layer L11 with target
layers L14 and L16. The report was saved by the run to
`contexts_v1/clera_positional_null_benchmark/positional_intent_null_benchmark.json`
in the user's Colab Drive workspace. The values below are transcribed from
that report; the JSON artifact and the full surrogate-generation code are not
part of this repository change.

The four-fold descriptive summaries are:

| Window | Target | Exact observed / null mean | Partial (\(\geq 2\)) observed / null mean | Mean fold Jaccard observed / null mean |
|---|---:|---:|---:|---:|
| \(b=4\) | L14 | 53 / 55.01 | 1,694 / 1,702.41 | 0.3078 / 0.3098 |
| \(b=4\) | L16 | 9 / 9.03 | 1,535 / 1,544.77 | 0.2633 / 0.2660 |
| \(b=8\) | L14 | 18 / 18.96 | 965 / 968.14 | 0.2543 / 0.2535 |
| \(b=8\) | L16 | 0 / 0.05 | 785 / 785.22 | 0.1870 / 0.1883 |

Exact and partial counts are sums across the four folds. Jaccard values are
unweighted arithmetic means of the four fold values. These aggregates are
descriptive and are not pooled inferential tests.

Across the 16 fold-target comparisons and three upper-tail statistics per
comparison (48 reported tests), every reported p-value is at least 0.257.
Thus, this benchmark supplies no evidence of upper-tail positional matching
enrichment over its tested null for either target layer, at either window
length. The observed matching levels are close to the corresponding null
means.

This is not an equivalence test: failure to detect upper-tail enrichment does
not establish that observed and null distributions are equivalent. It also
does not test whether L14 is more transportable than L16. The larger raw
matching values for L14 are descriptive; target intent-family sizes differ,
and the marginal per-target p-values do not test a paired layer contrast.

A valid L14-versus-L16 contrast should use the same null replicate within each
fold, recompute the full alignment/matching procedure for both targets in
every replicate, and test a predeclared paired difference (preferably with a
normalised statistic whose denominator and eligible-intent set are fixed or
recomputed consistently). The contrast should be calibrated across folds
without treating folds as independent surrogate replicates.

Finally, `MarginsOK=True` documents the reported margin check only. It does
not, by itself, establish Markov-chain mixing or effective sample size for
the exact, partial-match, and Jaccard statistics. Those diagnostics must be
reported before treating the surrogate distribution as a reliable sampling
reference.

These positional findings are specific to the expert-position window
representation and its reported null. They do not overturn or confirm
token-level tuple-transport findings obtained with a different FCA context,
statistic, and null model.
