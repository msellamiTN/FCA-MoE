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
