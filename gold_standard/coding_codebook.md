# Paper 1 Manual Coding Codebook

## Purpose

This package supports an intercoder reliability check for the paper's core disclosure-verifiability measure. Coders should classify each anonymized MD&A text segment without seeing the machine label, firm identifier, or year.

## Labels

- `verifiable`: the segment links GenAI language to observable implementation traces, such as deployment, launch, system/platform/product, application scenario, quantitative rollout, named partner, patent/tool, or current operational use.
- `soft_substantive`: the segment is application-oriented or operationally meaningful, but lacks stronger implementation evidence.
- `symbolic`: the segment is generic, aspirational, trend-following, policy-oriented, or concept-level language without concrete implementation traces.
- `unclear`: the segment is too short, truncated, ambiguous, or otherwise impossible to classify confidently.

## Coding Rules

1. Code only from the text shown in `coding_sample`.
2. Do not infer from firm identity, year, industry, or machine-generated labels.
3. Prefer `verifiable` only when there is a concrete implementation clue.
4. Prefer `soft_substantive` when the segment discusses use cases or business application but does not provide inspectable evidence.
5. Prefer `symbolic` when the segment only signals attention to GenAI or related technology narratives.
6. Use `unclear` sparingly and explain the reason in `notes`.

## Reliability Reporting

After two independent coders finish, report percent agreement and Cohen's kappa for the four-label classification. A reviewer-facing robustness table can also collapse `unclear` into exclusion and report kappa on the three primary labels.

## Sample Summary

- Strict first-disclosure adopters covered: 735
- Strict first-disclosure segment universe used for sampling: 1927

| machine_label | sample_segments | target_per_label |
| --- | --- | --- |
| soft_substantive | 60 | 60 |
| symbolic | 60 | 60 |
| verifiable | 60 | 60 |
