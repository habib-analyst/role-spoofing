# KILLGATE — project #106 `role-spoofing`

Verdict: **PASS — all gates passed on locked bars.** No bar was moved,
shrunk, or amended after data was seen.

- **G0 harness: PASS.** Deterministic, CPU-only, $0, no LLMs/APIs/GPUs.
  Pure Python standard library. 9/9 tests green. Head-to-head re-run
  byte-identical (SHA-256 of results/headline.json:
  51172db5bf6974846ab98928b5d53c6971c15ff80889a26220545a9d50913c5e).
- **G1 author-code recheck: PASS (mandatory, fresh 2026-10-30).**
  Abs page: no author code link (third-party finder toggles only).
  Full-text HTML (1519 lines): ZERO "github", ZERO "zenodo", ZERO
  "4open", ZERO "repository", ZERO "anonymous" matches. No author
  code or author replication/data package exists in the paper text.
  (Contrast #100, killed on an announced 4open replication package,
  and #97, killed for a post-hoc bar change -- neither failure mode
  applies here.)
- **Pilot gate: PASS.** Both pre-locked pilot bars passed:
  combined 96.7% > structure-random 88.0%, gap to content-only
  56.3pp (>= 10pp required); probe AUC 0.964 (>= 0.80), benign FPR
  6.8% (<= 10%). Pilot numbers are exploratory, never quoted as
  findings.
- **G2 disease: PASS.** All 4 locked bars passed (see PREREG.md):
  combined 96.9% (>= 0.90); combined − content-only CI lower bound
  64.4pp (> 10pp); combined − structure-random CI lower bound
  6.3pp (> 2pp); structure-random − content-only CI lower bound
  56.2pp (> 10pp).
- **G3 load-bearing mechanism: PASS.** All 8 locked bars passed:
  probe AUC 0.963 (>= 0.90); benign FPR 5.6% (<= 8%); pooled
  detection 86.4% (>= 70%); structure-random and combined detection
  100% each (>= 90%); content-robust content-only rate 0.6%
  (<= 5%); content-robust structure-random rate 99.0% (>= 85%);
  structure susceptibility change +10.1pp (drop >= -5pp required);
  content drop 29.8pp (>= 20pp required).
- **Scope honesty:** every number above is from the scripted analog.
  The paper's frontier-model figures are context only, everywhere
  (README, PAPER.md, draft).

## Staged payload
`~/workspace/build/role-spoofing/` — code (src/), tests, README,
KILLGATE.md, PREREG.md, results/pilot.json, results/headline.json.
Paper: `hidden_files/papers/role-spoofing/PAPER.md`.
LinkedIn draft (DRAFT ONLY, not queued): 
`hidden_files/linkedin_drafts/role-spoofing.md`.

## Repo request
**REPO NEEDED: habib-analyst/role-spoofing** — no repo was created
and nothing was pushed (GitHub App token cannot create repos).
