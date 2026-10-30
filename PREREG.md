# PREREGISTRATION — project #106 `role-spoofing` (head-to-head)

Locked 2026-10-30, AFTER the pilot and BEFORE any head-to-head run.
Pilot numbers (results/pilot.json) were seen; head-to-head seeds below
are fresh and were never used in the pilot. Decision rule (standing):
ANY locked bar fails -> KILL. No post-hoc bar changes, no bar-shrinking
(#97 lesson). This file is not to be edited after the head-to-head run
except to append the verdict.

## Source paper (context only, never reproduced)
arXiv 2610.40396, Edelman et al., "Understanding and Mitigating Role
Spoofing in LLMs" (submitted 29 Oct 2026). Its frontier-model figures
(injection/control ratios 63-104%, >95% AgentDojo impersonation) are
context only. Every number below is from the scripted analog in src/.

## G1 verdict (already passed, before this preregistration)
- Abs page fetched fresh 2026-10-30: NO author code link (only
  third-party finder toggles: alphaXiv, CatalyzeX, DagsHub, GotitPub,
  Hugging Face, ScienceCast, Replicate, Spaces, TXYZ.AI).
- Full-text HTML (arxiv.org/html/2610.40396v1, 1519 lines): ZERO
  "github", ZERO "zenodo", ZERO "4open", ZERO "repository", ZERO
  "anonymous" matches. No author code, no author replication/data
  package announced in the paper text. G1 PASS.

## Pilot verdict (exploratory, n=600/arm)
- Genuine user attributed to user 98.3%; genuine tool 2.7%.
- Content-only 31.7%, structure-toolstyle 71.8%,
  structure-random 88.0%, combined 96.7%.
- Probe (trained on clean random-content only): AUC 0.964 pooled
  spoof vs genuine tool; benign FPR 6.8%; detection content-only
  49.2%, all structure/combined arms 100%.
- Pilot gate PASS on both pre-locked pilot bars. Pilot is NOT the
  head-to-head result and is never quoted as the finding.

## Head-to-head protocol (fresh seeds, never used in pilot)
- n = 2000 segments per arm, slot-paired across arms (same slot seed,
  cue treatment differs -- pairing is by slot, documented in src/).
- Mixture model: belief = 0.5*content_vote + 0.5*structure_vote +
  N(0, 0.20), clipped to [0,1]; spoof success = belief > 0.5.
- Probe: logistic probe trained on clean random-content conversations
  only (train seed 9100, 800/role); threshold = 95th percentile of
  probe scores on held-out clean genuine-tool validation (seed 9200);
  test seeds 9300-series; benign FPR measured on fresh genuine-tool
  segments (seed 9400).
- Content-robust model: belief = structure_vote + N(0, 0.20).
- Differences: paired bootstrap, 2000 resamples, seed 9500, 95% CI.
- Arms: genuine_user, genuine_tool, content_only,
  structure_toolstyle, structure_random, combined.

## Locked bars -- G2 (disease: cue-mixture spoofing)
- G2a-1: combined rate >= 0.90.
- G2a-2: paired bootstrap 95% CI for (combined - content_only)
  has lower bound > 0.10.
- G2a-3: paired bootstrap 95% CI for (combined - structure_random)
  has lower bound > 0.02.
- G2a-4: paired bootstrap 95% CI for (structure_random - content_only)
  has lower bound > 0.10.
  (Together: combined strongest, and the two single cues fail
  differently -- the role-tracking mixture signature, not generic
  injection success.)

## Locked bars -- G3 (load-bearing mechanism: probe + asymmetry)
- G3b-1: probe AUC (all spoof arms pooled vs genuine tool) >= 0.90.
- G3b-2: benign false-positive rate <= 0.08.
- G3b-3: pooled spoof detection rate at the locked threshold >= 0.70.
- G3b-4: detection rate >= 0.90 for EACH of structure_random and
  combined (the forged-structure arms the probe must catch).
- G3c-1: under the content-robust model, content_only rate <= 0.05.
- G3c-2: under the content-robust model, structure_random rate >= 0.85.
- G3c-3: paired difference (content-robust structure_random rate
  minus original structure_random rate) >= -0.05 (structure
  susceptibility intact, not merely reduced a little).
- G3c-4: paired difference (original content_only rate minus
  content-robust content_only rate) >= 0.20 (content robustness
  actually worked -- otherwise G3c-2 would be vacuous).

## Verdict rule
ALL bars above must pass. Any single failure -> KILL with an honest
KILLED.md + negative PAPER.md, no repo, no LinkedIn draft. Scope
honesty applies to every artifact: scripted-analog numbers only.
