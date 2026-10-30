# role-spoofing — scripted analog of arXiv 2610.40396

Open analog of Edelman et al., "Understanding and Mitigating Role
Spoofing in LLMs" (UPenn / Princeton / Apple MLR, submitted 29 Oct 2026).

## What this is — and what it is not

**It is** a deterministic, CPU-only, $0, pure-stdlib scripted analog that
demonstrates the paper's *mechanism structure*:

1. Models attribute a segment's role from a **mixture** of content cues
   (writing style) and structure cues (the most recent role-token-like
   string, including forged ones planted inside a tool output).
2. Single-cue spoofs fail differently, and combined content+structure
   spoofing is strongest.
3. A **role-belief probe trained only on clean random-content
   conversations** (no spoof examples) separates spoofed from genuine
   tool segments with a low benign false-positive rate.
4. **Content-robustness training leaves structure susceptibility
   intact**: dropping the content cue kills content spoofs (30.4% →
   0.6%) while structure-random spoofing rises (89.0% → 99.0%).

**It is not** a reproduction on a real LLM. The paper's frontier-model
figures (injection/control ratios 63–104%, >95% impersonation in
AgentDojo workspace tasks) are context only and are never claimed as
reproduced. A scripted analog proves the mechanism structure, not that
any real model behaves this way.

## Why this is not just "another prompt-injection detector"

The task is role *attribution* ("who sent this segment?"), not
instruction following. The load-bearing claims are specific to
role-tracking: separate content vs structure failure modes, a probe
trained on random content, and the content-robustness asymmetry. A
generic injection detector would exhibit none of those three.

## Run it

```bash
python3 run_pilot.py          # exploratory pilot (n=600/arm)
python3 run_head_to_head.py   # preregistered head-to-head (n=2000/arm)
python3 -m unittest discover -s tests
```

Headline numbers live in `results/headline.json` (SHA-256
`51172db5bf6974846ab98928b5d53c6971c15ff80889a26220545a9d50913c5e`);
a re-run reproduces it byte-identically. Preregistration:
`PREREG.md` (locked before the head-to-head run; all 12 bars passed).
Kill-gate record: `KILLGATE.md`.

## Headline results (scripted analog, n=2000/arm, fresh seeds)

| Arm (true role: tool unless noted) | Attributed to user |
|---|---|
| Genuine user (control) | 96.9% |
| Genuine tool (control) | 2.2% |
| Content-only spoof | 30.4% |
| Structure-only, tool-style content | 69.1% |
| Structure-only, random content | 89.0% |
| Combined content+structure spoof | 96.9% |

Paired bootstrap 95% CIs: combined − content-only +66.5pp
[64.4, 68.5]; combined − structure-random +7.9pp [6.3, 9.5];
structure-random − content-only +58.6pp [56.2, 61.0].

Probe: AUC 0.963 (pooled spoof vs genuine tool), benign FPR 5.6%,
pooled detection 86.4%; detection 100% on structure-random and
combined, 45.6% on content-only (the probe reads the model's belief,
so a spoof that only partly fools the model is only partly flagged --
an honest limit, stated in PAPER.md).

Note on the identical 96.9% for genuine user and combined spoof:
by construction a combined spoof carries user-style content plus a
forged user token, i.e. exactly the cues of a genuine user segment,
so the analog cannot distinguish them. That indistinguishability is
the attack's point, not a bug -- and it is why the probe, which reads
belief, also flags combined spoofs only via their true-role mismatch
on tool segments, not from any cue difference.

## License

MIT. Author: Habib Ur Rehman (habib-analyst).
