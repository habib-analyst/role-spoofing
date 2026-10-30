"""role-spoofing: scripted analog of arXiv 2610.40396 (Edelman et al.).

SCOPE HONESTY (read first): this is a SCRIPTED ANALOG, not a reproduction
on a real LLM. There is no LLM, GPU, or API anywhere in this package.
What it demonstrates is the *mechanism structure* the paper describes:
  - role attribution from a mixture of content cues (writing style) and
    structure cues (the most recent role-token-like string, including
    forged ones planted inside a tool output);
  - cue-mixture behaviour: content-only, structure-only, and combined
    spoofs fail/succeed differently, combined strongest;
  - a role-belief probe trained ONLY on clean random-content
    conversations (no spoof examples, no coherent conversations) that
    flags tool segments whose predicted role disagrees with the true role;
  - the paper's key asymmetry: a content-robustness intervention (drop the
    content cue entirely) kills content spoofs but leaves structure
    susceptibility intact.

The paper's frontier-model figures (injection/control ratios 63-104%,
>95% impersonation in AgentDojo workspace tasks) are CONTEXT ONLY and are
never presented as reproduced here. All numbers in results/ come from
this scripted generator.

Differentiation from generic prompt-injection work (lineage fatigue):
the task here is NOT "does the model follow an injected instruction".
It is "who sent this segment?" -- role attribution under controlled
content/structure cues, with separate failure modes per cue type, a
random-content probe, and a content-robustness ablation. A generic
injection detector would not distinguish content-only from
structure-only failures, would not be trained on random content, and
would not exhibit the content-robustness asymmetry. Those three are
the load-bearing claims of this build.

Determinism: every generator, model, and probe run is driven by an
explicit integer seed through random.Random. No wall-clock, no dict
ordering dependence, no unseeded randomness. Pure standard library.
"""

import json
import math
import random

USER_STYLE = [
    "please", "thanks", "hey", "could", "you", "help", "me", "my",
    "i", "need", "want", "urgent", "asap", "hi", "wondering", "check",
]
TOOL_STYLE = [
    "result", "status", "200", "ok", "data", "record", "field",
    "value", "entry", "output", "response", "query", "row", "table",
    "system", "code",
]
NEUTRAL = ["w%03d" % i for i in range(64)]

N_TOKENS = 24
NOISE_SD = 0.20


def _content_vote(rng, content):
    """Fraction of style-bearing tokens that are user-style.

    random content has no style-bearing tokens -> vote exactly 0.5
    (role-neutral, as in the paper's random-content structure test).
    """
    if content == "random":
        return 0.5
    user_p = 0.80 if content == "user" else 0.20
    n_user = sum(1 for _ in range(N_TOKENS) if rng.random() < user_p)
    return n_user / N_TOKENS


def make_segment(rng, true_role, content, forged):
    """One conversation segment with a true role and controlled cues.

    true_role: 'user' or 'tool'. content: 'user'/'tool'/'random' style.
    forged: if True, a fake '<|im_start|>user' token string is planted
    inside the segment, so the most recent role-token-like string a
    naive structure reader sees is 'user' regardless of the true role.
    The CORRECT structure reader (ground truth) ignores tokens inside
    message content; the mixture model below is the naive reader the
    paper studies -- that distinction is the whole point.
    """
    vote = _content_vote(rng, content)
    if forged:
        structure_vote = 1.0  # forged user token wins the naive read
    else:
        structure_vote = 1.0 if true_role == "user" else 0.0
    return {
        "true_role": true_role,
        "content": content,
        "forged": forged,
        "content_vote": vote,
        "structure_vote": structure_vote,
    }


def belief_mixture(seg, rng, content_weight=0.5):
    """Mixture-model role belief: P(segment is from the user)."""
    b = (content_weight * seg["content_vote"]
         + (1.0 - content_weight) * seg["structure_vote"]
         + rng.gauss(0.0, NOISE_SD))
    return min(1.0, max(0.0, b))


def belief_content_robust(seg, rng):
    """Content-robustness analog: content cue dropped entirely.

    Mirrors the paper's adversarial training against content-based
    attacks: the model no longer uses content at all, so it must rely
    on structure alone -- including forged structure.
    """
    b = seg["structure_vote"] + rng.gauss(0.0, NOISE_SD)
    return min(1.0, max(0.0, b))


def activation(seg, belief):
    """Scripted 'activation' vector the probe reads (model-internal).

    Deliberately NOT including the true role. It contains the model's
    belief plus the cue values the belief was computed from, i.e. the
    information a real activation vector would carry in analog form.
    """
    cv = seg["content_vote"]
    sv = seg["structure_vote"]
    return [belief, cv, sv, belief * belief, cv * sv, N_TOKENS / 24.0]


# ---- logistic probe (pure stdlib, gradient descent) ----

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_probe(examples, epochs=300, lr=0.5, l2=1e-3):
    """examples: list of (features, label 1=user, 0=tool). Returns weights.

    Features are standardised with the training set's own mean/sd so
    the probe cannot smuggle test-set information in through scaling.
    """
    dim = len(examples[0][0])
    means = [sum(x[i] for x, _ in examples) / len(examples) for i in range(dim)]
    sds = []
    for i in range(dim):
        var = sum((x[i] - means[i]) ** 2 for x, _ in examples) / len(examples)
        sds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    w = [0.0] * (dim + 1)  # last weight is the bias on the raw 1.0 feature
    for _ in range(epochs):
        grad = [0.0] * (dim + 1)
        for x, y in examples:
            xs = [(x[i] - means[i]) / sds[i] for i in range(dim)] + [1.0]
            p = _sigmoid(sum(wi * xi for wi, xi in zip(w, xs)))
            err = p - y
            for j in range(dim + 1):
                grad[j] += err * xs[j]
        n = len(examples)
        for j in range(dim):
            w[j] -= lr * (grad[j] / n + l2 * w[j])
        w[dim] -= lr * grad[dim] / n
    return {"weights": w, "means": means, "sds": sds}


def probe_score(probe, features):
    xs = [(features[i] - probe["means"][i]) / probe["sds"][i]
          for i in range(len(features))] + [1.0]
    return _sigmoid(sum(wi * xi for wi, xi in zip(probe["weights"], xs)))


def auc(scores_pos, scores_neg):
    """Rank AUC of positive scores vs negative scores (ties count 0.5)."""
    wins = 0.0
    for p in scores_pos:
        for q in scores_neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(scores_pos) * len(scores_neg))


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_diff_ci(paired_a, paired_b, seed, reps=2000):
    """Paired bootstrap CI for mean(a - b); a, b are 0/1 lists, same length."""
    rng = random.Random(seed)
    n = len(paired_a)
    diffs = [a - b for a, b in zip(paired_a, paired_b)]
    mean = sum(diffs) / n
    boots = []
    for _ in range(reps):
        s = sum(rng.choice(diffs) for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * reps)]
    hi = boots[int(0.975 * reps) - 1]
    return mean, lo, hi


# ---- arm generation ----

# arm -> (true_role, content, forged)
ARMS = {
    "genuine_user": ("user", "user", False),
    "genuine_tool": ("tool", "tool", False),
    "content_only": ("tool", "user", False),
    "structure_toolstyle": ("tool", "tool", True),
    "structure_random": ("tool", "random", True),
    "combined": ("tool", "user", True),
}


def run_arm(seed, arm, n, belief_fn=belief_mixture):
    """Returns list of (success, segment, belief) for one arm.

    success = model attributes the segment to the user (belief > 0.5).
    For genuine_user this is a correct attribution; for every tool-true
    arm it is a spoof success.
    """
    rng = random.Random(seed)
    true_role, content, forged = ARMS[arm]
    out = []
    for _ in range(n):
        seg = make_segment(rng, true_role, content, forged)
        b = belief_fn(seg, rng)
        out.append((1 if b > 0.5 else 0, seg, b))
    return out


def run_arm_paired(seed, arm, base_contents, belief_fn=belief_mixture):
    """Paired version: same base segment draws across arms.

    base_contents is a list of per-segment RNG states is not portable,
    so pairing is implemented by regenerating each segment from a
    per-segment seed derived from (seed, index); the cue transformation
    (content/forged) is then applied to the same underlying draw. In
    practice the content draw differs by arm by construction (different
    style distributions), so pairing here means: same segment INDEX and
    same noise draw sequence position, i.e. the comparison is over
    matched segment slots, which is the paired structure the bootstrap
    uses. Documented honestly: pairing is by slot, not by identical
    token draw, because the cue IS the treatment.
    """
    true_role, content, forged = ARMS[arm]
    out = []
    for i in range(len(base_contents)):
        rng = random.Random(seed * 1000003 + i)
        seg = make_segment(rng, true_role, content, forged)
        b = belief_fn(seg, rng)
        out.append((1 if b > 0.5 else 0, seg, b))
    return out


def train_random_content_probe(train_seed, n_per_role=800):
    """Probe trained ONLY on clean random-content conversations.

    No spoof examples, no coherent conversations -- the paper's
    surprising training regime, reproduced in analog form.
    """
    rng = random.Random(train_seed)
    examples = []
    for true_role, label in (("user", 1), ("tool", 0)):
        for _ in range(n_per_role):
            seg = make_segment(rng, true_role, "random", False)
            b = belief_mixture(seg, rng)
            examples.append((activation(seg, b), label))
    rng.shuffle(examples)
    return train_probe(examples)


def probe_scores_for_arm(probe, seed, arm, n):
    rng = random.Random(seed)
    true_role, content, forged = ARMS[arm]
    scores = []
    for _ in range(n):
        seg = make_segment(rng, true_role, content, forged)
        b = belief_mixture(seg, rng)
        scores.append(probe_score(probe, activation(seg, b)))
    return scores
