import json
from pathlib import Path

raw_path = Path(
    "result/diagnose_raw_matched_mlphead_HDGM_d2_N2000_30trials.json"
)

wae_path = Path(
    "result/rl_c2st_wae_imq_HDGM_d2_power_"
    "pilot_z2_lam001_N2000_frozen_mlphead_seedmatched.json"
)

with open(raw_path) as f:
    raw = json.load(f)

with open(wae_path) as f:
    wae = json.load(f)


# ============================================================
# RAW trial records
# ============================================================

raw_by_trial = {}

for r in raw["records"]:

    trial = int(r["trial"])

    raw_by_trial[trial] = {
        "S": int(r["reject_S"]),
        "L": int(r["reject_L"]),
    }


# ============================================================
# WAE trial records
#
# Verified WAE payload structure:
#
#   payload["trial_records"]
#
# each containing:
#
#   outer_trial
#   trial_power_s
#   trial_power_l
#
# ============================================================

wae_records = wae["trial_records"]

wae_by_trial = {}

for r in wae_records:

    trial = int(r["outer_trial"])

    wae_by_trial[trial] = {
        "S": int(round(float(r["trial_power_s"]))),
        "L": int(round(float(r["trial_power_l"]))),
    }


# ============================================================
# Exact paired comparison on trials 0..9
# ============================================================

trials = list(range(10))

missing_raw = [
    t for t in trials
    if t not in raw_by_trial
]

missing_wae = [
    t for t in trials
    if t not in wae_by_trial
]

if missing_raw:
    raise RuntimeError(
        f"RAW missing trials: {missing_raw}"
    )

if missing_wae:
    raise RuntimeError(
        f"WAE missing trials: {missing_wae}"
    )


print()
print("===================================================")
print("PAIRED RAW vs WAE: EXACT SAME OUTER TRIALS 0..9")
print("===================================================")
print()

print("trial | RAW-S WAE-S | RAW-L WAE-L")
print("-----------------------------------")

for t in trials:

    rr = raw_by_trial[t]
    wr = wae_by_trial[t]

    print(
        f"{t:5d} |"
        f"   {rr['S']}     {wr['S']}  |"
        f"   {rr['L']}     {wr['L']}"
    )


# ============================================================
# Aggregate powers on the SAME 10 datasets
# ============================================================

raw_s = sum(
    raw_by_trial[t]["S"]
    for t in trials
) / len(trials)

wae_s = sum(
    wae_by_trial[t]["S"]
    for t in trials
) / len(trials)

raw_l = sum(
    raw_by_trial[t]["L"]
    for t in trials
) / len(trials)

wae_l = sum(
    wae_by_trial[t]["L"]
    for t in trials
) / len(trials)


print()
print("Aggregate on EXACT SAME 10 trials:")
print()

print(f"RAW-S = {raw_s:.3f}")
print(f"WAE-S = {wae_s:.3f}")

print()

print(f"RAW-L = {raw_l:.3f}")
print(f"WAE-L = {wae_l:.3f}")


# ============================================================
# Paired decision counts
#
# More informative than just comparing aggregate power.
# ============================================================

def paired_counts(metric):

    wae_only = 0
    raw_only = 0
    both = 0
    neither = 0

    for t in trials:

        raw_decision = raw_by_trial[t][metric]
        wae_decision = wae_by_trial[t][metric]

        if wae_decision == 1 and raw_decision == 0:
            wae_only += 1

        elif raw_decision == 1 and wae_decision == 0:
            raw_only += 1

        elif raw_decision == 1 and wae_decision == 1:
            both += 1

        else:
            neither += 1

    return {
        "WAE_only": wae_only,
        "RAW_only": raw_only,
        "both": both,
        "neither": neither,
    }


for metric in ["S", "L"]:

    counts = paired_counts(metric)

    print()
    print(f"{metric} paired outcomes:")
    print(
        "  WAE rejects / RAW does not : "
        f"{counts['WAE_only']}"
    )
    print(
        "  RAW rejects / WAE does not : "
        f"{counts['RAW_only']}"
    )
    print(
        "  both reject                : "
        f"{counts['both']}"
    )
    print(
        "  neither rejects             : "
        f"{counts['neither']}"
    )


# ============================================================
# Direct paired differences
# ============================================================

s_differences = [
    wae_by_trial[t]["S"] - raw_by_trial[t]["S"]
    for t in trials
]

l_differences = [
    wae_by_trial[t]["L"] - raw_by_trial[t]["L"]
    for t in trials
]

print()
print("Paired mean difference WAE - RAW:")
print(
    f"  S: {sum(s_differences) / len(s_differences):+.3f}"
)
print(
    f"  L: {sum(l_differences) / len(l_differences):+.3f}"
)

print()
print("Interpretation:")
print(
    "Positive WAE-only excess = evidence WAE helps on matched datasets."
)
print(
    "Positive RAW-only excess = evidence RAW head is better."
)
print(
    "Balanced counts = apparent aggregate differences are likely "
    "dataset / optimization noise."
)
