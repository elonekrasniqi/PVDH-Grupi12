"""
Handling incorrect detections and outlier treatment

Input (outputs of the two detection tasks):
  1A_UNIVARIATE_OUTLIERS_DATASET.csv   - 8A data + Z-score / IQR flags
  2E_MULTIVARIATE_OUTLIER_RESULTS.csv  - Isolation Forest / LOF flags (same row order)

Steps
  1. Domain check: is a flagged value really wrong (impossible), or only unusual?
     Impossible values are data errors -> replaced with the median + flag column.
  2. Review of every detection (feature x method): fraud rate, currency
     concentration, low-activity share, repeated customers -> verdict + reason.
     Detections caused by a wrong method/scale/level are marked as incorrect.
  3. Corrected detection, only for the features where it makes sense:
       - right-skewed money values  -> on the log1p scale
       - customer-level features    -> one row per customer (not per transaction)
       - an outlier is confirmed only when Z-score AND IQR agree
  4. Treatment: no rows are removed (outliers carry most of the fraud signal).
     Confirmed outliers are capped (winsorized) at the detection bounds in new
     *_capped columns and re-scaled like in Phase I; originals stay unchanged.

Outputs:
  3A_OUTLIER_TREATED_DATASET.csv     - 8A data + confirmed flags + capped/scaled columns
  3B_DETECTION_REVIEW_REPORT.csv     - verdict for every detection (feature x method)
  3C_OUTLIER_TREATMENT_REPORT.csv    - bounds, outliers and before/after stats per feature
  3D_OUTLIER_HANDLING_SUMMARY.csv    - row-level summary (before vs after the review)
  3E_OUTLIER_TREATMENT_PLOTS.png     - boxplots before / after capping
  3F_DETECTION_REVIEW_PLOT.png       - rows flagged by every detection, coloured by verdict
  3G_INCORRECT_DETECTION_EVIDENCE.png - masking, currency artefact, bounded count, repeated customers
  3H_OUTLIER_HANDLING_SUMMARY_PLOT.png - before vs after the review, fraud lost if removed, confidence
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE_2 = ROOT / "dataset" / "phase_2"
INPUT_UNIVARIATE = PHASE_2 / "1A_UNIVARIATE_OUTLIERS_DATASET.csv"
INPUT_MULTIVARIATE = PHASE_2 / "2E_MULTIVARIATE_OUTLIER_RESULTS.csv"
OUTPUT_FOLDER = PHASE_2

Z_THRESHOLD = 3
IQR_FACTOR = 1.5
LOW_ACTIVITY_QUANTILE = 0.05   # windows with very few previous transactions
CURRENCY_RATIO_LIMIT = 3       # flag rate in one currency > 3x the overall rate -> unit artefact
SWAMPING_LIMIT_PCT = 5         # one method flagging > 5% of rows is not "rare" any more

# kind  -> how the feature is reviewed and treated
# level -> "customer": the value is the same for every transaction of a customer
# scale -> how Phase I (task 8) scaled the feature; used again for the capped column
FEATURE_RULES = {
    "amount_usd":                 {"kind": "skewed_money",     "level": "transaction", "scale": "log_z"},
    "customer_avg_amount_usd":    {"kind": "skewed_money",     "level": "customer",    "scale": "log_z"},
    "customer_txn_count":         {"kind": "continuous",       "level": "customer",    "scale": "z"},
    "customer_away_share":        {"kind": "continuous",       "level": "customer",    "scale": "minmax"},
    "customer_night_share":       {"kind": "continuous",       "level": "customer",    "scale": "minmax"},
    "velocity_num_transactions":  {"kind": "continuous",       "level": "transaction", "scale": "z"},
    "customers_per_device":       {"kind": "discrete_mode",    "level": "transaction", "scale": None},
    "velocity_unique_merchants":  {"kind": "bounded_activity", "level": "transaction", "scale": None},
    "velocity_unique_countries":  {"kind": "bounded_activity", "level": "transaction", "scale": None},
    "velocity_total_amount":      {"kind": "mixed_currency",   "level": "transaction", "scale": None},
    "velocity_max_single_amount": {"kind": "mixed_currency",   "level": "transaction", "scale": None},
}
TREATED_KINDS = ("skewed_money", "continuous")
FEATURES = list(FEATURE_RULES)

METHOD_SUFFIXES = {
    "Z-score": "_z_outlier",
    "Z-score (log)": "_zlog_outlier",
    "IQR": "_iqr_outlier",
    "IQR (extreme)": "_iqr_extreme_outlier",
}
MULTIVARIATE_FLAGS = {
    "Isolation Forest": "isolation_forest_outlier",
    "LOF": "lof_outlier",
    "Isolation Forest + LOF": "both_methods_outlier",
}

# value ranges that are possible at all (outside them = data error, not an outlier)
VALID_RANGES = {
    "amount_usd": (0, None, False),                 # (min, max, min allowed?)
    "customer_avg_amount_usd": (0, None, False),
    "customer_txn_count": (1, None, True),
    "customer_away_share": (0, 1, True),
    "customer_night_share": (0, 1, True),
    "customers_per_device": (1, None, True),
    "velocity_num_transactions": (0, None, True),
    "velocity_unique_merchants": (0, None, True),
    "velocity_unique_countries": (0, None, True),
    "velocity_total_amount": (0, None, False),
    "velocity_max_single_amount": (0, None, False),
}
INTEGER_FEATURES = ("customer_txn_count", "customers_per_device", "velocity_num_transactions",
                    "velocity_unique_merchants", "velocity_unique_countries")


# ---------------------------------------------------------------- 1. domain check

def velocity_consistency(df):
    """velocity_* describe the last hour: num_transactions counts the PREVIOUS
    transactions, total/max also include the current one -> (n + 1) transactions."""
    n = df["velocity_num_transactions"] + 1
    total = df["velocity_total_amount"]
    largest = df["velocity_max_single_amount"]
    return {
        "unique_merchants <= num_transactions + 1": df["velocity_unique_merchants"] <= n,
        "unique_countries <= num_transactions + 1": df["velocity_unique_countries"] <= n,
        "max_single_amount <= total_amount": largest <= total * (1 + 1e-9) + 1e-6,
        "total_amount <= (num_transactions + 1) * max_single_amount": total <= n * largest * (1 + 1e-6) + 1e-6,
    }


def domain_check(df):
    """Impossible values -> NaN -> median, with a <feature>_domain_error flag."""
    df = df.copy()
    rows = []
    for feature, (low, high, low_allowed) in VALID_RANGES.items():
        values = pd.to_numeric(df[feature], errors="coerce")
        invalid = values.isna()
        if low is not None:
            invalid |= (values < low) if low_allowed else (values <= low)
        if high is not None:
            invalid |= values > high
        if feature in INTEGER_FEATURES:
            invalid |= values.notna() & (values % 1 != 0)
        if invalid.any():
            df[feature] = values.mask(invalid).fillna(values[~invalid].median())
            df[feature + "_domain_error"] = invalid.astype(int)
        rows.append({"rule": f"{feature} in valid range", "violations": int(invalid.sum()),
                     "handling": "replaced with median + flag column" if invalid.any() else "none needed"})

    for rule, ok in velocity_consistency(df).items():
        rows.append({"rule": rule, "violations": int((~ok).sum()),
                     "handling": "flagged (velocity_record_inconsistent)" if (~ok).any() else "none needed"})
    inconsistent = ~pd.concat(velocity_consistency(df), axis=1).all(axis=1)
    df["velocity_record_inconsistent"] = inconsistent.astype(int)

    if "amount_usd_invalid" in df.columns:
        rows.append({"rule": "amount_usd <= 0 (already imputed in Phase I, amount_usd_invalid)",
                     "violations": int(df["amount_usd_invalid"].sum()),
                     "handling": "already replaced with median in Phase I (task 8)"})
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. detection review

def evidence(df, flags):
    """Numbers that help decide whether the flagged rows are real outliers."""
    flags = flags.astype(bool)
    n = int(flags.sum())
    base = df["is_fraud"].mean()
    result = {"flagged_rows": n, "flagged_pct": round(100 * flags.mean(), 3),
              "flagged_customers": int(df.loc[flags, "customer_id"].nunique()),
              "fraud_rate": np.nan, "fraud_lift": np.nan, "top_currency": "",
              "top_currency_ratio": np.nan, "low_activity_share": np.nan}
    if n == 0:
        return result
    fraud_rate = df.loc[flags, "is_fraud"].mean()
    result["fraud_rate"] = round(fraud_rate, 3)
    result["fraud_lift"] = round(fraud_rate / base, 2) if base > 0 else np.nan

    if "currency_resolved" in df.columns:
        rate_by_currency = flags.groupby(df["currency_resolved"]).mean()
        result["top_currency"] = rate_by_currency.idxmax()
        result["top_currency_ratio"] = round(rate_by_currency.max() / flags.mean(), 2)

    low_activity = df["velocity_num_transactions"] <= df["velocity_num_transactions"].quantile(LOW_ACTIVITY_QUANTILE)
    result["low_activity_share"] = round(low_activity[flags].mean(), 3)
    return result


def verdict(feature, method, ev, df):
    """(verdict, reason) for one detection. Verdicts: accepted / corrected / rejected / supporting only."""
    rule = FEATURE_RULES[feature]
    kind, level = rule["kind"], rule["level"]
    values = df[feature]

    if ev["flagged_rows"] == 0:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        if method.startswith("IQR") and q1 == q3:
            return "rejected", "IQR = 0 (more than half of the values are identical), IQR cannot be used"
        return "accepted", "no outliers found"

    if kind == "discrete_mode":
        mode_share = values.eq(values.mode()[0]).mean()
        return ("rejected",
                f"discrete feature, {mode_share:.1%} of rows have the value {values.mode()[0]:g}; every other "
                f"value is flagged. These are valid shared-device cases (fraud rate {ev['fraud_rate']:.0%}), "
                f"kept as the is_shared_device category, not treated as outliers")

    if kind == "bounded_activity":
        return ("rejected",
                f"count with a natural upper limit ({values.max():g}); only the low side is flagged and "
                f"{ev['low_activity_share']:.0%} of these rows have very few previous transactions in the hour, "
                f"so a small count is expected, not an error (fraud lift {ev['fraud_lift']})")

    if kind == "mixed_currency":
        reason = ("adds up raw transaction amounts in different local currencies (not converted to USD), "
                  "so the value is not comparable between rows")
        if ev["top_currency_ratio"] > CURRENCY_RATIO_LIMIT:
            reason += (f"; flag rate in {ev['top_currency']} is {ev['top_currency_ratio']}x the overall "
                       f"rate -> currency artefact")
        else:
            reason += f"; {ev['low_activity_share']:.0%} of flagged rows are low-activity windows"
        return "rejected", reason + f" (fraud lift {ev['fraud_lift']})"

    customer_note = ""
    if level == "customer":
        customer_note = (f"; customer-level value repeated on every transaction ({ev['flagged_rows']} rows "
                         f"from only {ev['flagged_customers']} customers) -> detection repeated with one row "
                         f"per customer")

    if kind == "skewed_money":
        if method == "Z-score":
            return ("rejected",
                    f"masking: on the raw right-skewed scale the largest values inflate mean and std, "
                    f"only {ev['flagged_pct']}% flagged and fraud lift {ev['fraud_lift']}; replaced by the "
                    f"log scale")
        if method in ("IQR", "IQR (extreme)"):
            return ("rejected",
                    f"swamping: IQR on the raw right-skewed scale flags {ev['flagged_pct']}% of rows; "
                    f"replaced by IQR on the log scale")
        if level == "customer":
            return "corrected", "right scale (log)" + customer_note
        return "accepted", "right scale (log); confirmed only where IQR on the log scale agrees"

    # continuous
    if level == "customer":
        return "corrected", "method fits the distribution" + customer_note
    if method == "IQR (extreme)":
        return "accepted", "subset of the IQR outliers; covered by the Z-score + IQR agreement"
    if ev["flagged_pct"] > SWAMPING_LIMIT_PCT:
        return "corrected", f"flags {ev['flagged_pct']}% of rows; only rows where Z-score and IQR agree are kept"
    return "accepted", "confirmed only where Z-score and IQR agree"


def review_detections(df):
    rows = []
    for feature in FEATURES:
        for method, suffix in METHOD_SUFFIXES.items():
            column = feature + suffix
            if column not in df.columns:
                continue
            ev = evidence(df, df[column])
            decision, reason = verdict(feature, method, ev, df)
            rows.append({"feature": feature, "method": method, **ev, "verdict": decision, "reason": reason})

    for method, column in MULTIVARIATE_FLAGS.items():
        if column not in df.columns:
            continue
        ev = evidence(df, df[column])
        rows.append({"feature": "(multivariate)", "method": method, **ev, "verdict": "supporting only",
                     "reason": "contamination fixed at 5% (always flags the same number of rows) and the "
                               "inputs include mixed-currency velocity amounts; used only to grade the "
                               "confidence of the confirmed outliers, not to change values"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. corrected detection

def bounds(values):
    """Z-score and IQR bounds; a value is a confirmed outlier only outside both."""
    mean, std = values.mean(), values.std(ddof=0)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if std == 0 or iqr == 0:
        return None
    z_low, z_high = mean - Z_THRESHOLD * std, mean + Z_THRESHOLD * std
    iqr_low, iqr_high = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
    return min(z_low, iqr_low), max(z_high, iqr_high)


def detect_confirmed(df, feature):
    """Returns (flag per row, lower bound, upper bound) on the original scale."""
    rule = FEATURE_RULES[feature]
    unit = df.drop_duplicates("customer_id") if rule["level"] == "customer" else df
    values = unit[feature].astype(float)
    log_scale = rule["kind"] == "skewed_money"
    if log_scale:
        values = np.log1p(values)

    found = bounds(values)
    if found is None:
        return pd.Series(0, index=df.index), np.nan, np.nan
    low, high = found
    if log_scale:
        low, high = np.expm1(low), np.expm1(high)

    row_values = df[feature].astype(float)
    flags = ((row_values < low) | (row_values > high)).astype(int)
    return flags, low, high


def rescale(values, how):
    if how == "log_z":
        values = np.log1p(values)
    if how in ("log_z", "z"):
        return (values - values.mean()) / values.std(ddof=0)
    return (values - values.min()) / (values.max() - values.min())


def stats(values):
    return {"mean": values.mean(), "std": values.std(ddof=0), "skew": values.skew(),
            "min": values.min(), "max": values.max()}


def treat(df):
    df = df.copy()
    report = []
    for feature, rule in FEATURE_RULES.items():
        original = df[feature].astype(float)
        if rule["kind"] not in TREATED_KINDS:
            report.append({"feature": feature, "kind": rule["kind"], "level": rule["level"],
                           "detection": "none (detections rejected, see 3B)",
                           "treatment": "kept as is", "low_outliers": 0, "high_outliers": 0,
                           "values_changed": 0})
            continue

        flags, low, high = detect_confirmed(df, feature)
        capped = original.clip(lower=low, upper=high)
        df[feature + "_outlier"] = flags
        df[feature + "_capped"] = capped
        df[f"{feature}_capped_{rule['scale']}"] = rescale(capped, rule["scale"])

        before, after = stats(original), stats(capped)
        scale_text = "log1p" if rule["kind"] == "skewed_money" else "raw"
        report.append({
            "feature": feature, "kind": rule["kind"], "level": rule["level"],
            "detection": f"Z-score AND IQR, {scale_text} scale, one row per {rule['level']}",
            "treatment": f"capped (winsorized) -> {feature}_capped, {feature}_capped_{rule['scale']}",
            "lower_bound": round(low, 4), "upper_bound": round(high, 4),
            "low_outliers": int((original < low).sum()), "high_outliers": int((original > high).sum()),
            "values_changed": int((capped != original).sum()),
            "outlier_customers": int(df.loc[flags == 1, "customer_id"].nunique()),
            "fraud_rate_outliers": round(df.loc[flags == 1, "is_fraud"].mean(), 3) if flags.any() else np.nan,
            **{f"{k}_before": round(v, 4) for k, v in before.items()},
            **{f"{k}_after": round(v, 4) for k, v in after.items()},
        })

    flag_columns = [f + "_outlier" for f, r in FEATURE_RULES.items() if r["kind"] in TREATED_KINDS]
    df["confirmed_outlier_count"] = df[flag_columns].sum(axis=1)
    df["is_confirmed_outlier"] = (df["confirmed_outlier_count"] > 0).astype(int)
    if "is_univariate_outlier" in df.columns:
        # flagged in the detection task, but not confirmed after the review -> incorrect detection
        df["is_incorrect_detection"] = ((df["is_univariate_outlier"] == 1) &
                                        (df["is_confirmed_outlier"] == 0)).astype(int)
    df["outlier_confidence"] = confidence(df)
    return df, pd.DataFrame(report)


def confidence(df):
    votes = sum(df[c] for c in ("isolation_forest_outlier", "lof_outlier") if c in df.columns)
    confirmed = df["is_confirmed_outlier"] == 1
    return np.select(
        [confirmed & (votes >= 1), confirmed | (votes == 2), votes == 1],
        ["high", "medium", "low"], default="none")


# ---------------------------------------------------------------- summary / plots

def summary(df):
    total = len(df)
    fraud_total = df["is_fraud"].sum()
    rows = [("rows", total), ("fraud rate (all rows)", round(df["is_fraud"].mean(), 4))]

    def group(name, mask):
        mask = mask.astype(bool)
        rows.extend([
            (f"{name}: rows", int(mask.sum())),
            (f"{name}: % of rows", round(100 * mask.mean(), 2)),
            (f"{name}: fraud rate", round(df.loc[mask, "is_fraud"].mean(), 4) if mask.any() else np.nan),
            (f"{name}: % of all fraud cases lost if these rows were removed",
             round(100 * df.loc[mask, "is_fraud"].sum() / fraud_total, 2) if fraud_total else np.nan),
        ])

    if "is_univariate_outlier" in df.columns:
        group("univariate detection (Z-score or IQR, before review)", df["is_univariate_outlier"])
        group("incorrect detections (flagged before, not confirmed)", df["is_incorrect_detection"])
    group("confirmed outliers (after review)", df["is_confirmed_outlier"])
    for level in ("high", "medium", "low"):
        group(f"confidence = {level}", df["outlier_confidence"] == level)
    rows.append(("rows removed", 0))
    rows.append(("decision", "no rows removed: outliers are kept and capped, flags are kept as features"))
    return pd.DataFrame(rows, columns=["metric", "value"])


def horizontal_boxplot(ax, values):
    style = {"marker": "x", "markeredgecolor": "red"}
    try:
        ax.boxplot(values, orientation="horizontal", flierprops=style)   # matplotlib >= 3.10
    except TypeError:
        ax.boxplot(values, vert=False, flierprops=style)                 # older matplotlib


def plot(df, path):
    treated = [f for f, r in FEATURE_RULES.items() if r["kind"] in TREATED_KINDS]
    fig, axes = plt.subplots(len(treated), 2, figsize=(12, 2.2 * len(treated)))
    axes = np.atleast_2d(axes)
    for row, feature in zip(axes, treated):
        for ax, column, label in ((row[0], feature, "before"), (row[1], feature + "_capped", "after capping")):
            horizontal_boxplot(ax, df[column])
            if FEATURE_RULES[feature]["kind"] == "skewed_money":
                ax.set_xscale("log")
            ax.set_title(f"{feature} ({label})")
            ax.set_yticks([])
            ax.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- visualisations

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c3c2b7"
VERDICT_COLORS = {"accepted": BLUE, "corrected": AQUA, "rejected": ORANGE, "supporting only": VIOLET}


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_bars(ax, bars, texts, horizontal=True):
    for bar, text in zip(bars, texts):
        if horizontal:
            ax.annotate(text, (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        else:
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)


def plot_detection_review(review, path):
    """3F: every detection, how many rows it flagged and the verdict after the review."""
    data = review[review["flagged_rows"] > 0].iloc[::-1]
    names = data["feature"].str.replace("(multivariate)", "multivariate", regex=False) + "  |  " + data["method"]
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(data) + 1.8), facecolor=SURFACE)
    bars = ax.barh(names, data["flagged_pct"], height=0.7,
                   color=[VERDICT_COLORS[v] for v in data["verdict"]], edgecolor=SURFACE, linewidth=1)
    label_bars(ax, bars, [f"{p:.1f}%  (fraud lift {l:.2f})" for p, l in zip(data["flagged_pct"], data["fraud_lift"])])
    style(ax)
    ax.set_xlabel("rows flagged (%)", color=INK)
    ax.set_xlim(0, data["flagged_pct"].max() * 1.35)
    ax.set_title("Review of every outlier detection (feature | method), coloured by verdict\n"
                 "fraud lift = fraud rate of the flagged rows / overall fraud rate", loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in VERDICT_COLORS.values()]
    ax.legend(handles, VERDICT_COLORS.keys(), loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_incorrect_detection_evidence(df, path):
    """3G: four examples that show why some detections are incorrect."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    (a, b), (c, d) = axes

    # a) masking: raw Z-score bounds vs log-scale bounds on amount_usd
    amount = df["amount_usd"].astype(float)
    log_amount = np.log10(amount.clip(lower=0.01))
    edges = np.linspace(log_amount.min(), log_amount.max(), 60)
    a.hist([log_amount[df["is_fraud"] == 0], log_amount[df["is_fraud"] == 1]], bins=edges, stacked=True,
           color=[BLUE, ORANGE], edgecolor=SURFACE, linewidth=0.5, label=["not fraud", "fraud"])
    raw_upper = amount.mean() + Z_THRESHOLD * amount.std(ddof=0)
    flags, low, high = detect_confirmed(df, "amount_usd")
    a.axvline(np.log10(raw_upper), color=INK, linestyle="--", linewidth=1.5)
    top = a.get_ylim()[1]
    a.annotate("raw Z-score\nupper bound\n(lower bound < 0,\nmisses the low tail)", (np.log10(raw_upper), top * 0.55),
               xytext=(5, 0), textcoords="offset points", ha="left", fontsize=8, color=INK)
    for bound, text, side, offset in ((low, "log-scale\nlower bound", "left", 4),
                                      (high, "log-scale\nupper bound", "right", -4)):
        a.axvline(np.log10(bound), color=VIOLET, linewidth=1.5)
        a.annotate(text, (np.log10(bound), top * 0.8), xytext=(offset, 0), textcoords="offset points",
                   ha=side, fontsize=8, color=VIOLET)
    style(a, "y")
    a.set_xlabel("amount_usd (log10 USD)", color=INK)
    a.set_ylabel("transactions", color=INK)
    a.set_title("Masking: on the raw scale Z-score misses the micro-amounts (almost all fraud)", loc="left",
                fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=9, loc="upper left")

    # b) currency artefact: velocity_total_amount IQR flag rate per currency
    column = "velocity_total_amount_iqr_outlier"
    if column in df.columns and "currency_resolved" in df.columns:
        rate = (100 * df.groupby("currency_resolved")[column].mean()).sort_values()
        top = rate.idxmax()
        bars = b.barh(rate.index, rate.values, color=[ORANGE if c == top else GRAY for c in rate.index],
                      height=0.7, edgecolor=SURFACE)
        label_bars(b, bars, [f"{v:.1f}%" for v in rate.values])
        overall = 100 * df[column].mean()
        b.axvline(overall, color=INK, linestyle="--", linewidth=1)
        b.annotate(f"all rows {overall:.1f}%", (overall, -0.6), xytext=(4, 0), textcoords="offset points",
                   fontsize=8, color=INK)
        b.set_xlim(0, rate.max() * 1.2)
    style(b)
    b.set_xlabel("rows flagged by IQR on velocity_total_amount (%)", color=INK)
    b.set_title("Currency artefact: velocity amounts mix local currencies", loc="left", fontsize=10, color=INK)

    # c) bounded count: few unique merchants simply means few transactions in the hour
    column = "velocity_unique_merchants_z_outlier"
    flagged = df[column] == 1 if column in df.columns else pd.Series(False, index=df.index)
    x = df["velocity_num_transactions"] + 1
    c.scatter(x[~flagged], df.loc[~flagged, "velocity_unique_merchants"], s=8, color=BLUE, alpha=0.25,
              linewidths=0, label="not flagged")
    c.scatter(x[flagged], df.loc[flagged, "velocity_unique_merchants"], s=10, color=ORANGE, alpha=0.6,
              linewidths=0, label="flagged by Z-score")
    c.set_xscale("log")
    style(c, "both")
    c.set_xlabel("transactions in the last hour (num_transactions + 1, log scale)", color=INK)
    c.set_ylabel("velocity_unique_merchants", color=INK)
    c.set_title("Bounded count: flagged values are just low-activity windows", loc="left", fontsize=10, color=INK)
    c.legend(frameon=False, fontsize=9, loc="lower right", markerscale=2)

    # d) customer-level features: rows flagged vs distinct customers behind them
    customer_features = [f for f, r in FEATURE_RULES.items() if r["level"] == "customer"]
    rows, customers = [], []
    for feature in customer_features:
        column = feature + ("_zlog_outlier" if feature + "_zlog_outlier" in df.columns else "_z_outlier")
        mask = df[column] == 1
        rows.append(int(mask.sum()))
        customers.append(int(df.loc[mask, "customer_id"].nunique()))
    y = np.arange(len(customer_features))
    bars_rows = d.barh(y + 0.2, rows, height=0.38, color=BLUE, edgecolor=SURFACE, label="flagged rows")
    bars_cust = d.barh(y - 0.2, customers, height=0.38, color=AQUA, edgecolor=SURFACE, label="distinct customers")
    label_bars(d, bars_rows, [f"{v:,}" for v in rows])
    label_bars(d, bars_cust, [f"{v:,}" for v in customers])
    d.set_yticks(y, customer_features)
    d.set_xlim(0, max(rows + [1]) * 1.2)
    style(d)
    d.set_xlabel("count (Z-score detection; log-scale Z-score for money values)", color=INK)
    d.set_title("Customer-level values repeat on every transaction", loc="left", fontsize=10, color=INK)
    d.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Why some detections were incorrect", x=0.01, ha="left", fontsize=13, color=INK)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_handling_summary(df, path):
    """3H: before vs after the review - rows flagged, fraud rate, fraud lost if removed, confidence."""
    groups = {"flagged before\nreview": df["is_univariate_outlier"] == 1,
              "incorrect\ndetections": df["is_incorrect_detection"] == 1,
              "confirmed\noutliers": df["is_confirmed_outlier"] == 1}
    colors = [GRAY, ORANGE, BLUE]
    fraud_total = df["is_fraud"].sum()
    base = df["is_fraud"].mean()

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), facecolor=SURFACE)
    panels = [
        ("rows (% of dataset)", [100 * m.mean() for m in groups.values()], "{:.1f}%"),
        (f"fraud rate of the rows (%), dashed = all rows {100 * base:.1f}%", [100 * df.loc[m, "is_fraud"].mean() for m in groups.values()], "{:.1f}%"),
        ("fraud cases lost if removed (%)", [100 * df.loc[m, "is_fraud"].sum() / fraud_total
                                             for m in groups.values()], "{:.1f}%"),
    ]
    for ax, (title, values, fmt) in zip(axes, panels):
        bars = ax.bar(range(3), values, color=colors, width=0.65, edgecolor=SURFACE)
        label_bars(ax, bars, [fmt.format(v) for v in values], horizontal=False)
        ax.set_xticks(range(3), groups.keys(), fontsize=9)
        ax.set_ylim(0, max(values) * 1.18)
        style(ax, "y")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[1].axhline(100 * base, color=INK, linestyle="--", linewidth=1)

    ax = axes[3]
    levels = ["high", "medium", "low"]
    counts = [int((df["outlier_confidence"] == level).sum()) for level in levels]
    rates = [100 * df.loc[df["outlier_confidence"] == level, "is_fraud"].mean() for level in levels]
    bars = ax.bar(levels, counts, color=["#184f95", BLUE, "#86b6ef"], width=0.65, edgecolor=SURFACE)
    label_bars(ax, bars, [f"{c:,} rows\nfraud {r:.0f}%" for c, r in zip(counts, rates)], horizontal=False)
    ax.set_ylim(0, max(counts) * 1.3)
    style(ax, "y")
    ax.set_title("outlier confidence (with IF / LOF)", loc="left", fontsize=10, color=INK)

    fig.suptitle("Outlier handling: before vs after the review (no rows removed, confirmed outliers capped)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def load(univariate_path, multivariate_path):
    df = pd.read_csv(univariate_path, low_memory=False)
    multivariate = pd.read_csv(multivariate_path)
    if len(multivariate) != len(df):
        raise ValueError("1A and 2E have a different number of rows")
    if "transaction_id" in multivariate.columns and \
            not (multivariate["transaction_id"].to_numpy() == df["transaction_id"].to_numpy()).all():
        raise ValueError("1A and 2E rows are not in the same order (transaction_id differs)")
    for column in MULTIVARIATE_FLAGS.values():
        df[column] = multivariate[column].to_numpy()
    return df


def run(df):
    """Whole task on an already merged dataframe (1A + 2E flags)."""
    checked, domain_report = domain_check(df)
    review = review_detections(checked)
    treated, treatment_report = treat(checked)
    # the per-method flags stay in 1A; 3A keeps the summary flags only
    per_method = [f + s for f in FEATURES for s in METHOD_SUFFIXES.values() if f + s in treated.columns]
    treated = treated.drop(columns=per_method + ["outlier_count", "outlier_count_all_methods"], errors="ignore")
    return treated, review, treatment_report, domain_report


def main(univariate_path=INPUT_UNIVARIATE, multivariate_path=INPUT_MULTIVARIATE, output_folder=OUTPUT_FOLDER):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = load(univariate_path, multivariate_path)
    treated, review, treatment_report, domain_report = run(df)
    handling = pd.concat([
        summary(treated),
        pd.DataFrame({"metric": "domain check violations: " + domain_report["rule"],
                      "value": domain_report["violations"]}),
    ], ignore_index=True)

    treated.to_csv(output_folder / "3A_OUTLIER_TREATED_DATASET.csv", index=False)
    review.to_csv(output_folder / "3B_DETECTION_REVIEW_REPORT.csv", index=False)
    treatment_report.to_csv(output_folder / "3C_OUTLIER_TREATMENT_REPORT.csv", index=False)
    handling.to_csv(output_folder / "3D_OUTLIER_HANDLING_SUMMARY.csv", index=False)
    plot(treated, output_folder / "3E_OUTLIER_TREATMENT_PLOTS.png")
    plot_detection_review(review, output_folder / "3F_DETECTION_REVIEW_PLOT.png")
    plot_incorrect_detection_evidence(df, output_folder / "3G_INCORRECT_DETECTION_EVIDENCE.png")
    plot_handling_summary(treated, output_folder / "3H_OUTLIER_HANDLING_SUMMARY_PLOT.png")

    pd.set_option("display.width", 200)
    print("HANDLING INCORRECT DETECTIONS AND OUTLIER TREATMENT\n")
    print(review[["feature", "method", "flagged_rows", "fraud_lift", "verdict"]].to_string(index=False))
    print()
    print(treatment_report[["feature", "treatment", "low_outliers", "high_outliers", "values_changed"]]
          .to_string(index=False))
    print()
    print(handling.to_string(index=False))
    return treated, review, treatment_report, handling


if __name__ == "__main__":
    main()
