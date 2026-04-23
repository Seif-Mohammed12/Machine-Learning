"""
Milestone 1 - Predicting RecommendationCount for Online Games

We tried 4 models:
    1. Ridge Regression - basic linear model with regularization (our baseline)
    2. Random Forest - ensemble of trees, handles non-linear stuff well
    3. LightGBM - gradient boosting, usually best on tabular data
    4. Stacked Ensemble - combines RF + LightGBM predictions using Ridge on top
"""

import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH              = "data/train_data.csv"
PLOTS_DIR              = "artifacts/plots"
METRICS_PATH           = "artifacts/milestone1_metrics.json"
SELECTED_FEATURES_PATH = "artifacts/selected_features.csv"
TARGET                 = "RecommendationCount"
RANDOM_STATE           = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def make_date_features(df):
    # parse release date and get calendar features
    # release_age_days is useful because older games have had more time to
    # accumulate recommendations (corr=0.36 with log target)
    # note: QueryID correlates at -0.47 with log target and is 0.91 correlated
    # with release_age_days - both basically measure game age. we keep both
    # since QueryID has no missing values.
    release  = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    ref_date = release.dropna().max()

    return pd.DataFrame({
        "release_year":      release.dt.year,
        "release_month":     release.dt.month,
        "release_dayofweek": release.dt.dayofweek,
        "release_age_days":  (ref_date - release).dt.days,
    }, index=df.index)


def make_price_features(df):
    # log compress both price columns (heavy tailed)
    # is_free and discount ratio capture pricing strategy
    # side note: 1392 games have PriceFinal=0 but IsFree=False which is weird,
    # we flag that as price_free_mismatch
    initial = df["PriceInitial"].fillna(0).clip(lower=0)
    final   = df["PriceFinal"].fillna(0).clip(lower=0)
    ratio   = np.where(initial > 0, (initial - final) / initial, 0.0)

    return pd.DataFrame({
        "log1p_price_final":    np.log1p(final),
        "log1p_price_initial":  np.log1p(initial),
        "price_discount_ratio": np.clip(ratio, 0, 1),
        "is_free":              (final == 0).astype(int),
        "is_premium":           (final >= 20).astype(int),
        "price_free_mismatch":  ((final == 0) & (~df["IsFree"])).astype(int),
    }, index=df.index)


def make_steamspy_features(df):
    # steamspy is the strongest predictor group by far
    #
    # important thing to know about steamspy: ownership numbers come in
    # discrete brackets (0-20k, 20k-50k, 50k-100k etc.). the raw value
    # is just the midpoint of whatever bracket the game falls in, not the
    # real number. so two games with 19k and 49k actual owners get the
    # same reported number - that's a lot of noise.
    #
    # using percentile rank (0 to 1) instead fixes this:
    #   - players_rank: 0.786 correlation with log target
    #   - owners_rank:  0.750 correlation with log target
    # both beat the raw log-transformed versions (0.66)
    #
    # we also compute bracket bounds (Owners +/- Variance) so the model
    # knows the actual range, not just the midpoint.
    #
    # keeping all 4 raw steamspy columns - variance columns were accidentally
    # dropped in v1 even though they have 0.53-0.55 corr with log target.
    owners   = df["SteamSpyOwners"].fillna(0).clip(lower=0)
    players  = df["SteamSpyPlayersEstimate"].fillna(0).clip(lower=0)
    own_var  = df["SteamSpyOwnersVariance"].fillna(0).clip(lower=0)
    play_var = df["SteamSpyPlayersVariance"].fillna(0).clip(lower=0)

    lower_bound = np.maximum(0, owners - own_var)
    upper_bound = owners + own_var

    return pd.DataFrame({
        "log1p_steamspy_owners":    np.log1p(owners),
        "log1p_steamspy_players":   np.log1p(players),
        "log1p_steamspy_own_var":   np.log1p(own_var),
        "log1p_steamspy_play_var":  np.log1p(play_var),
        "log1p_ownership_lower":    np.log1p(lower_bound),
        "log1p_ownership_upper":    np.log1p(upper_bound),
        "owners_rank":              owners.rank(pct=True),
        "players_rank":             players.rank(pct=True),
        "own_var_rank":             own_var.rank(pct=True),
        "engagement_ratio":         np.where(owners > 0, players / owners, 0.0),
        "variance_to_owners_ratio": np.where(owners > 0, own_var / (owners + 1), 0.0),
    }, index=df.index)


def make_metacritic_features(df):
    # metacritic score is 0 for 83% of games - that means "not rated",
    # not an actual score of 0. so we need a flag for whether the game
    # even has a metacritic score (corr=0.46 with log target).
    # games with a score average 5020 recommendations vs 435 for unrated.
    score = df["Metacritic"].fillna(0)
    return pd.DataFrame({
        "metacritic_score": score,
        "has_metacritic":   (score > 0).astype(int),
    }, index=df.index)


def make_content_features(df):
    # content richness: movies, screenshots, DLC, achievements, packages
    # all log-compressed because of right skew
    # PackageCount, DeveloperCount, PublisherCount were missing from v1
    # (corr 0.27, 0.15, 0.16 respectively)
    movies  = np.log1p(df["MovieCount"].fillna(0))
    shots   = np.log1p(df["ScreenshotCount"].fillna(0))
    dlc     = np.log1p(df["DLCCount"].fillna(0))
    ach     = np.log1p(df["AchievementCount"].fillna(0))
    ach_h   = np.log1p(df["AchievementHighlightedCount"].fillna(0))
    pkg     = np.log1p(df["PackageCount"].fillna(0))
    dev     = np.log1p(df["DeveloperCount"].fillna(0))
    pub     = np.log1p(df["PublisherCount"].fillna(0))

    return pd.DataFrame({
        "log1p_movie_count":       movies,
        "log1p_screenshot_count":  shots,
        "log1p_dlc_count":         dlc,
        "log1p_achievement_count": ach,
        "log1p_achievement_hl":    ach_h,
        "log1p_package_count":     pkg,
        "log1p_developer_count":   dev,
        "log1p_publisher_count":   pub,
        "content_score":           movies + shots + dlc,
    }, index=df.index)


def make_language_features(df):
    # SupportedLanguages is space-delimited (not comma separated)
    # after removing the full-audio-support suffix, word count = language count
    # range is 0-29, corr=0.21 with log target
    cleaned = df["SupportedLanguages"].fillna("").str.replace(
        r"\*[^*]*\*?languages[^*]*", "", regex=True
    )
    lang_count = cleaned.str.split().apply(len)

    return pd.DataFrame({
        "lang_count":       lang_count,
        "log1p_lang_count": np.log1p(lang_count),
        "is_multilingual":  (lang_count > 1).astype(int),
    }, index=df.index)


def make_platform_features(df):
    # linux and mac support correlates positively with recommendations
    # (linux games get 3.5x more recs on average, mac gets 2.7x vs windows only)
    # probably because cross-platform games tend to be better funded/polished
    win = df["PlatformWindows"].astype(int)
    lin = df["PlatformLinux"].astype(int)
    mac = df["PlatformMac"].astype(int)

    return pd.DataFrame({
        "platform_count": win + lin + mac,
        "supports_linux": lin,
        "supports_mac":   mac,
    }, index=df.index)


def make_multiplayer_features(df):
    # multiplayer games average 4-5x more recommendations than single player
    # action + multiplayer is especially strong (CS, TF2, Dota etc.)
    is_multi = (
        df["CategoryMultiplayer"] | df["CategoryCoop"] | df["CategoryMMO"]
    ).astype(int)

    lang_count = df["SupportedLanguages"].fillna("").str.replace(
        r"\*[^*]*\*?languages[^*]*", "", regex=True
    ).str.split().apply(len)

    log_owners = np.log1p(df["SteamSpyOwners"].fillna(0).clip(lower=0))

    return pd.DataFrame({
        "is_multiplayer":       is_multi,
        "lang_x_multiplayer":   lang_count * is_multi,
        "owners_x_multiplayer": log_owners * is_multi,
        "action_x_multiplayer": df["GenreIsAction"].astype(int) * is_multi,
    }, index=df.index)


def make_age_features(df):
    # RequiredAge is 0 for 95% of games
    # the useful signal is really just whether a game is mature rated (17+)
    age = df["RequiredAge"].fillna(0)
    return pd.DataFrame({
        "required_age": age,
        "is_mature":    (age >= 17).astype(int),
    }, index=df.index)


def make_text_richness_features(df):
    # longer descriptions = more polished store page = more marketing effort
    # which correlates with game quality and popularity
    cols = ["AboutText", "ShortDescrip", "DetailedDescrip",
            "PCMinReqsText", "PCRecReqsText", "Reviews"]
    out = {}
    for col in cols:
        s = df[col].fillna("").astype(str)
        out[f"{col}_len"] = np.log1p(s.str.len())
        out[f"{col}_has"] = (s.str.strip() != "").astype(int)

    out["has_support_email"] = (df["SupportEmail"].fillna("") != "").astype(int)
    out["has_website"]       = (df["Website"].fillna("") != "").astype(int)
    out["has_reviews"]       = (df["Reviews"].fillna("").str.strip() != "").astype(int)

    return pd.DataFrame(out, index=df.index)


def make_interaction_features(df):
    # hand-crafted interaction terms between strongest predictors
    # owners_x_metacritic (corr=0.51): lots of owners AND critical acclaim
    # basically guarantees a blockbuster
    # var_x_owners: high variance relative to ownership = near a steamspy
    # bracket boundary, so actual popularity might be higher than midpoint
    log_owners  = np.log1p(df["SteamSpyOwners"].fillna(0).clip(lower=0))
    log_players = np.log1p(df["SteamSpyPlayersEstimate"].fillna(0).clip(lower=0))
    log_own_var = np.log1p(df["SteamSpyOwnersVariance"].fillna(0).clip(lower=0))
    metacritic  = df["Metacritic"].fillna(0)
    has_meta    = (metacritic > 0).astype(float)

    return pd.DataFrame({
        "owners_x_metacritic":     log_owners * metacritic,
        "players_x_metacritic":    log_players * metacritic,
        "owners_x_has_metacritic": log_owners * has_meta,
        "var_x_owners":            log_own_var * log_owners,
    }, index=df.index)


def engineer_features(df):
    # put everything together
    parts = [
        make_date_features(df),
        make_price_features(df),
        make_steamspy_features(df),
        make_metacritic_features(df),
        make_content_features(df),
        make_language_features(df),
        make_platform_features(df),
        make_multiplayer_features(df),
        make_age_features(df),
        make_text_richness_features(df),
        make_interaction_features(df),
    ]

    # boolean category/genre columns -> integers
    bool_cols = df.select_dtypes(include="bool").columns
    parts.append(df[bool_cols].astype(int))

    # any remaining numeric columns we haven't already handled
    already_handled = {
        "PriceInitial", "PriceFinal",
        "SteamSpyOwners", "SteamSpyOwnersVariance",
        "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
        "Metacritic", "MovieCount", "ScreenshotCount",
        "DLCCount", "AchievementCount", "AchievementHighlightedCount",
        "RequiredAge", "PackageCount", "DeveloperCount", "PublisherCount",
        TARGET,
    }
    extra = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in already_handled]
    parts.append(df[extra])

    # one-hot encode currency
    parts.append(pd.get_dummies(df["PriceCurrency"].fillna("Unknown"),
                                prefix="currency"))

    features = pd.concat(parts, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features


# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(y_true_log, y_pred_log):
    # compute metrics on both log scale (for model comparison) and
    # original scale (so we can actually interpret the numbers)
    y_pred_log = np.clip(y_pred_log, -20, 20)
    y_true     = np.expm1(y_true_log)
    y_pred     = np.clip(np.expm1(y_pred_log), 0, None)

    return {
        "r2_log":  round(float(r2_score(y_true_log, y_pred_log)), 4),
        "r2_orig": round(float(r2_score(y_true, y_pred)), 4),
        "r2_%":    round(max(0.0, r2_score(y_true_log, y_pred_log) * 100), 2),
        "mae":     round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse":    round(float(mean_squared_error(y_true, y_pred) ** 0.5), 2),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_target_dist(y, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(y, bins=60, ax=axes[0], color="#0f766e", kde=False)
    axes[0].set_title("RecommendationCount — raw (skew=68)")
    axes[0].set_xlabel("RecommendationCount")

    sns.histplot(np.log1p(y), bins=60, ax=axes[1], color="#1d4ed8", kde=False)
    axes[1].set_title("log1p(RecommendationCount) — after transform")
    axes[1].set_xlabel("log1p(RecommendationCount)")

    fig.suptitle("Target Distribution: raw vs log-transformed", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(X, path):
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.3, annot=False, ax=ax)
    ax.set_title("Feature Correlation Matrix (selected features)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_importance(model, feature_names, title, color, path):
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).nlargest(20, "importance").iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color=color)
    ax.set_title(f"{title} — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_actual_vs_predicted(y_true_log, y_pred_log, title, path, log_scale=True):
    y_pred_log = np.clip(y_pred_log, -20, 20)
    if log_scale:
        actual, predicted = y_true_log, y_pred_log
        xlabel, ylabel    = "log1p(Actual)", "log1p(Predicted)"
    else:
        actual    = np.expm1(y_true_log)
        predicted = np.clip(np.expm1(y_pred_log), 0, None)
        xlabel, ylabel = "Actual RecommendationCount", "Predicted RecommendationCount"

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, s=18, alpha=0.35, color="#2563eb")
    ax.plot([lo, hi], [lo, hi], color="#dc2626", linestyle="--", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residuals(y_true_log, y_pred_log, title, path):
    y_pred_log = np.clip(y_pred_log, -20, 20)
    residuals  = y_true_log - y_pred_log

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_log, residuals, s=18, alpha=0.35, color="#7c3aed")
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title(f"{title} — Residuals vs Fitted")
    ax.set_xlabel("Fitted log1p(RecommendationCount)")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_comparison(results, path):
    df     = pd.DataFrame(results)
    melted = df.melt(id_vars="model", value_vars=["mae", "rmse"],
                     var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Model Comparison — Test Set Error (original scale)")
    ax.set_ylabel("Error")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_r2_comparison(results, path):
    df = pd.DataFrame(results)[["model", "r2_%"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="model", y="r2_%", ax=ax, palette="Blues_d")
    ax.set_title("Model R2 Comparison (log scale)")
    ax.set_ylabel("R2 %")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# =============================================================================
# STACKING HELPER
# =============================================================================

def get_oof_predictions(model, X, y, kfold):
    # out-of-fold predictions for stacking
    # each row gets predicted by a model that never saw it during training
    # this stops the meta-learner from exploiting base model overfitting
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr        = y.iloc[train_idx]
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
    return oof_preds


# =============================================================================
# MAIN
# =============================================================================

def main():
    # load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # feature engineering
    X = engineer_features(df)
    y = np.log1p(df[TARGET])

    print(f"Engineered feature matrix: {X.shape}")
    plot_target_dist(df[TARGET], f"{PLOTS_DIR}/target_distribution.png")

    # 80/20 train test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # median imputation - more robust than mean for skewed columns
    # fit on train only to avoid leakage
    imputer     = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw),
                               columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw),
                               columns=X_test_raw.columns, index=X_test_raw.index)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # feature selection for Ridge only (tree models handle this internally)
    # extended to k=80 because k=60 was still winning last time
    print("\nSelecting features for Ridge via mutual information ...")
    best_k, best_k_r2 = 20, -np.inf
    for k in [20, 30, 40, 50, 60, 70, 80]:
        pipe = Pipeline([
            ("sel",   SelectKBest(mutual_info_regression, k=k)),
            ("sc",    StandardScaler()),
            ("model", Ridge()),
        ])
        r2 = cross_val_score(pipe, X_train_imp, y_train, cv=kfold, scoring="r2").mean()
        print(f"  k={k:>3}  CV R2 = {r2:.4f}")
        if r2 > best_k_r2:
            best_k_r2, best_k = r2, k

    selector = SelectKBest(mutual_info_regression, k=best_k)
    selector.fit(X_train_imp, y_train)
    sel_mask   = selector.get_support()
    sel_scores = pd.DataFrame({
        "feature": X_train_imp.columns[sel_mask],
        "score":   selector.scores_[sel_mask],
    }).sort_values("score", ascending=False)
    sel_scores.to_csv(SELECTED_FEATURES_PATH, index=False)

    sel_cols    = sel_scores["feature"].tolist()
    X_train_sel = X_train_imp[sel_cols]
    X_test_sel  = X_test_imp[sel_cols]

    print(f"Best k = {best_k}  |  Top features: {sel_cols[:8]} ...")
    plot_correlation_heatmap(X_train_sel, f"{PLOTS_DIR}/feature_correlation.png")

    all_results = []

    # =========================================================================
    # MODEL 1 - Ridge Regression
    # =========================================================================
    # adds L2 regularization to linear regression so large coefficients
    # get penalized. more stable than plain OLS when features are correlated.
    # tuning alpha via cross-validation.
    print("\n-- Ridge Regression --")
    best_alpha, best_ridge_r2 = 1.0, -np.inf
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe  = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])
        score = cross_val_score(pipe, X_train_sel, y_train, cv=kfold, scoring="r2").mean()
        print(f"  alpha={alpha:>7}  CV R2 = {score:.4f}")
        if score > best_ridge_r2:
            best_ridge_r2, best_alpha = score, alpha

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=best_alpha))])
    ridge_cv   = cross_val_score(ridge_pipe, X_train_sel, y_train, cv=kfold, scoring="r2")
    ridge_pipe.fit(X_train_sel, y_train)
    ridge_pred = ridge_pipe.predict(X_test_sel)
    ridge_m    = compute_metrics(y_test.values, ridge_pred)
    ridge_m.update({"cv_r2_mean": round(ridge_cv.mean(), 4),
                    "cv_r2_std":  round(ridge_cv.std(), 4)})
    print(f"\n  Best alpha = {best_alpha}")
    print(f"  CV R2 = {ridge_m['cv_r2_mean']} +/- {ridge_m['cv_r2_std']}")
    print(f"  Test  R2(log) = {ridge_m['r2_log']}  MAE = {ridge_m['mae']:,.0f}")

    plot_actual_vs_predicted(y_test.values, ridge_pred,
                             f"Ridge (alpha={best_alpha}) — log scale",
                             f"{PLOTS_DIR}/ridge_scatter_log.png")
    plot_actual_vs_predicted(y_test.values, ridge_pred,
                             f"Ridge (alpha={best_alpha}) — original scale",
                             f"{PLOTS_DIR}/ridge_scatter.png", log_scale=False)
    plot_residuals(y_test.values, ridge_pred,
                   f"Ridge (alpha={best_alpha})", f"{PLOTS_DIR}/ridge_residuals.png")

    all_results.append({"model": f"Ridge (α={best_alpha})", **ridge_m})

    # =========================================================================
    # MODEL 2 - Random Forest
    # =========================================================================
    # builds lots of trees in parallel on random subsets of data and features
    # averaging them reduces variance without hurting bias too much
    # uses all features - RF naturally ignores irrelevant ones by not splitting on them
    print("\n-- Random Forest --")
    best_rf_params, best_rf_r2 = {"n_estimators": 200, "max_depth": 20}, -np.inf

    for n_est in [100, 200]:
        for max_d in [10, 20, None]:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                       min_samples_leaf=4, n_jobs=-1,
                                       random_state=RANDOM_STATE)
            score = cross_val_score(rf, X_train_imp, y_train,
                                    cv=kfold, scoring="r2").mean()
            print(f"  n_est={n_est}  max_depth={str(max_d):>5}  CV R2 = {score:.4f}")
            if score > best_rf_r2:
                best_rf_r2     = score
                best_rf_params = {"n_estimators": n_est, "max_depth": max_d}

    rf_model = RandomForestRegressor(**best_rf_params, min_samples_leaf=4,
                                     n_jobs=-1, random_state=RANDOM_STATE)
    rf_cv    = cross_val_score(rf_model, X_train_imp, y_train, cv=kfold, scoring="r2")
    rf_model.fit(X_train_imp, y_train)
    rf_pred  = rf_model.predict(X_test_imp)
    rf_m     = compute_metrics(y_test.values, rf_pred)
    rf_m.update({"cv_r2_mean": round(rf_cv.mean(), 4),
                 "cv_r2_std":  round(rf_cv.std(), 4)})
    print(f"\n  Best params: {best_rf_params}")
    print(f"  CV R2 = {rf_m['cv_r2_mean']} +/- {rf_m['cv_r2_std']}")
    print(f"  Test  R2(log) = {rf_m['r2_log']}  MAE = {rf_m['mae']:,.0f}")

    plot_actual_vs_predicted(y_test.values, rf_pred,
                             "Random Forest — log scale",
                             f"{PLOTS_DIR}/rf_scatter_log.png")
    plot_actual_vs_predicted(y_test.values, rf_pred,
                             "Random Forest — original scale",
                             f"{PLOTS_DIR}/rf_scatter.png", log_scale=False)
    plot_residuals(y_test.values, rf_pred,
                   "Random Forest", f"{PLOTS_DIR}/rf_residuals.png")
    plot_feature_importance(rf_model, X_train_imp.columns.tolist(),
                            "Random Forest", "#16a34a",
                            f"{PLOTS_DIR}/rf_feature_importance.png")

    all_results.append({"model": "Random Forest", **rf_m})

    # =========================================================================
    # MODEL 3 - LightGBM
    # =========================================================================
    # grows trees leaf-wise instead of level-wise so it finds better splits faster
    # key difference from RF: sequential, each tree corrects previous tree's errors
    # num_leaves controls complexity more directly than max_depth
    # has built in L1/L2 reg and feature subsampling like RF
    print("\n-- LightGBM --")
    best_lgb_params = {"n_estimators": 500, "num_leaves": 31}
    best_lgb_r2     = -np.inf

    for n_est in [300, 500, 700]:
        for num_leaves in [31, 63, 127]:
            lgb = LGBMRegressor(
                n_estimators=n_est, num_leaves=num_leaves,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=20,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )
            score = cross_val_score(lgb, X_train_imp, y_train,
                                    cv=kfold, scoring="r2").mean()
            print(f"  n_est={n_est}  num_leaves={num_leaves:<4}  CV R2 = {score:.4f}")
            if score > best_lgb_r2:
                best_lgb_r2     = score
                best_lgb_params = {"n_estimators": n_est, "num_leaves": num_leaves}

    lgb_model = LGBMRegressor(
        **best_lgb_params,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    lgb_cv   = cross_val_score(lgb_model, X_train_imp, y_train, cv=kfold, scoring="r2")
    lgb_model.fit(X_train_imp, y_train)
    lgb_pred = lgb_model.predict(X_test_imp)
    lgb_m    = compute_metrics(y_test.values, lgb_pred)
    lgb_m.update({"cv_r2_mean": round(lgb_cv.mean(), 4),
                  "cv_r2_std":  round(lgb_cv.std(), 4)})
    print(f"\n  Best params: {best_lgb_params}")
    print(f"  CV R2 = {lgb_m['cv_r2_mean']} +/- {lgb_m['cv_r2_std']}")
    print(f"  Test  R2(log) = {lgb_m['r2_log']}  MAE = {lgb_m['mae']:,.0f}")

    plot_actual_vs_predicted(y_test.values, lgb_pred,
                             "LightGBM — log scale",
                             f"{PLOTS_DIR}/lgb_scatter_log.png")
    plot_actual_vs_predicted(y_test.values, lgb_pred,
                             "LightGBM — original scale",
                             f"{PLOTS_DIR}/lgb_scatter.png", log_scale=False)
    plot_residuals(y_test.values, lgb_pred,
                   "LightGBM", f"{PLOTS_DIR}/lgb_residuals.png")
    plot_feature_importance(lgb_model, X_train_imp.columns.tolist(),
                            "LightGBM", "#b45309",
                            f"{PLOTS_DIR}/lgb_feature_importance.png")

    all_results.append({"model": "LightGBM", **lgb_m})

    # =========================================================================
    # MODEL 4 - Stacked Ensemble (RF + LightGBM -> Ridge meta-learner)
    # =========================================================================
    # instead of training directly on features, the meta-model trains on the
    # predictions of the two base models.
    #
    # the trick is using out-of-fold predictions so there's no leakage.
    # for each fold we train on the other 4 folds and predict on the held out
    # one - so no prediction ever uses data the model was trained on.
    # those OOF predictions become the training data for the meta Ridge model.
    #
    # for test predictions we just retrain the base models on all of train.
    print("\n-- Stacked Ensemble (RF + LightGBM -> Ridge meta-learner) --")

    rf_for_stack  = RandomForestRegressor(**best_rf_params, min_samples_leaf=4,
                                          n_jobs=-1, random_state=RANDOM_STATE)
    lgb_for_stack = LGBMRegressor(
        **best_lgb_params,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )

    print("  Getting OOF predictions for RF ...")
    rf_oof  = get_oof_predictions(rf_for_stack,  X_train_imp, y_train, kfold)
    print("  Getting OOF predictions for LightGBM ...")
    lgb_oof = get_oof_predictions(lgb_for_stack, X_train_imp, y_train, kfold)

    meta_train = np.column_stack([rf_oof, lgb_oof])

    # retrain on full training set for final test predictions
    rf_for_stack.fit(X_train_imp,  y_train)
    lgb_for_stack.fit(X_train_imp, y_train)
    meta_test = np.column_stack([
        rf_for_stack.predict(X_test_imp),
        lgb_for_stack.predict(X_test_imp),
    ])

    meta_scaler = StandardScaler()
    meta_ridge  = Ridge(alpha=1.0)
    meta_ridge.fit(meta_scaler.fit_transform(meta_train), y_train)

    stack_pred = meta_ridge.predict(meta_scaler.transform(meta_test))
    stack_m    = compute_metrics(y_test.values, stack_pred)

    meta_cv_scores = cross_val_score(
        Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        meta_train, y_train, cv=kfold, scoring="r2"
    )
    stack_m.update({"cv_r2_mean": round(meta_cv_scores.mean(), 4),
                    "cv_r2_std":  round(meta_cv_scores.std(), 4)})

    print(f"  CV R2 (meta) = {stack_m['cv_r2_mean']} +/- {stack_m['cv_r2_std']}")
    print(f"  Test  R2(log) = {stack_m['r2_log']}  MAE = {stack_m['mae']:,.0f}")

    plot_actual_vs_predicted(y_test.values, stack_pred,
                             "Stacked Ensemble — log scale",
                             f"{PLOTS_DIR}/stack_scatter_log.png")
    plot_actual_vs_predicted(y_test.values, stack_pred,
                             "Stacked Ensemble — original scale",
                             f"{PLOTS_DIR}/stack_scatter.png", log_scale=False)
    plot_residuals(y_test.values, stack_pred,
                   "Stacked Ensemble", f"{PLOTS_DIR}/stack_residuals.png")

    all_results.append({"model": "Stacked Ensemble", **stack_m})

    # comparison plots
    plot_model_comparison(all_results, f"{PLOTS_DIR}/model_comparison.png")
    plot_r2_comparison(all_results,    f"{PLOTS_DIR}/r2_comparison.png")

    # save metrics to json
    summary = {
        "dataset_shape":               list(df.shape),
        "train_size":                  int(len(X_train_imp)),
        "test_size":                   int(len(X_test_imp)),
        "features_engineered":         int(X.shape[1]),
        "features_selected_for_ridge": sel_cols,
        "best_ridge_alpha":            float(best_alpha),
        "best_rf_params":              best_rf_params,
        "best_lgb_params":             best_lgb_params,
        "models":                      {r["model"]: r for r in all_results},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in sorted(all_results, key=lambda x: -x["r2_log"]):
        print(f"  {r['model']:<30}  R2={r['r2_%']}%  MAE={r['mae']:>10,.0f}  CV={r['cv_r2_mean']}")
    print(f"\nMetrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()