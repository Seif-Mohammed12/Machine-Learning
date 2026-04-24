"""
Milestone 1 - Predicting RecommendationCount for Online Games

We tried 4 models:
    1. Ridge Regression - basic linear model with regularization (our baseline)
    2. Random Forest - ensemble of trees, handles non-linear stuff well
    3. LightGBM - gradient boosting, usually best on tabular data
    4. Stacked Ensemble - combines RF + LightGBM predictions using Ridge on top

Structure:
    Step 1 - preprocess_all_columns(): cleans every single column in the raw
             dataset, even ones we won't use later. This is required before
             any feature selection or model building.
    Step 2 - engineer_features(): builds the actual model features from the
             cleaned data. some columns get dropped here with justification.
    Step 3 - models
"""

import json
import os
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
# STEP 1 - PREPROCESSING (every column in the raw dataset)
# =============================================================================

def preprocess_all_columns(df):
    """
    Cleans and standardizes every column in the raw dataset.
    We do this before any feature engineering or model building.
    Columns we won't model on are still cleaned here - we just document
    why we drop them later in engineer_features().

    Returns a new dataframe with all columns cleaned.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # ID columns: QueryID, ResponseID
    # ------------------------------------------------------------------
    # numeric, no nulls, no cleaning needed
    # QueryID has -0.47 corr with log target (lower ID = older/more
    # established game = more time to accumulate recs).
    # 99.3% of rows have QueryID == ResponseID (game queried against itself)
    # so these are basically just the steam app ID.

    # ------------------------------------------------------------------
    # Name columns: QueryName, ResponseName
    # ------------------------------------------------------------------
    # strip whitespace, fill the one null in QueryName with ResponseName
    df["QueryName"]    = df["QueryName"].fillna(df["ResponseName"]).str.strip()
    df["ResponseName"] = df["ResponseName"].str.strip()
    # not used in modeling - 11301 unique values in 11357 rows means
    # almost every game name appears only once, making encoding useless.
    # the ID columns already capture game identity.

    # ------------------------------------------------------------------
    # ReleaseDate
    # ------------------------------------------------------------------
    # stored as string, parse to datetime. no nulls in raw data.
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    # 0 nulls after parsing (all dates are valid)

    # ------------------------------------------------------------------
    # RequiredAge
    # ------------------------------------------------------------------
    # already integer, no nulls. values: 0, 7, 12, 13, 16, 17, 18, 21
    # 95% of games are 0 (no age restriction). leave as-is.

    # ------------------------------------------------------------------
    # DemoCount
    # ------------------------------------------------------------------
    # integer 0-2, no nulls. low corr with target (0.09). cleaned, kept.
    # we log-compress it in feature engineering.

    # ------------------------------------------------------------------
    # DeveloperCount, PublisherCount
    # ------------------------------------------------------------------
    # integers, no nulls. small positive corr with target (~0.15).
    # leave as-is, log-compress in feature engineering.

    # ------------------------------------------------------------------
    # DLCCount, MovieCount, ScreenshotCount, PackageCount
    # ------------------------------------------------------------------
    # integers, no nulls, right-skewed counts.
    # leave as-is, log-compress in feature engineering.

    # ------------------------------------------------------------------
    # AchievementCount, AchievementHighlightedCount
    # ------------------------------------------------------------------
    # integers, no nulls, right-skewed counts.
    # leave as-is, log-compress in feature engineering.

    # ------------------------------------------------------------------
    # Metacritic
    # ------------------------------------------------------------------
    # integer 0-97, no nulls. BUT 83% of games have score=0 which means
    # "not rated", not an actual score of 0. we handle this with a flag
    # in feature engineering. leave the raw column as-is here.

    # ------------------------------------------------------------------
    # SteamSpy columns (all 4)
    # ------------------------------------------------------------------
    # SteamSpyOwners, SteamSpyOwnersVariance,
    # SteamSpyPlayersEstimate, SteamSpyPlayersVariance
    # all integers, no nulls. extremely right-skewed (max in millions).
    # note: steamspy reports in brackets so these are midpoints, not
    # exact counts. leave raw values as-is, handle in feature engineering.

    # ------------------------------------------------------------------
    # RecommendationCount (target)
    # ------------------------------------------------------------------
    # integer, no nulls. extremely right-skewed (skew=68, 63% are zeros).
    # we apply log1p transform during modeling, not here.

    # ------------------------------------------------------------------
    # PriceCurrency
    # ------------------------------------------------------------------
    # only 2 values: "USD" and empty string. fill empty with "Unknown".
    df["PriceCurrency"] = df["PriceCurrency"].replace("", "Unknown").fillna("Unknown")

    # ------------------------------------------------------------------
    # PriceInitial, PriceFinal
    # ------------------------------------------------------------------
    # floats, no nulls. clip to >= 0 (no negative prices make sense).
    # 1392 games have PriceFinal=0 but IsFree=False - that's a data
    # inconsistency we flag in feature engineering.
    df["PriceInitial"] = df["PriceInitial"].clip(lower=0)
    df["PriceFinal"]   = df["PriceFinal"].clip(lower=0)

    # ------------------------------------------------------------------
    # Boolean columns (all of them)
    # ------------------------------------------------------------------
    # ControllerSupport, IsFree, FreeVerAvail, PurchaseAvail,
    # SubscriptionAvail, PlatformWindows, PlatformLinux, PlatformMac,
    # PCReqsHaveMin, PCReqsHaveRec, LinuxReqsHaveMin, LinuxReqsHaveRec,
    # MacReqsHaveMin, MacReqsHaveRec, CategorySinglePlayer,
    # CategoryMultiplayer, CategoryCoop, CategoryMMO, CategoryInAppPurchase,
    # CategoryIncludeSrcSDK, CategoryIncludeLevelEditor, CategoryVRSupport,
    # GenreIsNonGame, GenreIsIndie, GenreIsAction, GenreIsAdventure,
    # GenreIsCasual, GenreIsStrategy, GenreIsRPG, GenreIsSimulation,
    # GenreIsEarlyAccess, GenreIsFreeToPlay, GenreIsSports, GenreIsRacing,
    # GenreIsMassivelyMultiplayer
    # all already bool dtype with no nulls. convert to int for modeling.
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # ------------------------------------------------------------------
    # SupportEmail
    # ------------------------------------------------------------------
    # string, 2 nulls. fill nulls with empty string.
    # high cardinality (5359 unique) so we only use it as a has/hasn't flag.
    df["SupportEmail"] = df["SupportEmail"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # SupportURL
    # ------------------------------------------------------------------
    # string, 1 null. fill with empty string.
    # used as has/hasn't flag only - actual URLs are not useful for modeling.
    df["SupportURL"] = df["SupportURL"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # Website
    # ------------------------------------------------------------------
    # string, 2713 nulls (24% missing). fill nulls with empty string.
    # used as has/hasn't flag.
    df["Website"] = df["Website"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # AboutText, ShortDescrip, DetailedDescrip
    # ------------------------------------------------------------------
    # strings, no nulls. strip whitespace. used as text length features.
    for col in ["AboutText", "ShortDescrip", "DetailedDescrip"]:
        df[col] = df[col].fillna("").str.strip()

    # ------------------------------------------------------------------
    # PCMinReqsText, PCRecReqsText
    # ------------------------------------------------------------------
    # strings, no nulls. strip whitespace. used as text length features.
    df["PCMinReqsText"] = df["PCMinReqsText"].fillna("").str.strip()
    df["PCRecReqsText"] = df["PCRecReqsText"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # LinuxMinReqsText, LinuxRecReqsText
    # ------------------------------------------------------------------
    # strings, no nulls. strip whitespace. used as has/hasn't flags.
    # not used as length features since Linux reqs are usually copy-paste.
    df["LinuxMinReqsText"] = df["LinuxMinReqsText"].fillna("").str.strip()
    df["LinuxRecReqsText"] = df["LinuxRecReqsText"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # MacMinReqsText, MacRecReqsText
    # ------------------------------------------------------------------
    # same as Linux. strip and keep as flags only.
    df["MacMinReqsText"] = df["MacMinReqsText"].fillna("").str.strip()
    df["MacRecReqsText"] = df["MacRecReqsText"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------
    # string, no nulls (but many are empty/whitespace = no review).
    # used as text length feature and has/hasn't flag.
    df["Reviews"] = df["Reviews"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # SupportedLanguages
    # ------------------------------------------------------------------
    # string, no nulls. space-delimited (NOT comma-delimited like it looks).
    # needs special parsing to count languages - done in feature engineering.
    df["SupportedLanguages"] = df["SupportedLanguages"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # LegalNotice
    # ------------------------------------------------------------------
    # string, 1 null. fill with empty string. used as has/hasn't flag only.
    # actual legal text is not useful for modeling.
    df["LegalNotice"] = df["LegalNotice"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # DRMNotice
    # ------------------------------------------------------------------
    # string, no nulls (but many empty). fill and use as has/hasn't flag.
    # 41 unique values - games with DRM notices tend to be bigger releases.
    df["DRMNotice"] = df["DRMNotice"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # ExtUserAcctNotice
    # ------------------------------------------------------------------
    # string, no nulls (many empty). fill and use as has/hasn't flag.
    # indicates games requiring external accounts (e.g. Ubisoft Connect).
    df["ExtUserAcctNotice"] = df["ExtUserAcctNotice"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    # string - this is just a URL to the game's store page background image.
    # not useful for modeling (just a CDN link, no semantic content).
    # cleaned here (fill null), will be dropped in feature engineering.
    df["Background"] = df["Background"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # HeaderImage
    # ------------------------------------------------------------------
    # string - URL to the header image. same situation as Background.
    # cleaned here (fill null), will be dropped in feature engineering.
    df["HeaderImage"] = df["HeaderImage"].fillna("").str.strip()

    return df


# =============================================================================
# STEP 2 - FEATURE ENGINEERING (runs on cleaned data)
# =============================================================================

def make_date_features(df):
    # release_age_days: older games have had more time to accumulate recs
    # corr=0.36 with log target. QueryID is 0.91 correlated with this but
    # we keep both since QueryID has no missing values.
    ref_date = df["ReleaseDate"].dropna().max()

    return pd.DataFrame({
        "release_year":      df["ReleaseDate"].dt.year,
        "release_month":     df["ReleaseDate"].dt.month,
        "release_dayofweek": df["ReleaseDate"].dt.dayofweek,
        "release_age_days":  (ref_date - df["ReleaseDate"]).dt.days,
    }, index=df.index)


def make_price_features(df):
    # log compress both price columns (heavy tailed)
    # is_free and discount ratio capture pricing strategy
    # price_free_mismatch: 1392 games have PriceFinal=0 but IsFree=False
    initial = df["PriceInitial"]
    final   = df["PriceFinal"]
    ratio   = np.where(initial > 0, (initial - final) / initial, 0.0)

    return pd.DataFrame({
        "log1p_price_final":    np.log1p(final),
        "log1p_price_initial":  np.log1p(initial),
        "price_discount_ratio": np.clip(ratio, 0, 1),
        "is_free_price":        (final == 0).astype(int),
        "is_premium":           (final >= 20).astype(int),
        "price_free_mismatch":  ((final == 0) & (df["IsFree"] == 0)).astype(int),
    }, index=df.index)


def make_steamspy_features(df):
    # steamspy is the strongest predictor group
    #
    # important: steamspy reports ownership in discrete brackets
    # (0-20k, 20k-50k, 50k-100k etc.). raw value is the bracket midpoint,
    # not the exact count. two games with 19k and 49k actual owners get the
    # same number - that's real noise in the data.
    #
    # percentile rank fixes this - converts noisy midpoints to clean 0-1 ranks:
    #   players_rank: 0.786 corr with log target
    #   owners_rank:  0.750 corr with log target
    # both beat the raw log-transformed versions (~0.66)
    #
    # bracket bounds (Owners +/- Variance) give the model the actual range
    # edges explicitly instead of just the midpoint.
    #
    # all 4 raw steamspy columns are kept - variance columns had 0.53-0.55
    # corr with log target and were wrongly dropped in an earlier version.
    owners   = df["SteamSpyOwners"]
    players  = df["SteamSpyPlayersEstimate"]
    own_var  = df["SteamSpyOwnersVariance"]
    play_var = df["SteamSpyPlayersVariance"]

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
    # score=0 means "not rated" for 83% of games, not an actual 0 score
    # has_metacritic flag has corr=0.46 with log target
    # games with a score average 5020 recs vs 435 for unrated
    score = df["Metacritic"]
    return pd.DataFrame({
        "metacritic_score": score,
        "has_metacritic":   (score > 0).astype(int),
    }, index=df.index)


def make_content_features(df):
    # content richness: movies, screenshots, DLC, achievements, packages
    # all log-compressed because of right skew
    # DemoCount has low corr (0.09) but still included and let feature
    # selection decide for Ridge; tree models will naturally ignore it
    return pd.DataFrame({
        "log1p_movie_count":       np.log1p(df["MovieCount"]),
        "log1p_screenshot_count":  np.log1p(df["ScreenshotCount"]),
        "log1p_dlc_count":         np.log1p(df["DLCCount"]),
        "log1p_achievement_count": np.log1p(df["AchievementCount"]),
        "log1p_achievement_hl":    np.log1p(df["AchievementHighlightedCount"]),
        "log1p_package_count":     np.log1p(df["PackageCount"]),
        "log1p_developer_count":   np.log1p(df["DeveloperCount"]),
        "log1p_publisher_count":   np.log1p(df["PublisherCount"]),
        "log1p_demo_count":        np.log1p(df["DemoCount"]),
        "content_score":           np.log1p(df["MovieCount"]) + np.log1p(df["ScreenshotCount"]) + np.log1p(df["DLCCount"]),
    }, index=df.index)


def make_language_features(df):
    # SupportedLanguages is space-delimited (looks comma-separated but isn't)
    # strip the full-audio suffix pattern then count words = language count
    # range 0-29, corr=0.21 with log target
    cleaned = df["SupportedLanguages"].str.replace(
        r"\*[^*]*\*?languages[^*]*", "", regex=True
    )
    lang_count = cleaned.str.split().apply(len)

    return pd.DataFrame({
        "lang_count":       lang_count,
        "log1p_lang_count": np.log1p(lang_count),
        "is_multilingual":  (lang_count > 1).astype(int),
    }, index=df.index)


def make_platform_features(df):
    # linux and mac support correlates with more recs
    # (linux games get 3.5x more recs on average, mac 2.7x vs windows only)
    # probably because cross-platform games tend to be better funded
    return pd.DataFrame({
        "platform_count": df["PlatformWindows"] + df["PlatformLinux"] + df["PlatformMac"],
        "supports_linux": df["PlatformLinux"],
        "supports_mac":   df["PlatformMac"],
    }, index=df.index)


def make_multiplayer_features(df):
    # multiplayer games average 4-5x more recs than single player
    # action + multiplayer especially strong (CS, TF2, Dota etc.)
    is_multi = (
        df["CategoryMultiplayer"] | df["CategoryCoop"] | df["CategoryMMO"]
    ).astype(int)

    lang_count = df["SupportedLanguages"].str.replace(
        r"\*[^*]*\*?languages[^*]*", "", regex=True
    ).str.split().apply(len)

    log_owners = np.log1p(df["SteamSpyOwners"])

    return pd.DataFrame({
        "is_multiplayer":       is_multi,
        "lang_x_multiplayer":   lang_count * is_multi,
        "owners_x_multiplayer": log_owners * is_multi,
        "action_x_multiplayer": df["GenreIsAction"] * is_multi,
    }, index=df.index)


def make_age_features(df):
    # RequiredAge is 0 for 95% of games (no restriction)
    # mature flag (17+) is the useful signal - bigger budget games
    age = df["RequiredAge"]
    return pd.DataFrame({
        "required_age": age,
        "is_mature":    (age >= 17).astype(int),
    }, index=df.index)


def make_text_richness_features(df):
    # longer descriptions = more polished store page = more marketing effort
    # = correlates with game quality. we use log-compressed length as a proxy.
    # for text we won't use as length features (URLs, legal, DRM), we still
    # create a has/hasn't flag since presence alone carries signal.
    out = {}

    # length + presence for main text columns
    for col in ["AboutText", "ShortDescrip", "DetailedDescrip",
                "PCMinReqsText", "PCRecReqsText", "Reviews"]:
        out[f"{col}_len"] = np.log1p(df[col].str.len())
        out[f"{col}_has"] = (df[col] != "").astype(int)

    # presence flags only for secondary text columns
    out["has_linux_min_reqs"]    = (df["LinuxMinReqsText"] != "").astype(int)
    out["has_linux_rec_reqs"]    = (df["LinuxRecReqsText"] != "").astype(int)
    out["has_mac_min_reqs"]      = (df["MacMinReqsText"] != "").astype(int)
    out["has_mac_rec_reqs"]      = (df["MacRecReqsText"] != "").astype(int)
    out["has_support_email"]     = (df["SupportEmail"] != "").astype(int)
    out["has_support_url"]       = (df["SupportURL"] != "").astype(int)
    out["has_website"]           = (df["Website"] != "").astype(int)
    out["has_legal_notice"]      = (df["LegalNotice"] != "").astype(int)
    out["has_drm_notice"]        = (df["DRMNotice"] != "").astype(int)
    out["has_ext_acct_notice"]   = (df["ExtUserAcctNotice"] != "").astype(int)

    # Background and HeaderImage are CDN URLs with no useful content
    # we verify they exist (non-empty) but won't use the URLs themselves
    out["has_background_img"]    = (df["Background"] != "").astype(int)
    out["has_header_img"]        = (df["HeaderImage"] != "").astype(int)

    return pd.DataFrame(out, index=df.index)


def make_interaction_features(df):
    # hand-crafted interactions between the strongest predictors
    # owners_x_metacritic (corr=0.51): lots of owners + critical acclaim
    # var_x_owners: high variance near a steamspy bracket boundary
    log_owners  = np.log1p(df["SteamSpyOwners"])
    log_players = np.log1p(df["SteamSpyPlayersEstimate"])
    log_own_var = np.log1p(df["SteamSpyOwnersVariance"])
    metacritic  = df["Metacritic"]
    has_meta    = (metacritic > 0).astype(float)

    return pd.DataFrame({
        "owners_x_metacritic":     log_owners * metacritic,
        "players_x_metacritic":    log_players * metacritic,
        "owners_x_has_metacritic": log_owners * has_meta,
        "var_x_owners":            log_own_var * log_owners,
    }, index=df.index)


def engineer_features(df):
    """
    Builds model features from the already-cleaned dataframe.
    Note: all columns were already preprocessed in preprocess_all_columns().
    Columns dropped here (with reason):
      - QueryName, ResponseName: ~11300 unique values in 11357 rows,
        encoding would just be noise. QueryID already identifies the game.
      - Background, HeaderImage: just CDN URLs, no semantic content.
        presence flags are included in make_text_richness_features().
      - SupportURL: same as above, presence flag included.
    """
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

    # boolean genre/category columns (already converted to int in preprocessing)
    bool_cols = [c for c in df.select_dtypes(include=[int, "int64"]).columns
                 if c.startswith(("Category", "Genre", "Platform", "PCReqs",
                                  "LinuxReqs", "MacReqs", "IsFree", "FreeVer",
                                  "Purchase", "Subscription", "Controller"))]
    if bool_cols:
        parts.append(df[bool_cols])

    # remaining numeric columns not already handled
    already_handled = {
        "PriceInitial", "PriceFinal", "RequiredAge",
        "SteamSpyOwners", "SteamSpyOwnersVariance",
        "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
        "Metacritic", "MovieCount", "ScreenshotCount", "DLCCount",
        "AchievementCount", "AchievementHighlightedCount",
        "PackageCount", "DeveloperCount", "PublisherCount", "DemoCount",
        TARGET,
    }
    extra_num = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in already_handled]
    if extra_num:
        parts.append(df[extra_num])

    # currency one-hot
    parts.append(pd.get_dummies(df["PriceCurrency"], prefix="currency"))

    features = pd.concat(parts, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features


# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(y_true_log, y_pred_log):
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
        xlabel, ylabel = "log1p(Actual)", "log1p(Predicted)"
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
    df_res = pd.DataFrame(results)
    melted = df_res.melt(id_vars="model", value_vars=["mae", "rmse"],
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
    df_res = pd.DataFrame(results)[["model", "r2_%"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df_res, x="model", y="r2_%", ax=ax, palette="Blues_d")
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
    # out-of-fold predictions for stacking - each row is predicted by a model
    # that never saw it during training, preventing leakage into the meta-learner
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
    # ── Load raw data ─────────────────────────────────────────────────────────
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # ── Step 1: Preprocess ALL columns ────────────────────────────────────────
    # every column gets cleaned here before anything else happens
    print("Preprocessing all columns ...")
    df = preprocess_all_columns(df_raw)
    print(f"  Done. Shape unchanged: {df.shape}")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    X = engineer_features(df)
    y = np.log1p(df[TARGET])
    print(f"Engineered feature matrix: {X.shape}")

    plot_target_dist(df[TARGET], f"{PLOTS_DIR}/target_distribution.png")

    # ── 80/20 train/test split ────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ── Imputation ────────────────────────────────────────────────────────────
    # median imputation - robust to skewed count/price columns
    # fit only on train to avoid any leakage into test
    imputer     = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw),
                               columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw),
                               columns=X_test_raw.columns, index=X_test_raw.index)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ── Feature selection for Ridge only ──────────────────────────────────────
    # tree models handle irrelevant features by just not splitting on them,
    # so SelectKBest only matters for Ridge. extended search to k=80 since
    # k=60 was still the winner at the boundary of the previous run.
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
    # adds L2 regularization to OLS so correlated features don't blow up
    # coefficients. tune alpha with cross-validation.
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
    # lots of trees in parallel on random data/feature subsets, average them
    # uses all features - RF naturally ignores irrelevant ones
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
    # grows trees leaf-wise so it finds better splits faster than level-wise
    # sequential: each tree corrects the previous tree's residuals
    # num_leaves controls complexity, has built-in L1/L2 reg
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
    # meta-model trains on predictions of the base models instead of raw features
    # out-of-fold predictions ensure no leakage: each row gets predicted by a
    # model that never saw it during training. those OOF preds become the
    # training data for the Ridge meta-learner.
    # for test, we retrain base models on all of train then feed to meta Ridge.
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

    # save metrics
    summary = {
        "dataset_shape":               list(df_raw.shape),
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
