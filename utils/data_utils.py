import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.stats import mannwhitneyu, pointbiserialr
from scipy.stats.contingency import association
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             recall_score,precision_score, accuracy_score, fbeta_score)
from math import ceil
from sklearn.model_selection import StratifiedKFold


def missing_value_summary(df):
    """
    Returns a summary of missing values for each column in the DataFrame.

    Parameters:
    - df: pd.DataFrame

    Returns:
    - pd.DataFrame with:
        - Number of missing values per column
        - Percentage of missing values relative to total rows
    """
    miss_per_col = df.isnull().sum()
    missing_val_percent = np.round(100 * miss_per_col / len(df), 2)

    missing = pd.concat([miss_per_col, missing_val_percent], axis=1)
    missing.columns = ['Number of Missing Values', 'Percent of Total Values']
    missing = missing.sort_values('Percent of Total Values', ascending=False)

    return missing

def plot_histograms(df, numerical_vars, n_cols=2):
    """
    Plot histograms for multiple numerical variables from a DataFrame.

    This function creates a grid of histograms for the specified numerical variables
    using seaborn and matplotlib. Each subplot shows the distribution of one variable.
    Unused subplots (if any) are removed for cleaner presentation.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    numerical_vars : list of str
        A list of column names in `df` representing the numerical variables to plot.
    n_cols : int, optional (default=2)
        Number of columns in the grid layout of subplots.

    Returns
    -------
    None
        The function displays the histograms but does not return any value.
    """
    n = len(numerical_vars)
    
    fig, axes = plt.subplots(n, n_cols, figsize=(12, 4 * n))
    axes = axes.flatten()
    
    for i, num_var in enumerate(numerical_vars):
        ax = axes[i]
        sns.histplot(df, x=num_var, ax=ax)
        ax.set_title(f'Histogram of {num_var.replace("_", " ")}')
        ax.set_ylabel('Count')
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_density_by_category(df, numerical_vars, target, n_cols=2):
    """
    Plot density plots of numerical variables grouped by a categorical target.

    This function creates a grid of kernel density plots (KDE) for each variable in 
    `numerical_vars`, overlaid by groups defined by the `target` column. Each subplot 
    shows the distribution of the variable by category, with separate densities and 
    filled areas for comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    numerical_vars : list of str
        A list of column names in `df` representing the numerical variables to plot.
    target : str
        The name of the categorical column in `df` used to group the KDEs by hue.
    n_cols : int, optional (default=2)
        Number of columns in the grid layout of subplots.

    Returns
    -------
    None
        The function displays the KDE plots but does not return any value.
    """
    n = len(numerical_vars)
    
    fig, axes = plt.subplots(n, n_cols, figsize=(12, 4 * n))
    axes = axes.flatten()
    
    for i, num_var in enumerate(numerical_vars):
        ax = axes[i]
        sns.kdeplot(df, x=num_var, hue=target, fill=True, common_norm=False, ax=ax)
        ax.set_title(f'{num_var.replace("_", " ")} by {target}')
        ax.set_ylabel('Density')
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_boxplots_by_category(df, numerical_vars, target, n_cols=2):
    """
    Plot horizontal boxplots of numerical variables grouped by a categorical target.

    This function creates a grid of boxplots, one for each variable in `numerical_vars`,
    with the target variable on the y-axis and numerical values on the x-axis. Each
    subplot shows the distribution and potential outliers for each category of the
    target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    numerical_vars : list of str
        A list of numerical column names in `df` to be plotted.
    target : str
        The name of the categorical column in `df` to group the boxplots by.
    n_cols : int, optional (default=2)
        Number of columns in the grid layout of subplots.

    Returns
    -------
    None
        Displays the boxplots but does not return any value.
    """
    n = len(numerical_vars)
    fig, axes = plt.subplots(n, n_cols, figsize=(12, 4 * n))
    axes = axes.flatten()

    for i, num_var in enumerate(numerical_vars):
        sns.set_style("white")
        ax = axes[i]
        sns.boxplot(data=df, x=num_var, y=target, width=0.2, ax=ax)
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def barplots_by_category(df, categorical_vars, target, n_cols=2):
    """
    Plot bar plots of proportions for multiple categorical variables by a target variable.

    This function generates a grid of bar plots that show the proportion of each
    target category within the levels of several categorical variables. It helps in
    visualizing how the distribution of the target variable varies across different
    categories.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    categorical_vars : list of str
        List of categorical column names in `df` to be used as x-axis variables.
    target : str
        The name of the categorical column in `df` to use as the hue (grouping variable).
    n_cols : int, optional (default=2)
        Number of columns in the grid layout of subplots.

    Returns
    -------
    None
        Displays the bar plots but does not return any value.
    """
    n = len(categorical_vars)
    n_rows = (n + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for i, cat_var in enumerate(categorical_vars):
        ax = axes[i]
    
        counts = (
            df
            .groupby([cat_var, target], observed=True)
            .size()
            .to_frame(name='Count')
            .reset_index()
        )
    
        counts['Proportion'] = counts.groupby(cat_var)['Count'].transform(lambda x: x / x.sum())
    
        sns.barplot(data=counts, x=cat_var, y='Proportion', hue=target, ax=ax)
        ax.set_title(f'Proportion of {target} by {cat_var.replace("_", " ")}')
        ax.set_xlabel(cat_var.replace('_', ' '))
        ax.set_ylabel('Proportion')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(False)
        sns.despine(ax=ax)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  
    plt.show()



def loess_diagnostic_grid2(
    df,
    numeric_vars,
    target,
    *,
    frac=0.25,
    logistic_C=1.0,
    n_cols=2,
    scatter_alpha=0.15,
    scatter_size=12,
    figsize_per_plot=(6, 4),
):
    """
    LOWESS–vs–linear-logit diagnostic for multiple numeric predictors.
    Shows two plots per row, automatically expanding as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing predictors and the binary target.
    numeric_vars : list of str
        Column names of numeric predictors to diagnose.
    target : str
        Binary response column (0/1).
    frac : float, optional
        LOWESS smoothing span (default 0.25).
    logistic_C : float, optional
        Inverse regularisation parameter for sklearn LogisticRegression.
    n_cols : int, optional
        Number of plots per row (default 2).
    scatter_alpha : float, optional
        Transparency of the raw scatter points.
    scatter_size : int, optional
        Marker size of the raw scatter points.
    figsize_per_plot : tuple, optional
        Base (width, height) for each subplot; overall figure size
        scales with number of rows.
    """
    n = len(numeric_vars)
    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_plot[0] * n_cols
    fig_h = figsize_per_plot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, x in enumerate(numeric_vars):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        X = df[[x]].values.astype(float)
        y_vec = df[target].values

        # scale predictor for stability
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        
        lo_out = lowess(y_vec, df[x], frac=frac, return_sorted=True)
        x_lo, y_lo = lo_out[:, 0], lo_out[:, 1]

        
        log_reg = LogisticRegression(C=logistic_C, solver="lbfgs")
        log_reg.fit(Xs, y_vec)
        x_grid = np.linspace(df[x].min(), df[x].max(), 300)[:, None]
        y_hat = log_reg.predict_proba(scaler.transform(x_grid))[:, 1]

       
        sns.scatterplot(x=df[x], y=y_vec, alpha=scatter_alpha, s=scatter_size, ax=ax)
        ax.plot(x_lo, y_lo, lw=2, label="LOWESS")
        ax.plot(x_grid.ravel(), y_hat, lw=2, linestyle="--", label="Logit (linear)")
        ax.set_xlabel(x)
        ax.set_ylabel(f"P({target}=1)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Non-linearity check: {x}")
        ax.legend()

    
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.show()

class CholesterolCleaner(BaseEstimator, TransformerMixin):
    """
    Replace Cholesterol==0 with NaN, then impute with fold-specific median.
    Does not change the set or order of columns.
    """
    def __init__(self, col="Cholesterol"):
        self.col = col  # keep params verbatim for sklearn cloning

    def fit(self, X, y=None):
        X = self._to_df(X)
        if self.col not in X:
            raise KeyError(f"{self.col} not found in X columns.")
        x_col = X[self.col].replace(0, np.nan)
        median = x_col.median()
        if pd.isna(median):
            raise ValueError(f"Median of {self.col} is NaN (all zeros/missing in this fold).")
        self.median_cholesterol_ = median
        # remember input feature names for get_feature_names_out
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ["median_cholesterol_", "feature_names_in_"])
        X = self._to_df(X)
        if self.col not in X:
            raise KeyError(f"{self.col} not found in X columns during transform.")
        X = X.copy()
        X[self.col] = X[self.col].replace(0, np.nan).fillna(self.median_cholesterol_)
        return X

    def get_feature_names_out(self, input_features=None):
        """
        If sklearn passes names in, return them; otherwise return the names seen in fit.
        This keeps the pipeline name propagation intact.
        """
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    
    def _to_df(self, X):
        """Ensure a DataFrame, preserving names when available."""
        if isinstance(X, pd.DataFrame):
            return X
        cols = getattr(self, "feature_names_in_", None)
        if cols is None:
            # fallback anonymous names for first call on ndarray
            cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    
class PolynomialFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, degree=2, include_bias=False, interaction_only=False):
        self.feature_names = feature_names
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.poly = PolynomialFeatures(degree=degree, 
                                       include_bias=include_bias, 
                                       interaction_only=interaction_only)
        self.poly_feature_names = []

    def fit(self, X, y=None):
        self.poly.fit(X[self.feature_names])
        self.poly_feature_names = self.poly.get_feature_names_out(self.feature_names)
        return self

    def transform(self, X):
        X_poly = self.poly.transform(X[self.feature_names])
        X_poly_df = pd.DataFrame(X_poly, columns=self.poly_feature_names, index=X.index)

       
        X_remaining = X.drop(columns=self.feature_names)

        
        return pd.concat([X_poly_df, X_remaining], axis=1)

    def get_feature_names_out(self, input_features=None):
        # Return names from fitted polynomial transformer
        return self.poly_feature_names
    



def outlier_summary(df, cols, k=1.5, z_thresh=3.0, round_digits=2):
    """
    Compute outlier diagnostics for numeric columns using both
    Tukey's IQR method and Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing numeric features.
    cols : list of str
        List of column names to analyze.
    k : float, optional (default=1.5)
        Multiplier for IQR to set the outlier fences.
    z_thresh : float, optional (default=3.0)
        Threshold for absolute z-scores to flag outliers.
    round_digits : int, optional (default=2)
        Number of decimals for limits.

    Returns
    -------
    pd.DataFrame
        Table with both IQR and Z-score limits and counts of outliers.
    """
    results = []

    for c in cols:
        s = df[c].dropna()

        # IQR method
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lower_iqr, upper_iqr = q1 - k * iqr, q3 + k * iqr
        n_low_iqr = int((s < lower_iqr).sum())
        n_high_iqr = int((s > upper_iqr).sum())

        # Z-score method
        mean, std = s.mean(), s.std(ddof=0)
        if std == 0:
            lower_z, upper_z = np.nan, np.nan
            n_low_z, n_high_z = 0, 0
        else:
            z_scores = (s - mean) / std
            lower_z, upper_z = mean - z_thresh * std, mean + z_thresh * std
            n_low_z = int((z_scores < -z_thresh).sum())
            n_high_z = int((z_scores > z_thresh).sum())

        results.append({
            "feature": c,

            # IQR results
            "iqr_lower": round(lower_iqr, round_digits),
            "iqr_upper": round(upper_iqr, round_digits),
            "iqr_low_n": n_low_iqr,
            "iqr_high_n": n_high_iqr,

            # Z-score results
            "z_lower": round(lower_z, round_digits) if lower_z is not np.nan else np.nan,
            "z_upper": round(upper_z, round_digits) if upper_z is not np.nan else np.nan,
            "z_low_n": n_low_z,
            "z_high_n": n_high_z,
        })

    return pd.DataFrame(results).set_index("feature")

def numeric_effect_sizes(X, y, num_vars, thr_pb=0.20, thr_rb=0.20):
    """
    Compute point–biserial r (r_pb) and rank–biserial r (r_rb) for numeric features vs binary target.
    Returns (full_table, important_table) where 'important' = |r_pb|>=thr_pb or |r_rb|>=thr_rb.
    """
    y = pd.Series(y).astype(int)
    rows = []
    for c in num_vars:
        x = X[c]
        # point–biserial
        r_pb, _ = pointbiserialr(x, y)

        # rank–biserial from Mann–Whitney U (class1 vs class0)
        x0 = x[y == 0].dropna()
        x1 = x[y == 1].dropna()
        n0, n1 = len(x0), len(x1)
        if n0 and n1:
            U, _ = mannwhitneyu(x1, x0, alternative="two-sided")
            ps = U / (n0 * n1)             # P(X1 > X0)
            r_rb = 2 * ps - 1              # rank-biserial in [-1,1]
        else:
            ps, r_rb = np.nan, np.nan

        rows.append({
            "feature": c,
            "r_pointbiserial": r_pb,
            "rank_biserial": r_rb,
            "P(X1>X0)": ps,
            "median_0": float(np.median(x0)) if n0 else np.nan,
            "median_1": float(np.median(x1)) if n1 else np.nan,
        })

    full = (pd.DataFrame(rows)
              .assign(abs_rpb=lambda d: d["r_pointbiserial"].abs(),
                      abs_rrb=lambda d: d["rank_biserial"].abs())
              .sort_values(["abs_rpb","abs_rrb"], ascending=False)
              .drop(columns=["abs_rpb","abs_rrb"])
              .reset_index(drop=True)
              .round({"r_pointbiserial":3, "rank_biserial":3, "P(X1>X0)":3,
                      "median_0":2, "median_1":2}))

    important = (full[(full["r_pointbiserial"].abs() >= thr_pb) |
                      (full["rank_biserial"].abs() >= thr_rb)]
                 .reset_index(drop=True))

    return full, important





def _proba_from_logit(model, xgrid):
    # model is a sklearn pipeline (scaler + logistic)
    return model.predict_proba(xgrid.reshape(-1, 1))[:, 1]

def loess_diagnostic_grid(
    df,
    numeric_vars,
    target,
    *,
    frac=0.25,                
    logistic_C=1.0,            
    n_cols=2,
    scatter_alpha=0.15,
    scatter_size=12,
    figsize_per_plot=(6, 4),
    cv_splits=5,
    rng=0,
    handle_zeros_for=("Cholesterol",),  
):
    """
    LOWESS-vs-logistic diagnostic for multiple numeric predictors, with a
    per-variable nonlinearity score and optional CV AUC comparison (deg1 vs deg2).

    Returns
    -------
    summary : pd.DataFrame
        Columns: variable, rmse_lowess_vs_logit, auc_deg1, auc_deg2, auc_gain,
                 flagged_nonlinear (bool), n_eff (rows used)
    """
    y = df[target].values.astype(int)
    n = len(numeric_vars)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize_per_plot[0]*n_cols, figsize_per_plot[1]*n_rows),
        squeeze=False
    )

    rng = np.random.default_rng(rng)
    records = []

    for idx, var in enumerate(numeric_vars):
        ax = axes[idx // n_cols, idx % n_cols]

       
        x = df[var].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if var in handle_zeros_for:
            mask &= (x != 0)

        x_use, y_use = x[mask], y[mask]
        n_eff = int(mask.sum())
        if n_eff < 20:
            ax.text(0.5, 0.5, f"{var}\n(insufficient data)", ha="center", va="center")
            ax.axis("off")
            records.append({
                "variable": var, "rmse_lowess_vs_logit": np.nan,
                "auc_deg1": np.nan, "auc_deg2": np.nan, "auc_gain": np.nan,
                "flagged_nonlinear": False, "n_eff": n_eff
            })
            continue

       
        xgrid = np.linspace(np.nanpercentile(x_use, 1), np.nanpercentile(x_use, 99), 200)

        
        lo = lowess(endog=y_use, exog=x_use, frac=frac, return_sorted=True)
        
        lo_grid = np.interp(xgrid, lo[:,0], lo[:,1])

        # Logistic (degree=1)
        logit_deg1 = make_pipeline(StandardScaler(), LogisticRegression(C=logistic_C, solver="lbfgs", max_iter=200))
        logit_deg1.fit(x_use.reshape(-1,1), y_use)
        p_deg1_grid = _proba_from_logit(logit_deg1, xgrid)

        # Nonlinearity score: RMSE between LOWESS and logistic curves
        rmse = float(np.sqrt(np.mean((lo_grid - p_deg1_grid)**2)))

        # change in AUC (degree 1 vs 2)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        auc1, auc2 = [], []
        for tr, te in skf.split(x_use.reshape(-1,1), y_use):
            Xtr, Xte = x_use[tr].reshape(-1,1), x_use[te].reshape(-1,1)
            ytr, yte = y_use[tr], y_use[te]

            # deg 1
            m1 = make_pipeline(StandardScaler(),
                               LogisticRegression(C=logistic_C, solver="lbfgs", max_iter=200))
            m1.fit(Xtr, ytr)
            auc1.append(roc_auc_score(yte, m1.predict_proba(Xte)[:,1]))

            # deg 2
            poly2 = PolynomialFeatures(degree=2, include_bias=False)
            m2 = make_pipeline(poly2, StandardScaler(with_mean=False),
                               LogisticRegression(C=logistic_C, solver="lbfgs", max_iter=200))
            m2.fit(Xtr, ytr)
            auc2.append(roc_auc_score(yte, m2.predict_proba(Xte)[:,1]))

        auc1 = float(np.mean(auc1))
        auc2 = float(np.mean(auc2))
        auc_gain = float(auc2 - auc1)

        #either clear curvature (RMSE) or useful AUC gain
        flagged = (rmse >= 0.03) or (auc_gain >= 0.01)

        xj = x_use + rng.normal(0, 0.01 * np.std(x_use), size=x_use.size)
        yj = y_use + rng.normal(0, 0.03, size=y_use.size)
        ax.scatter(xj, yj, s=scatter_size, alpha=scatter_alpha, edgecolor="none")

        # Curves
        ax.plot(xgrid, lo_grid, linewidth=2, label="LOWESS p̂(y=1|x)")
        ax.plot(xgrid, p_deg1_grid, linewidth=1.5, linestyle="--", label="Logistic (linear)")

        ax.set_title(f"{var}  •  RMSE={rmse:.3f}  •  Change in AUC={auc_gain:.3f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(var)
        ax.set_ylabel("Probability of HeartDisease")
        ax.legend(loc="best", fontsize=8, frameon=False)

        records.append({
            "variable": var,
            "rmse_lowess_vs_logit": rmse,
            "auc_deg1": auc1,
            "auc_deg2": auc2,
            "auc_gain": auc_gain,
            "flagged_nonlinear": flagged,
            "n_eff": n_eff
        })

    total_axes = n_rows * n_cols
    for k in range(n, total_axes):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.tight_layout()
    summary = pd.DataFrame.from_records(records).sort_values("flagged_nonlinear", ascending=False)
    return summary


def barplots_cat_univariate(
    df,
    categorical_vars,
    *,
    n_cols=2,
    decimals=0,
    show_counts=True,
    order="desc",           
    dropna=False,
    figsize_per_plot=(6, 4),
):
    """
    Grid of bar plots showing % distribution for each categorical variable.
    Each bar is annotated with the % (and count, optionally).

    Returns
    -------
    summary : pd.DataFrame
        Columns: variable, category, count, percent
    """
    records = []
    for var in categorical_vars:
        vc = df[var].value_counts(dropna=dropna)
        total = vc.sum()
        for cat, cnt in vc.items():
            pct = cnt / total * 100.0
            records.append({"variable": var, "category": cat, "count": int(cnt), "percent": pct})

    summary = pd.DataFrame(records)

    # sort within each variable
    if order in {"desc", "asc"}:
        ascending = (order == "asc")
        summary["rank"] = summary.groupby("variable")["percent"].rank(method="first", ascending=ascending)
        summary = summary.sort_values(["variable", "rank"]).drop(columns="rank")

    # plot
    n = len(categorical_vars)
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False
    )
    axes = axes.ravel()

    for i, var in enumerate(categorical_vars):
        ax = axes[i]
        dsub = summary[summary["variable"] == var].copy()
        dsub["category"] = dsub["category"].astype(str)
        order_cats = dsub["category"].tolist()

        sns.barplot(data=dsub, x="category", y="percent", order=order_cats, ax=ax)
        ax.set_title(f"{var.replace('_',' ')} — % by category")
        ax.set_xlabel(var.replace("_", " "))
        ax.set_ylabel("Percent")
        ax.set_ylim(0, max(5, dsub["percent"].max() * 1.15))
        ax.tick_params(axis="x", rotation=15)
        sns.despine(ax=ax)

        # annotate bars
        for p, (_, row) in zip(ax.patches, dsub.iterrows()):
            y = p.get_height()
            label = f"{row['percent']:.{decimals}f}%"
            if show_counts:
                label += f"\n(n={row['count']})"
            ax.annotate(
                label,
                (p.get_x() + p.get_width() / 2, y),
                ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                fontsize=9
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return summary

def barplots_by_category_labeled(df, categorical_vars, target, n_cols=2, decimals=0):
    """
    Grouped bar plots: proportion of target categories within each level of the
    categorical variable. Adds % labels on each bar.
    """
    n = len(categorical_vars)
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4.5*n_rows), squeeze=False)
    axes = axes.ravel()

    for i, cat_var in enumerate(categorical_vars):
        ax = axes[i]

        counts = (
            df.groupby([cat_var, target], observed=True)
              .size()
              .to_frame(name='Count')
              .reset_index()
        )
        counts['Proportion'] = counts.groupby(cat_var)['Count'].transform(lambda x: x / x.sum())
        counts['Percent'] = counts['Proportion'] * 100

        cat_order = (df[cat_var].value_counts().index.astype(str).tolist())
        counts[cat_var] = counts[cat_var].astype(str)

        sns.barplot(
            data=counts, x=cat_var, y='Proportion', hue=target,
            order=cat_order, ax=ax
        )

        ax.set_title(f'{target} proportion by {cat_var.replace("_", " ")}')
        ax.set_xlabel(cat_var.replace('_', ' '))
        ax.set_ylabel('Proportion')
        ax.tick_params(axis='x', rotation=15)
        ax.set_ylim(0, min(1.0, counts['Proportion'].max() * 1.15))
        sns.despine(ax=ax)

        # Add % labels to each bar
        #patches are ordered by hue group within each x; we map them back to the counts rows.
        patch_idx = 0
        for _, row in counts.iterrows():
            if patch_idx >= len(ax.patches):
                break
            bar = ax.patches[patch_idx]
            pct = row['Percent']
            ax.annotate(
                f"{pct:.{decimals}f}%",
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                fontsize=9
            )
            patch_idx += 1

        ax.legend(title=target, fontsize=9, frameon=False)

    for k in range(i + 1, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()

def cramers_v_table_scipy(df, categorical_vars, target, correction=True):
    rows = []
    for var in categorical_vars:
        ct = pd.crosstab(df[var], df[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            rows.append({"variable": var, "cramers_v": None})
            continue
        v = association(ct, method="cramer", correction=correction)
        rows.append({"variable": var, "levels": ct.shape[0], "cramers_v": round(v, 3)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)

class OldpeakBinner(BaseEstimator, TransformerMixin):
    def __init__(self, col="Oldpeak", new_col="Oldpeak_bin",
                 bins=(-np.inf, 0, 1, 2, 3, np.inf),
                 labels=("Negative/Zero","(0,1]","(1,2]","(2,3]","(3,+)")):
        self.col, self.new_col, self.bins, self.labels = col, new_col, bins, labels

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.copy()
        X[self.new_col] = pd.cut(X[self.col], bins=self.bins, labels=self.labels, include_lowest=True, ordered=True)
        return X

    def get_feature_names_out(self, input_features=None):
        base = np.asarray(input_features, dtype=object) if input_features is not None else self.feature_names_in_
        return np.concatenate([base, np.array([self.new_col], dtype=object)])


class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = tuple(columns)   # store as immutable, no mutation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        cols = list(self.columns)
        idx = getattr(X, "index", None)
        return pd.DataFrame(X, columns=cols, index=idx)

def plot_perm_importance_box(result, X_cols, title="Permutation Importances", xlabel="Decrease in F2 score"):
    """
    result: sklearn.inspection.permutation_importance(...) output
    X_cols: sequence of column names (aligned with the X passed to permutation_importance)
    """
    # sort features by mean importance (ascending so most important at bottom/top of plot)
    sorted_idx = result.importances_mean.argsort()
    cols_sorted = np.array(X_cols)[sorted_idx]

    importances_df = pd.DataFrame(result.importances[sorted_idx].T, columns=cols_sorted)

    ax = importances_df.plot.box(vert=False, whis=10, figsize=(8, 6))
    ax.set_title(title)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def summarize_tuned(name, tuned_clf, X, y):
    y_pred = tuned_clf.predict(X)  
    return {
        "model": name,
        "threshold": round(float(tuned_clf.best_threshold_), 3),
        "recall": recall_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "accuracy": accuracy_score(y, y_pred),
        "F2": fbeta_score(y, y_pred, beta=2, zero_division=0),
    }
def plot_confusion_matrices_grid(
    models_tuned,
    X,
    y,
    *,
    labels=(0, 1),
    figsize=(14, 6),
    cmap="Blues",
    title_counts_suffix="Counts",
    title_norm_suffix="Row-normalized",
    fontsize=9,
    vspace=0.35,
    hspace=0.15,
    show=True,         
):
    """
    Plot confusion matrices for multiple tuned classifiers in a compact 2xN grid.

    Top row  : raw count confusion matrices
    Bottom row: row-normalized confusion matrices (each row sums to 1.0)

    Parameters
    ----------
    models_tuned : list[tuple[str, estimator]]
        Sequence of (name, tuned_estimator) where tuned_estimator.predict(X)
        uses the learned threshold.
    X, y : array-like / pandas objects
        Validation (or evaluation) features and labels.
    labels : sequence, default=(0, 1)
        Order of class labels for the confusion matrices.
    figsize : tuple, default=(14, 6)
        Figure size passed to matplotlib.
    cmap : str, default="Blues"
        Colormap for the heatmaps.
    title_counts_suffix : str, default="Counts"
        Suffix appended to each model title for the counts row.
    title_norm_suffix : str, default="Row-normalized"
        Suffix appended to each model title for the normalized row.
    fontsize : int, default=9
        Font size for subplot titles and axis labels.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 2xN grid of confusion matrices.
    """
    n_models = len(models_tuned)
    if n_models == 0:
        raise ValueError("models_tuned is empty.")

    # confusion matrices
    cms_counts, cms_norm = [], []
    for _, clf in models_tuned:
        y_pred = clf.predict(X)
        cms_counts.append(confusion_matrix(y, y_pred, labels=labels))
        cms_norm.append(confusion_matrix(y, y_pred, labels=labels, normalize="true"))

    fig, axes = plt.subplots(
        nrows=2, ncols=n_models, figsize=figsize, constrained_layout=True,
    )
    try:
        fig.set_constrained_layout_pads(w_pad=hspace, h_pad=vspace)
    except Exception:
        pass

    if n_models == 1:
        axes = axes.reshape(2, 1)

    for i, (name, _) in enumerate(models_tuned):
        # Counts 
        disp_c = ConfusionMatrixDisplay(confusion_matrix=cms_counts[i], display_labels=labels)
        disp_c.plot(ax=axes[0, i], values_format="d", cmap=cmap, colorbar=False)
        axes[0, i].set_title(f"{name}\n{title_counts_suffix}", fontsize=fontsize)
        axes[0, i].set_xlabel(""); axes[0, i].set_ylabel("")

        # Normalized
        disp_n = ConfusionMatrixDisplay(confusion_matrix=cms_norm[i], display_labels=labels)
        disp_n.plot(ax=axes[1, i], values_format=".2f", cmap=cmap, colorbar=False)
        axes[1, i].set_title(f"{name}\n{title_norm_suffix}", fontsize=fontsize)
        axes[1, i].set_xlabel(""); axes[1, i].set_ylabel("")

    
    axes[0, 0].set_ylabel("Actual", fontsize=fontsize)
    axes[1, 0].set_ylabel("Actual", fontsize=fontsize)
    for ax in axes[1, :]:
        ax.set_xlabel("Predicted", fontsize=fontsize)

    if show:
        plt.show()
        return None
    else:
        plt.close(fig)
        return fig


def plot_confusion_matrices_grid(
    models_tuned,
    X,
    y,
    *,
    labels=(0, 1),
    figsize=(14, 6),
    cmap="Blues",
    title_counts_suffix="Counts",
    title_norm_suffix="Row-normalized",
    fontsize=9,
    vspace=0.35,
    hspace=0.15,
    show=True,          
):
    n_models = len(models_tuned)
    if n_models == 0:
        raise ValueError("models_tuned is empty.")

    # confusion matrices
    cms_counts, cms_norm = [], []
    for _, clf in models_tuned:
        y_pred = clf.predict(X)
        cms_counts.append(confusion_matrix(y, y_pred, labels=labels))
        cms_norm.append(confusion_matrix(y, y_pred, labels=labels, normalize="true"))

    
    fig, axes = plt.subplots(
        nrows=2, ncols=n_models, figsize=figsize, constrained_layout=True,
    )
    try:
        fig.set_constrained_layout_pads(w_pad=hspace, h_pad=vspace)
    except Exception:
        pass

    if n_models == 1:
        axes = axes.reshape(2, 1)

    for i, (name, _) in enumerate(models_tuned):
        # Counts (top)
        disp_c = ConfusionMatrixDisplay(confusion_matrix=cms_counts[i], display_labels=labels)
        disp_c.plot(ax=axes[0, i], values_format="d", cmap=cmap, colorbar=False)
        axes[0, i].set_title(f"{name}\n{title_counts_suffix}", fontsize=fontsize)
        axes[0, i].set_xlabel(""); axes[0, i].set_ylabel("")

        # Normalized (bottom)
        disp_n = ConfusionMatrixDisplay(confusion_matrix=cms_norm[i], display_labels=labels)
        disp_n.plot(ax=axes[1, i], values_format=".2f", cmap=cmap, colorbar=False)
        axes[1, i].set_title(f"{name}\n{title_norm_suffix}", fontsize=fontsize)
        axes[1, i].set_xlabel(""); axes[1, i].set_ylabel("")

   
    axes[0, 0].set_ylabel("Actual", fontsize=fontsize)
    axes[1, 0].set_ylabel("Actual", fontsize=fontsize)
    for ax in axes[1, :]:
        ax.set_xlabel("Predicted", fontsize=fontsize)

    if show:
        plt.show()
        return None
    else:
        
        plt.close(fig)
        return fig
class ThresholdWrapper:
    def __init__(self, base_estimator, threshold):
        self.base_estimator = base_estimator
        self.best_threshold_ = float(threshold)
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.best_threshold_).astype(int)

def show_dist(s):
    vc = s.value_counts().sort_index()
    pct = (vc / vc.sum() * 100).round(2)
    return pd.DataFrame({"count": vc, "percent": pct})

