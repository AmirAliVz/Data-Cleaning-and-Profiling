"""
main.py

Performs data profiling and cleansing on the employee dataset.

- Includes visualizations for identifying and analyzing outliers (can be turned off).
- Saves the final cleaned dataset to a CSV file.

Author: Amir Vaziri
Date: 05-03-2025
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Module-level constants ─────────────────────────────────────────────────────
RANDOM_SEED    = 42
WEEKS_PER_YEAR = 52
DATA_PATH      = "data/Employee Turnover Dataset.csv"
OUTPUT_PATH    = "data/Cleaned_Employee_Turnover_Dataset.csv"
FIGURES_DIR    = "Figures"


def plot_histogram_boxplot(
    data, filename, kde=True, display=True,
    figsize=(10, 6), fontsize=8, fontcolor="black",
):
    """
    Plot a combined box plot and histogram for a given data Series.

    Parameters
    ----------
    data : pd.Series
        Input column of data.
    filename : str
        Column label used as the plot title and saved filename.
    kde : bool, optional
        Overlay a kernel density estimate on the histogram. Default True.
    display : bool, optional
        Render and save the figure. Default True.
    figsize : tuple, optional
        Figure dimensions (width, height) in inches. Default (10, 6).
    fontsize : int, optional
        Font size for box-plot annotations. Default 8.
    fontcolor : str, optional
        Font colour for box-plot annotations. Default 'black'.

    Returns
    -------
    lower_whisker : float
        Lower IQR bound (Q1 - 1.5 * IQR, floored at the data minimum).
    upper_whisker : float
        Upper IQR bound (Q3 + 1.5 * IQR, capped at the data maximum).

    Notes
    -----
    The figure is saved as a .jpg in the FIGURES_DIR folder.
    """
    q1            = np.percentile(data, 25)
    median        = np.median(data)
    q3            = np.percentile(data, 75)
    iqr           = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        sharex=True,
    )

    cmap = plt.get_cmap("inferno")

    # ── Box plot ──────────────────────────────────────────────────────────────
    sns.boxplot(x=data, ax=ax_box, color="skyblue")

    stats = {
        "Min":    lower_whisker,
        "Q1":     q1,
        "Median": median,
        "Q3":     q3,
        "Max":    upper_whisker,
    }
    for label, val in stats.items():
        ax_box.text(
            val, 0.02,
            f"{label}\n{val:.2f}",
            ha="center", va="bottom",
            fontsize=fontsize, color=fontcolor, rotation=45,
        )
    ax_box.set(xlabel="")

    # ── Histogram ─────────────────────────────────────────────────────────────
    hist_plot = sns.histplot(
        data, bins="auto", kde=kde,
        color="steelblue", edgecolor="black", ax=ax_hist,
    )
    patches = hist_plot.patches
    counts  = [patch.get_height() for patch in patches]

    # Apply inferno colour gradient based on bar height
    for patch, count in zip(patches, counts):
        patch.set_facecolor(cmap(0.3 + 0.7 * count / max(counts)))

    # Annotate each bar with its count
    for patch, count in zip(patches, counts):
        if count > 0:
            ax_hist.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center", va="bottom", fontsize=8,
            )

    ax_hist.set_title(f"Distribution of {filename}", fontweight="bold")
    ax_hist.set_xlabel(filename, fontweight="bold")
    ax_hist.set_ylabel("Count", fontweight="bold")

    plt.tight_layout()

    # Create the output folder if it does not already exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.jpg"), dpi=300)
    plt.show()

    return lower_whisker, upper_whisker


def main(display=False, verbose=False):
    """
    Orchestrate the full data-profiling and cleaning pipeline.

    Steps
    -----
    1. Load the raw employee turnover dataset.
    2. Remove duplicate records keyed on EmployeeNumber.
    3. Impute missing values using column-specific strategies.
    4. Standardise inconsistent categorical entries.
    5. Detect and correct outliers in numerical columns.
    6. Save the cleaned dataset to CSV.

    Parameters
    ----------
    display : bool, optional
        Generate and save all visualisations. Default False.
    verbose : bool, optional
        Print intermediate diagnostics to stdout. Default False.
    """
    plt.rcParams["font.family"] = "Georgia"
    plt.rcParams["font.size"]   = 12

    # Fix the random seed so that the random-sampling imputation step
    # produces identical results on every run
    np.random.seed(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    df       = pd.read_csv(DATA_PATH)
    df_clean = df.copy()

    # ── 1. Duplicate removal ──────────────────────────────────────────────────
    if verbose:
        print("=" * 50)
        print(" Duplicate Entries ")
        print("=" * 50)
        print(f"--> Duplicated rows: {df.duplicated().sum()}")

    # EmployeeNumber is a surrogate key; any duplicate signals a data quality issue
    df_clean = df_clean.drop_duplicates(subset=["EmployeeNumber"], keep="first")

    # ── 2. Missing value imputation ───────────────────────────────────────────
    if verbose:
        print("=" * 50)
        print(" MISSING VALUE SUMMARY ")
        print("=" * 50)
        print(df_clean.isnull().sum())

    # NumCompaniesPreviouslyWorked: impute with the median rather than the mean
    # because the column is right-skewed and the median is more robust to outliers
    column_name = "NumCompaniesPreviouslyWorked"
    plot_histogram_boxplot(df[column_name], column_name + " Before Cleaning", display=display)
    df_clean[column_name] = df_clean[column_name].fillna(df_clean[column_name].median())

    # AnnualProfessionalDevHrs: impute with random sampling from observed values
    # rather than the median, because the original distribution is approximately
    # uniform — median imputation would introduce an artificial spike
    column_name = "AnnualProfessionalDevHrs"
    plot_histogram_boxplot(df[column_name], column_name + " Before Cleaning", display=display)
    df_clean[column_name] = df_clean[column_name].apply(
        lambda x: np.random.choice(df_clean[column_name].dropna()) if pd.isnull(x) else x
    )

    # TextMessageOptIn: preserve missingness as a distinct category rather than
    # imputing, since who did not respond is behaviourally different from Yes/No
    df_clean["TextMessageOptIn"] = df["TextMessageOptIn"].fillna("Unknown")

    # ── 3. Inconsistent formatting ────────────────────────────────────────────
    if verbose:
        print("=" * 50)
        print(" Inconsistent Formatting ")
        print("=" * 50)
        print(f"--> JobRoleArea unique entries:\n{df_clean['JobRoleArea'].unique()}")
        print(f"--> PaycheckMethod unique entries:\n{df_clean['PaycheckMethod'].unique()}")

    # Remove any accidental leading/trailing whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # Strip the dollar sign and convert HourlyRate to a numeric type
    # so it can be used in arithmetic and statistical operations
    df_clean["HourlyRate"] = (
        df_clean["HourlyRate"].str.replace("$", "", regex=False).astype(float)
    )

    # Collapse all spelling variants of each job role into a single canonical value
    job_role_map = {
        "InformationTechnology": "Information_Technology",
        "Information Technology": "Information_Technology",
        "HumanResources":        "Human_Resources",
        "Human Resources":       "Human_Resources",
    }
    df_clean["JobRoleArea"] = df_clean["JobRoleArea"].replace(job_role_map)

    # Collapse all spelling variants of each payment method into a single canonical value
    paycheck_map = {
        "Mail Check":     "Mail_Check",
        "Mailed Check":   "Mail_Check",
        "MailedCheck":    "Mail_Check",
        "DirectDeposit":  "Direct_Deposit",
        "Direct Deposit": "Direct_Deposit",
    }
    df_clean["PaycheckMethod"] = df_clean["PaycheckMethod"].replace(paycheck_map)

    # Convert object-type text columns to the pandas StringDtype for consistency
    df_clean = df_clean.convert_dtypes()

    if verbose:
        print("=" * 50)
        print(" Data Types ")
        print("=" * 50)
        print(df_clean.dtypes)

    # ── 4. Outlier handling ───────────────────────────────────────────────────

    # Visualise the distribution of each numeric column before targeted corrections
    for column_name in df_clean.select_dtypes(include="number").columns[1:]:
        plot_histogram_boxplot(df_clean[column_name], filename=column_name, display=display)

    # AnnualSalary ─────────────────────────────────────────────────────────────
    column_name = "AnnualSalary"

    # Negative salary values are physically impossible and are treated as sign errors
    df_clean[column_name] = df_clean[column_name].abs()

    # Cross-validate the recorded salary against the expected value derived from
    # hourly rate, weekly hours, and weeks per year
    calculated_annual_salary = round(
        df_clean["HourlyRate"] * df_clean["HoursWeekly"] * WEEKS_PER_YEAR, 1
    )

    if display:
        # x = expected salary (derived from the formula)
        # y = recorded salary (as stored in the dataset)
        x = calculated_annual_salary
        y = df_clean[column_name]

        fig, ax = plt.subplots(figsize=(7, 8))

        scatter = ax.scatter(
            x, y,
            alpha=1, c=np.abs(x - y), cmap="coolwarm", s=8, edgecolor="none",
            label="Employee",
        )

        # The diagonal reference line marks where actual salary equals the formula
        ax.plot(
            [min(x), max(x)], [min(x), max(x)],
            linestyle="--", color="red", label="Actual = Calculated", linewidth=1,
        )

        plt.colorbar(scatter, ax=ax, label="Difference (Actual - Calculated)")

        # Annotation explaining how to read the chart; using ASCII '->'
        # because the Georgia font does not include the Unicode arrow glyph
        ax.text(
            0.04, 0.97,
            "- Each point represents one employee\n"
            "- On the line   -> salary matches the formula\n"
            "- Above the line -> actual > formula (e.g. bonus)\n"
            "- Below the line -> actual < formula (e.g. unpaid leave)",
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Calculated Annual Salary", fontweight="bold")
        ax.set_ylabel("Actual Annual Salary", fontweight="bold")
        ax.set_title("Comparison of Actual vs Calculated Annual Salary", fontweight="bold")
        ax.set_aspect("equal", "box")
        ax.legend()

        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "Comparison of Actual vs Calculated Annual Salary.jpg"),
            dpi=300,
        )
        plt.show()

    plot_histogram_boxplot(
        df_clean[column_name], filename=column_name + " Cleaned", display=display
    )

    # DrivingCommuterDistance ─────────────────────────────────────────────────
    column_name = "DrivingCommuterDistance"

    # Negative distances are sign errors — take the absolute value
    df_clean[column_name] = df_clean[column_name].abs()

    # Cap values above the upper IQR bound to suppress extreme outliers
    # without dropping any rows; values above the ceiling are replaced
    # with the upper whisker so they remain the largest in the dataset
    _, upper_bound = plot_histogram_boxplot(df_clean[column_name], column_name, display=False)
    df_clean[column_name] = df_clean[column_name].apply(
        lambda x: upper_bound if x > upper_bound else x
    )

    plot_histogram_boxplot(
        df_clean[column_name], filename=column_name + " Cleaned", display=display
    )

    # ── Save cleaned dataset ──────────────────────────────────────────────────
    df_clean.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main(display=True, verbose=True)
