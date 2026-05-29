# Employee Turnover Data Cleaning & Profiling

This project performs **data profiling and cleaning** on an employee turnover dataset to prepare it for downstream analysis or predictive modeling. The dataset contains 10,199 employee records across 16 variables — including demographic, compensation, and behavioral attributes — and requires several targeted cleaning steps before it can be reliably used.

The pipeline covers:
- Identifying and removing duplicate records
- Handling missing values with column-specific imputation strategies
- Standardizing inconsistent categorical entries
- Detecting and correcting outliers in numerical columns

---

## Dataset Overview

| Property | Detail |
|---|---|
| Rows | 10,199 employee records |
| Columns | 16 variables |
| Mix | Numerical (continuous & discrete) + Categorical (nominal) |
| Target Variable | `Turnover` (Yes / No) |

---

## Cleaning Steps Summary

### 1. Duplicate Removal
Duplicates identified via `EmployeeNumber` — **99 duplicate rows removed**.

### 2. Missing Value Imputation

| Column | Strategy |
|---|---|
| `AnnualProfessionalDevelopmentHours` | Random sampling from observed values (preserves uniform distribution) |
| `NumberOfCompaniesWorked` | Median imputation (robust to right-skew) |
| `TextMessageOptIn` | New `"Unknown"` category (preserves missingness signal) |

### 3. Inconsistent Entries
- **Column names** cleaned of extra whitespace
- **`HourlyRate`** — stripped `$` symbol, converted to `float`
- **`JobRoleArea`** — standardized to 8 canonical values: `Research`, `Information_Technology`, `Sales`, `Human_Resources`, `Laboratory`, `Manufacturing`, `Healthcare`, `Marketing`
- **`PaycheckMethod`** — standardized to: `Direct_Deposit`, `Mail_Check`

### 4. Outlier Handling

| Column | Issue | Strategy |
|---|---|---|
| `AnnualSalary` | Negative values | Absolute value transformation |
| `AnnualSalary` | Discrepancy vs. calculated salary | Retained — likely reflect bonuses/leave |
| `DrivingCommuterDistance` | Negative values | Absolute value transformation |
| `DrivingCommuterDistance` | Extreme values (250–950 mi) | IQR-based capping at Q3 + 1.5×IQR |

---

## Visualizations

The analysis generates the following plots:


### Outlier Detection
| Column | Visualization |
|---|---|
| `Age` | ![Age](Figures/Age.jpg) |
| `Tenure` | ![Tenure](Figures/Age.jpg) |
| `HourlyRate` | ![HourlyRate](Figures/Age.jpg) |
| `AnnualSalary` | ![AnnualSalary](Figures/Age.jpg) |
| `DrivingCommuterDistance` | ![DrivingCommuterDistance](Figures/Age.jpg) |

### Salary Discrepancy Analysis
| Actual vs. Calculated Annual Salary |
|---|
| _scatter plot placeholder_ |

---

## How to Run

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup

Place the dataset file inside the `data/` folder:

```
project/
├── data/
│   └── your_dataset.csv        ← put it here
├── main.py
├── requirements.txt
└── README.md
```

### Run

```bash
python main.py
```

That's it. The script will handle all profiling, cleaning, and visualization steps automatically. Cleaned output will be saved to the `data/` folder and plots will be saved to the `Figures/` folder.

---

## Project Structure

```
project/
├── data/                   # Raw input dataset + Cleaned dataset 
├── Figures/                # Generated plots
├── main.py                 # Entry point — run this
├── requirements.txt
└── README.md
```
