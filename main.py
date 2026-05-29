"""
main.py

Performs data profiling and cleansing on the employee dataset.

- Includes visualizations for identifying and analyzing outliers (can be turned off).
- Saves the final cleaned dataset to a CSV file.


Author: Amirali Vaziribeiraghdar
Date: 05-03-2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_histogram_boxplot(data, filename, kde=True, display=True, figsize=(10, 6), fontsize=8, fontcolor='black'):

    """
    Plots a combined figure of a box plot and a histogram for a specified column of data in a Series.

    Parameters:
        data (pd.Series): The input column of data. Must be provided.
        filename (str): The name of the column to visualize. Must be provided.
        kde (bool, optional): Whether to use a kernel density estimator or not.
        display (bool): Whether or not to display the figure.
        ylog (bool, optional): If True, sets the histogram's y-axis to a logarithmic scale. Defaults to False.
        figsize (tuple, optional): The size of the figure shown. Defaults to (10, 6).
        fontsize (int, optional): Font size for annotations within the box plot. Defaults to 8.
        fontcolor (str, optional): Font color for annotations within the box plot. Defaults to 'black'.



    Returns:
        lower_bound (float): The lower bound used in the box plot for outlier detection.
        upper_bound (float): The upper bound used in the box plot for outlier detection.

    Note:
        The plot is saved as a .jpg image in the 'Figures' folder, named after the column with the specified suffix.
    """

    # Calculate boxplot stats
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    # Setup the figure
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": (0.25, 0.75)}, sharex=True
    )

    cmap = plt.get_cmap('inferno')

    # --- Boxplot
    sns.boxplot(x=data, ax=ax_box, color='skyblue')

    # Annotate stats on boxplot
    stats = {
        'Min': lower_whisker,
        'Q1': q1,
        'Median': median,
        'Q3': q3,
        'Max': upper_whisker
    }

    for label, val in stats.items():
        ax_box.text(val, 0.02, f'{label}\n{val:.2f}',
                    ha='center', va='bottom', fontsize=fontsize, color=fontcolor, rotation=45)

    ax_box.set(xlabel='')

    # --- Histogram
    hist_plot  = sns.histplot(data, bins='auto', kde=kde, color='steelblue', edgecolor='black', ax=ax_hist)

    # Accessing the patches (bars) and bins from the plot
    patches = hist_plot.patches  # These are the bars of the histogram

    # To get the counts, you can use `patches` to calculate the heights (counts)
    n = [patch.get_height() for patch in patches]

    # Gradient fill on histogram bars
    for patch, count in zip(patches, n):
        color = cmap(0.3 + 0.7 * count / max(n))
        patch.set_facecolor(color)

    # Annotate count values on bars
    for patch, count in zip(patches, n):
        if count > 0:
            ax_hist.text(patch.get_x() + patch.get_width() / 2,
                         count,
                         f'{int(count)}',
                         ha='center', va='bottom', fontsize=8)

    # Titles and labels
    ax_hist.set_title(f'Distribution of {filename}', fontweight='bold')
    ax_hist.set_xlabel(filename, fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')

    plt.tight_layout()

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    plt.show()

    return lower_whisker, upper_whisker


def main(display=False, verbose=False):

    # Change global font to 'Arial', size 14
    plt.rcParams['font.family'] = 'Georgia'
    plt.rcParams['font.size'] = 12

    # Read the 'Employee Turnover Dataset' CSV file into a DataFrame
    df = pd.read_csv("Employee Turnover Dataset.csv")
    # Create a copy of the DataFrame for performing all data cleaning operations
    df_clean = df.copy()

    # ============ Handling Duplicates =================
    if verbose:
        print("="*50)
        print(" Duplicate Entries ")
        print("="*50)
        print('--> Number of Duplicated rows:')
        print(df.duplicated().sum())
    # Drop duplicate rows based on the 'EmployeeNumber' column
    df_clean = df_clean.drop_duplicates(subset=['EmployeeNumber'], keep='first')  # keep='first' keeps the first occurrence

    # ============ Handling Missing Values =================
    if verbose:
        print("="*50)
        print(" MISSING VALUE SUMMARY ")
        print("="*50)
        print("--> Number of Missing Values per Column:")
        print(df_clean.isnull().sum())

    # Visualizing the distribution of 'NumCompaniesPreviouslyWorked' before applying data cleaning
    column_name = 'NumCompaniesPreviouslyWorked'
    plot_histogram_boxplot(df[column_name], column_name+' Before Cleaning', display=display)

    # Fill missing values in the 'NumCompaniesPreviouslyWorked' column with the median
    df_clean[column_name] = df_clean[column_name].fillna(
        df_clean[column_name].median())


    # Visualizing the distribution of 'AnnualProfessionalDevHrs' before applying data cleaning
    column_name = 'AnnualProfessionalDevHrs'
    plot_histogram_boxplot(df[column_name], column_name+' Before Cleaning', display=display)

    # Fill missing values in the 'AnnualProfessionalDevHrs' column with random samples from the existing data
    df_clean[column_name] = df_clean[column_name].apply(
        lambda x: np.random.choice(df_clean[column_name].dropna()) if pd.isnull(x) else x)

    # Fill missing values in the 'TextMessageOptIn' column with the category 'Unknown'
    df_clean['TextMessageOptIn'] = df['TextMessageOptIn'].fillna('Unknown')

    # ============ Handling Inconsistent Formatting =================
    if verbose:
        print("="*50)
        print(" Inconsistent Formatting ")
        print("="*50)
        print('--> JobRoleArea unique entries:')
        print(df_clean['JobRoleArea'].unique())
        print("-"*50)
        print('--> PaycheckMethod unique entries')
        print(df_clean['PaycheckMethod'].unique())

    # Remove any extra spaces from the column names
    df_clean.columns = df_clean.columns.str.strip()

    # Remove the dollar sign ('$') from the 'HourlyRate' column and convert the data type to 'float64'
    df_clean['HourlyRate'] = df_clean['HourlyRate'].str.replace('$', '', regex=False).astype(float)

    # Replace the entry 'InformationTechnology' with 'Information_Technology'
    df_clean['JobRoleArea'] = df_clean['JobRoleArea'].replace({'InformationTechnology': 'Information_Technology'})
    # Replace the entry 'Information Technology' to 'Information_Technology'
    df_clean['JobRoleArea'] = df_clean['JobRoleArea'].replace({'Information Technology': 'Information_Technology'})
    # Replace the entry 'HumanResources' to 'Human_Resources'
    df_clean['JobRoleArea'] = df_clean['JobRoleArea'].replace({'HumanResources': 'Human_Resources'})
    # Replace the entry 'Human Resources' to 'Human_Resources'
    df_clean['JobRoleArea'] = df_clean['JobRoleArea'].replace({'Human Resources': 'Human_Resources'})

    # Replace the entry 'Mail Check' to 'Mail_Check'
    df_clean['PaycheckMethod'] = df_clean['PaycheckMethod'].replace({'Mail Check': 'Mail_Check'})
    # Replace the entry 'Mailed Check' to 'Mail_Check'
    df_clean['PaycheckMethod'] = df_clean['PaycheckMethod'].replace({'Mailed Check': 'Mail_Check'})
    # Replace the entry 'MailedCheck' to 'Mail_Check'
    df_clean['PaycheckMethod'] = df_clean['PaycheckMethod'].replace({'MailedCheck': 'Mail_Check'})
    # Replace the entry 'DirectDeposit' to 'Direct_Deposit'
    df_clean['PaycheckMethod'] = df_clean['PaycheckMethod'].replace({'DirectDeposit': 'Direct_Deposit'})
    # Replace the entry 'Direct Deposit' to 'Direct_Deposit'
    df_clean['PaycheckMethod'] = df_clean['PaycheckMethod'].replace({'Direct Deposit': 'Direct_Deposit'})

    # Converting all object-type columns containing text data to the 'string' dtype
    df_clean = df_clean.convert_dtypes()

    if verbose:
        print("="*50)
        print(" Data Types ")
        print("="*50)
        print(df_clean.dtypes)


    # ============ Handling Outliers =================

    # Visualizing the value distributions of selected columns using histograms and boxplots
    # Iterating over all numeric columns
    for columnName in df_clean.select_dtypes(include='number').columns[1:]:
        plot_histogram_boxplot(df_clean[columnName], filename=columnName, display=display)

    #  ======== Handling outliers in the 'AnnualSalary' column ========
    column_name = 'AnnualSalary'
    # Converting all values to their absolute values to handle negative entries
    df_clean[column_name] = df_clean[column_name].abs()
    # Comparing the 'CalculatedAnnualSalary' column with the 'AnnualSalary' column to check for discrepancies
    CalculatedAnnualSalary = round(df_clean['HourlyRate'] * df_clean['HoursWeekly'] * 52, 1)

    if display:
        column_name = 'AnnualSalary'
        x = CalculatedAnnualSalary
        y = df_clean[column_name]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(7, 8))

        # Scatter plot with semi-transparent points
        scatter = ax.scatter(x, y, alpha=1, c=np.abs(x - y), cmap='coolwarm', s=8, edgecolor='none')

        # Add the x=y dashed line for reference
        ax.plot([min(x), max(x)], [min(x), max(x)], linestyle='-', dashes=(5, 10), color='red', label='x = y', linewidth=1)

        # Add color bar to show the difference color map
        plt.colorbar(scatter, ax=ax, label='Difference')

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Set axis labels and title
        ax.set_xlabel('Actual Annual Salary', fontweight='bold')
        ax.set_ylabel('Calculated Annual Salary', fontweight='bold')
        ax.set_title('Comparison of Actual vs Calculated Annual Salary', fontweight='bold')

        # Set the aspect ratio to be equal for better visual comparison
        ax.set_aspect('equal', 'box')

        # Show plot
        plt.tight_layout()
        plt.legend()
        plt.savefig('Figures/' + 'Comparison of Actual vs Calculated Annual Salary' + '.jpg', dpi=300)
        plt.show()

    # Plotting boxplot and histogram of 'AnnualSalary' again after data cleaning
    lower_bound, upper_bound = plot_histogram_boxplot(df_clean[column_name], filename=column_name + ' Cleaned', display=display)

    #  ======== Handling outliers in the 'DrivingCommuterDistance' column ========
    column_name = 'DrivingCommuterDistance'
    # Converting all values to their absolute values to handle negative entries
    df_clean[column_name] = df_clean[column_name].abs()

    lower_bound, upper_bound = plot_histogram_boxplot(df_clean[column_name], column_name, display=False)
    # Capping all values above the upper IQR bound to reduce the effect of extreme outliers
    df_clean[column_name] = df_clean[column_name].apply(
        lambda x: upper_bound if  x > upper_bound else x)


    # Plotting boxplot and histogram of 'DrivingCommuterDistance' again after data cleaning
    plot_histogram_boxplot(df_clean[column_name], filename=column_name + ' Cleaned', display=display)

    # Save the cleaned DataFrame to a CSV file
    df_clean.to_csv("Cleaned_Employee_Turnover_Dataset.csv", index=False)


if __name__ == "__main__":
    main(display=True, verbose=True)












