# Capstone Project - CAP44
# Team Members: CHING-YUAN PENG (cp4516), WEI-CHENG HSU (wh2757), and CHENG-JUI YANG (cy2941)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
MY_SEED = 18358526
random.seed(MY_SEED)
np.random.seed(MY_SEED)
rng = np.random.default_rng(MY_SEED)

print(f"Random Seed set to: {MY_SEED}")

# ==========================================
# Data Loading and Preparation
# ==========================================

# Read the CSV file
num = pd.read_csv("rmpCapstoneNum.csv", header=None)
num.columns = ['Avg_Rating', 'Avg_Difficulty', 'NumofRatings', 'Pepper', 'Again?', 'From online', 'Male', 'Female']

# Question 1 Data Preparation
cols_num = ['AvgRating', 'AvgDifficulty', 'NumRatings', 'Pepper', 'ProportionRetake', 'OnlineRatings', 'Male', 'Female']
cols_tags = [
    "Tough grader", "Good feedback", "Respected", "Lots to read", "Participation matters",
    "Don't skip class", "Lots of homework", "Inspirational", "Pop quizzes!", "Accessible",
    "So many papers", "Clear grading", "Hilarious", "Test heavy", "Graded by few things",
    "Amazing lectures", "Caring", "Extra credit", "Group projects", "Lecture heavy"
]

# Load CSV files
df_num = pd.read_csv('rmpCapstoneNum.csv', header=None, names=cols_num)
df_tags = pd.read_csv('rmpCapstoneTags.csv', header=None, names=cols_tags)

# Merge datasets
df = pd.concat([df_num, df_tags], axis=1)

# Data Cleaning
df = df.dropna(subset=['AvgRating', 'NumRatings', 'Male', 'Female'])
df = df[(df['Male'] == 1) | (df['Female'] == 1)]

# [Key Filter] Only keep professors with >= 5 ratings
df_clean = df[df['NumRatings'] >= 5].copy()

# Create a single Gender column
df_clean['Gender'] = df_clean.apply(lambda x: 'Male' if x['Male'] == 1 else 'Female', axis=1)

# Separate datasets
male_ratings = df_clean[df_clean['Gender'] == 'Male']['AvgRating']
female_ratings = df_clean[df_clean['Gender'] == 'Female']['AvgRating']

# ==========================================
# Question 1: Gender Rating Comparison
# ==========================================

print("=" * 60)
print("QUESTION 1: Gender Rating Comparison")
print("=" * 60)

# Perform statistical test: Mann-Whitney U test
u_stat, p_val_q1 = stats.mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')
median_male = np.median(male_ratings)
median_female = np.median(female_ratings)

print(f"--- Q1 Results ---")
print(f"Male Median: {median_male}, Female Median: {median_female}")
print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"P-value: {p_val_q1:.5e}")

# Visualization: KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_clean, x='AvgRating', hue='Gender', fill=True, common_norm=False, palette=['blue', 'red'], alpha=0.3)
plt.axvline(median_male, color='blue', linestyle='--', label=f'Male Median: {median_male}')
plt.axvline(median_female, color='red', linestyle='--', label=f'Female Median: {median_female}')

plt.title('Distribution of Average Ratings by Gender (Q1 Analysis)', fontsize=14)
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.legend()
plt.show()

# ==========================================
# Question 2: Distribution Comparison
# ==========================================

print("=" * 60)
print("QUESTION 2: Distribution Comparison")
print("=" * 60)

# Perform KS Test
ks_stat, p_val_q2 = stats.ks_2samp(male_ratings, female_ratings)
std_male = np.std(male_ratings)
std_female = np.std(female_ratings)

print(f"--- Q2 Results ---")
print(f"KS Statistic: {ks_stat:.5f}")
print(f"P-value: {p_val_q2:.5e}")
print(f"Male Std Dev: {std_male:.4f}, Female Std Dev: {std_female:.4f}")

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df_clean, x='AvgRating', hue='Gender', palette=['blue', 'red'], linewidth=2)
plt.title(f'Cumulative Distribution of Ratings (KS Test D={ks_stat:.3f})', fontsize=14)
plt.xlabel('Average Rating')
plt.ylabel('Cumulative Probability')
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# Question 3: Bootstrap Analysis
# ==========================================

print("=" * 60)
print("QUESTION 3: Bootstrap Analysis")
print("=" * 60)

# Bootstrap method
n_boot = 10000
boot_median_diffs = []
boot_std_diffs = []

# Convert to numpy array for better performance
m_data = male_ratings.values
f_data = female_ratings.values

for _ in range(n_boot):
    # Resample with replacement
    m_sample = np.random.choice(m_data, size=len(m_data), replace=True)
    f_sample = np.random.choice(f_data, size=len(f_data), replace=True)
    
    # Calculate the difference in statistics
    boot_median_diffs.append(np.median(m_sample) - np.median(f_sample))
    boot_std_diffs.append(np.std(m_sample) - np.std(f_sample))

# Calculate 95% confidence interval
ci_median = np.percentile(boot_median_diffs, [2.5, 97.5])
ci_std = np.percentile(boot_std_diffs, [2.5, 97.5])

print(f"--- Q3 Results ---")
print(f"95% CI for Median Diff (Rating Bias): [{ci_median[0]:.4f}, {ci_median[1]:.4f}]")
print(f"95% CI for Std Dev Diff (Spread Bias): [{ci_std[0]:.4f}, {ci_std[1]:.4f}]")

# Visualization: Bootstrapped Differences Histogram
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Median Difference
sns.histplot(boot_median_diffs, kde=True, ax=ax[0], color='purple', alpha=0.5)
ax[0].axvline(ci_median[0], color='red', linestyle='--')
ax[0].axvline(ci_median[1], color='red', linestyle='--')
ax[0].set_title('Bootstrap Dist. of Median Difference (Male - Female)')
ax[0].set_xlabel('Difference in Median Rating')

# Plot 2: Std Dev Difference
sns.histplot(boot_std_diffs, kde=True, ax=ax[1], color='green', alpha=0.5)
ax[1].axvline(ci_std[0], color='red', linestyle='--')
ax[1].axvline(ci_std[1], color='red', linestyle='--')
ax[1].set_title('Bootstrap Dist. of Std Dev Difference (Male - Female)')
ax[1].set_xlabel('Difference in Standard Deviation')

plt.tight_layout()
plt.show()

# ==========================================
# Question 4: Chi-Square for Tags
# ==========================================

print("=" * 60)
print("QUESTION 4: Chi-Square for Tags")
print("=" * 60)

# Calculate total population (for normalization)
total_ratings_male = df_clean[df_clean['Gender'] == 'Male']['NumRatings'].sum()
total_ratings_female = df_clean[df_clean['Gender'] == 'Female']['NumRatings'].sum()

results = []

# Loop through each tag to perform the test
for tag in cols_tags:
    # Count how many times each gender received this tag
    male_count = df_clean[df_clean['Gender'] == 'Male'][tag].sum()
    female_count = df_clean[df_clean['Gender'] == 'Female'][tag].sum()
    
    # Create contingency table
    obs = np.array([
        [male_count, total_ratings_male - male_count],
        [female_count, total_ratings_female - female_count]
    ])
    
    # Perform Chi-Square test
    chi2, p_val, dof, expected = stats.chi2_contingency(obs)
    
    # Calculate frequency difference
    m_rate = male_count / total_ratings_male
    f_rate = female_count / total_ratings_female
    diff = m_rate - f_rate
    
    results.append({
        'Tag': tag,
        'P-value': p_val,
        'Diff': diff,
        'Bias': 'Male' if diff > 0 else 'Female'
    })

# Convert to DataFrame and sort by P-value
df_res = pd.DataFrame(results).sort_values(by='P-value')

print("\nTop 3 Most Gendered (Lowest P-value):")
print(df_res.head(3)[['Tag', 'P-value', 'Bias']])

print("\nTop 3 Least Gendered (Highest P-value):")
print(df_res.tail(3)[['Tag', 'P-value', 'Bias']])

# ==========================================
# Question 5: Gender vs Difficulty
# ==========================================

print("=" * 60)
print("QUESTION 5: Gender vs Difficulty")
print("=" * 60)

# Combine two gender columns into one for the original num dataframe
num['Gender'] = num['Male'] + num['Female'] * -1
num['Gender'] = num['Gender'].astype(int)

# Replace invalid values with NaN
num['Gender'] = num['Gender'].replace(0, pd.NA)
num['Gender'] = num['Gender'].replace(2, pd.NA)

# Drop null values
num = num.dropna(subset=['Avg_Difficulty', 'Gender'])

# Test with different rating thresholds
thresholds = [1, 2, 3, 6, 10]
for threshold in thresholds:
    male_difficulty = num.loc[(num['Gender'] == 1) & (num['NumofRatings'] >= threshold), 'Avg_Difficulty']
    female_difficulty = num.loc[(num['Gender'] == -1) & (num['NumofRatings'] >= threshold), 'Avg_Difficulty']
    
    stat, p = mannwhitneyu(male_difficulty, female_difficulty, alternative='two-sided')
    print(f"Using data with at least {threshold} ratings:")
    print(f"Mann-Whitney U test statistic: {stat}, p-value: {p}")

# Visualization: KDE Plot
male_median = np.median(num[num['Gender'] == 1]['Avg_Difficulty'])
female_median = np.median(num[num['Gender'] == -1]['Avg_Difficulty'])

plt.figure(figsize=(10, 6))
sns.kdeplot(data=num, x='Avg_Difficulty', hue='Gender', fill=True, common_norm=False, palette=['blue', 'red'], alpha=0.3)
plt.axvline(male_median, color='blue', linestyle='--', label=f'Male Median: {male_median}')
plt.axvline(female_median, color='red', linestyle='--', label=f'Female Median: {female_median}')

plt.title('Distribution of Average Difficulty by Gender (Q5 Analysis for all data)', fontsize=14)
plt.xlabel('Average Difficulty')
plt.ylabel('Density')
plt.legend()
plt.show()

# ==========================================
# Question 6: Bootstrap Effect Size
# ==========================================

print("=" * 60)
print("QUESTION 6: Bootstrap Effect Size")
print("=" * 60)

def bootstrap_effect_size(num, threshold):
    # Prepare data for Bootstrap
    male_data = num.loc[(num['Gender'] == 1) & (num['NumofRatings'] >= threshold), 'Avg_Difficulty'].values
    female_data = num.loc[(num['Gender'] == -1) & (num['NumofRatings'] >= threshold), 'Avg_Difficulty'].values

    # Calculate the observed effect size
    actual_diff = np.mean(male_data) - np.mean(female_data)

    # Bootstrap process
    n_boot = 10000
    boot_diffs = []

    for _ in range(n_boot):
        # Resample with replacement
        male_sample = rng.choice(male_data, size=len(male_data), replace=True)
        female_sample = rng.choice(female_data, size=len(female_data), replace=True)

        # Calculate the difference for this bootstrap sample
        diff = np.mean(male_sample) - np.mean(female_sample)
        boot_diffs.append(diff)

    # Convert to array
    boot_diffs = np.array(boot_diffs)

    # Calculate 95% confidence interval
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    print(f"Effect Size: {actual_diff:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

for threshold in [1, 2, 3, 6, 10]:
    print(f"\nBootstrap Effect Size for NumofRatings >= {threshold}:")
    bootstrap_effect_size(num, threshold)

# Detailed bootstrap for threshold 1 with visualization
male_data = num.loc[(num['Gender'] == 1) & (num['NumofRatings'] >= 1), 'Avg_Difficulty'].values
female_data = num.loc[(num['Gender'] == -1) & (num['NumofRatings'] >= 1), 'Avg_Difficulty'].values

actual_diff = np.mean(male_data) - np.mean(female_data)

n_boot = 10000
boot_diffs = []

for _ in range(n_boot):
    male_sample = rng.choice(male_data, size=len(male_data), replace=True)
    female_sample = rng.choice(female_data, size=len(female_data), replace=True)
    diff = np.mean(male_sample) - np.mean(female_sample)
    boot_diffs.append(diff)

boot_diffs = np.array(boot_diffs)
ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)

print(f"\nDetailed Bootstrap (all data):")
print(f"Effect Size: {actual_diff:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(boot_diffs, kde=True, color='purple', alpha=0.5)
plt.axvline(ci_lower, color='red', linestyle='--', label='95% CI Lower Bound')
plt.axvline(ci_upper, color='red', linestyle='--', label='95% CI Upper Bound')
plt.title(f'Bootstrap Distribution of Mean Difference (NumofRatings >= 1)')
plt.xlabel('Difference in Mean Difficulty (Male - Female)')
plt.legend()
plt.show()

# ==========================================
# Question 7: Numerical Predictors Regression
# ==========================================

print("=" * 60)
print("QUESTION 7: Numerical Predictors Regression")
print("=" * 60)

# Check for multicollinearity
features_for_collinearity = num.drop(columns=['Avg_Rating'])
feature_corr_matrix = features_for_collinearity.corr()

# Find highly correlated feature pairs
high_corr_pairs = []
for i in range(len(feature_corr_matrix.columns)):
    for j in range(i+1, len(feature_corr_matrix.columns)):
        if abs(feature_corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature 1': feature_corr_matrix.columns[i],
                'Feature 2': feature_corr_matrix.columns[j],
                'Correlation': feature_corr_matrix.iloc[i, j]
            })

print("\nMulticollinearity Analysis:")
print("=" * 60)
print(f"Found {len(high_corr_pairs)} feature pairs with |r| > 0.8")

if len(high_corr_pairs) > 0:
    print("\nTop 10 most correlated feature pairs:")
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df.head(10))

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(feature_corr_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Multicollinearity Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Build regression model (exclude Male and Female due to collinearity)
predictors = ['Avg_Difficulty', 'NumofRatings', 'Pepper', 'Again?', 'From online', 'Gender']
target = 'Avg_Rating'

regression_data = num[predictors + [target]].dropna()

print(f"Original: {len(num)}")
print(f"After removing null values: {len(regression_data)}")

X = regression_data[predictors]
y = regression_data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=MY_SEED)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transform back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=predictors, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=predictors, index=X_test.index)

# Add constant term
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

# Fit regression model
model = sm.OLS(y_train, X_train_scaled).fit()

print(model.summary())

# Calculate RMSE
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")
print("Model R-squared on training set:", model.rsquared)

r2_score_test = r2_score(y_test, y_pred)
print("Model R-squared on test set:", r2_score_test)

# Find strongest predictor
params = model.params.drop('const')
max_impact_feature = params.abs().idxmax()
max_impact_value = params[max_impact_feature]

print(f"The feature with the highest impact on Avg_Rating is '{max_impact_feature}' with a coefficient of {max_impact_value:.4f}.")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, y_pred, alpha=0.3, color='blue', label='Data Points')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_title('Actual vs. Predicted Ratings', fontsize=15)
axes[0].set_xlabel('Actual Avg_Rating', fontsize=12)
axes[0].set_ylabel('Predicted Avg_Rating', fontsize=12)
axes[0].legend()

# Plot 2: Strongest predictor relationship
strongest_factor = 'Again?'
plot_sample = regression_data.sample(n=min(2000, len(regression_data)), random_state=42)

sns.regplot(
    x=strongest_factor, 
    y='Avg_Rating', 
    data=plot_sample, 
    ax=axes[1], 
    scatter_kws={'alpha':0.3, 'color':'green'}, 
    line_kws={'color':'red'} 
)

axes[1].set_title(f'Relationship: {strongest_factor} vs. Avg_Rating', fontsize=15)
axes[1].set_xlabel(f'{strongest_factor} (Original Scale)', fontsize=12)
axes[1].set_ylabel('Avg_Rating', fontsize=12)

plt.tight_layout()
plt.show()

# ==========================================
# Bonus: NY vs Other States
# ==========================================

print("=" * 60)
print("BONUS: NY vs Other States")
print("=" * 60)

# Read qualitative data
num_bonus = pd.read_csv("rmpCapstoneNum.csv", header=None)
qual = pd.read_csv("rmpCapstoneQual.csv", header=None)

num_bonus.columns = ['Avg_Rating', 'Avg_Difficulty', 'NumofRatings', 'Pepper', 'Again?', 'From online', 'Male', 'Female']
qual.columns = ['Field', 'University', 'US State']

print("Qualitative data sample:")
print(qual.head())
print(f"\nNY professors count:")
print(qual[qual['US State'] == 'NY'].shape[0])

# Merge data
merged_data = num_bonus.merge(qual, left_index=True, right_index=True)
print(f"\nMerged data sample:")
print(merged_data.head())

merged_data = merged_data.dropna(subset=['Avg_Rating', 'US State'])

print(f"\nTotal professors after cleaning: {len(merged_data)}")
print(f"NY professors: {len(merged_data[merged_data['US State'] == 'NY'])}")
print(f"Other states professors: {len(merged_data[merged_data['US State'] != 'NY'])}")

# Test with different thresholds
thresholds = [1, 2, 3, 6, 10]
bonus_results = []

for threshold in thresholds:
    ny_ratings = merged_data.loc[(merged_data['US State'] == 'NY') & (merged_data['NumofRatings'] >= threshold), 'Avg_Rating']
    other_ratings = merged_data.loc[(merged_data['US State'] != 'NY') & (merged_data['NumofRatings'] >= threshold), 'Avg_Rating']
    
    if len(ny_ratings) > 0 and len(other_ratings) > 0:
        stat, p = mannwhitneyu(ny_ratings, other_ratings, alternative='two-sided')
        ny_median = np.median(ny_ratings)
        other_median = np.median(other_ratings)
        
        print(f"\nUsing data with at least {threshold} ratings:")
        print(f"NY sample size: {len(ny_ratings)}, Other states sample size: {len(other_ratings)}")
        print(f"NY median: {ny_median:.3f}, Other states median: {other_median:.3f}")
        print(f"Mann-Whitney U test statistic: {stat}, p-value: {p:.5e}")
        
        bonus_results.append({
            'threshold': threshold,
            'p_value': p,
            'ny_median': ny_median,
            'other_median': other_median,
            'ny_n': len(ny_ratings),
            'other_n': len(other_ratings)
        })
    else:
        print(f"\nInsufficient data for threshold {threshold}")

# Store the most comprehensive result (threshold=1) for final summary
bonus_main_result = bonus_results[0] if bonus_results else None

# Visualization
ny_median = np.median(merged_data.loc[merged_data['US State'] == 'NY', 'Avg_Rating'])
other_median = np.median(merged_data.loc[merged_data['US State'] != 'NY', 'Avg_Rating'])

plt.figure(figsize=(10, 6))
# Create binary state column for visualization
merged_data['State_Group'] = merged_data['US State'].apply(lambda x: 'NY' if x == 'NY' else 'Other States')
sns.kdeplot(data=merged_data, x='Avg_Rating', hue='State_Group', fill=True, common_norm=False, palette=['blue', 'red'], alpha=0.3)
plt.axvline(ny_median, color='blue', linestyle='--', label=f'NY Median: {ny_median:.3f}')
plt.axvline(other_median, color='red', linestyle='--', label=f'Other States Median: {other_median:.3f}')

plt.title('Distribution of Average Rating by US State (Bonus Analysis for all data)', fontsize=14)
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"\n--- Bonus Summary ---")
print(f"NY vs Other States median difference: {ny_median - other_median:.4f}")
if bonus_main_result:
    print(f"Statistical significance (all data): p = {bonus_main_result['p_value']:.5e}")
    print(f"Significant difference: {'Yes' if bonus_main_result['p_value'] < 0.05 else 'No'}")

# ==========================================
# Question 8: Tags Regression (Rating)
# ==========================================

print("=" * 60)
print("QUESTION 8: Tags Regression (Rating)")
print("=" * 60)

# Load tags data
tags = pd.read_csv("rmpCapstoneTags.csv", header=None)

tag_names = [
    'Tough_grader', 'Good_feedback', 'Respected', 'Lots_to_read', 'Participation_matters',
    'Dont_skip_class', 'Lots_of_homework', 'Inspirational', 'Pop_quizzes', 'Accessible',
    'So_many_papers', 'Clear_grading', 'Hilarious', 'Test_heavy', 'Graded_by_few_things',
    'Amazing_lectures', 'Caring', 'Extra_credit', 'Group_projects', 'Lecture_heavy'
]

tags.columns = tag_names
print(f"Tags data shape: {tags.shape}")

# Reload num data
num_full = pd.read_csv("rmpCapstoneNum.csv", header=None)
num_full.columns = ['Avg_Rating', 'Avg_Difficulty', 'NumofRatings', 'Pepper', 'Again?', 'From online', 'Male', 'Female']

# Merge tags with numerical data
tags_num = tags.merge(num_full[['Avg_Rating', 'NumofRatings']], left_index=True, right_index=True)

# Filter by minimum ratings
MIN_RATINGS_THRESHOLD = 3
print(f"Setting minimum ratings threshold: {MIN_RATINGS_THRESHOLD}")
print(f"Data before filtering: {len(tags_num)}")
tags_num = tags_num[tags_num['NumofRatings'] >= MIN_RATINGS_THRESHOLD]
print(f"Data after filtering (>= {MIN_RATINGS_THRESHOLD} ratings): {len(tags_num)}")

# Normalize tags
for tag in tag_names:
    tags_num[f'{tag}_norm'] = tags_num[tag] / tags_num['NumofRatings'].replace(0, np.nan)

print("\nSample of normalized tags:")
print(tags_num[['Avg_Rating', 'NumofRatings'] + tag_names[:5] + [f'{tag}_norm' for tag in tag_names[:5]]].head())

# Prepare regression data
tag_predictors = [f'{tag}_norm' for tag in tag_names]
target = 'Avg_Rating'

regression_data_tags = tags_num[tag_predictors + [target, 'NumofRatings']].dropna()

print(f"\nOriginal data: {len(tags_num)}")
print(f"After removing null values: {len(regression_data_tags)}")

# Check multicollinearity
tag_corr_matrix = regression_data_tags[tag_predictors].corr()

high_corr_pairs = []
for i in range(len(tag_corr_matrix.columns)):
    for j in range(i+1, len(tag_corr_matrix.columns)):
        if abs(tag_corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Tag 1': tag_corr_matrix.columns[i],
                'Tag 2': tag_corr_matrix.columns[j],
                'Correlation': tag_corr_matrix.iloc[i, j]
            })

print(f"\nFound {len(high_corr_pairs)} tag pairs with |r| > 0.8")
if len(high_corr_pairs) > 0:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df.head(10))

# Visualize correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(tag_corr_matrix, cmap='coolwarm', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Tag Multicollinearity Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Build regression model
X_tags = regression_data_tags[tag_predictors]
y_tags = regression_data_tags[target]

# Split data
X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(
    X_tags, y_tags, test_size=0.2, random_state=MY_SEED
)

# Standardize features
scaler_tags = StandardScaler()
X_train_tags_scaled = scaler_tags.fit_transform(X_train_tags)
X_test_tags_scaled = scaler_tags.transform(X_test_tags)

# Convert back to DataFrame
X_train_tags_scaled = pd.DataFrame(X_train_tags_scaled, columns=tag_predictors, index=X_train_tags.index)
X_test_tags_scaled = pd.DataFrame(X_test_tags_scaled, columns=tag_predictors, index=X_test_tags.index)

# Add constant
X_train_tags_scaled = sm.add_constant(X_train_tags_scaled)
X_test_tags_scaled = sm.add_constant(X_test_tags_scaled)

# Fit OLS model
model_tags = sm.OLS(y_train_tags, X_train_tags_scaled).fit()

print(model_tags.summary())

# Calculate metrics
y_pred_tags = model_tags.predict(X_test_tags_scaled)
rmse_tags = np.sqrt(mean_squared_error(y_test_tags, y_pred_tags))
r2_test_tags = r2_score(y_test_tags, y_pred_tags)
print(f"\nTest RMSE: {rmse_tags:.4f}")
print(f"Test R-squared: {r2_test_tags:.4f}")
print(f"Training R-squared: {model_tags.rsquared:.4f}")

# Find most predictive tag
ALPHA = 0.005
params_tags = model_tags.params.drop('const')
pvalues_tags = model_tags.pvalues.drop('const')

significant_tags = pvalues_tags[pvalues_tags < ALPHA]
print(f"\nSignificant tags (p < {ALPHA}): {len(significant_tags)} out of {len(params_tags)}")

max_impact_tag = params_tags.abs().idxmax()
max_impact_value = params_tags[max_impact_tag]
max_impact_pvalue = pvalues_tags[max_impact_tag]

print(f"\nThe tag with the highest impact on Avg_Rating is '{max_impact_tag}'")
print(f"  Coefficient: {max_impact_value:.4f}")
print(f"  P-value: {max_impact_pvalue:.10f}")
print(f"  Significant at α={ALPHA}: {'Yes' if max_impact_pvalue < ALPHA else 'No'}")

print(f"\nComparison with Question 7 model:")
print(f"Tags model R²: {model_tags.rsquared:.4f}")
print(f"Tags model RMSE: {rmse_tags:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test_tags, y_pred_tags, alpha=0.3, color='purple', label='Data Points')
min_val = min(y_test_tags.min(), y_pred_tags.min())
max_val = max(y_test_tags.max(), y_pred_tags.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_title('Actual vs. Predicted Ratings (Tags Model)', fontsize=15)
axes[0].set_xlabel('Actual Avg_Rating', fontsize=12)
axes[0].set_ylabel('Predicted Avg_Rating', fontsize=12)
axes[0].legend()

# Plot 2: Top 5 most predictive tags
top_tags = params_tags.abs().nlargest(5)
top_tags_sorted = top_tags.sort_values(ascending=True)
axes[1].barh(range(len(top_tags_sorted)), top_tags_sorted.values, color='steelblue')
axes[1].set_yticks(range(len(top_tags_sorted)))
axes[1].set_yticklabels([tag.replace('_norm', '').replace('_', ' ') for tag in top_tags_sorted.index])
axes[1].set_xlabel('Absolute Coefficient Value', fontsize=12)
axes[1].set_title('Top 5 Most Predictive Tags', fontsize=15)
plt.tight_layout()
plt.show()

# ==========================================
# Question 9: Tags Regression (Difficulty)
# ==========================================

print("=" * 60)
print("QUESTION 9: Tags Regression (Difficulty)")
print("=" * 60)

# Prepare data for difficulty prediction
target_difficulty = 'Avg_Difficulty'

tags_num_difficulty = tags.merge(num_full[['Avg_Difficulty', 'NumofRatings']], left_index=True, right_index=True)

print(f"Data before filtering: {len(tags_num_difficulty)}")
tags_num_difficulty = tags_num_difficulty[tags_num_difficulty['NumofRatings'] >= MIN_RATINGS_THRESHOLD]
print(f"Data after filtering (>= {MIN_RATINGS_THRESHOLD} ratings): {len(tags_num_difficulty)}")

# Create normalized tag columns
for tag in tag_names:
    tags_num_difficulty[f'{tag}_norm'] = tags_num_difficulty[tag] / tags_num_difficulty['NumofRatings'].replace(0, np.nan)

# Prepare regression data
regression_data_difficulty = tags_num_difficulty[tag_predictors + [target_difficulty, 'NumofRatings']].dropna()

print(f"Data for difficulty prediction: {len(regression_data_difficulty)} observations")

# Build model
X_diff = regression_data_difficulty[tag_predictors]
y_diff = regression_data_difficulty[target_difficulty]

# Split data
X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(
    X_diff, y_diff, test_size=0.2, random_state=MY_SEED
)

# Standardize features
scaler_diff = StandardScaler()
X_train_diff_scaled = scaler_diff.fit_transform(X_train_diff)
X_test_diff_scaled = scaler_diff.transform(X_test_diff)

# Convert back to DataFrame
X_train_diff_scaled = pd.DataFrame(X_train_diff_scaled, columns=tag_predictors, index=X_train_diff.index)
X_test_diff_scaled = pd.DataFrame(X_test_diff_scaled, columns=tag_predictors, index=X_test_diff.index)

# Add constant
X_train_diff_scaled = sm.add_constant(X_train_diff_scaled)
X_test_diff_scaled = sm.add_constant(X_test_diff_scaled)

# Fit OLS model
model_difficulty = sm.OLS(y_train_diff, X_train_diff_scaled).fit()

print(model_difficulty.summary())

# Calculate metrics
y_pred_diff = model_difficulty.predict(X_test_diff_scaled)
rmse_diff = np.sqrt(mean_squared_error(y_test_diff, y_pred_diff))
r2_test_diff = r2_score(y_test_diff, y_pred_diff)
print(f"\nTest RMSE: {rmse_diff:.4f}")
print(f"Test R-squared: {r2_test_diff:.4f}")
print(f"Training R-squared: {model_difficulty.rsquared:.4f}")

# Find most predictive tag
params_diff = model_difficulty.params.drop('const')
pvalues_diff = model_difficulty.pvalues.drop('const')

significant_tags_diff = pvalues_diff[pvalues_diff < ALPHA]
print(f"\nSignificant tags (p < {ALPHA}): {len(significant_tags_diff)} out of {len(params_diff)}")

max_impact_tag_diff = params_diff.abs().idxmax()
max_impact_value_diff = params_diff[max_impact_tag_diff]
max_impact_pvalue_diff = pvalues_diff[max_impact_tag_diff]

print(f"\nThe tag with the highest impact on Avg_Difficulty is '{max_impact_tag_diff}'")
print(f"  Coefficient: {max_impact_value_diff:.4f}")
print(f"  P-value: {max_impact_pvalue_diff:.6f}")
print(f"  Significant at α={ALPHA}: {'Yes' if max_impact_pvalue_diff < ALPHA else 'No'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test_diff, y_pred_diff, alpha=0.3, color='orange', label='Data Points')
min_val = min(y_test_diff.min(), y_pred_diff.min())
max_val = max(y_test_diff.max(), y_pred_diff.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_title('Actual vs. Predicted Difficulty (Tags Model)', fontsize=15)
axes[0].set_xlabel('Actual Avg_Difficulty', fontsize=12)
axes[0].set_ylabel('Predicted Avg_Difficulty', fontsize=12)
axes[0].legend()

# Plot 2: Top 5 most predictive tags
top_tags_diff = params_diff.abs().nlargest(5)
top_tags_diff_sorted = top_tags_diff.sort_values(ascending=True)
axes[1].barh(range(len(top_tags_diff_sorted)), top_tags_diff_sorted.values, color='coral')
axes[1].set_yticks(range(len(top_tags_diff_sorted)))
axes[1].set_yticklabels([tag.replace('_norm', '').replace('_', ' ') for tag in top_tags_diff_sorted.index])
axes[1].set_xlabel('Absolute Coefficient Value', fontsize=12)
axes[1].set_title('Top 5 Most Predictive Tags for Difficulty', fontsize=15)
plt.tight_layout()
plt.show()

# ==========================================
# Question 10: Pepper Classification
# ==========================================

print("=" * 60)
print("QUESTION 10: Pepper Classification")
print("=" * 60)

# Prepare data for classification
num_class = num_full.copy()
num_class.columns = ['Avg_Rating', 'Avg_Difficulty', 'NumofRatings', 'Pepper', 'Again?', 'From online', 'Male', 'Female']

print(f"Data before filtering: {len(num_class)}")
num_class = num_class[num_class['NumofRatings'] >= MIN_RATINGS_THRESHOLD]
print(f"Data after filtering (>= {MIN_RATINGS_THRESHOLD} ratings): {len(num_class)}")

# Create Gender column
num_class['Gender'] = num_class['Male'] + num_class['Female'] * -1
num_class['Gender'] = num_class['Gender'].replace(0, pd.NA)
num_class['Gender'] = num_class['Gender'].replace(2, pd.NA)

# Merge tags with numerical data
tags_class = tags.copy()
tags_class.columns = tag_names

# Normalize tags
for tag in tag_names:
    tags_class[f'{tag}_norm'] = tags_class[tag] / num_class['NumofRatings'].replace(0, np.nan)

# Combine all features
all_features = num_class[['Avg_Rating', 'Avg_Difficulty', 'NumofRatings', 'Again?', 'From online', 'Gender']].copy()
all_features = all_features.merge(tags_class[[f'{tag}_norm' for tag in tag_names]], 
                                   left_index=True, right_index=True)

# Target variable
target_class = num_class['Pepper'].copy()

# Combine and drop missing values
classification_data = all_features.copy()
classification_data['Pepper'] = target_class
classification_data = classification_data.dropna()

print(f"\nClassification data shape: {classification_data.shape}")
print(f"\nClass distribution:")
print(classification_data['Pepper'].value_counts())
print(f"\nClass proportions:")
print(classification_data['Pepper'].value_counts(normalize=True))

# Prepare features and target
feature_cols = [col for col in classification_data.columns if col != 'Pepper']
X_class = classification_data[feature_cols]
y_class = classification_data['Pepper'].astype(int)

# Split data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=MY_SEED, stratify=y_class
)

print(f"Training set class distribution:")
print(y_train_class.value_counts())
print(f"\nTest set class distribution:")
print(y_test_class.value_counts())

# Standardize features
scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

# Convert back to DataFrame
X_train_class_scaled = pd.DataFrame(X_train_class_scaled, columns=feature_cols, index=X_train_class.index)
X_test_class_scaled = pd.DataFrame(X_test_class_scaled, columns=feature_cols, index=X_test_class.index)

# Build Logistic Regression model
lr_model = LogisticRegression(random_state=MY_SEED, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_class_scaled, y_train_class)

# Predictions
y_pred_lr = lr_model.predict(X_test_class_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_class_scaled)[:, 1]

# Calculate metrics
auc_lr = roc_auc_score(y_test_class, y_pred_proba_lr)
print("Logistic Regression Results:")
print(f"AUROC: {auc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_lr))

# Feature importance
lr_feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test_class, y_pred_proba_lr)

axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2, color='blue')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=15)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Confusion Matrix
cm_lr = confusion_matrix(y_test_class, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
            xticklabels=['No Pepper', 'Pepper'], yticklabels=['No Pepper', 'Pepper'])
axes[1].set_title('Confusion Matrix - Logistic Regression', fontsize=15)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

# Plot 3: Top 10 Feature Importances
top_10_features_lr = lr_feature_importance.head(10)
axes[2].barh(range(len(top_10_features_lr)), top_10_features_lr['Abs_Coefficient'].values, color='steelblue')
axes[2].set_yticks(range(len(top_10_features_lr)))
axes[2].set_yticklabels([feat.replace('_norm', '').replace('_', ' ') for feat in top_10_features_lr['Feature']])
axes[2].set_xlabel('Absolute Coefficient Value', fontsize=12)
axes[2].set_title('Top 10 Most Important Features', fontsize=15)
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

print(f"\nModel Summary:")
print(f"Logistic Regression AUROC: {auc_lr:.4f}")

# ==========================================
# Final Summary
# ==========================================

print("=" * 80)
print("FINAL SUMMARY OF ALL RESULTS")
print("=" * 80)
print(f"Q1 - Gender rating difference: p = {p_val_q1:.5e}")
print(f"Q2 - Distribution difference: p = {p_val_q2:.5e}")
print(f"Q3 - Bootstrap CI for median diff: [{ci_median[0]:.4f}, {ci_median[1]:.4f}]")
print(f"Q4 - Most gendered tag: {df_res.iloc[0]['Tag']} (p = {df_res.iloc[0]['P-value']:.5e})")
print(f"Q5 - Gender difficulty difference (all data): p-value varies by threshold")
print(f"Q6 - Bootstrap effect size CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Q7 - Numerical predictors R²: {model.rsquared:.4f}, RMSE: {rmse:.4f}")
print(f"Q8 - Tags model R²: {model_tags.rsquared:.4f}, RMSE: {rmse_tags:.4f}")
print(f"Q9 - Difficulty tags model R²: {model_difficulty.rsquared:.4f}, RMSE: {rmse_diff:.4f}")
print(f"Q10 - Pepper classification AUROC: {auc_lr:.4f}")
if bonus_main_result:
    print(f"BONUS - NY vs Other States: p = {bonus_main_result['p_value']:.5e}")
    print(f"        Median difference: {bonus_main_result['ny_median'] - bonus_main_result['other_median']:.4f}")
    print(f"        Significant: {'Yes' if bonus_main_result['p_value'] < 0.05 else 'No'}")
else:
    print(f"BONUS - NY vs Other States: Data insufficient for analysis")
print("=" * 80)