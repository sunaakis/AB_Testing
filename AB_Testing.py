import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.float_format", lambda x: "%.4f" % x)
pd.set_option("display.max_columns", None)
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy import stats

# Testing if average bidding return > maximum bidding return
# control group will have maximum bidding
# test group will have average bidding

# HO: Mean of Avr.Bid.Return = Mean of Max.Bid.Return
# H1: Mean of Avr.Bid.Return ≠ Mean of Max.Bid.Return

# Reading Data
cont_grp = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name="Control Group")
test_grp = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name="Test Group")

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(cont_grp)
check_df(test_grp)

# Data Set:
# Impression -> # of watch
# Click -> #  of clicks
# Purchase -> # of purchase
# Earning -> earnings

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def outliers(dataframe, col):
    low_limit, up_limit = outlier_thresholds(dataframe, col)
    number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
    print(col + " has " + str(number_of_outliers) + " outliers ")

print (outliers(cont_grp,"Impression"), outliers(cont_grp,"Click"),
       outliers(cont_grp,"Purchase"), outliers(cont_grp,"Earning"))

# Impression has 0 outliers
# Click has 0 outliers
# Purchase has 0 outliers
# Earning has 0 outliers

print (outliers(test_grp,"Impression"), outliers(test_grp,"Click"),
       outliers(test_grp,"Purchase"), outliers(test_grp,"Earning"))

# Impression has 0 outliers
# Click has 0 outliers
# Purchase has 0 outliers
# Earning has 0 outliers

test_grp.head()
cont_grp.head()

# Main indicator of the change of policy is the number of purchase:

mean_cont_grp = cont_grp["Purchase"].mean()
# 550.8940587702316
mean_test_grp = test_grp["Purchase"].mean()
# 582.1060966484675

# We see that there is difference between the two groups but is it statistically meaningful?
# It is seen that there is a difference between the mean values of the two groups, however;
# AB test should be done to measure whether it is also statistically significant:

################################################
# 1-How to define the hypothesis testing of this A/B Test?
################################################

# H0: There is no statistically significant difference between the averages
# of purchases made when maximum bidding and average bidding methods are applied.
# H1: ..... there is a statistically significant difference.

# HO: Mean_Purchase of Avr.Bid. =  Mean_Purchase of Max.Bid
# H1: Mean_Purchase of Avr.Bid. ≠ Mean_Purchase of Max.Bid

################################################
# 2- Can we draw statistically significant results?
################################################

# Normality Test:
control = cont_grp["Purchase"]
test = test_grp["Purchase"]
normal_c = shapiro(control)[1] # --> p-value
normal_t = shapiro(test)[1] # --> p-value

if (normal_c < 0.05) & (normal_t < 0.05):
    # reject H0: normality is not satisfied
    # Do non-parametric test:
    ttest = stats.mannwhitneyu(control, test)[1]
else:
    # do not reject HO: normality cond. is satisfied
    # Variance homogeneity test:
    LeveneTest = stats.levene(control, test)[1]
    if LeveneTest < 0.05:
        ttest = stats.ttest_ind(control, test, equal_var=False)[1]
    else:
        ttest = stats.ttest_ind(control, test, equal_var=True)[1]
    table = pd.DataFrame({"p-value":[ttest]})
    table["Normality Assumption"] = np.where((normal_c < 0.05) & (normal_t < 0.05), "Not Satisfied", "Satisfied")
    table["Homogenous Variance"] = np.where((LeveneTest < 0.05), "No", "Yes")
    table["Test Type"] = np.where((normal_c < 0.05) & (normal_t < 0.05),"Non-Parametric", "Parametric")
    print(table)

#    p-value   Test Type Homogenous Variance Normality Assumption
# 0   0.3493  Parametric                 Yes            Satisfied

# According to the above tests we found that the difference between purchases
# in av.bid and max.bid is not statistically meaningful.

# We see that when the maximum bidding method is compared with the average bidding method,
# there is no statistically significant difference between the purchase rates.

################################################
# 3- Which test we used and why?
################################################

# For the normality condition check we used "shapiro"
# For the variance homogeneity we used "levene" test
# If the normality and homogeneity assumptions are satisfied;
# parametric test we used is "independent two samples T-test"
# If the normality and homogeneity assumptions are NOT satisfied;
# non-prametric test we used is "mannwhitneyu" test

################################################
# 4- Advise to the customer?
################################################

# we found that the difference between purchases in av.bid and max.bid is not statistically meaningful.
# However this does not mean that the difference will stay meaningless.
# We can find a more precise answer if we repeat the test over time and add more data to our set
# just by looking at the purchase behavior the customer can use both methods for the bidding.
# Again we also need to look at he number of clicks, revenue by click and other variables.













