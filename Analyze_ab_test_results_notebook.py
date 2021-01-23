#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, I will be working to analyse results of an A/B test run by an e-commerce website.  my goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


#Read the dataset
abo_df = pd.read_csv('ab_data.csv')
abo_df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


#call teh shape to knoe the rows and cloumns.
abo_df.shape


# c. The number of unique users in the dataset.

# In[4]:


# call unique to extract the number of unique ids
abo_df.user_id.nunique()


# d. The proportion of users converted.

# In[5]:


#find the proportion by taking the mean multiplie it by 100
print("Proportion of user converted is:")
abo_df.converted.mean() *100


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


#Extract the total where newpage and treatment dont match
nepg = abo_df.query("group == 'treatment' and landing_page == 'old_page'").shape[0]
treat = abo_df.query("group == 'control' and landing_page == 'new_page'").shape[0]
notmatch = nepg + treat
notmatch


# f. Do any of the rows have missing values?

# In[7]:


#Check missing values by dislpying the info and check for any missing data.
abo_df.info()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


#drop rows that dont match and create new dataset

drpod = abo_df.query('group == "treatment"').query('landing_page == "new_page"')
df2 = drpod.append(abo_df.query('group == "control"').query('landing_page == "old_page"'))


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


# call unique to extract the number of unique ids
df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


#Check duplicat!
df2[df2.duplicated(['user_id'])].user_id


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2.duplicated(['user_id'],keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


#remove the repated id using drop 
df2.drop_duplicates('user_id', inplace=True)


# In[14]:


#Check the data sahpe rows
df2.shape


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


#calculate the mean to get probability of converting 
df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[16]:


#groupby converted control

df2.query('group == "control"')['converted'].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


#groupby converted treatment

df2.query('group == "treatment"')['converted'].mean()


# d. What is the probability that an individual received the new page?

# In[18]:



df2.query('landing_page == "new_page"').shape[0]/df2.landing_page.shape[0]


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.
# 
# 

# **There is not enough difference between the control group and treatment 12% and 11.8%, so it is hard to prove that the old page had more success.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **$H_{0}$ : $p_{new}$ <= $p_{old}$<br>
# $H_{1}$ : $p_{new}$ > $p_{old}$**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[19]:


p_new = df2['converted'].mean()
print("conversion rate for p_new  is: ",p_new)


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[20]:


p_old = df2['converted'].mean()
print("conversion rate for p_old  is: ",p_old)


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[21]:


n_new = df2.query("group == 'treatment'").user_id.nunique()
print("uumber of users in tratment group: ",n_new)


# d. What is $n_{old}$, the number of individuals in the control group?

# In[22]:


n_old=df2.query('group=="control"').shape[0]
print("number of users in control page: ", n_old)


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[23]:


new_page_converted = np.random.binomial(n_new, p_new)


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[24]:


old_page_converted = np.random.binomial(n_old, p_old)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[25]:


difr_pd = new_page_converted/n_new - old_page_converted/n_old
print("diffrence is ", difr_pd)


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[26]:


p_diffs =[]

for _ in range(1,10000):
    new_page_converted = np.random.binomial(n_new, p_new)
    old_page_converted = np.random.binomial(n_old, p_old)
    difr_pd = new_page_converted / n_new - old_page_converted / n_old
    p_diffs.append(difr_pd)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[44]:


#plot line for observed:
plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[28]:


# The actual difference  :
a_diff = df2.query("group == 'treatment'")['converted'].mean() - df2.query("group == 'control'")['converted'].mean()
print("diffrence = ",a_diff)


# In[29]:


#calculate P-value
p_v = (p_diffs>a_diff).mean()
print("p_value = ",p_v)


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Since we calculte the p-value is 0.9 we faild to reject the null hypothesis <br>
# Which mean that the Null is true, so the old page should remain**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[30]:


import statsmodels.api as sm

convert_old = df2.query("landing_page == 'old_page' and converted == 1").shape[0]
print("old page with conversion: ",convert_old)
convert_new = df2.query("landing_page == 'new_page' and converted == 1").shape[0] 
print("new page with conversion: ",convert_new)
n_old = df2[df2['group'] == 'control'].shape[0]
print("The rows associated with old page: ",n_old)
n_new = df2[df2['group'] == 'treatment'].shape[0]
print("The rows associated with new page: ",n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[31]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
print("z_score:", z_score, "\np_value:", p_value)
#z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='smaller')
#z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **The result is simlier to the finding above in j,k, we fail to rejct the null hypothiss .**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **as there are only two outcome i will be using logstic regression**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[32]:


#create intercept column 
df2['intercept']=1
#create dummies
df2[['control','treatment']] = pd.get_dummies(df2['group'])

df2.head()


# In[33]:


df_ab = df2.rename(columns={'treatment': 'ab_page'})
df_ab.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[34]:


import statsmodels.api as sm

log_b = sm.Logit(df_ab['converted'],df_ab[['intercept' ,'ab_page']])
results = log_b.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[35]:


results.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **The P-value is 0.1899 Hence,the new landing page is not  significant in customers decision whether to convert or not.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **If we added another factor it would increase the accuracy of our result. Factor such as time can make a difference in our analysis to understand user behaviour. On the other hand and if we keep adding other factor to our sample it will increase the chance of multicollinearity which consider as a disadvantage**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[36]:


countryfile_df = pd.read_csv('./countries.csv')
updt_df = countryfile_df.set_index('user_id').join(df_ab.set_index('user_id'), how='inner')
updt_df.head()


# In[37]:


#get the new unique value 
updt_df['country'].unique()


# In[38]:


#create dummies variable with result of uniqe value
updt_df[['UK', 'US', 'CA']] = pd.get_dummies(updt_df['country'])[['UK', 'US', 'CA']]


updt_df.head()


# In[39]:


log_b = sm.Logit(updt_df['converted'], updt_df[['intercept','UK','US']]).fit()


# In[40]:


log_b.summary2()


# 
# p_value of the dummies variables coclude that they are not 
# 
# The influence of landing_page in Uk & US is not different to the influence of landing_page in the other countries.
# 
# we fail to reject the null hypothesis.
# 

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[41]:


#Create interacton between  UK and US
updt_df['UK_ab_page'] = updt_df['UK'] * updt_df['ab_page']
updt_df['US_ab_page'] = updt_df['US'] * updt_df['ab_page']
updt_df.head()


# In[42]:


#Create logistic regression for the intereaction variable between ab_page and country using dummy variable
abo_df['intercept'] = 1 # add column for interpret
log_b = sm.Logit(updt_df['converted'], updt_df[['intercept', 'ab_page','UK', 'US', 'UK_ab_page',  'US_ab_page']])
results = log_b.fit()
results.summary2()


# 
# Again the result proves that there is no evidince intraction between the page and countries 
# 
# 
# 
# 
# ## Conclusion
#     
# To conclude, There is no not suffint evidence that the new page add any value than the old page beside all countries did not affect the the conversion rate so once again we failed to rejct the hypothesis using many ways amd methods ro ptove that, 
# 
# Final word that the old page should remained based on the A/B test analysis have been prefomed. 
# 
# ## Refrence
# *Github <br>
# *the resource posted on the notebbok
# 

# In[43]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

