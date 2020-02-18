#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
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
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
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
import collections
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[4]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[5]:


df.groupby('converted')['user_id'].nunique()[1] / df['user_id'].nunique()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


df[(df['group']=='treatment') & (df['landing_page'] != 'new_page')].count()[0]


# f. Do any of the rows have missing values?

# In[7]:


df.isnull().values.any() #no


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2 = df[((df['group']=='treatment') & (df['landing_page'] == 'new_page')) | ((df['group']=='control') & (df['landing_page'] == 'old_page'))]


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# In[10]:


df2.head()


# a. How many unique **user_id**s are in **df2**?

# In[11]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[12]:


dup = df2[df2['user_id'].duplicated()]
dup['user_id']


# c. What is the row information for the repeat **user_id**? 

# In[13]:


dup


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[14]:


df2 = df2.drop_duplicates(subset='user_id', keep='first', inplace=False)


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


pconvert = df2.loc[df2['converted']==1].count()[0] / df2.count()[0]
pconvert


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[16]:


pcontrol = df2.loc[df2['group']=='control'].count()[0] / df2.count()[0]
pcon_control = (pcontrol * pconvert) / pcontrol
pcon_control


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


ptreatment = df2.loc[df2['group']=='treatment'].count()[0] / df2.count()[0]
ptreatment
pcon_treatment = (pcontrol * ptreatment) / ptreatment
pcon_treatment


# d. What is the probability that an individual received the new page?

# In[18]:


pnewpage = df2.loc[df2['landing_page']=='new_page'].count()[0] / df2.count()[0]
pnewpage


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# **Yes, people are more likely to convert if they received the treatment instead of the control.**

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

# h0 : pnew - pold ≤ 0
# 
# h1 : pnew - pold > 0

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[19]:


pnew = df2.loc[df2['converted']==1].count()[0] / df2.count()[0]
pnew


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[20]:


pold = df2.loc[df2['converted']==1].count()[0] / df2.count()[0]
pold


# c. What is $n_{new}$?

# In[21]:


df2.groupby(['group']).count()
#145310 users received treatment
nnew = 145310


# d. What is $n_{old}$?

# In[22]:


df2.groupby(['group']).count()
#145274 users received control
nold = 145274


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[23]:


new_page_converted = np.random.choice([0, 1], size=(nnew,), p=[1-pnew, pnew])
new_page_converted


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[24]:


old_page_converted = np.random.choice([0, 1], size=(nold,), p=[1-pold, pold])
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[25]:


pnew_sim = collections.Counter(new_page_converted)[1] / nnew
pnew_sim


# In[26]:


pold_sim = collections.Counter(old_page_converted)[1] / nold
pold_sim


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[27]:


p_diffs = []
for i in range (100): #10000 makes it run really slow
    new_page_converted = np.random.choice([0, 1], size=(nnew,), p=[1-pnew, pnew])
    pnew_sim = collections.Counter(new_page_converted)[1] / nnew
    old_page_converted = np.random.choice([0, 1], size=(nold,), p=[1-pold, pold])
    pold_sim = collections.Counter(old_page_converted)[1] / nold
    p_dif = pnew_sim - pold_sim
    p_diffs.append(p_dif)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[28]:


plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[29]:


count = 0
for diff in p_diffs:
    if diff > pcon_treatment - pcon_control:
        count +=1
count/len(p_diffs)


# In[30]:


plt.hist(p_diffs)
plt.axvline(x=pcon_treatment - pcon_control,color='red')


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **The p-value is the proportion of values in the null distribution that were greater than the observed difference. Since the p value is less than 0.05, we conclude that the new page has a significantly higher CTR than the old page.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[31]:


from statsmodels.stats.proportion import proportions_ztest

convert_old = df2[(df2['group']=='control') & (df2['converted'] == 1)].count()[0]
convert_new = df2[(df2['group']=='treatment') & (df2['converted'] == 1)].count()[0]
n_old = df2[df2['group']=='control'].count()[0]
n_new = df2[df2['group']=='treatment'].count()[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[32]:


stat, pval = proportions_ztest(convert_old,n_old,0.05)
stat, pval


# In[33]:


stat, pval = proportions_ztest(convert_new,n_new,0.05)
stat, pval


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Yes, the p value is < 0.05 which indicates that the difference between the old pages and the new pages is significant.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[34]:


df2['intercept'] = 1
df2['ab_page'] = np.where(df2['group']=='control',0,1) 
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[39]:


import statsmodels.api as sm
glm = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results = glm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[40]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **The p-value is 0.19. This p value is different because it is measuring the value of the ab_page coefficient in predicting conversion. Because the p value is greater than 0.05, ab_page may not be significant in predicting conversion**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Adding other factors into the regression model may help you better predict the response variable. However multicollinearity, when predictor variables are correlated with one another, may bias the model.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[41]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[67]:


### Create the necessary dummy variables
df_new = pd.concat([df_new, pd.get_dummies(df_new['country'],prefix = 'country')],axis = 1)
glm = sm.Logit(df_new['converted'],df_new[['intercept','country_CA','country_UK','country_US']])
results = glm.fit()
results.summary()


# **No country is significant in predicting conversion.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[66]:


glm = sm.Logit(df_new['ab_page'],df_new[['intercept','country_UK','country_CA','country_US']])
results = glm.fit()
results.summary()


# **Country is significant in predicting which page they were given**

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.

# In[ ]:




