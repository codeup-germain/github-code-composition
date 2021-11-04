<!-- Add banner here -->
![Banner](https://github.com/codeup-germain/github-code-composition/blob/main/Codeup%20(1280%20x%20650%20px).jpg)

# github-code-composition

<!-- Add buttons here -->

![GitHub release (latest by date including pre-releases)](https://img.shields.io/badge/release-draft-yellow)
![GitHub last commit](https://img.shields.io/badge/last%20commit-Oct%202021-green)

<!-- Describe your project in brief -->

# What is a github README.md?
- You can add a README file to a repository to communicate important information about your project. A README, along with a repository license, citation file, contribution guidelines, and a code of conduct, communicates expectations for your project and helps you manage contributions.
- A README is often the first item a visitor will see when visiting your repository. README files typically include information on: What the project does, why the project is useful, how users can get started with the project, where users can get help with your project, and who maintains and contributes to the project.

# Executive Summary
<!-- Add a demo for your project -->
- Conclusion:  Our model could accuratley predict the language of the repos by 62% which had beat double the baseline model of 31%. Further cleaning of common words or stop words may improve the accuracy of the model. C# gave our models the hardest time with misclassifications of every other langauge falling under the C# moniker.

- Key taweaways: 123 entries in total.  List of repos we gathered were mostly coded HTML, C#, Python, and JavaScript.  C# gave or models the biggest problem causing the most misclassifications because was shared common words, could try filtering those common words to improve accuracy.

- Recommendation: We recommend adding more GitHub repositories README.md files to the dataset to help prevent overfitting and to possibly increase the model accuracy.

# Project Goals
- To construct a classification model that would accurately predict the programming language that was used in a GitHub repository by scraping their respective readme.md files.

# Table of contents
<!-- Add a table of contents for your project -->

- [Project Title](#project-title)
- [What is a github README.md?](#What-is-a-github-README.md?)
- [Executive Summary](#executive-summary)
- [Table of contents](#table-of-contents)
- [Data Dictionary](#data-dictionary)
- [Data Science Pipeline](#data-science-pipline)
    - [Acquire](#acquire)
    - [Prepare](#prepare)
    - [Explore](#explore)
    - [Model](#model)
    - [Evaluate](#evaluate)
- [Conclusion](#conclusion)
- [Given More Time](#given-more-time)
- [Recreate This Project](#recreate-this-project)
- [Footer](#footer)

# Data Dictionary
[(Back to top)](#table-of-contents)
<!-- Drop that sweet sweet dictionary here-->
| Feature         | Datatype             | Definition                                  |
|:----------------|:---------------------|:--------------------------------------------|
| repo            | 123 non-null: object | name of repository                          |
| language        | 121 non-null: object | programming language project was written in |
| readme_contents | 123 non_null: object | text contents of the readme                 |




| Egineered Features | Datatype             | Definition                                  |
|:----------------   |:---------------------|:--------------------------------------------|
| clean              | 108 non-null: object | readme_contents cleaned                     |
| stemmed            | 106 non-null: object | readme_contents stemmed                     |
| lemmatized         | 108 non_null: object | readme_contents lemmatized                  |

# Data Science Pipeline
[(Back to top)](#table-of-contents)
<!-- Describe your Data Science Pipeline process -->

### Acquire
[(Back to top)](#table-of-contents)
<!-- Describe your acquire process -->
- The data is coming from repos on github.  A list of repos has already been saved to github_list.py.  The acquire.py will use the api to return the desired information. 
- Go to github and generate a personal access token https://github.com/settings/tokens.  Create a env.py and save the personal access token in your env.py file under the variable `github_token`.  Add your github username to your env.py file under the variable `github_username`.
- The final function will return a pandas DataFrame
- Import the acquire function from the acquire.py module
- Complete some initial data summarization (`.info()`, `.describe()`, `.head()`, ...).
- Plot distributions of individual variables.

### Prepare
[(Back to top)](#table-of-contents)
<!-- Describe your prepare process -->
- Store functions needed to prepare the github readme data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
- Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.

### Explore
[(Back to top)](#table-of-contents)
<!-- Describe your explore process -->
- Answer key questions, our hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, language. 
- Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
- Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are correlated to language (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
- Summarize our conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
    
### Model
[(Back to top)](#table-of-contents)
<!-- Describe your modeling process -->
- Established a baseline model of 31% accuacy using the most common language from our train data set, C#.
- Created and optimized over 256 different models, including decision trees with varying branch depth, random forest with varying tree depth, knn models with varying neighbors, Support Vector Classification(one vs one, and many vs one),  Naive-Bayes Multinomial, and Logistic Regression models. 
- Performed models on the entirety of the df and on Recursive Feature Engineered features with the top 28 features. Models were also split into ngrams of 1, 2, 3 and 4. The best performing model actually performed best with all features and and ngram size of 1. 
- Best performing model was a decisicion tree with a max depth of three perfomed on lemmatized data using TFIDF vectorization. It had a 72% accuracy on Validate and 86% on train. This is what we are going to run the final unseen data test on.   

### Evaluate
[(Back to top)](#table-of-contents)
<!-- Describe your evaluation process -->
- Ran the best performing model on unseen test data. It doubled baseline prediction with an accuracy score of 62%. It looks like C# gives our models the largest problem. Most misclasifications where because of C#. In the future we may look at further cleaning of common words to reduce this misclassification.

# Conclusion
[(Back to top)](#table-of-contents)
<!-- Wrap up with conclusions and takeaways -->
Our model could accuratley predict the language of the repos by 62% which had beat double the baseline model of 31%. Further cleaning of common words or stop words may improve the accuracy of the model. C# gave our models the hardest time with misclassifications of every other langauge falling under the C# moniker. 

# Given More Time
[(Back to top)](#table-of-contents)
<!-- LET THEM KNOW WHAT YOU WISH YOU COULD HAVE DONE-->
- With more time, we would like to use more repositories to potentially find more words that can help predict the programming language used.

# Recreate This Project
[(Back to top)](#table-of-contents)
<!-- How can they do what you do?-->
- [x] Read this README.md
- [ ] Make a github personal access token
- [ ] Go [here](https://github.com/settings/tokens) and generate a personal access token.  You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
- [ ] Save it in your env.py file under the variable `github_token`
- [ ] Add your github username to your env.py file under the variable `github_username`
- [ ] Download the aquire.py, prepare.py, functions.py, model.py, github_list.py, and final.ipynb files into your working directory
- [ ] From the terminal, run python acquire.py 
- [ ] Run the final_report.ipynb notebook

# Footer
[(Back to top)](#table-of-contents)


If you have any questions please feel free to reach out to any member of the team.