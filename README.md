<!-- Add banner here -->
![Banner]

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
- Conclustion: Coming Soon
- Key taweaways: Coming Soon
- Recommendations: Coming Soon

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
- Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, language. 
- Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
- Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are correlated to language (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
- Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
    
### Model
[(Back to top)](#table-of-contents)
<!-- Describe your modeling process -->
- Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.
- Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
- Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
- Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
- Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.

### Evaluate
[(Back to top)](#table-of-contents)
<!-- Describe your evaluation process -->


# Conclusion
[(Back to top)](#table-of-contents)
<!-- Wrap up with conclusions and takeaways -->


# Given More Time
[(Back to top)](#table-of-contents)
<!-- LET THEM KNOW WHAT YOU WISH YOU COULD HAVE DONE-->


# Recreate This Project
[(Back to top)](#table-of-contents)
<!-- How can they do what you do?-->

# Footer
[(Back to top)](#table-of-contents)
<!-- LET THEM KNOW WHO YOU ARE (linkedin links) close with a joke. -->

If you have anyquestions please feel free to reach out to me.