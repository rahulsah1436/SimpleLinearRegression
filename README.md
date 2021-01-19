# Simple Linear Regression Project
## By Rahul Sah
### Under The Sparks Foundation
> In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

## Project overview
> In this project, I build a Simple Linear Regression model to study the linear relationship between studying hours and percentage of marks obtained.

### Linear Regression
Linear Regression is a statistical technique which is used to find the linear relationship between dependent and one or more independent variables. This technique is applicable for Supervised learning Regression problems where we try to predict a continuous variable.

Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. In this project, I employ Simple Linear Regression technique where I have one independent and one dependent variable. It is the simplest form of Linear Regression where we fit a straight line to the data.

Simple Linear Regression (SLR) Simple Linear Regression (or SLR) is the simplest model in machine learning. It models the linear relationship between the independent and dependent variables.

In this project, there is one independent or input variable which represents the Hours of study and is denoted by X. Similarly, there is one dependent or output variable which represents the Scored percentage and is denoted by y. We want to build a linear relationship between these variables. This linear relationship can be modelled by mathematical equation of the form:-

         Y = β0   + β1*X    -------------   (1)
In this equation, X and Y are called independent and dependent variables respectively,

β1 is the coefficient for independent variable and

β0 is the constant term.

β0 and β1 are called parameters of the model.

For simplicity, we can compare the above equation with the basic line equation of the form:-

           y = ax + b       ----------------- (2)
We can see that

slope of the line is given by, a = β1, and

intercept of the line by b = β0.

In this Simple Linear Regression model, we want to fit a line which estimates the linear relationship between X and Y. So, the question of fitting reduces to estimating the parameters of the model β0 and β1.
Modelling the linear relationship between Sales and Advertising Dataset
       


### Python libraries
I have Anaconda Python distribution installed on my system. It comes with most of the standard Python libraries I need for this project. The basic Python libraries used in this project are:-

• Numpy – It provides a fast numerical array structure and operating functions.

• pandas – It provides tools for data storage, manipulation and analysis tasks.

• Scikit-Learn – The required machine learning library in Python.

• Matplotlib – It is the basic plotting library in Python. It provides tools for making plots.

       

### The problem statement
The aim of building a machine learning model is to solve a problem and to define a metric to measure model performance. So, first of all I have to define the problem to be solved in this project. As described earlier, the problem is to model and investigate the linear relationship between hours of study and Percentage Scored . I have used performance metrics RMSE to compute our model performance.

- Independent and Dependent Variables
In this project, I refer Independent variable as Feature variable and Dependent variable as Target variable. These variables are also recognized by different names as follows: -

### Independent variable
Independent variable is also called Input variable and is denoted by X. In practical applications, independent variable is also called Feature variable or Predictor variable. We can denote it as: - Independent or Input variable (X) = Feature variable = Predictor variable

### Dependent variable
Dependent variable is also called Output variable and is denoted by y. Dependent variable is also called Target variable or Response variable. It can be denote it as follows: -

Dependent or Output variable (y) = Target variable = Response variable


## Reading data from remote link
- url = "http://bit.ly/w-data"
### Observation:
Mean Absolute Error: 4.183859899002975
RMSE value: 4.6474
