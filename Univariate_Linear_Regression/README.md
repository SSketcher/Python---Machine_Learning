# Univariate Linear Regression
## General info
This project is about creating simple linear regression class. What linear regression is ?

>In statistics, linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression. This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.
[Linear regression Wiki](https://en.wikipedia.org/wiki/Linear_regression)

* [linear_regression.py](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Linear--Regression/linear_regression.py) - main python file containing class performing linear regression.
* [test_example.py](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Linear--Regression/test_example.py) - python script using the linear regression model to fit regression line for simple linear dataset with random noise.

Test example consist of a few linearly distributed points with random noise. On this dataset later it's performed linear regression.
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Univariate_Linear_Regression/resources/Figure_1.png?raw=true)

The algorithm worked for 50 epoch to fit the regression line into the dataset.
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Univariate_Linear_Regression/resources/Figure_2.png?raw=true)

Visualization of change in value of cost function.        
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Univariate_Linear_Regression/resources/Figure_3.png?raw=true)

## Technologies
* Python 3.7.3

Libraries:
* NumPy
* Matplotlib

## Sources and helpful materials
[Univariate Linear Regression From Scratch With Python](https://satishgunjal.com/univariate_lr/)

[Linear Regression using Gradient Descent](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)

[Linear regression Wiki](https://en.wikipedia.org/wiki/Linear_regression)

[Line plot styles in Matplotlib](https://www.pythoninformer.com/python-libraries/matplotlib/line-plots/)

[Pyplot Doc](https://matplotlib.org/tutorials/introductory/pyplot.html)
