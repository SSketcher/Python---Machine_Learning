# Multivariate Linear Regression
## General info
This project is about creating multidimensional linear regression class, also nown as multivariate linear regression. What linear regression is ?

>In statistics, linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression. This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.
[Linear regression Wiki](https://en.wikipedia.org/wiki/Linear_regression)

* [linear_regression.py](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/linear_regression.py) - main python file containing class performing linear regression.
* [Real_estate_price_prediction.py](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/Real_estate_price_prediction.py) - python script using the linear regression model to predict real estate price.
* [resources/Real_estate.csv](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/resources/Real_estate.csv) - csv file with data.

Visualization the influence of each feature in the dataset, on the real estate price.
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/resources/Figure_1.png?raw=true)

Plotting in 3D space relation between the geographic coordinates, and the real estate price. We can see there is a coraelation between location and price.
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/resources/Figure_2.png?raw=true)

Training the model on the dataset, in 100 epochs with learning rate equal to 0.05, and batch size of 105. Visualization of change in value of cost function.        
![alt text](https://github.com/SSketcher/Python---Machine_Learning/blob/master/Multivariate_Linear_Regression/resources/Figure_3.png?raw=true)


## Technologies
* Python 3.7.10

Libraries:
* NumPy
* Matplotlib
* Pandas

## Sources and helpful materials
[Multivariate Linear Regression From Scratch With Python](https://satishgunjal.com/multivariate_lr/)

[Data Science Simplified Part 5: Multivariate Regression Models](https://towardsdatascience.com/data-science-simplified-part-5-multivariate-regression-models-7684b0489015)

[Linear regression Wiki](https://en.wikipedia.org/wiki/Linear_regression)

[Line plot styles in Matplotlib](https://www.pythoninformer.com/python-libraries/matplotlib/line-plots/)

[Pyplot Doc](https://matplotlib.org/tutorials/introductory/pyplot.html)
