import matplotlib.pyplot as plt
import seaborn as sns


def plot_variable_pairs(train, cols, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()

def plot_pairplot(train, cols, hue=None):
    '''
    Take in train df, list of columns to plot, and hue=None
    and display scatter plots and hists.
    '''
    sns.pairplot(train[cols], corner=True, hue=hue)
    plt.show()
    

def correlation_exploration(train, feature_x, feature_y, t, p):
    '''
    This function takes in a df, a string for an x-axis variable in the df, 
    and a string for a y-axis variable in the df and displays a scatter plot, the r-
    squared value, and the p-value. It explores the correlation between input the x 
    and y variables.
    '''
    train.plot.scatter(feature_x, feature_y)
    plt.title(f"{feature_x}'s Relationship with {feature_y}")
    print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f't = {round(t, 2)}')
    plt.show()

def distribution_plot(df,feature_lst):
    '''
    This function will take in a dataframe(train) and features to create a barplot for us to check distributions
    of our selected features/univeriate exploration
    '''
    plt.figure(figsize=(13,25))
    plt.subplot(5,1,1, xlabel = 'Property Square Footage', ylabel= 'No. Properties', title='Distribution of Sq Ft')
    plt.hist(data=df, x=feature_lst[0], bins = 30,ec='black')

    plt.subplot(5,1,2, xlabel = 'No. of Bathrooms on Property', ylabel= 'No. Properties', title='Distribution of No. of Bathrooms')
    plt.hist(data=df, x=feature_lst[1], ec='black')

    plt.subplot(5,1,3, xlabel = 'No. of Bedrooms on Property', ylabel= 'No. Properties', title='Distribution of No. of Bedrooms')
    plt.hist(data=df, x=feature_lst[2],ec='black')

    plt.subplot(5,1,4, xlabel = 'County', ylabel= 'No. Properties', title='Distribution of FIPS')
    plt.hist(data=df, x=feature_lst[3],ec='black')

    plt.subplot(5,1,5, xlabel = 'Age of Property', ylabel= 'No. Properties', title='Distribution of House Age')
    plt.hist(data=df, x=feature_lst[4],ec='black')

    plt.subplots_adjust(hspace=1)
    plt.show()

def get_zillow_heatmap(train):
    '''
    returns a heatmap and correlations of how each feature relates to tax_value
    '''
    plt.figure(figsize=(8,12))
    heatmap = sns.heatmap(train.corr()[['tax_value']].sort_values(by='tax_value', ascending=False), vmin=-.5, vmax=.5, annot=True)
    heatmap.set_title('Feautures Correlating with Value')
    
    return heatmap

def plot_categorical_and_continuous_vars (df, categorical, continuous):
    '''
    takes in a df, a list of categorical columns, list
    '''
    print('Discrete with Continuous')
    plt.figure(figsize=(13, 6))
    for cat in categorical:
        for cont in continuous:
            sns.boxplot(x= cat, y=cont, data=df)
            plt.show()
            sns.swarmplot(x=cat, y=cont, data=df)
            plt.show()
    print('Continuous with Continuous')        
    sns.pairplot(df[continuous], kind="reg", plot_kws={'line_kws':{'color':'red'}}, corner=True)
    return

def distribution_single_var (df, columns):
    '''
    Take in a train_df and return a distributions of single varibles
    '''

    for col in columns:
            #plot
            plt.show()
            plt.figure(figsize=(10, 6))
            sns.displot(df[col])
            plt.title(col)
            plt.show()

    return

def plot_residuals(y_validate):
    '''
    take in a df, display a scatter plot. 
    The closer a dot is to the line means that the closer the prediction was to the actual value
    '''
    # plot the residuals for the best performing model
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lm - y_validate.tax_value, 
            alpha=.5, color="blue", s=100, label="Model: Linear Regression")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lars - y_validate.tax_value, 
            alpha=.5, color="yellow", s=100, label="Model: LassoLars Regression")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_glm - y_validate.tax_value, 
            alpha=.5, color="green", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lm3 - y_validate.tax_value, 
            alpha=.5, color="red", s=100, label="Model 3rd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual tax Value")
    plt.ylabel("Residual/Error: Predicted Tax value - Actual")
    plt.title("Residuals")

    plt.show()

def plot_polynomial(y_test, y_validate):
    plt.figure(figsize=(20,8))

    sns.regplot(data=y_test, x=y_validate.tax_value, y=y_validate.tax_value_pred_lm3, 
            scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    plt.xlabel("Actual Tax Value of Property", fontdict={'fontsize':15})
    plt.ylabel("Predicted Tax Value of Property W/ Polynomial Regression Model", fontdict={'fontsize':15})
    plt.title("Polynomial Regression Model", fontdict={'fontsize': 20})

    plt.show()