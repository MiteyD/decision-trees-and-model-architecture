import pandas
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/san_fran_crime.csv

#Import the data from the .csv file
dataset = pandas.read_csv('san_fran_crime.csv', delimiter="\t")

#Let's have a look at the data and the relationship we are going to model
print(dataset.head())
print(dataset.shape)

import graphing # custom graphing code. See our GitHub repo for details
import numpy as np

# Crime category
graphing.multiple_histogram(dataset, label_x='Category', label_group="Resolution", histfunc='sum', show=True)

# District
graphing.multiple_histogram(dataset, label_group="Resolution", label_x="PdDistrict", show=True)

# Map of crimes
graphing.scatter_2D(dataset, label_x="X", label_y="Y", label_colour="Resolution", title="GPS Coordinates", size_multiplier=0.8, show=True)

# Day of the week
graphing.multiple_histogram(dataset, label_group="Resolution", label_x="DayOfWeek", show=True)

# day of the year
# For graphing we simplify this to week or the graph becomes overwhelmed with bars
dataset["week_of_year"] = np.round(dataset.day_of_year / 7.0)
graphing.multiple_histogram(dataset, 
                    label_x='week_of_year',
                    label_group='Resolution',
                    histfunc='sum', show=True)
del dataset["week_of_year"]

# One-hot encode categorical features
dataset = pandas.get_dummies(dataset, columns=["Category", "PdDistrict"], drop_first=False)

print(dataset.head())

from sklearn.model_selection import train_test_split

# Split the dataset in an 90/10 train/test ratio. 
# We can afford to do this here because our dataset is very very large
# Normally we would choose a more even ratio
train, test = train_test_split(dataset, test_size=0.1, random_state=2, shuffle=True)

print("Data shape:")
print("train", train.shape)
print("test", test.shape)

from sklearn.metrics import balanced_accuracy_score

# Make a utility method that we can re-use throughout this exercise
# To easily fit and test out model

features = [c for c in dataset.columns if c != "Resolution"]


def fit_and_test_model(model):
    '''
    Trains a model and tests it against both train and test sets
    '''  
    global features

    # Train the model
    model.fit(train[features], train.Resolution)

    # Assess its performance
    # -- Train
    predictions = model.predict(train[features])
    train_accuracy = balanced_accuracy_score(train.Resolution, predictions)

    # -- Test
    predictions = model.predict(test[features])
    test_accuracy = balanced_accuracy_score(test.Resolution, predictions)

    return train_accuracy, test_accuracy


print("Ready to go!")


import sklearn.tree

# fit a simple tree using only three levels
model = sklearn.tree.DecisionTreeClassifier(random_state=2, max_depth=3) 
train_accuracy, test_accuracy = fit_and_test_model(model)

print("Model trained!")
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)


#--------------
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

plot = plt.subplots(figsize = (4,4), dpi=300)[0]
plot = plot_tree(model,
                fontsize=3,
                feature_names = features, 
                class_names = ['0','1'], # class_names in ascending numerical order 
                label="root",
                impurity=False,
                filled=True) 
plt.show()



# fit a very deep tree
model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=100)

train_accuracy, test_accuracy = fit_and_test_model(model)
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)




# Temporarily shrink the training set to something
# more realistic
full_training_set = train
train = train[:100]

# fit the same tree as before
model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=100)

# Assess on the same test set as before
train_accuracy, test_accuracy = fit_and_test_model(model)
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)

# Roll the training set back to the full set
train = full_training_set


"""
Pruning a tree
Pruning is the process of simplifying a decision tree so that it gives the best classification results while simultaneously reducing overfitting. There are two types of pruning: pre-pruning and post-pruning.

Pre-pruning involves restricting the model during training, so that it does not grow larger than is useful. We will cover this below.

Post-pruning is when we simplify the tree after training it. It does not involve the making of any design decision ahead of time, but simply optimizing the exisiting model. This is a valid technique but is quite involved, and so we do not cover it here due time constraints.

Prepruning
We can perform pre-pruning, by generating many models, each with different max_depth parameters. For each, we recording the balanced accuracy for the test set. To show that this is important even with quite large datasets, we will work with 10000 samples here.
"""


# Temporarily shrink the training set to 10000
# for this exercise to see how pruning is important
# even with moderately large datasets
full_training_set = train
train = train[:10000]


# Loop through the values below and build a model
# each time, setting the maximum depth to that value 
max_depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,15, 20, 50, 100]
accuracy_trainset = []
accuracy_testset = []
for depth in max_depth_range:
    # Create and fit the model
    prune_model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=depth)

    # Calculate and record its sensitivity
    train_accuracy, test_accuracy = fit_and_test_model(prune_model)
    accuracy_trainset.append(train_accuracy)
    accuracy_testset.append(test_accuracy)

# Plot the sensitivity as a function of depth  
pruned_plot = pandas.DataFrame(dict(max_depth=max_depth_range, accuracy=accuracy_trainset))

fig = graphing.line_2D(dict(train=accuracy_trainset, test=accuracy_testset), x_range=max_depth_range, show=True)

# Roll the training set back to the full thing
train = full_training_set




# Temporarily shrink the training set to 10000
# for this exercise to see how pruning is important
# even with moderately large datasets
full_training_set = train
train = train[:10000]


# Not-pruned
model = sklearn.tree.DecisionTreeClassifier(random_state=1)
train_accuracy, test_accuracy = fit_and_test_model(model)
print("Unpruned Train accuracy", train_accuracy)
print("Unpruned Test accuracy", test_accuracy)


# re-fit our final tree to print out its performance
model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=10)
train_accuracy, test_accuracy = fit_and_test_model(model)
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)

# Roll the training set back to the full thing
train = full_training_set


"""
Our new and improved pruned model shows a marginally better balanced accuracy on the test set and much worse performance on the training set than the model that is not pruned. This means our pruning has significantly reduced overfitting.

If you would like, go back and change the number of samples to 100, and notice how the optimal max_depth changes. Think about why this might be (hint: model complexity versus sample size)

Another option that you may like to play with is how many features are entered into the tree. Similar patterns of overfitting can be observed by manipulating this. In fact, the number and type of the features provided to a decision tree can be even more important than its sheer size.
"""
