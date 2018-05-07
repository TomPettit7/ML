from sklearn import tree

#the first number in the bracket represents a feature of the object you want the AI to recognise.
#in this case, the first number represents weight (kg), and the second represents height (cm)
features = [[75, 180], [70, 190], [50,160], [55,165]]

#labels are used to state what the features are describing
#in this case, 1 = man, 0 = woman
labels = [1, 1, 0, 0]

#then train a decision tree classifier using the features and labels provided above
classifier = tree.DecisionTreeClassifier()

#the 'fit' command finds patterns in the data provided above
classifier = classifier.fit(features, labels)

#use the 'predict' command to make the AI predict whether it is a man or a woman based off of the features you provide
#for example here, I make it guess what an 80kg, 185cm tall person is
print(classifier.predict([[80,185]]))
