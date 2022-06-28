from random import randint
from sklearn.linear_model import LinearRegression


# Creating a training dataset
train_set_limit = 1000
train_set_count = 100

train_input = list()
train_output = list()

for i in range(train_set_limit):
	b1 = randint(0, train_set_limit)
	b2 = randint(0, train_set_limit)
	b3 = randint(0, train_set_limit)
	op = b1 + (5*b2) + (8*b3)
	train_input.append([b1, b2, b3])
	train_output.append(op)

for i in range(20):
	print(train_input[i], train_output[i])

# Training
predictor = LinearRegression()
predictor.fit(X=train_input, y=train_output)

# Prediction
x_test = [[10, 20, 30]]
outcome = predictor.predict(X=x_test)
coefficient = predictor.coef_

print('Outcome :', outcome)
print('Coefficient :', coefficient)


