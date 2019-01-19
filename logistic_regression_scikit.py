from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="white") # this configures preset theme for plot

# read in the data as dataframe, and into numpy array
data = pd.read_csv('ex2data1.txt').values



X, y = data[:, :2], data[:, -1]

## plot the data first
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y.ravel())
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

## construct the model
model = LogisticRegression(random_state=0, solver='lbfgs')

## now train!
model.fit(X, y)
result = model.predict(X)


acc = np.where(y == result, 1, 0).mean() * 100
print("Accuracy of the model is {}%".format(round(acc, 1)))

#### Plot decision boundary with contour plot #### 
## create grid and make probability prediction

## min edge of the graph, and max edge of the graph, here 30 and 100, and step, 0.01
# ** the smaller the step, the more time it'll take for the model to make prediction for proba
xx, yy = np.mgrid[30:100:.1, 30:100:.1]
## Above code creates every possible combinations of points for x and y, resulting mesh like 
## structure of coordinates that we'll use later in contour
## LINK: http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/

grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

## plot it!
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X[:,0], X[:, 1], c=y[:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(30, 100), ylim=(30, 100),
       xlabel="$X_1$", ylabel="$X_2$")

plt.show()