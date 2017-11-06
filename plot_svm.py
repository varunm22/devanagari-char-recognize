import matplotlib.pyplot as plt
import seaborn as sns

X = [10, 20, 50, 100, 200, 500, 1000, 1500]
data = {}
data["ovo"] = [0.48, 0.58043478, 0.67391304, 0.71728261, 0.75532609, 0.782934782609, 0.796304347826, 0.796195652174]

data['ovr'] = [0.4123913, 0.45815217, 0.48032609, 0.4775, 0.50043478, 0.53869565, 0.5351087, 0.53293478] 

sns.set_style("darkgrid")
plt.plot(X, data['ovo'])
plt.plot(X, data['ovr'])
plt.xlabel("Number of Training Examples per Character")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy of Multiclass SVM")
plt.legend(["One vs One", "One vs Rest"], loc='lower right')
plt.savefig('svm-plot.png')
