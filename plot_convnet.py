import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style('darkgrid')
epoch_nums = np.array(list(range(26)))
val_accuracy = [0.0, .7826, .9283, .9554, .9613, .9664, .9753, .9772, .9726, .9763, .9850, .9820, .9817, .9839, .9896, .9875, .9932, .9918, .9900, .9916, .9921, .9911, .9882, .9911, .9925, .9928]
validation, = plt.plot(epoch_nums, val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()

