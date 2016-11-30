
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


d = [[5, 6], [3, 0], [1, 0], [4, 0], [1, 5], [4, 0], [6, 6], [4, 5], [2, 1], [0, 1], [6, 6], [0, 5], [2, 3], [1, 0], [4, 2], [5, 2], [0, 0], [4, 0], [6, 0], [4, 0], [1, 5], [2, 0], [5, 6], [2, 0], [3, 2], [1, 2], [2, 4], [0, 4], [0, 5], [5, 4], [1, 5], [3, 2], [4, 1], [0, 2], [5, 5], [1, 6], [2, 2], [5, 6], [1, 3], [4, 0], [1, 4], [0, 6], [3, 3], [5, 1], [1, 6], [1, 4], [2, 0], [2, 6], [0, 0], [6, 5], [1, 4], [2, 3], [0, 0], [3, 5], [3, 5], [0, 1], [6, 5], [0, 1], [5, 5], [2, 1], [3, 1], [4, 0], [5, 4], [4, 2], [5, 5], [0, 0], [2, 1], [2, 3], [0, 2], [3, 3], [1, 4], [6, 0], [6, 6], [1, 5], [5, 5], [0, 0], [4, 1], [5, 0], [3, 2], [5, 0], [3, 1], [5, 0], [0, 3], [5, 1], [5, 6], [1, 6], [3, 1], [3, 5], [1, 0], [5, 5], [6, 5], [1, 6], [6, 1], [6, 2], [0, 6], [4, 1], [4, 1], [6, 5], [1, 2], [5, 1], [4, 6], [6, 3], [3, 0], [0, 5], [3, 5], [4, 2], [4, 6], [1, 3], [1, 0], [6, 4], [3, 2], [4, 3], [5, 6], [3, 3], [6, 1], [0, 4], [5, 4], [5, 4], [2, 1], [6, 2], [3, 3], [2, 0], [1, 0], [6, 1], [0, 3], [1, 0], [3, 6], [2, 4], [5, 4], [3, 3], [2, 3], [1, 3], [5, 3], [0, 1], [0, 5], [6, 3], [5, 0], [4, 0], [6, 1], [2, 5], [3, 4], [0, 6], [1, 4], [6, 0], [0, 0], [2, 2], [6, 5], [1, 0], [1, 4], [0, 1], [5, 6], [1, 5], [3, 4], [2, 1], [2, 4], [1, 6], [4, 0], [6, 5], [0, 2], [3, 2], [3, 6], [5, 1], [1, 2], [3, 6], [4, 0], [6, 3], [6, 0], [5, 1], [5, 3], [1, 2], [0, 0], [5, 0], [3, 1], [2, 4], [4, 4], [6, 4], [5, 0], [1, 5], [3, 3], [4, 1], [0, 3], [1, 5], [2, 1], [1, 4], [1, 5], [2, 0], [2, 1], [0, 1], [5, 0], [4, 2], [6, 3], [4, 3], [3, 4], [1, 5], [6, 5], [5, 0], [0, 6], [1, 3], [6, 3], [1, 6], [1, 5], [2, 1], [5, 3], [3, 3], [1, 0], [3, 4], [1, 6], [3, 1], [4, 4], [5, 0], [4, 2], [6, 0], [5, 4], [1, 0], [5, 0], [2, 0], [6, 5], [4, 0], [3, 3], [4, 5], [0, 6], [3, 5], [3, 1], [0, 1], [2, 0], [4, 5], [3, 0], [3, 2], [4, 3], [3, 2], [2, 1]]

a,b = np.rot90(d)
# df  = pd.DataFrame(np.asarray(d))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(a, b)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names =('0_anger', '1_disgust', '2_fear', '3_happy',   '4_sad' ,'5_surprise', '6_neutral')
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


print 


# # Load the datset of correlations between cortical brain networks

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(12, 9))

# # Draw the heatmap using seaborn
# sns.heatmap(cor, vmax=.8, square=True)

# # Use matplotlib directly to emphasize known networks

# f.tight_layout()

# sns.plt.show()


            

