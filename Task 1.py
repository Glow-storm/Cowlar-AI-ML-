import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from collections import Counter
import random
import time
import matplotlib.pyplot as plt


def euclidiean(x1, x2):
    return np.sqrt(np.sum((x1-x2))**2)

def mahalanobis(x1, x2): # tried to implement it but was having problems with for now
    diff = (x1 - x2).reshape(1, -1)
    cov=np.cov(X_train)
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))


def cosine_similarity(vec1, vec2): # tried to implement it but was having problems as accuracuy is very low
    dot_product = np.dot(vec1, vec2)
    norm_product = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return dot_product / norm_product


class KNN:

    def __init__(self, k): # k is the number of nearest neighbors to consider
        self.k=k
    def fit(self,X,Y): # used to training data to the class
        self.X_train = X
        self.Y_train = Y

    def predict(self,X): # here x is testing data X_test
        predicted_labels = [self._predict(a) for a in X] # here we are predicating the labels of all the the test data
        #one by one , by taking each example one at a time, this is called list comprehension
        return predicted_labels
    def _predict(self,x): # here x is one sample from X test
        distances = [euclidiean(x,x_train) for x_train in self.X_train] # compare our current point to each point in training data
        k_indices = np.argsort(distances)[:self.k] # this will sort the distance indices and give the 3 nearst distances
        k_labels  = [self.Y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


def preprocess_data(file_path):
    data=pd.read_csv(file_path,header=None)
    values=data.values
    val_magnitude=np.linalg.norm(values,axis=1)
    norm_values=values/val_magnitude[:,np.newaxis]
    return norm_values



norm_values=preprocess_data("fea.csv")

num_splits=5 # this is for the number of splits in the data
accuracy_score=[]
classes=10 # this is for the number of classes in the data
k_values=[1,3,5,7,9,11,13,15,17,19,21,23,25] # this is for the k values in the knn to manually change the k values
train_img=150 # this is for the number of training images
test_img=20 # this is for the number of testing images
#for k in k_values:  # this is for the k values in the knn , for testing the k values you can either use this or the above one manual way

for split in range(num_splits):
    accuracries=[]
    start_time=time.time()
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for subject in range(classes):
        subject_data=norm_values[subject*170:(subject+1)*170,:] # this is to get the data of each subject
        labels=np.full(170,subject) # this is to get the labels of each subject

        train_indices = np.random.choice(170, train_img, replace=False)
        test_indices = np.random.choice(170, test_img, replace=False)

        # Get the training and testing data for the current subject
        X_train.append(subject_data[train_indices])
        y_train.append(labels[train_indices])
        X_test.append(subject_data[test_indices])
        y_test.append(labels[test_indices])

    single_X_train=np.concatenate(X_train)
    single_y_train=np.concatenate(y_train)
    single_X_test=np.concatenate(X_test)
    single_y_test=np.concatenate(y_test)

    training_set=list(zip(single_X_train,single_y_train))
    testing_set=list(zip(single_X_test,single_y_test))
    # Set the random seed
    # Shuffle the training set
    random.shuffle(training_set)
    shuffled_X_train, shuffled_y_train = zip(*training_set)
    # Shuffle the testing set
    random.shuffle(testing_set)
    shuffled_X_test, shuffled_y_test = zip(*testing_set)
    shuffled_X_train=np.array(shuffled_X_train)
    shuffled_y_train=np.array(shuffled_y_train)
    shuffled_X_test=np.array(shuffled_X_test)
    shuffled_y_test=np.array(shuffled_y_test)


    k1=KNN(k=1)
    k1.fit(shuffled_X_train,shuffled_y_train)
    predictions=k1.predict(shuffled_X_test)
    accuracy=np.sum(predictions==shuffled_y_test)/len(shuffled_y_test)
    print("Accuracy:", accuracy)
    accuracries.append(accuracy)
    endt=time.time()
    print(f"time taken without PCA in split {split+1}  :",endt-start_time)
    std=np.std(accuracy)



    start_time=time.time()
    pca = PCA(n_components=50)  # Choose the number of principal components
    pca.fit(shuffled_X_train)
    reduced_X_train = pca.transform(shuffled_X_train)
    reduced_X_test = pca.transform(shuffled_X_test)


    k1=KNN(k=1)
    k1.fit(reduced_X_train,shuffled_y_train)
    predictions=k1.predict(reduced_X_test)
    accuracy=np.sum(predictions==shuffled_y_test)/len(shuffled_y_test)
    print("Accuracy:", accuracy)
    accuracries.append(accuracy)
    endt=time.time()
    print(f"time taken with  PCA in split {split+1} :",endt-start_time)
    std=np.std(accuracy)








average_accuracy = np.mean(accuracries)
std_deviation = np.std(accuracries)

print("Average Accuracy:", average_accuracy)
print("Standard Deviation:", std_deviation)

covariance_before = np.cov(shuffled_X_train.T)
covariance_after = np.cov(reduced_X_train.T)


# Plot the covariance matrices as images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(covariance_before, cmap='hot', interpolation='nearest')
axs[0].set_title("Covariance Matrix (Before PCA)")
axs[0].axis('off')

axs[1].imshow(covariance_after, cmap='hot', interpolation='nearest')
axs[1].set_title("Covariance Matrix (After PCA)")
axs[1].axis('off')

plt.tight_layout()
plt.show()


