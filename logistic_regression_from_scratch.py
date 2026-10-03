import pandas as pd 
import numpy as np
#creating the required data frame
train_df = pd.read_csv('data/train(2).csv')
test_df = pd.read_csv('data/test(3).csv')
X_train = train_df.drop(columns=['id','target']).values.astype(float)
Y_train = train_df['target'].values
X_test = test_df.drop(columns=['id']).values.astype(float)
#we standardize the data using the train values
mu,sd = X_train.mean(0),X_train.std(0)
Xtr = (X_train-mu)/sd
Xte = (X_test-mu)/sd
n,d = Xtr.shape;K=3#to get the dimension of the training set (150,4)
#then we need to add a column full of 1 in the X_tr to add artificially the bias (b) term. => w*x + b
Xb = np.hstack([Xtr,np.ones((n,1))])
Xbe = np.hstack([Xte,np.ones((len(Xte),1))])
#we create a 1 hot encoding matrix using the Identity 3x3
Y = np.eye(K)[Y_train]#we end up with a 105x3 matrix with a 1 only in the true class column

#1 we aplly softmax to have probabilities. Before, we apply the exp to have only positive output. 
def softmax(Z):
    #np.exp overflow easily => goes fast to inf so we substract the biggest value to get a biggest argument of 0 so we have no overflow.
    #we can do that because the softmax is shift invariant
    Z = Z - Z.max(axis=1, keepdims=True) #axis=1 means operates across columns to find the biggest score per flower, keepdim=True means to keep the dimension to (n,1) so we can have (n,3) - (n,1)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

#2 the objective function: J(w) = cross-entropy + 0.5*ridge
def objective(W):
    P = softmax(Xb @ W.T) # it makes the matrix multiplication (105, 5) @ (5, 3) and transform it into a probability distribution for each rows
    data = -np.log(P[np.arange(n),Y_train]).sum() #the cross entropy with numpy pairs elementwise the rows indices and the train label
    reg = 0.5 * (W[:, :d] ** 2).sum() #we select only the features => we exclude the col d
    return data+reg

#3 the gradient: (predicted - truth) aggregated over samples + ridge 
def gradient(W):
    P = softmax(Xb @ W.T)
    G = (P - Y).T @ Xb #we take the difference of the predicted vs the Truth with P - Y, we multiply by the weight matrix to have the residual error weighted by the flower feature value 
    Wreg = W.copy(); Wreg[:, d] = 0.0              # intercept stays free
    return G + Wreg

#4 the gradient descent:
W = np.zeros((K,d+1))
lr, epochs = 0.5,2000
losses = [objective(W)]
for epoch in range(epochs):
    W = W - (lr*gradient(W))/n #we use the 1/n because we are averaging the loss on each data point
    if epoch % 100 == 0:
        losses.append(objective(W))
        if len(losses)>2 and abs(losses[-2]-losses[-1]) < 1e-12:
            break
    print(f"converged after {epoch} epochs, J = {objective(W):.8f}")

#5 predictions: argmax of the softmax probabilities
preds = np.argmax(Xbe @ W.T, axis=1)
submission = pd.DataFrame({'id': test_df['id'], 'target': preds})
submission.to_csv('submission_scratch.csv', index=False)