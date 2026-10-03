import pandas as pd 
from sklearn.linear_model import LogisticRegression 
train_df = pd.read_csv('data/train(2).csv')
X_train = train_df.drop(columns=['id','target'])
Y_train = train_df['target']
clf = LogisticRegression(random_state = 0,verbose=1)
clf.fit(X_train,Y_train)
test_df = pd.read_csv('data/test(3).csv')
predictions = clf.predict(test_df.drop(columns=['id']))
#the number of steps for the function minimization
print("number of iteration:",clf.n_iter_)
#train accuracy after the training
train_acc = clf.score(X_train,Y_train)
print(f"\ntraining accuracy: {train_acc:.3f}")
submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)