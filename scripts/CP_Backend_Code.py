import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import os

RANDOM_SEED = 42


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the images
dataset_path = os.path.join(current_dir, "..", "data", "Crop_recommendation.csv")
saved_model_path = os.path.join(current_dir, "..", "models")

# define a function to preprocess the dataset
def preprocess_dataset(df):

    # split the dataset into features and target variable
    X = df.iloc[:, :-2].values
    y1 = df.iloc[:, -2].values  # yield_type
    y2 = df.iloc[:, -1].values  # yield

    # # encode the target variable using LabelEncoder
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)

    # # define the categorical and numerical features
    # categorical_features = []
    # numerical_features = [0, 1, 2, 3, 4, 5, 6]
    #
    # # define the preprocessing steps for numerical and categorical features
    # numerical_transformer = StandardScaler()
    # categorical_transformer = OneHotEncoder()
    # default_transformer = SimpleImputer(strategy='constant', fill_value=0)
    #
    # # combine the preprocessing steps using ColumnTransformer
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", numerical_transformer, numerical_features),
    #         ("cat", categorical_transformer, categorical_features),
    #         ("default", default_transformer, numerical_features)
    #     ])
    #
    # # preprocess the dataset
    # X = preprocessor.fit_transform(X)

    return X, y1, y2

# load the dataset
dataset = pd.read_csv(dataset_path)
# Data anslysis
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# preprocess the dataset
X, y1, y2 = preprocess_dataset(dataset)
# split the dataset into training and testing sets for yield_type
X_train_yt, X_test_yt, y_train_yt, y_test_yt = train_test_split(X, y1, test_size=0.2, random_state=RANDOM_SEED)

# split the dataset into training and testing sets for yield
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y2, test_size=0.2, random_state=RANDOM_SEED)

# train a random forest classifier on the training set for yield_type
rf_classifier_yt = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_classifier_yt.fit(X_train_yt, y_train_yt)

# train a KNN classifier on the training set for yield_type
knn_classifier_yt = KNeighborsClassifier(n_neighbors=5)
knn_classifier_yt.fit(X_train_yt, y_train_yt)

# train a support vector classifier on the training set for yield_type
svc_classifier_yt = SVC(kernel='linear', random_state=42)
svc_classifier_yt.fit(X_train_yt, y_train_yt)

# train a gradient boosting classifier on the training set for yield_type
gb_classifier_yt = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
gb_classifier_yt.fit(X_train_yt, y_train_yt)

# make predictions on the testing set for yield_type
rf_y_pred_yt = rf_classifier_yt.predict(X_test_yt)
knn_y_pred_yt = knn_classifier_yt.predict(X_test_yt)
svc_y_pred_yt = svc_classifier_yt.predict(X_test_yt)
gb_y_pred_yt = gb_classifier_yt.predict(X_test_yt)

# evaluate the models for yield_type
rf_cm_yt = confusion_matrix(y_test_yt, rf_y_pred_yt)
rf_accuracy_yt = accuracy_score(y_test_yt, rf_y_pred_yt)
rf_precision_yt = precision_score(y_test_yt, rf_y_pred_yt, average='weighted')
rf_recall_yt = recall_score(y_test_yt, rf_y_pred_yt, average='weighted')
rf_f1_yt = f1_score(y_test_yt, rf_y_pred_yt, average='weighted')

knn_cm_yt = confusion_matrix(y_test_yt, knn_y_pred_yt)
knn_accuracy_yt = accuracy_score(y_test_yt, knn_y_pred_yt)
knn_precision_yt = precision_score(y_test_yt, knn_y_pred_yt, average='weighted')
knn_recall_yt = recall_score(y_test_yt, knn_y_pred_yt, average='weighted')
knn_f1_yt = f1_score(y_test_yt, knn_y_pred_yt, average='weighted')

svc_cm_yt = confusion_matrix(y_test_yt, svc_y_pred_yt)
svc_accuracy_yt = accuracy_score(y_test_yt, svc_y_pred_yt)
svc_precision_yt = precision_score(y_test_yt, svc_y_pred_yt, average='weighted')
svc_recall_yt = recall_score(y_test_yt, svc_y_pred_yt, average='weighted')
svc_f1_yt = f1_score(y_test_yt, svc_y_pred_yt, average='weighted')

gb_cm_yt = confusion_matrix(y_test_yt, gb_y_pred_yt)
gb_accuracy_yt = accuracy_score(y_test_yt, gb_y_pred_yt)
gb_precision_yt = precision_score(y_test_yt, gb_y_pred_yt, average='weighted')
gb_recall_yt = recall_score(y_test_yt, gb_y_pred_yt, average='weighted')
gb_f1_yt = f1_score(y_test_yt, gb_y_pred_yt, average='weighted')

print("Random Forest Classifier for yield_type")
print("Confusion Matrix:\n", rf_cm_yt)
print("Accuracy: {:.2f}%".format(rf_accuracy_yt*100))
print("Precision: {:.2f}%".format(rf_precision_yt*100))
print("Recall: {:.2f}%".format(rf_recall_yt*100))
print("F1 Score: {:.2f}%".format(rf_f1_yt*100))

print("\nKNN Classifier for yield_type")
print("Confusion Matrix:\n", knn_cm_yt)
print("Accuracy: {:.2f}%".format(knn_accuracy_yt*100))
print("Precision: {:.2f}%".format(knn_precision_yt*100))
print("Recall: {:.2f}%".format(knn_recall_yt*100))
print("F1 Score: {:.2f}%".format(knn_f1_yt*100))

print("\nSVC Classifier for yield_type")
print("Confusion Matrix:\n", svc_cm_yt)
print("Accuracy: {:.2f}%".format(svc_accuracy_yt*100))
print("Precision: {:.2f}%".format(svc_precision_yt*100))
print("Recall: {:.2f}%".format(svc_recall_yt*100))
print("F1 Score: {:.2f}%".format(svc_f1_yt*100))

print("\nGradient Boosting Classifier")
print("Confusion Matrix:\n", gb_cm_yt)
print("Accuracy: {:.2f}%".format(gb_accuracy_yt*100))
print("Precision: {:.2f}%".format(gb_precision_yt*100))
print("Recall: {:.2f}%".format(gb_recall_yt*100))
print("F1 Score: {:.2f}%".format(gb_f1_yt*100))

# train a random forest classifier on the training set for yield
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_classifier.fit(X_train_y, y_train_y)

# train a KNN classifier on the training set
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_y, y_train_y)

# train a support vector classifier on the training set
svc_classifier = SVC(kernel='linear', random_state=RANDOM_SEED)
svc_classifier.fit(X_train_y, y_train_y)

# train a gradient boosting classifier on the training set
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
gb_classifier.fit(X_train_y, y_train_y)

# make predictions on the testing set
rf_y_pred = rf_classifier.predict(X_test_y)
knn_y_pred = knn_classifier.predict(X_test_y)
svc_y_pred = svc_classifier.predict(X_test_y)
gb_y_pred = gb_classifier.predict(X_test_y)

# evaluate the models
rf_cm = confusion_matrix(y_test_y, rf_y_pred)
rf_accuracy = accuracy_score(y_test_y, rf_y_pred)
rf_precision = precision_score(y_test_y, rf_y_pred, average='weighted')
rf_recall = recall_score(y_test_y, rf_y_pred, average='weighted')
rf_f1 = f1_score(y_test_y, rf_y_pred, average='weighted')

knn_cm = confusion_matrix(y_test_y, knn_y_pred)
knn_accuracy = accuracy_score(y_test_y, knn_y_pred)
knn_precision = precision_score(y_test_y, knn_y_pred, average='weighted')
knn_recall = recall_score(y_test_y, knn_y_pred, average='weighted')
knn_f1 = f1_score(y_test_y, knn_y_pred, average='weighted')

svc_cm = confusion_matrix(y_test_y, svc_y_pred)
svc_accuracy = accuracy_score(y_test_y, svc_y_pred)
svc_precision = precision_score(y_test_y, svc_y_pred, average='weighted')
svc_recall = recall_score(y_test_y, svc_y_pred, average='weighted')
svc_f1 = f1_score(y_test_y, svc_y_pred, average='weighted')

gb_cm = confusion_matrix(y_test_y, gb_y_pred)
gb_accuracy = accuracy_score(y_test_y, gb_y_pred)
gb_precision = precision_score(y_test_y, gb_y_pred, average='weighted')
gb_recall = recall_score(y_test_y, gb_y_pred, average='weighted')
gb_f1 = f1_score(y_test_y, gb_y_pred, average='weighted')

# print the evaluation metrics for each model
print("Random Forest Classifier:")
print("Confusion Matrix:")
print(rf_cm)
print("Accuracy: {:.2f}%".format(rf_accuracy*100))
print("Precision: {:.2f}%".format(rf_precision*100))
print("Recall: {:.2f}%".format(rf_recall*100))
print("F1 Score: {:.2f}%".format(rf_f1*100))

print("\nKNN Classifier:")
print("Confusion Matrix:")
print(knn_cm)
print("Accuracy: {:.2f}%".format(knn_accuracy*100))
print("Precision: {:.2f}%".format(knn_precision*100))
print("Recall: {:.2f}%".format(knn_recall*100))
print("F1 Score: {:.2f}%".format(knn_f1*100))

print("\nSupport Vector Classifier:")
print("Confusion Matrix:")
print(svc_cm)
print("Accuracy: {:.2f}%".format(svc_accuracy*100))
print("Precision: {:.2f}%".format(svc_precision*100))
print("Recall: {:.2f}%".format(svc_recall*100))
print("F1 Score: {:.2f}%".format(svc_f1*100))

print("\nGradient Boosting Classifier:")
print("Confusion Matrix:")
print(gb_cm)
print("Accuracy: {:.2f}%".format(gb_accuracy*100))
print("Precision: {:.2f}%".format(gb_precision*100))
print("Recall: {:.2f}%".format(gb_recall*100))
print("F1 Score: {:.2f}%".format(gb_f1*100))

#save the models for yield
joblib.dump(rf_classifier, os.path.join(saved_model_path, "random_forest_model_yield.joblib"))
joblib.dump(knn_classifier, os.path.join(saved_model_path, "knn_model_yield.joblib"))
joblib.dump(svc_classifier, os.path.join(saved_model_path, "svc_model_yield.joblib"))
joblib.dump(gb_classifier, os.path.join(saved_model_path, "gradient_boost_model_yield.joblib"))

#save the models for yield type
joblib.dump(rf_classifier_yt, os.path.join(saved_model_path, "random_forest_model_yield_type.joblib"))
joblib.dump(knn_classifier_yt, os.path.join(saved_model_path, "knn_model_yield_type.joblib"))
joblib.dump(svc_classifier_yt, os.path.join(saved_model_path, "svc_model_yield_type.joblib"))
joblib.dump(gb_classifier_yt, os.path.join(saved_model_path, "gradient_boost_model_yield_type.joblib"))

# #save the models evaluation meterics
# joblib.dump([rf_accuracy, rf_precision, rf_recall, rf_f1], 'random_forest_model_yield_em.joblib')
# joblib.dump([knn_accuracy, knn_precision, knn_recall, knn_f1], 'knn_model_yield_em.joblib')
# joblib.dump([svc_accuracy, svc_precision, svc_recall, svc_f1], 'svc_model_yield_em.joblib')
# joblib.dump([gb_accuracy, gb_precision, gb_recall, gb_f1], 'gradient_boost_model_yield_em.joblib')

