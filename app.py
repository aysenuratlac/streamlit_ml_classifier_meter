import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
    def run(self):
        self.get_dataset()
        self.generate()
    def Init_Streamlit_Page(self):
        st.title('Classifiers Performance Meter')

        st.write("""
        ## Explore different classifier for various datasets
        """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"### {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )
    def get_dataset(self):
        data = None
        if self.dataset_name == 'Breast Cancer':
            data = pd.read_csv('data.csv')
        else:
            data = pd.read_csv('data.csv')

        st.write(data.head(10))
        st.write('Columns :',data.columns.tolist())

        data.drop(columns=['Unnamed: 32','id'],inplace=True)
        data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
        y=data["diagnosis"]
        data2 = data.drop(columns=["diagnosis"])
        X = data2
        X=(X-X.min())/(X.max()-X.min()) #normalization
        self.X = X
        self.y = y
        st.write('Shape of dataset:', self.X.shape)
        st.write('Number of classes:', len(np.unique(self.y)))
        st.write('Data Preprocessing steps are implemented.', data.tail(10))

        fig = px.imshow(data.corr(), color_continuous_scale="Emrld")
        fig.update_layout(title="Breast Cancer Correlation Matrix",font=dict(size=14), margin=dict(l=20, r=20, b=20, t=20))
        st.plotly_chart(fig)

        malignant = data[data["diagnosis"] == 1]
        benign = data[data["diagnosis"] == 0]
        plt.scatter(malignant["radius_mean"], malignant["texture_mean"], color="red", label="Malignant")
        plt.scatter(benign["radius_mean"], benign["texture_mean"], color="green", label="Benign")
        plt.xlabel("radius_mean")
        plt.ylabel("texture_mean")
        plt.legend()
        st.pyplot(plt)

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            svm_param_grid = {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.001, 0.01, 0.1, 1]
            }
            self.clf  = GridSearchCV(SVC(), param_grid=svm_param_grid, cv=5)
        elif self.classifier_name == 'KNN':
            knn_param_grid = {
                "n_neighbors": np.arange(1, 11),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            }
            self.clf  = GridSearchCV(KNeighborsClassifier(), param_grid=knn_param_grid, cv=5)
        elif self.classifier_name == 'Naive Bayes':
            nb_param_grid = {
                "var_smoothing": np.logspace(0, -9, 10)
            }
            self.clf = GridSearchCV(GaussianNB(), param_grid=nb_param_grid, cv=5)

    def generate(self):
        self.get_classifier()
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)

        self.clf.fit(X_train, y_train)
        st.write(f'### Classifier = {self.classifier_name}')
        st.write(f'### Best Parameters:', self.clf.best_params_)

        y_pred = self.clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        

        st.title("Model Evaluation")

        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Greens")
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(plt)
    