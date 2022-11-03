# Import the required Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import seaborn as sns
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# instantiate classifier with default hyperparameters

# Page layout
## Page expands to full width
st.set_page_config(page_title='Project for CE 548 _ MD. NABIL ZAWAD',
    layout='wide')
def data_Cover():
    st.title('Interactive Application for Data Analysis using Machine Learning Models')
    st.write("""
    ## MD. NABIL ZAWAD (443106604)
    #### A Project work for the course 'CE 548: ML and AI Applications in Transportation Systems', Dept. of Civil Engineering, King Saud University.
    
    """)

# Functions for each of the pages



def data_Linear_Regression():
    # ---------------------------------#
    # Model building
    def build_model(df):
        X = df.iloc[:, :-1]  # Using all column except for the last column as X
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100, random_state=0)

        stat = st.selectbox('Info of Dataframe:',
                            ('1.1. Overview of Dataset', '1.2. Data splits', '1.3. Variable details',
                             '1.4. Correlation'))

        if stat == '1.1. Overview of Dataset':
            # st.markdown('**1.1. Overview of Dataset**')
            st.write('Data size')
            st.info(df.shape)

        # st.markdown('**1.2. Data splits**')
        elif stat == '1.2. Data splits':
            st.write('Training set')
            st.info(X_train.shape)
            st.write('Test set')
            st.info(X_test.shape)

        # st.markdown('**1.3. Variable details**:')
        elif stat == '1.3. Variable details':
            st.write('X variable')
            st.info(list(X.columns))
            st.write('Y variable')
            st.info(Y.name)
        elif stat == '1.4. Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())

        ml = LinearRegression()
        ml.fit(X_train, Y_train)

        st.subheader('2. Model Performance')
        stat = st.selectbox('Info of Dataframe:',
                            ('2.1. Training set', '2.2. Test set', '2.3. Scatter Plot'))
        # st.markdown('**2.1. Training set**')
        if stat == '2.1. Training set':
            Y_pred_train = ml.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info(r2_score(Y_train, Y_pred_train))

            st.write('Error (MSE or MAE):')
            st.info(mean_squared_error(Y_train, Y_pred_train))

        # st.markdown('**2.2. Test set**')
        elif stat == '2.2. Test set':
            Y_pred_test = ml.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info(r2_score(Y_test, Y_pred_test))

            st.write('Error (MSE or MAE):')
            st.info(mean_squared_error(Y_test, Y_pred_test))

        elif stat == '2.3. Scatter Plot':
            Y_pred_test = ml.predict(X_test)
            fig = plt.figure()
            plt.scatter(Y_test, Y_pred_test)
            plt.xlabel('Actual Data')
            plt.ylabel('Predicted Data')
            plt.grid()
            plt.show()
            st.pyplot(fig)

    # ---------------------------------#
    st.write("""
        # Simple/ Multiple Linear Regression
        In this implementation, the *LinearRegression()* function is used to build a **Linear Regression Model**.
                """)

    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    # ---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')
    # st.selectbox('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Sample Dataset'):
            df = pd.read_csv(
                'https://raw.githubusercontent.com/nzawad/accident_severity_sample/main/npvproject_concrete_cleaned.csv')

            st.markdown('The Compressive Strength of Concrete dataset is used as an example.')
            st.write(df.head(5))

            build_model(df)
    # end of Linear Regression

def data_Logistic_Regression():
    # Model building
    def build_model(df):
        X = df.iloc[:, :-1]  # Using all column except for the last column as X
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100, random_state=0)

        stat = st.selectbox('Info of Dataframe:',
                            ('1.1. Overview of Dataset', '1.2. Data splits', '1.3. Variable details',
                             '1.4. Correlation'))

        if stat == '1.1. Overview of Dataset':
            # st.markdown('**1.1. Overview of Dataset**')
            st.write('Data size')
            st.info(df.shape)

        # st.markdown('**1.2. Data splits**')
        elif stat == '1.2. Data splits':
            st.write('Training set')
            st.info(X_train.shape)
            st.write('Test set')
            st.info(X_test.shape)

        # st.markdown('**1.3. Variable details**:')
        elif stat == '1.3. Variable details':
            st.write('X variable')
            st.info(list(X.columns))
            st.write('Y variable')
            st.info(Y.name)


        # st.markdown('**1.4. Correlation**:')
        elif stat == '1.4. Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())

        lr = LogisticRegression(max_iter=12000)
        lr = lr.fit(X_train, Y_train)

        st.subheader('2. Model Performance')
        stat = st.selectbox('Info of Dataframe:',
                            ('2.1. Training set', '2.2. Test set', '2.3. Confusion Matrix', '2.4. ROC Curve',
                             '2.5. Precision-Recall Curve', '2.6. Feature Importance'))
        # st.markdown('**2.1. Training set**')
        if stat == '2.1. Training set':
            st.write('Coefficient of determination:')
            st.info(lr.score(X_train, Y_train))

        # st.markdown('**2.2. Test set**')
        elif stat == '2.2. Test set':
            st.write('Coefficient of determination:')
            st.info(lr.score(X_test, Y_test))

        elif stat == '2.3. Confusion Matrix':
            class_names = ['0', '1']
            cnf_matrix = confusion_matrix(y_true=Y_test, y_pred=lr.predict(X_test))
            fig, ax = plt.subplots()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            # create heatmap
            sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
            ax.xaxis.set_label_position("top")
            plt.tight_layout()
            plt.title('Confusion matrix', y=1.1)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)

        elif stat == '2.4. ROC Curve':
            fig, ax = plt.subplots()
            st.subheader("ROC Curve")
            plot_roc_curve(lr, X_test, Y_test, ax=ax)
            st.pyplot(fig)

        elif stat == '2.5. Precision-Recall Curve':
            fig, ax = plt.subplots()
            plot_precision_recall_curve(lr, X_test, Y_test, ax=ax)
            st.pyplot(fig)

        elif stat == '2.6. Feature Importance':
            fig, ax = plt.subplots()
            lr.fit(X, Y)
            odds = np.exp(lr.coef_[0])
            pd.DataFrame(odds,
                         X.columns,
                         columns=['coef']) \
                .sort_values(by='coef', ascending=False)

            # https://stackoverflow.com/a/66750803/19886681

            feat_importances = pd.Series(odds, index=list(X.columns))
            feat_importances.nlargest(30).plot(kind='barh', title='Feature Importance', alpha=.9, figsize=(15, 10))

            # set parameters for tick labels
            plt.tick_params(axis='y', which='major', labelsize=10)
            plt.grid()
            st.pyplot(fig)

    # ---------------------------------#
    st.write("""
        # Logistic Regression
        In this implementation, the *LogisticRegression()* function is used to build a **Logistic Regression Model**.
                """)

    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    # ---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')
    # st.selectbox('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Sample Dataset'):
            df = pd.read_csv(
                'https://raw.githubusercontent.com/nzawad/accident_severity_sample/main/accident_severity_cleaned_numeric.csv')

            st.markdown('Accident Severity dataset is used as an example.')
            st.write(df.head(5))

            build_model(df)
    # end of Logistic Regression

def data_Random_Forest_rg():
    # ---------------------------------#
    st.write("""
    # Random Forest (Regressor)
    In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm. The main focus is on **Hyperparameter Optimization**.
    """)

    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.sidebar.header('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.subheader('Learning Parameters')
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
    st.sidebar.write('---')
    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1, 3), 1)
    st.sidebar.number_input('Step size for max_features', 1)
    st.sidebar.write('---')
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)',
                                                   options=[True, False])
    parameter_oob_score = st.sidebar.select_slider(
        'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1] + parameter_n_estimators_step,
                                   parameter_n_estimators_step)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1] + 1, 1)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    # ---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('Dataset')

    # ---------------------------------#
    # Model building

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
        return href

    def build_model(df):
        X = df.iloc[:, :-1]  # Using all column except for the last column as X
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        st.markdown('A model is being built to predict the following **Y** variable:')
        st.info(Y.name)

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100, random_state=0)
        # X_train.shape, Y_train.shape
        # X_test.shape, Y_test.shape

        rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                   random_state=parameter_random_state,
                                   max_features=parameter_max_features,
                                   criterion=parameter_criterion,
                                   min_samples_split=parameter_min_samples_split,
                                   min_samples_leaf=parameter_min_samples_leaf,
                                   bootstrap=parameter_bootstrap,
                                   oob_score=parameter_oob_score,
                                   n_jobs=parameter_n_jobs)

        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        grid.fit(X_train, Y_train)

        st.subheader('Model Performance')

        Y_pred_test = grid.predict(X_test)
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_test, Y_pred_test))

        st.write('Error (MSE or MAE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))

        st.write("The best parameters are %s with a score of %0.2f"
                 % (grid.best_params_, grid.best_score_))

        # -----Process grid data-----#
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),
                                  pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
        # Segment data into groups based on the 2 hyperparameters
        grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
        # Pivoting the data
        grid_reset = grid_contour.reset_index()
        grid_reset.columns = ['max_features', 'n_estimators', 'R2']
        grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
        x = grid_pivot.columns.levels[1].values
        y = grid_pivot.index.values
        z = grid_pivot.values

        # -----Plot-----#
        layout = go.Layout(
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text='n_estimators')
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text='max_features')
            ))
        fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
        fig.update_layout(title='Hyperparameter tuning',
                          scene=dict(
                              xaxis_title='n_estimators',
                              yaxis_title='max_features',
                              zaxis_title='R2'),
                          autosize=False,
                          width=800, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig)

        # -----Save grid data-----#
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        z = pd.DataFrame(z)
        df = pd.concat([x, y, z], axis=1)
        st.markdown(filedownload(grid_results), unsafe_allow_html=True)

    # ---------------------------------#
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            df = pd.read_csv(
                'https://raw.githubusercontent.com/nzawad/accident_severity_sample/main/npvproject_concrete_cleaned.csv')

            st.markdown('The Compressive Strength of Concrete dataset is used as an example.')
            st.write(df.head(5))

            build_model(df)

    #end of RF_reg

def data_Random_Forest_cls():
    # ---------------------------------#
    st.write("""
        # Random Forest (Classifier)
        In this implementation, the *RandomForestClassifier()* function is used in this app for build a classification model using the **Random Forest** algorithm. The main focus is on **Feature Selection**.
        """)

    ######################################
    ## Sidebar
    ######################################
    # Input your csv
    st.sidebar.header('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step=1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    metrics = st.sidebar.multiselect("What metrics to plot?",
                                     ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    ######################################
    # Main panel
    ######################################
    st.subheader('Dataset')

    ######################################
    def build_model(df):
        x = df.iloc[:, :-1]  # Using all column except for the last column as X
        y = df.iloc[:, -1]  # Selecting the last column as Y

        # Data splitting
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100 - split_size) / 100, random_state=0)

        def impPlot(imp, name):
            figure = px.bar(imp,
                            x=imp.values,
                            y=imp.keys(), labels={'x': 'Importance Value', 'index': 'Columns'},
                            text=np.round(imp.values, 2),
                            title=name + ' Feature Selection Plot (RF_classifier)',
                            width=1000, height=600)
            figure.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            st.plotly_chart(figure)

        model = RandomForestClassifier()
        model.fit(x, y)
        feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
        impPlot(feat_importances, 'Random Forest Classifier')
        # st.write(feat_importances)
        st.write('\n')

        def plot_metrics(metrics_list):
            st.subheader('Model Performance')
            model1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                            n_jobs=-1)
            model1.fit(x_train, y_train)
            y_pred_test = model1.predict(x_test)
            if "Confusion Matrix" in metrics_list:
                st.subheader("Confusion Matrix")
                cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
                fig, ax = plt.subplots()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names)
                plt.yticks(tick_marks, class_names)
                # create heatmap
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Confusion matrix', y=1.1)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                st.pyplot(fig)
            if "ROC Curve" in metrics_list:
                st.subheader("ROC Curve")
                plot_roc_curve(model1, x_test, y_test)
                st.pyplot()
            if "Precision-Recall Curve" in metrics_list:
                st.subheader("Precision-Recall Curve")
                plot_precision_recall_curve(model1, x_test, y_test)
                st.pyplot()

        class_names = ['0', '1']

        plot_metrics(metrics)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(5))
        build_model(df)

    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            df = pd.read_csv(
                'https://raw.githubusercontent.com/nzawad/accident_severity_sample/main/accident_severity_cleaned_numeric.csv')
            st.markdown('The **Accident Severity** dataset is used as the example.')
            st.write(df.head(5))
            build_model(df)

    # end of RF_cls

def data_SVM():
    #beginning svm

    #st.set_page_config(layout="wide")
    st.header('Support Vector Machine (SVM)')
    st.write("""
            In this implementation, the *SVC()* function is used to build a **SVM Model**.
                    """)
    st.text("")
    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        classifier_name = st.sidebar.radio('Select kernel: ', ('rbf', 'poly', 'sigmoid', 'linear'))
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

    def print_score(clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
            st.write('Train Result:n================================================')
            st.write(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            st.write("_______________________________________________")
            st.write('CLASSIFICATION REPORT')
            st.table(clf_report)
            st.write("_______________________________________________")
            st.write(f"Confusion Matrix: n {confusion_matrix(y_train, pred)}n")
        elif train == False:
            pred = clf.predict(X_test)
            clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
            st.write("Test Result:n================================================")
            st.write(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            st.write("_______________________________________________")
            st.write('CLASSIFICATION REPORT')
            st.table(clf_report)
            st.write("_______________________________________________")
            st.write(f"Confusion Matrix: n {confusion_matrix(y_test, pred)}n")

    def get_dataset(dataset_name):
        if dataset_name == 'Breast Cancer':
            data = datasets.load_breast_cancer()
        elif dataset_name == 'Iris':
            data = datasets.load_iris()
        else:
            data = datasets.load_digits()
        X = data.data
        y = data.target
        return data, X, y

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))

        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))
        C0 = st.slider('C', 0.01, 10.00, step=0.1)
        clf = SVC(kernel=classifier_name, C=C0, gamma=gamma)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],
                          'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
                          'kernel': ['rbf', 'poly', 'linear']}
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)

        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.colorbar()
        st.pyplot(fig)

    else:
        st.write("""
                **Upload CSV file or use Sample Datasets from below:**
                        """)
        dataset_name = st.selectbox('Select Sample Dataset: ',
                                    ('Breast Cancer', 'Iris', 'Digits'))
        data, X, y = get_dataset(dataset_name)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write("You select prepared data.")
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))
        C0 = st.slider('C', 0.01, 10.00, step=0.1)
        clf = SVC(kernel=classifier_name, C=C0, gamma=gamma)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy Score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],
                          'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
                          'kernel': ['rbf', 'poly', 'linear']}
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)

        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.colorbar()
        st.pyplot(fig)

    #end svm

def data_NN():
    st.header('Neural Network')
    st.write("""
            In this implementation, the *MLPClassifier()* function is used to build a **NN Model**.
                    """)
    st.text("")
    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        C0 = st.slider('max_iter', 1, 10000)
        CA = st.slider('hidden_layer_sizes', 1, 1000)
        activation_unit = st.sidebar.radio('Select activation function for the hidden layer: ',
                                           ('identity', 'logistic', 'tanh', 'relu'))

    def print_score(clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
            st.write('Train Result:n================================================')
            st.write(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            st.write("_______________________________________________")
            st.write('CLASSIFICATION REPORT')
            st.table(clf_report)
            st.write("_______________________________________________")
            st.write(f"Confusion Matrix: n {confusion_matrix(y_train, pred)}n")
        elif train == False:
            pred = clf.predict(X_test)
            clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
            st.write("Test Result:n================================================")
            st.write(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            st.write("_______________________________________________")
            st.write('CLASSIFICATION REPORT')
            st.table(clf_report)
            st.write("_______________________________________________")
            st.write(f"Confusion Matrix: n {confusion_matrix(y_test, pred)}n")

    def get_dataset(dataset_name):
        if dataset_name == 'Breast Cancer':
            data = datasets.load_breast_cancer()
        elif dataset_name == 'Iris':
            data = datasets.load_iris()
        else:
            data = datasets.load_digits()
        X = data.data
        y = data.target
        return data, X, y

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))

        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))

        clf = MLPClassifier(hidden_layer_sizes=CA, activation=activation_unit, learning_rate_init=0.1, max_iter=C0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'max_iter': [1, 100, 1000, 10000],
                          'hidden_layer_sizes': [1, 5, 10, 50, 100, 200, 500, 1000],
                          'activation': ['identity', 'logistic', 'tanh', 'relu']}
            grid = GridSearchCV(MLPClassifier(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)



    else:
        st.write("""
                **Upload CSV file or use Sample Datasets from below:**
                        """)
        dataset_name = st.selectbox('Select Sample Dataset: ',
                                    ('Breast Cancer', 'Iris', 'Digits'))
        data, X, y = get_dataset(dataset_name)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write("You select prepared data.")
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))

        clf = MLPClassifier(hidden_layer_sizes=CA, activation=activation_unit, learning_rate_init=0.1, max_iter=C0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy Score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'max_iter': [1, 100, 1000, 10000],
                          'hidden_layer_sizes': [1, 5, 10, 50, 100, 200, 500, 1000],
                          'activation': ['identity', 'logistic', 'tanh', 'relu']}
            grid = GridSearchCV(MLPClassifier(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)

def data_KNN():
    st.header('K-Nearest Neighbor')
    st.write("""
        In this implementation, the *KNeighborsClassifier()* function is used to build a **KNN Model**.
                """)
    st.text("")
    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        C0 = st.slider('No. of neighbours', 1, 100, 5, 1)
        CA = st.sidebar.slider('distance: Manhattan (1), Euclidean (2) ', 1, 2, 1, 2)
        weight = st.sidebar.radio('how weights will be measured for data points: ',
                                  ('uniform', 'distance'))

    def get_dataset(dataset_name):
        if dataset_name == 'Breast Cancer':
            data = datasets.load_breast_cancer()
        elif dataset_name == 'Iris':
            data = datasets.load_iris()
        else:
            data = datasets.load_digits()
        X = data.data
        y = data.target
        return data, X, y

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))

        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))

        clf = KNeighborsClassifier(n_neighbors=C0, weights=weight, metric='minkowski', p=CA)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'n_neighbors': [1, 5, 10, 15, 20, 50, 100, 1000],
                          'weights': ['uniform', 'distance'],
                          'p': [1, 2],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        # error rate
        error_rate = []
        # Will take some time
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
                markersize=10)
        ax.xaxis.set_label_position("top")
        ax.set_title('Error Rate vs. K Value\n\n')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.grid()
        st.pyplot(fig)

    else:
        st.write("""
                **Upload CSV file or use Sample Datasets from below:**
                        """)
        dataset_name = st.selectbox('Select Sample Dataset: ',
                                    ('Breast Cancer', 'Iris', 'Digits'))
        data, X, y = get_dataset(dataset_name)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write("You select prepared data.")
        stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))

        clf = KNeighborsClassifier(n_neighbors=C0, weights=weight, metric='minkowski', p=CA)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy Score = {acc:.5f}')
        st.write('Classification Report')
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        st.table(report)

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)

        if st.button('Press to run Hyper-parameter tuning'):
            st.write("""
                    **Hyper-parameter tuning**
                            """)
            param_grid = {'n_neighbors': [1, 5, 10, 15, 20, 50, 100, 1000],
                          'weights': ['uniform', 'distance'],
                          'p': [1, 2],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            st.write(f"Best parameters: {best_params}")

        # error rate
        error_rate = []
        # Will take some time
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
                markersize=10)
        ax.xaxis.set_label_position("top")
        ax.set_title('Error Rate vs. K Value\n\n')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.grid()
        st.pyplot(fig)
    #end of KNN

# Add a title and intro text
#st.title('Machine Learning')
#st.text('This is a web app to allow exploration of Machine Learning')


# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select ML Model:', ['Cover', 'Linear Regression', 'Logistic Regression', 'Random Forest (Regressor)', 'Random Forest (Classifier)', 'SVM', 'Neural Network', 'K-Nearest Neighbor'])

# Navigation options
if options == 'Cover':
    data_Cover()
elif options == 'Linear Regression':
    data_Linear_Regression()
elif options == 'Logistic Regression':
    data_Logistic_Regression()
elif options == 'Random Forest (Regressor)':
    data_Random_Forest_rg()
elif options == 'Random Forest (Classifier)':
    data_Random_Forest_cls()
elif options == 'SVM':
    data_SVM()
elif options == 'Neural Network':
    data_NN()
elif options == 'K-Nearest Neighbor':
    data_KNN()
