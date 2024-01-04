import streamlit as st
import pandas as pd
from visualization import EasyVisualize
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

@st.cache_data
def load_data():
    table = pd.read_excel('Titanic_dataset.xlsx')
    return table


class Encoder(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X =pd.get_dummies(X, columns=['pclass', 'embarked', 'title', 'family_size', 'sex'], dtype=int)
                return X

def business_understanding():
    st.write('## 1. Business Understanding')
    st.write(
        """
        Our objective is to predict passenger survival using the Titanic dataset, which is sourced from Kaggle and widely utilized in data science and machine learning applications. 

        Source: https://www.kaggle.com/datasets/marouandaghmoumi/titanic-dataset.

        The Titanic dataset is a popular dataset in the field of data science and machine learning. It contains information about the passengers aboard the RMS Titanic, which sank on its maiden voyage in 1912 after hitting an iceberg. The dataset is often used for predictive modeling and classification tasks.

        Here are the key features or columns in the Titanic dataset:

        - **PassengerId**: A unique identifier assigned to each passenger.
        - **Survived**: A binary variable indicating whether the passenger survived (1) or did not survive (0).
        - **Pclass (Passenger Class)**: The ticket class of the passenger, which can be 1st (1), 2nd (2), or 3rd (3) class.
        - **Name**: The name of the passenger.
        - **Sex**: The gender of the passenger (male or female).
        - **Age**: The age of the passenger in years. It may contain missing values.
        - **SibSp**: The number of siblings or spouses the passenger had aboard the Titanic.
        - **Parch**: The number of parents or children the passenger had aboard the Titanic.
        - **Ticket**: The ticket number.
        - **Fare**: The amount of money the passenger paid for the ticket.
        - **Cabin**: The cabin number where the passenger stayed. It may contain missing values.
        - **Embarked**: The port at which the passenger boarded the Titanic (C for Cherbourg, Q for Queenstown, S for Southampton).
        """
    )

def data_understanding(df):
    st.write('## 2. Data Understanding')
    # Extract title from the 'name' column
    df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Create a family_size column
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['family_size'] = df['family_size'].astype('object')
    # Drop unecessary columns
    df.drop(['name', 'sibsp', 'parch', 'ticket','cabin', 'body', 'home.dest', 'boat'], axis=1, inplace=True)
    with st.expander('Univariate Analysis', expanded=False):
        # Initialize visualization library
        vis = EasyVisualize(df)
        # Define graph type
        vis_dict = {
        'pclass':'count',
        'survived':'count',
        'sex':'count',
        'fare':'dist',
        'embarked':'count',
        'family_size':'count'
        }
        fig = vis.create_mixed_subplots_st(vis_dict, 3, 2)
        st.write(
            '''
                - **pclass**: indicates that the majority of passengers were in class 3, implying that a significant proportion of individuals belonged to the lower class.
                - **survived**: it reveals that the majority of passengers did not survive the incident.
                - **sex**: the data suggests that the majority of passengers were male.
                - **fare**: it becomes evident that most passengers paid a relatively low fare for their tickets, which aligns with the information from 'pclass.'
                - **embarked**: the data points out that the majority of passengers embarked from Southampton.
                - **family_size**: highlights that most passengers were traveling alone.

            '''
        )
        st.pyplot(fig)
    with st.expander('Bivariate Analysis', expanded=False):
        vis_dict = {
            'pclass':'count',
            'sex':'count',
            'embarked':'count',
            'family_size':'count'
        }
        fig = vis.create_mixed_subplots_st(vis_dict, 2, 2, hue='survived')
        st.write(
            '''
                **Analysis of Variables in Relation to 'Survived'** 
                - **pclass**: There is a notable correlation between passenger class and survival rates, with a clear trend of higher-class passengers having better odds of survival.
                - **sex**: An observable pattern emerges as the "women and children first" protocol appears to have been applied during the incident, resulting in a significantly higher likelihood of survival for female passengers.
                - **embarked**: Passengers who embarked from Cherbourg exhibit a higher survival rate, possibly attributed to a greater proportion of 1st class ticket holders originating from this location.
                - **family_size**: Interestingly, passengers traveling in groups of 2 to 4 individuals not only had better survival chances but also had the highest proportion of 1st class tickets, which may have contributed to their increased likelihood of survival.
            '''
        )
        st.pyplot(fig)
        fig2 = vis.plot_count_st('embarked', hue='pclass')
        st.pyplot(fig2)
        fig3 = vis.plot_count_st('family_size', hue='pclass')
        st.pyplot(fig3)
    with st.expander('Conclusion', expanded=True):
        st.write(
            '''
                ##### Passenger Demographics and Fare
                - **Pclass**: A significant portion of the passengers were in the lower economic class (class 3).
                - **Sex**: The majority were male, suggesting a gender imbalance among the passengers.
                - **Fare**: Most passengers paid a low fare, consistent with the high number of third-class passengers.
                - **Embarked**: The majority embarked from Southampton, indicating a possible geographic concentration of the passengers.

                ##### Survival Analysis
                - **Pclass and Survival**: There is a strong correlation between a passenger's class and their survival chances, with first-class passengers having a higher survival rate.
                - **Gender and Survival**: Women had a higher chance of survival, aligning with the historical accounts of the "women and children first" protocol.
                - **Embarkation Point and Survival**: Passengers who embarked from Cherbourg showed a higher survival rate, potentially due to a higher number of first-class passengers from this location.
                - **Family Size and Survival**: Traveling in small groups (2-4 people) seems to have been advantageous for survival. This group not only had better survival rates but also a higher proportion of first-class tickets.

                **Overall**, the data suggests that socio-economic status (reflected in passenger class and fare), gender, embarkation point, and family size were significant factors in the survival of passengers during the incident. This analysis reflects the complex interplay of social, economic, and demographic factors in determining outcomes in such a catastrophic event.

            '''
        )
    
def data_preparation():
    st.write('## 3. Data Preparation')
    with st.expander('Data Prep Description',expanded=False):
        st.write(
            '''
                This data preprocessing pipeline aims to prepare the dataset for further analysis or modeling. It involves a series of structured steps:

                #### 1. Data Type Conversion

                - Convert the 'pclass' column data type to 'object'.

                #### 2. Feature Extraction

                - Extract titles from the 'name' column.

                #### 3. Feature Engineering

                - Create a new 'family size' column by summing 'sibsp', 'parch', and 1.
                - Convert the 'family size' column data type to 'object'.

                #### 4. Feature Selection

                - Drop the following columns: 'name', 'sibsp', 'parch', 'ticket', 'cabin', 'boat', 'body', and 'home.dest'.

                #### 5. Missing Data Handling

                - Impute missing ages with the median.
                - Impute missing fare values with the mode.
                - Impute missing embarked values with the mode.

                #### 6. Encoding Categorical Features

                - Encode the categorical columns 'pclass', 'embarked', 'title', 'family_size', and 'sex' into dummy variables.

                #### 7. Feature Scaling

                - Scale the 'age' and 'fare' columns using StandardScaler.

                This comprehensive preprocessing pipeline ensures that the dataset is appropriately transformed and cleaned, making it suitable for subsequent data analysis or modeling tasks.

            '''
        )
    with st.expander('Imports', expanded=False):
        st.code('''
                    import preprocessing as pp
                    from sklearn.base import BaseEstimator, TransformerMixin
                    from sklearn.impute import SimpleImputer
                    from sklearn.preprocessing import StandardScaler, OneHotEncoder
                    from sklearn.pipeline import Pipeline, make_pipeline
                    from sklearn.compose import ColumnTransformer''', 'python')
    with st.expander('Custom Pipelines Definition', expanded=False):
        code = '''
                    class TitleExtractor(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X['title'] = X['name'].str.extract(' ([A-Za-z]+)\.', expand=False).to_frame()
                            return X

                    class FamilySizeCreator(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X= X.assign(family_size=X['sibsp'] + X['parch'] + 1)
                            return X

                    class ColumnDropper(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X =  X.drop(['name', 'sibsp', 'parch', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1)
                            return X

                    class AgeImputer(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            imputer = SimpleImputer(strategy='median')
                            X['age'] = imputer.fit_transform(X[['age']])
                            return X

                    class FareImputer(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            imputer = SimpleImputer(strategy='most_frequent')
                            X['fare'] = imputer.fit_transform(X[['fare']])
                            X['fare'] = X['fare'].replace(0, X['fare'].median())
                            return X

                    class EmbarkedImputer(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            mode_value = X['embarked'].mode()[0]
                            X['embarked'].fillna(mode_value, inplace=True)
                            return X

                    class PclassDtype(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X['pclass'] = X['pclass'].astype('object')
                            return X

                    class FamilySizeDtype(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X['family_size'] = X['family_size'].astype('object')
                            return X

                    class Encoder(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            return self

                        def transform(self, X):
                            X =pd.get_dummies(X, columns=['pclass', 'embarked', 'title', 'family_size', 'sex'], dtype=int)
                            return X

                    class Scaler(BaseEstimator, TransformerMixin):
                        def fit(self, X, y=None):
                            self.scaler = StandardScaler()
                            self.scaler.fit(X[['age', 'fare']])
                            return self

                        def transform(self, X):
                            X[['age', 'fare']] = self.scaler.transform(X[['age', 'fare']])
                            return X
                '''
        st.code(code, 'python')
    with st.expander('Pipeline Definition', expanded=False):
        code2 = '''
                    preprocessing = Pipeline(
                        [
                            ('pclass_dtype', PclassDtype()),
                            ('title_extractor', TitleExtractor()),
                            ('family_size_creator', FamilySizeCreator()),
                            ('family_size_dtype', FamilySizeDtype()),
                            ('column_dropper', ColumnDropper()),
                            ('age_imputer', AgeImputer()),
                            ('fare_imputer', FareImputer()),
                            ('embarked_imputer', EmbarkedImputer()),
                            ('encoder', Encoder()),
                            ('scaler', Scaler())
                        ]
                    )
                '''
        st.code(code2, 'python')

def modeling():
    st.write('## 4. Modeling')
    with st.expander('Modeling Methodology', expanded=True):
        st.write('''
            In the modeling phase, I applied a diverse set of machine learning algorithms to tackle the predictive task. These included:

            - **Random Forest**: Utilized for its versatility and ability to harness multiple decision trees for robust predictions.

            - **Gradient Boosting**: Employed this boosting technique to combine weak learners into a strong model, often resulting in high predictive accuracy.

            - **Logistic Regression**: Leveraged for its simplicity and interpretability, especially suitable for binary classification tasks.

            - **Support Vector Machine (SVM)**: Utilized this powerful classification algorithm to identify an optimal hyperplane for effective data separation.

            - **Neural Network**: Implemented a neural network, likely a deep learning architecture, capable of capturing intricate data relationships.

            Additionally, to fine-tune the performance of these models and maximize predictive accuracy, I employed **GridSearch**. GridSearch is a hyperparameter tuning technique that systematically explores a predefined set of hyperparameters, ensuring that the models are optimized for the given dataset.

            This comprehensive approach demonstrates a commitment to achieving the best possible predictive performance by leveraging a variety of algorithms and optimizing their performance through hyperparameter tuning.
        ''')
    with st.expander('Imports', False):
        st.code('''
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.neural_network import MLPClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.model_selection import GridSearchCV''', 'python')
    with st.expander('Preprocessing', False):
        st.code('''
                X= preprocessing.fit_transform(titanic_df.drop('survived', axis=1))
                y = titanic_df['survived']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)''', 'python')
    with st.expander('Model Pipeline', False):
        st.code('''
                model_pipelines = {
                'rf':make_pipeline(RandomForestClassifier(random_state=27)),
                'gb':make_pipeline(GradientBoostingClassifier(random_state=27)),
                'lr':make_pipeline(LogisticRegression(random_state=27)),
                'svm':make_pipeline(SVC(random_state=27)),
                'nlp':make_pipeline(MLPClassifier(random_state=27))
                }''', 'python')
    with st.expander('Hypertune Definition', False):
        st.code('''
                hypergrid = {
                'lr': {
                    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'logisticregression__solver': ['lbfgs', 'sag', 'saga']
                },
                'rf': {
                    'randomforestclassifier__n_estimators': [10, 50, 100, 200],
                    'randomforestclassifier__max_depth': [None, 10, 20, 30, 40],
                    'randomforestclassifier__min_samples_split': [2, 5, 10],
                    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
                },
                'gb': {
                    'gradientboostingclassifier__n_estimators': [100, 200, 300],
                    'gradientboostingclassifier__learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'gradientboostingclassifier__max_depth': [3, 5, 7, 9]
                },
                'svm': {
                    'svc__C': [0.1, 1, 10, 100],
                    'svc__gamma': [1, 0.1, 0.01, 0.001],
                    'svc__kernel': ['rbf']
                },
                'nlp': {
                    'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                    'mlpclassifier__activation': ['tanh', 'relu'],
                    'mlpclassifier__solver': ['sgd', 'adam'],
                    'mlpclassifier__alpha': [0.0001, 0.05],
                    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
                }
                }''', 'python')
    with st.expander('GridSearch with Hypertune Execution', False):
        st.code('''
                fit_models = {}
                for m, pipeline in model_pipelines.items():
                    model = GridSearchCV(pipeline, hypergrid[m], cv=10, n_jobs=-1, scoring='accuracy')
                    try:
                        print(f'Start training for {m}')
                        model.fit(X_train, y_train)
                        fit_models[m] = model
                        print(f'{m} has been successfully fit.')
                    except Exception as e:
                        print(f"An error occurred while fitting {m}: {repr(e)}")''', 'python')

def evaluation():
    st.write('## 5. Evaluation')
    with st.expander('Evaluation Methodology', True):
        st.write('''
                 **Model Evaluation**

                    To evaluate the performance of the models, I employed the following metrics:

                    - **Classification Report**: This report provides a comprehensive overview of precision, recall, F1-score, and support for each class within the classification task.

                    - **ROC_AUC Score**: ROC (Receiver Operating Characteristic) Area Under the Curve (AUC) is a valuable metric for binary classification tasks. It measures the model's ability to distinguish between classes, with a higher value indicating better performance.

                    - **Log Loss**: Logarithmic Loss (log loss) measures the accuracy of probability predictions. A lower log loss indicates more accurate predictions.

                    After thorough evaluation, it was determined that the Random Forest model outperformed the others. It achieved the highest ROC_AUC score and the lowest log loss, indicating superior predictive performance.
                    ''')
    with st.expander('Imports', False):
        st.code('''
                from sklearn.metrics import classification_report
                from sklearn.metrics import roc_auc_score, log_loss''', 'python')
    with st.expander('Classification Report', False):
        st.code('''
                for m, model in fit_models.items():
                yhat = model.predict(X_test)
                print(m, classification_report(y_test, yhat))''', 'python')
    with st.expander('Metrics Evaluation - ROC-AUC/LOG LOSS', False):
        st.code('''
                for m, model in fit_models.items():
                    try:
                        y_pred = model.predict(X_test)
                        roc_auc = roc_auc_score(y_test, y_pred)
                        logloss = log_loss(y_test, y_pred)
                        print(f"{m} - ROC-AUC: {roc_auc}, Log Loss: {logloss}")
                    except Exception as e:
                        print(f"An error occurred while evaluating {m}: {repr(e)}")''', 'python')

def prediction(scaler):
       
    def load_model():
        # Load your trained model from a .pkl file
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    
    st.write('## 6. Prediction Playground')

    # Load the model
    model = load_model()

    title = [
            'Miss',
            'Master',
            'Mr',
            'Mrs',
            'Col',
            'Mme',
            'Dr',
            'Major',
            'Capt',
            'Lady',
            'Sir',
            'Mlle',
            'Dona',
            'Jonkheer',
            'Countess',
            'Don',
            'Rev',
            'Ms'
        ]

    with st.form("passenger_form"):
        
        preprocessing = Pipeline(
                [
                    ('encoder', Encoder())
                ]
            )

        # Form fields (excluding 'Survived')
        pclass = st.selectbox("Pclass", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
        fare = st.number_input("Fare", min_value=0.0, step=0.1)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
        title = st.selectbox("Title", title)
        family_size = st.number_input("Family Size", min_value=1, max_value=10)

        # Form submission button
        submitted = st.form_submit_button("Submit")

        expected_columns = ['age',
                        'fare',
                        'pclass_1',
                        'pclass_2',
                        'pclass_3',
                        'embarked_C',
                        'embarked_Q',
                        'embarked_S',
                        'title_Capt',
                        'title_Col',
                        'title_Countess',
                        'title_Don',
                        'title_Dona',
                        'title_Dr',
                        'title_Jonkheer',
                        'title_Lady',
                        'title_Major',
                        'title_Master',
                        'title_Miss',
                        'title_Mlle',
                        'title_Mme',
                        'title_Mr',
                        'title_Mrs',
                        'title_Ms',
                        'title_Rev',
                        'title_Sir',
                        'family_size_1',
                        'family_size_2',
                        'family_size_3',
                        'family_size_4',
                        'family_size_5',
                        'family_size_6',
                        'family_size_7',
                        'family_size_8',
                        'family_size_11',
                        'sex_female',
                        'sex_male']

        if submitted:
            # Create DataFrame from the input data
            data = pd.DataFrame({
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'fare': [fare],
                'embarked': [embarked],
                'title': [title],
                'family_size': [family_size]
            })
            data = preprocessing.transform(data)
            data[['age','fare']] = scaler.transform(data[['age','fare']])
            expected_data = pd.DataFrame(columns=expected_columns)
            for col in expected_columns:
                if col in data.columns:
                    expected_data[col] = data[col]
                else:
                    expected_data[col] = 0
            # Make prediction
            prediction = model.predict(expected_data)

            # Display the result
            result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
            st.write(f"The model predicts that the passenger would have: {result}")