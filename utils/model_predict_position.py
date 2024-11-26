import pandas as pd
import numpy as np
from flask import jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
# from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from prettytable import PrettyTable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import missingno as msno
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from nltk.stem.snowball import SnowballStemmer


from utils.pickle_utils import load_pickle, save_pickle


pickle_directory = 'pickle_files' 
emp = pd.read_csv("./uploads/employees_info.csv", skipinitialspace = True, sep=",")
positions = pd.read_csv("./uploads/positions_info.csv", skipinitialspace = True, sep=",")
new_employees = pd.read_csv("./uploads/new_employees.csv", skipinitialspace = True, sep=",")


pickle_directory = 'pickle_files' 


def get_employee_info_chart():
    null=emp.isnull().sum().sum()
    notNull=len(emp)-null

    y = np.array([null,notNull])
    mylabels = ["Missing", "Present"]
    total = sum(y)
    percentages = [(value / total * 100).round(1) for value in y]
    return jsonify({'labels': mylabels, 'data': percentages})


def generate_employee_msno_chart(filepath):
    msno.bar(emp, color = 'y', figsize = (10,8))
    employee_msno_chart_path = filepath + '/employee_msno_chart.png'
    employee_msno_chart = msno.bar(emp, color='y', figsize=(16, 16))
    employee_msno_chart.figure.savefig(employee_msno_chart_path)
    plt.show()
 
def get_employee_msno_chart(filepath):
    employee_msno_chart_path = filepath + '/employee_msno_chart.png'
    return send_file(employee_msno_chart_path, mimetype='image/png') 
        
def get_all_positions():
    list = positions["Position Name"].dropna().unique().tolist()
    return jsonify(list)
        
# needed it in retrain
def generate_features_distribution(filepath):
    bar_categorical_feature_is_promoted_path = filepath + '/bar_categorical_feature_is_promoted.png'
    plt.figure(figsize=(5,5))
    sns.countplot(x='Is promoted', data=emp)
    plt.savefig(bar_categorical_feature_is_promoted_path)
    plt.close()

    pie_categorical_feature_is_promoted_path = filepath + '/pie_categorical_feature_is_promoted.png'
    emp["Is promoted"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_is_promoted_path)
    plt.close()
    
    bar_categorical_feature_education_path = filepath + '/bar_categorical_feature_education.png'
    plt.figure(figsize=(8,5))
    sns.countplot(x='Education Background', data=emp)
    plt.savefig(bar_categorical_feature_education_path)
    plt.close()

    pie_categorical_feature_education_path = filepath + '/pie_categorical_feature_education.png'
    emp["Education Background"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_education_path)
    plt.close()
    
    bar_categorical_feature_gender_path = filepath + '/bar_categorical_feature_gender.png'
    plt.figure(figsize=(8,5))
    sns.countplot(x='Gender', data=emp)
    plt.savefig(bar_categorical_feature_gender_path)
    plt.close()

    pie_categorical_feature_gender_path = filepath + '/pie_categorical_feature_gender.png'
    emp["Gender"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_gender_path)
    plt.close()
    
    bar_categorical_feature_position_level_path = filepath + '/bar_categorical_feature_position_level.png'
    plt.figure(figsize=(8,5))
    sns.countplot(x='Position Level', data=emp)
    plt.savefig(bar_categorical_feature_position_level_path)
    plt.close()

    pie_categorical_feature_position_level_path = filepath + '/pie_categorical_feature_position_level.png'
    emp["Position Level"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_position_level_path)
    plt.close()

    bar_categorical_feature_profile_path = filepath + '/bar_categorical_feature_profile.png'
    plt.figure(figsize=(15,5))
    sns.countplot(x='Profile', data=emp)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(bar_categorical_feature_profile_path)
    plt.close()

    pie_categorical_feature_profile_path = filepath + '/pie_categorical_feature_profile.png'
    emp["Profile"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_profile_path)
    plt.close()

    bar_categorical_feature_previous_position_path = filepath + '/bar_categorical_feature_previous_position.png'
    plt.figure(figsize=(15,5))
    sns.countplot(x='Previous Position', data=emp)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(bar_categorical_feature_previous_position_path)
    plt.close()

    pie_categorical_feature_previous_position_path = filepath + '/pie_categorical_feature_previous_position.png'
    emp["Previous Position"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(9, 9), startangle=0).legend()
    plt.savefig(pie_categorical_feature_previous_position_path)
    plt.close()


def get_bar_features_distribution(filepath, featurename):
    bar_categorical_feature_path = filepath + '/bar_categorical_feature_' + featurename +'.png'
    return send_file(bar_categorical_feature_path, mimetype='image/png')


def get_pie_features_distribution(filepath, featurename):
    pie_categorical_feature_path = filepath + '/pie_categorical_feature_'+ featurename +'.png'
    return send_file(pie_categorical_feature_path, mimetype='image/png')

def generate_numeric_features(filepath):
    numeric_feature_path = filepath +'/numeric_feature.png'
    # Visulazing the distibution of the data for every numerical feature
    emp.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
    plt.savefig(numeric_feature_path)
    plt.close()

def get_numeric_features(filepath):
     numeric_feature_path = filepath +'/numeric_feature.png'
     return send_file(numeric_feature_path, mimetype='image/png')

# needed it in retrain
def generate_target_variable_dependencies(filepath):
    target_variable_dependency_gender_is_promoted_path = filepath + '/target_variable_dependency_gender_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Gender"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_gender_is_promoted_path)
    plt.close()

    target_variable_dependency_age_is_promoted_path = filepath + '/target_variable_dependency_age_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Age"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_age_is_promoted_path)
    plt.close()

    target_variable_dependency_prev_is_promoted_path = filepath + '/target_variable_dependency_prev_is_promoted.png'
    plt.rcParams['figure.figsize'] = [20, 15]
    score_bin = pd.crosstab(emp["Previous Position"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_prev_is_promoted_path)
    plt.close()

    target_variable_dependency_education_is_promoted_path = filepath + '/target_variable_dependency_education_is_promoted.png'
    plt.rcParams['figure.figsize'] = [15, 10]
    score_bin = pd.crosstab(emp["Education Background"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_education_is_promoted_path)
    plt.close()

    target_variable_dependency_position_level_is_promoted_path = filepath + '/target_variable_dependency_position_level_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Position Level"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_position_level_is_promoted_path)
    plt.close()

    target_variable_dependency_year_service_is_promoted_path = filepath + '/target_variable_dependency_year_service_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Years Of Service"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_year_service_is_promoted_path)
    plt.close()

    target_variable_dependency_year_experience_is_promoted_path = filepath + '/target_variable_dependency_year_experience_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Years Of Experience"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_year_experience_is_promoted_path)
    plt.close()

    target_variable_dependency_nbr_promotion_is_promoted_path = filepath + '/target_variable_dependency_nbr_promotion_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["Number Of Promotions"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_nbr_promotion_is_promoted_path)
    plt.close()

    target_variable_dependency_nbr_project_is_promoted_path = filepath + '/target_variable_dependency_nbr_project_is_promoted.png'
    plt.rcParams['figure.figsize'] = [10, 5]
    score_bin = pd.crosstab(emp["number of projects"],emp["Is promoted"],normalize='index')
    score_bin.plot.bar(stacked=True)
    plt.legend(title='Is promoted',loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(target_variable_dependency_nbr_project_is_promoted_path)
    plt.close()
    

def get_target_dependency(filepath, dependencyname):
    pie_categorical_feature_path = filepath + '/target_variable_dependency_'+ dependencyname +'_is_promoted.png'
    return send_file(pie_categorical_feature_path, mimetype='image/png')


        
def generate_preprocessing_employee_position_data(filepath):
#Display the number of columns with missing values for each employee:
    preprocessing_position_path = filepath + '/preprocessing_position.png'

    emp_columns = emp.drop('EMP ID', axis=1)

    # Count null and not null values for each employee
    null_values = emp_columns.isnull().sum(axis=1)
    not_null_values = emp_columns.notnull().sum(axis=1)
    plt.figure(figsize=(20, 10))
    plt.bar(emp['EMP ID'], null_values, label='Null Values', color='red')
    plt.bar(emp['EMP ID'], not_null_values, bottom=null_values, label='Not null Values', color='black')

    plt.xlabel('Employee ID')
    plt.ylabel('Count')
    plt.title('Columns with missing values for each Employee')
    plt.legend()
    plt.xticks(emp['EMP ID'])

    plt.savefig(preprocessing_position_path)
    plt.close()
        
def get_preprocessing_employee_position_data(filepath):
    preprocessing_position_path = filepath + '/preprocessing_position.png'
    return send_file(preprocessing_position_path, mimetype='image/png') 

def get_position_missing_values():
    #  emp_columns = emp.drop('EMP ID', axis=1)
    # data = drop_target_point_column(file)
    my_list = []
    # Iterate through unique EMP IDs
    for employee in emp["EMP ID"].unique():
        # expert_data_list = []  # Initialize a list to store expert data for this employee
        # for expert in emps[emp["EMP ID"] == emps]:
        expert_dict = {
            **{
                column: 'Missing' if emp[column].isnull().any() else 'Present'
                for column in emp.columns
            },
            'emp_id': str(employee)
        }
        
        my_list.append(expert_dict)
    return  jsonify(my_list)

def get_missing_columns_position_value_name(): 
    null_columns=list()
    total_columns=emp.columns
    for employee in emp["EMP ID"].unique():
        test1=emp[emp["EMP ID"]==employee].isnull().sum().sum() >=1
        if (test1==True) :
            for col in total_columns:
                test2=emp[emp["EMP ID"]==employee][col].isnull().values[0]
                if (test2==True):
                    expert_dict = {
                    'column': col,
                    'emp_id': str(employee)
                }
                    null_columns.append(expert_dict)
    return jsonify(null_columns)

def ordinal_encoder_categrical_variables(filepath):
    correlation_path = filepath + '/correlation.png'
    #emp=emp[~emp['EMP ID'].isin(rows_to_drop)]
    categorical_columns=list(emp.drop("Profile",axis=1).select_dtypes(include='object').columns)
    d=emp.drop("Profile",axis=1)
    # Initialize an OrdinalEncoder instance
    ordinal_encoder = OrdinalEncoder()
    # Fit and transform the data using the OrdinalEncoder
    d[categorical_columns] = ordinal_encoder.fit_transform(emp[categorical_columns])
    #Apply same scale for all the variables
    columns=d.drop("Is promoted",axis=1).columns
    scaler = MinMaxScaler()
    d = scaler.fit_transform(d.drop("Is promoted",axis=1))
    d = pd.DataFrame(d, columns = columns)
    d.index=emp.index
    d["Is promoted"]=emp["Is promoted"]
    plt.figure(figsize=(15,6))
    correlation = d.corr( method='pearson' )
    sns.heatmap( correlation, annot=True )
    plt.savefig(correlation_path)
    plt.close()
    save_pickle('d_correlation.pkl', d, pickle_directory )
    save_pickle('ordinal_encoder.pkl', ordinal_encoder, pickle_directory )
    save_pickle('scaler.pkl', scaler, pickle_directory )

def get_correlation(filepath):
    correlation_path = filepath + '/correlation.png'
    return send_file(correlation_path, mimetype='image/png') 


def model_training(filepath):
    model_evaluation_path =  filepath+'/model_evaluation.png'
    classification_report_path =  filepath+'/classification_report.png'
    classification_prediction_error_path =  filepath+'/classification_prediction_error.png'

    d = load_pickle('d_correlation.pkl', pickle_directory )
    X=d.drop(["Is promoted","EMP ID","Education Background","Previous Position","Gender","Age"],axis=1)
    y=d["Is promoted"]

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
    #Initial Model

    #Transforming the Data into an optimized format:

    # Create DMatrix for training and test data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    #Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        #'n_estimators': 100,
        'seed': 42
    }

    # Train the model and capture the training and test loss history
    evals_result = {}

    model_initial = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result,
        verbose_eval=10
    )


    # Make predictions on the test set
    predictions = model_initial.predict(xgb.DMatrix(X_test))
    pred_initial=(predictions > 0.5).astype(int)
    #Save the Model File
    model_initial.save_model(pickle_directory+'/best_model.model')
    save_pickle('loss_history.pkl', evals_result, pickle_directory)

    # Calculate the accuracy score
    initial_accuracy = accuracy_score(y_test,pred_initial )

    print(f'Initial Accuracy: {initial_accuracy}')
    # Calculate metrics
    accuracy_xgb = accuracy_score(y_test, pred_initial)
    precision = precision_score(y_test, pred_initial)
    recall = recall_score(y_test, pred_initial)
    f1_xgb = f1_score(y_test, pred_initial)
    roc_auc = roc_auc_score(y_test, pred_initial)
    mse_xgb=mean_squared_error(y_test, pred_initial)


    # Display metrics
    print(f"Accuracy: {accuracy_xgb:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_xgb:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")
    print(f"MSE: {mse_xgb:.2f}")

    cfm=metrics.confusion_matrix(y_test,pred_initial)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = [False, True])
    cm_display.plot()
    plt.show()  
    plt.savefig(model_evaluation_path)
    plt.close()

    # Generate a classification report
    report = classification_report(y_test, pred_initial, target_names=['0', '1'], output_dict=True)

    # Create a visualization using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = ['0', '1']

    # Plot precision, recall, and F1-score for each class
    for i, class_name in enumerate(classes):
        precision = report[class_name]['precision']*100
        recall = report[class_name]['recall']*100
        f1_score_ = report[class_name]['f1-score']*100
        x = np.arange(3)
        ax.bar(x + i * 0.2, [precision, recall, f1_score_], width=0.2, label=class_name)

    ax.set_xticks(np.arange(3) + 0.2)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_title('Custom Classification Report')
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig(classification_report_path)
    plt.close()

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_test, pred_initial)

    # Extract true positives, true negatives, false positives, and false negatives
    tn, fp, fn, tp = conf_matrix.ravel()

    # Create a custom bar chart for classification prediction errors
    categories = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    counts = [tn, fp, fn, tp]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=['green', 'red', 'red', 'green'])
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Custom Classification Prediction Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig(classification_prediction_error_path)
    plt.close()

    model_importance_path = filepath + '/model_importance.png'
    xgb.plot_importance(model_initial)
    plt.savefig(model_importance_path)
    plt.close()

def get_model_evaluation_data(filepath):
    model_evaluation_path =  filepath+'/model_evaluation.png'
    return send_file(model_evaluation_path, mimetype='image/png') 


def generate_model_performance(filepath):
    model_performance_path =  filepath+'/model_performance.png'

    # Load the loss history from the saved file
    loaded_history = load_pickle('loss_history.pkl', pickle_directory)

    # Extract and plot the training and test loss curves
    train_losses = loaded_history['train']['logloss']
    test_losses = loaded_history['test']['logloss']
    # plot learning curves
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')

    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.grid(True)

    # show the plot
    plt.show()
    plt.savefig(model_performance_path)
    plt.close()
   

def get_model_performance(filepath):
    model_performance_path =  filepath+'/model_performance.png'
    return send_file(model_performance_path, mimetype='image/png') 
   

def get_classification_report(filepath):
    classification_report_path =  filepath+'/classification_report.png'
    return send_file(classification_report_path, mimetype='image/png') 
   

def get_classification_prediction_error(filepath):
    classification_prediction_error_path =  filepath+'/classification_prediction_error.png'
    return send_file(classification_prediction_error_path, mimetype='image/png') 

   

def get_model_importance(filepath):
    model_importance_path = filepath + '/model_importance.png'
    return send_file(model_importance_path, mimetype='image/png') 


def get_positions_missing_values():
    sum_null=positions.isnull().sum().sum()
    print(sum_null)
    notNull=len(emp)-sum_null

    y = np.array([sum_null,notNull])
    mylabels = ["Missing", "Present"]
    total = sum(y)
    percentages = [(value / total * 100).round(1) for value in y]
    return jsonify({'labels': mylabels, 'data': percentages})
    # return sum_null
    # test=sum_null>=1
    # if (test):
    #     print(f"The positions Information Dataset contains {sum_null} missing value(s)!")
    # else:
    #     print("The positions Information Dataset is clean (no missing values)!")
   

def enter_employee_info():
    emp_info={}
    df=emp.drop("Is promoted",axis=1)
    categorical_columns=df.select_dtypes(include='object').columns
    numeric_columns=df.select_dtypes(include={'int','float'}).columns
    for col in numeric_columns:
        emp_info[col]=float(input(f"Enter {col} : "))
    for col in categorical_columns:
        emp_info[col]=str(input(f"Enter {col} : "))

    emp_info=pd.DataFrame([emp_info])
    origin_columns=emp.drop(["Is promoted"],axis=1).columns
    emp_info=emp_info[origin_columns]
    print(emp_info)
    return emp_info

def text_preprocessing(t):
    stopWords=set(stopwords.words("english"))
    t=t.lower()
    t=re.sub(r"[^a-zA-Z]", " ", t)
    words=word_tokenize(t)
    words=[word for word in words if word not in stopWords]
    stemmer = SnowballStemmer(language='english')
    words= [stemmer.stem(word) for word in words]
    return " ".join(words)

def calcul_numeric_similarity(pos_num_features,emp_numeric_features):
    list_emp=[emp_numeric_features[i][0] for i in emp_numeric_features.columns]
    list_pos=[i for i in pos_num_features[0]]
    number_similar_metrics=0

    if len(list_pos) == len(list_emp):
     for i in range(len(list_emp)):
        if list_emp[i]>=list_pos[i]:
            number_similar_metrics=number_similar_metrics+1
    result=number_similar_metrics/len(list_emp)
    return result

def most_compatible_positions(employee_info,positions_df):
    emp_level=employee_info["Position Level"][0]
    emp_text_features=(employee_info["Profile"]).apply(text_preprocessing)
    emp_numeric_features=employee_info[["Years Of Experience","Support", "Reporting","Team", "Evaluation" ,"Organisation", "Performance"]]
    position_text_features={}
    position_numeric_features={}

    #Drop missing values
    positions_df=positions_df.dropna()

    for pos in positions_df[positions_df["Level"]>emp_level]["Position Name"]:
        position_text_features[pos]=text_preprocessing(list(positions[positions["Position Name"]==pos]["Key Words"].values)[0])
        position_numeric_features[pos]=list((positions_df[positions_df["Position Name"]==pos][["Minimum Years Of Experience","Min Support Rating","Min Reporting Rating","Min Team Rating","Min Evaluation Rating","Min Organisation Rating","Min Performance Rating"]].values)[0])

    pos_similarity={}
    tfidf_vectorizer = TfidfVectorizer()
    matrix=tfidf_vectorizer.fit_transform(position_text_features.values())
    emp_text_vector=tfidf_vectorizer.transform(emp_text_features)


    for pos in position_text_features.keys():
        position_text_vector=tfidf_vectorizer.transform(pd.Series(position_text_features[pos]))
        textual_cosine_similarity = cosine_similarity(emp_text_vector, position_text_vector)

        pos_num_features=np.array(position_numeric_features[pos]).reshape(1, -1)
        numeric_similarity=calcul_numeric_similarity(pos_num_features,emp_numeric_features)

        final_similarity_result=textual_cosine_similarity*0.2+numeric_similarity*0.8
        pos_similarity[pos]=round((final_similarity_result[0])[0]*100,2)

    result=pd.DataFrame.from_dict(pos_similarity,orient ='index', columns=["Percentage"]).sort_values(by="Percentage",ascending=False)
    # Convert the DataFrame to a list of dictionaries
    result_list = result.reset_index().to_dict(orient='records')

    # Now, 'result_list' is a list of dictionaries
    print(result_list)
    return(result_list)

def get_new_prediction(emp_new_data):
    
    ordinal_encoder = load_pickle('ordinal_encoder.pkl', pickle_directory )
    scaler = load_pickle('scaler.pkl', pickle_directory )
    # index = ['Row1']  

    # emp_new_prediction=pd.DataFrame([emp_new_data])
    # origin_columns=emp.drop(["Is promoted"],axis=1).columns
    # emp_new_prediction=emp_new_prediction[origin_columns]

    emp_info = emp_new_data
    emp_info=pd.DataFrame([emp_info])
    origin_columns=emp.drop(["Is promoted"],axis=1).columns
    emp_info=emp_info[origin_columns]
    emp_new_prediction = emp_info

    # emp_new_prediction = pd.DataFrame(emp_new_data, index=index)
    print('emp_new_prediction', emp_new_prediction)
    # emp_info_as_input = enter_employee_info()
    # print(emp_info_as_input)
    
    emp_id=int(emp_new_prediction["EMP ID"][0])
    #Drop the Profile:
    copy_info=emp_new_prediction.drop("Profile",axis=1)

    #Ordinal Encoding:
    categorical_columns=emp_new_prediction.drop("Profile",axis=1).select_dtypes(include='object').columns
    copy_info[categorical_columns]=ordinal_encoder.transform(copy_info[categorical_columns])

    #Scaling:
    copy_info = scaler.transform(copy_info)

    #Transform to DataFrame after Scaling:
    columns=emp_new_prediction.drop("Profile",axis=1).columns
    copy_info=pd.DataFrame(copy_info, columns = columns)

    #Drop Unused columns during the training of the model:
    copy_res=copy_info.drop(["EMP ID","Education Background","Previous Position","Gender","Age"],axis=1)


    # Create a DMatrix for the new observation
    new_observation_dmatrix = xgb.DMatrix(copy_res)

    #Load the Saved Model
    model = xgb.Booster(model_file= pickle_directory+"/best_model.model")

    # Make a prediction
    prediction_proba = model.predict(new_observation_dmatrix)
    new = 1 if prediction_proba > 0.5 else 0

    #If employee predicted ad promoted(==1) then display the list of most compatible positions for him
    if new==1:
     print(f"Employee with ID:{emp_id} is predicted as PROMOTED! \n")
     print("Most Compatible positions for the Employee ID: ", emp_id)
     emp_compatible_positions=most_compatible_positions(emp_new_prediction,positions)
     print(emp_compatible_positions)
     return jsonify(emp_compatible_positions)
    else:
     print(f"Employee with ID:{emp_id} is predicted as NOT promoted!")
     print(copy_res)
     return""
 