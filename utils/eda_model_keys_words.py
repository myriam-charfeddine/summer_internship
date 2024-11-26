import pandas as pd
import numpy as np
from flask import jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (Agg)
from utils.pickle_utils import load_pickle, save_pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans

three_months = pd.read_csv("./uploads/3_months_experts.csv", skipinitialspace = True, sep=",")
six_months = pd.read_csv("./uploads/6_months_experts.csv", skipinitialspace = True, sep=",")
yearly = pd.read_csv("./uploads/yearly_experts.csv", skipinitialspace = True, sep=",")
tech_skills = pd.read_csv("./uploads/technical_skills.csv", sep=",")
df = None
df1 = None
df2 = None
df3 = None
columns = ["EMP ID","EXPERT ID","Ttler's Target Points"]

pickle_directory = 'pickle_files' 

def load_data(filePath):
    three_months = pd.read_csv(filePath + "/3_months_experts.csv", skipinitialspace = True, sep=",")
    six_months = pd.read_csv(filePath + "/6_months_experts.csv", skipinitialspace = True, sep=",")
    yearly = pd.read_csv(filePath + "/yearly_experts.csv", skipinitialspace = True, sep=",")
    tech_skills = pd.read_csv(filePath + "/technical_skills.csv", sep=",")
    three_months.head(2)
    six_months.head(2)
    yearly.head(2)
    tech_skills.head(2)


def data_preprocessing(filePath):
    data = pd.read_csv(filePath, skipinitialspace = True, sep=",")
    # Select the concerned columns for this model
    # columns = ["EMP ID","EXPERT ID","Ttler's Target Points"]
    # EDA for the 3 months data :
    df1 = data[columns]
    df1.shape
    #Info about features
    df1.info()
    #Missing values
    df1.isnull().sum()
    #Display the amount of missing values among all the data of this model
    null = df1.isnull().sum().sum()
    notNull = len(df1)-null
    y = np.array([null,notNull])
    mylabels = ["Missing", "Present"]
    total = sum(y)
    percentages = [(value / total * 100).round(1) for value in y]
    return jsonify({'labels': mylabels, 'data': percentages})
    # plt.pie(y, autopct='%1.1f%%', labels=mylabels)
    # plt.legend(title="3 months Evaluation Textual Data")
    # # plt.show()

    #  # Save the pie chart as an image file
    # chart_image_path = filePath+'/path_to_save_pie_chart.png'
    # plt.savefig(chart_image_path)
    # plt.close()

def get_df():
    return df
        

def concatenate_data():
    # print('EROS', nltk.data.path)
    # load_data('./uploads')
    global df
    df = pd.DataFrame()
    global df1
    df1 = three_months[columns]
    global df2
    df2 = six_months[columns]
    global df3
    df3 = yearly[columns]

    #Get the "Ttler's Target Points" from the 3 months dataset:
    df["Emp ID"] = [i for i in df3["EMP ID"].unique()]
    df["Ttler's Target Points_1"]=[item for item in df1.groupby("EMP ID")["Ttler's Target Points"].apply(lambda x: " ".join(x.dropna()))]
    # df.head(3)

    #Get the "Ttler's Target Points" from the 6 months dataset:
    df["Ttler's Target Points_2"]=[item for item in df2.groupby("EMP ID")["Ttler's Target Points"].apply(lambda x: " ".join(x.dropna()))]
    # df.head(3)

    #Get the "Ttler's Target Points" from the Yearly dataset:
    df["Ttler's Target Points_3"]=[item for item in df3.groupby("EMP ID")["Ttler's Target Points"].apply(lambda x: " ".join(x.dropna()))]
    # df.head(3)

    #Concatenate the 3 previous columns in one column named "Ttler's Target Points" + remove the NAN values
    df["Ttler's Target Points"]=df[["Ttler's Target Points_1","Ttler's Target Points_2","Ttler's Target Points_3"]].astype(str).apply(lambda row: '_'.join(row.dropna()), axis=1)

    #Remove the unwanted columns
    df=df.drop(["Ttler's Target Points_1","Ttler's Target Points_2","Ttler's Target Points_3"],axis=1)
    # df.head()
    # df.shape
    # df.info()
    # df.isnull().sum()

# apply_text_preprocessing
    apply_text_preprocessing(df)


#Text preprocessing function
def text_preprocessing(t):
    stopWords=set(stopwords.words("english"))
    t=t.lower()
    t=re.sub(r"[^a-zA-Z]", " ", t)
    words=word_tokenize(t)
    words=[word for word in words if word not in stopWords]
    lemmatizer = nltk.WordNetLemmatizer()
    words= [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def apply_text_preprocessing(df): 
    df["Ttler's Target Points"]=df["Ttler's Target Points"].apply(text_preprocessing)


#Apply TF-IDF    
def apply_tf_idf(): 
    concatenate_data()
    tfidf_vectorizer_2 = TfidfVectorizer()
    tfidf_matrix_2 = tfidf_vectorizer_2.fit_transform(df["Ttler's Target Points"])
    feature_names_2 = tfidf_vectorizer_2.get_feature_names_out()
    tfidf_matrix_2.shape
    idf_values = tfidf_vectorizer_2.idf_
    tfidf_matrix_2.toarray()
    
     # Save the TF-IDF vectorizer and transformed data using pickle
    save_pickle('tfidf_vectorizer.pkl', tfidf_vectorizer_2, pickle_directory)
    save_pickle('tfidf_matrix.pkl', tfidf_matrix_2, pickle_directory)
    save_pickle('feature_names.pkl', feature_names_2, pickle_directory)
    save_pickle('idf_values.pkl', idf_values, pickle_directory)

def get_key_words():
    my_list = []
    # Load the TF-IDF vectorizer using the custom load_pickle function
    idf_values = load_pickle('idf_values.pkl', pickle_directory)
    feature_names = load_pickle('feature_names.pkl', pickle_directory)
    # Print the IDF values for each word
    for word, idf in zip(feature_names, idf_values):
        idf_formatted = round(idf, 4)
        my_list.append({'word': word, 'idf': idf_formatted})
    return my_list


def drop_target_point_column(file):
    if file == '3_months_experts':
        return three_months.drop([ "Ttler's Target Points"], axis=1)
    elif file == '6_months_experts':
        return six_months.drop([ "Ttler's Target Points"], axis=1)
    else:
        return yearly.drop([ "Ttler's Target Points"], axis=1)


def data_preprocessing_soft_skills(file):
    data = drop_target_point_column(file)
    return jsonify(data_preprocessing_files(data))


def data_preprocessing_files(data):
    null = data.isnull().sum().sum()
    notNull = len(data)-null
    y = np.array([null,notNull])
    mylabels = ["Missing", "Present"]
    total = sum(y)
    percentages = [(value / total * 100).round(1) for value in y]
    return {'labels': mylabels, 'data': percentages}


def get_missing_values(file):
    data = drop_target_point_column(file)
    my_list = []
    # Iterate through unique EMP IDs
    for emp in data["EMP ID"].unique():
        expert_data_list = []  # Initialize a list to store expert data for this employee
        for expert in data[data["EMP ID"] == emp]["EXPERT ID"]:
            filtered_data = data[(data["EMP ID"] == emp) & (data["EXPERT ID"] == expert)].drop(['EMP ID', 'EXPERT ID'], axis=1)
            expert_dict = {
                'expert_id': str(expert),
                **{
                    column: 'Missing' if filtered_data[column].isnull().any() else 'Present'
                    for column in filtered_data.columns
                },
                'emp_id': str(emp)
            }
           
            my_list.append(expert_dict)
    return  jsonify(my_list)

# def get_missing_values(file):
#     data = drop_target_point_column(file)
#     my_list = []
#     # Iterate through unique EMP IDs
#     for emp in data["EMP ID"].unique():
#         expert_data_list = []  # Initialize a list to store expert data for this employee
#         for expert in data[data["EMP ID"] == emp]["EXPERT ID"]:
#             filtered_data = data[(data["EMP ID"] == emp) & (data["EXPERT ID"] == expert)].drop(['EMP ID', 'EXPERT ID'], axis=1)
#             # expert_dict = {'expert_id': str(expert)}
#             # result_dict = {
#             #     column: 'Missing' if filtered_data[column].isnull().any() else 'Present'
#             #     for column in filtered_data.columns
#             # }
#             expert_dict = {
#                 'expert_id': str(expert),
#                 **{
#                     column: 'Missing' if filtered_data[column].isnull().any() else 'Present'
#                     for column in filtered_data.columns
#                 }
#             }
           
#             expert_data_list.append(expert_dict)
#         my_list.append({'emp_id': str(emp), 'expert_data': expert_data_list})
#     print('my_list ', my_list[0])
#     return  jsonify(my_list)

#is needed to retrain model
def concatenate_numerical_data():
    df3 = yearly[columns]
    df_= pd.DataFrame()
    df_["EMP ID"]=[i for i in df3["EMP ID"].unique()]
    
    calculate_average_skills_three_months(df_)
    calculate_average_skills_six_months(df_)
    calculate_diff_six_three_months(df_)
    remove_unwanted_columns(df_)
    #Fill the diffrence == NaN, by ZERO:
    df_= df_.fillna(0)
    calculate_average_skills_yearly(df_)
    save_pickle('df_.pkl', df_, pickle_directory)
    

def calculate_average_skills_three_months(df_):
    df_["Efficiency&Autonomy_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Efficiency&Autonomy"].mean() if item is not None]
    df_["Analytical Skills_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Analytical Skills"].mean() if item is not None]
    df_["Communication_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Communication"].mean() if item is not None]
    df_["Technical Skills_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Technical Skills"].mean() if item is not None]
    df_["Discipline&Quality Of Work_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Discipline&Quality Of Work"].mean() if item is not None]
    df_["Initiative_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Initiative"].mean() if item is not None]
    df_["Cooperation&Teamwork_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Cooperation&Teamwork"].mean() if item is not None]
    df_["Availability_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Availability"].mean() if item is not None]
    df_["Punctuality&Regular Attendance_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Punctuality&Regular Attendance"].mean() if item is not None]
    df_["Self-Control&Stress Tolerance_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Self-Control&Stress Tolerance"].mean() if item is not None]
    df_["Collaboration With The Hierarchy_1"]=[round(item, 2) for item in three_months.groupby("EMP ID")["Collaboration With The Hierarchy"].mean() if item is not None]

def calculate_average_skills_six_months(df_):
    df_["Efficiency&Autonomy_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Efficiency&Autonomy"].mean() if item is not None]
    df_["Analytical Skills_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Analytical Skills"].mean() if item is not None]
    df_["Communication_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Communication"].mean() if item is not None]
    df_["Technical Skills_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Technical Skills"].mean() if item is not None]
    df_["Discipline&Quality Of Work_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Discipline&Quality Of Work"].mean() if item is not None]
    df_["Initiative_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Initiative"].mean() if item is not None]
    df_["Cooperation&Teamwork_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Cooperation&Teamwork"].mean() if item is not None]
    df_["Availability_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Availability"].mean() if item is not None]
    df_["Punctuality&Regular Attendance_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Punctuality&Regular Attendance"].mean() if item is not None]
    df_["Self-Control&Stress Tolerance_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Self-Control&Stress Tolerance"].mean() if item is not None]
    df_["Collaboration With The Hierarchy_2"]=[round(item, 2) for item in six_months.groupby("EMP ID")["Collaboration With The Hierarchy"].mean() if item is not None]


def calculate_diff_six_three_months(df_):
    df_['Efficiency&Autonomy_3/6_Diff'] = [i for i in (df_['Efficiency&Autonomy_2'] - df_['Efficiency&Autonomy_1']) ]
    df_['Analytical Skills_3/6_Diff'] = [i for i in (df_['Analytical Skills_2'] - df_['Analytical Skills_1']) ]
    df_['Communication_3/6_Diff'] = [i for i in (df_['Communication_2'] - df_['Communication_1']) ]
    df_['Technical Skills_3/6_Diff'] = [i for i in (df_['Technical Skills_2'] - df_['Technical Skills_1']) ]
    df_['Discipline&Quality Of Work_3/6_Diff'] = [i for i in (df_['Discipline&Quality Of Work_2'] - df_['Discipline&Quality Of Work_1']) ]
    df_['Initiative_3/6_Diff'] = [i for i in (df_['Initiative_2'] - df_['Initiative_1']) ]
    df_['Cooperation&Teamwork_3/6_Diff'] = [i for i in (df_['Cooperation&Teamwork_2'] - df_['Cooperation&Teamwork_1']) ]
    df_['Availability_3/6_Diff'] = [i for i in (df_['Availability_2'] - df_['Availability_1']) ]
    df_['Punctuality&Regular Attendance_3/6_Diff'] = [i for i in (df_['Punctuality&Regular Attendance_2'] - df_['Punctuality&Regular Attendance_1']) ]
    df_['Self-Control&Stress Tolerance_3/6_Diff'] = [i for i in (df_['Self-Control&Stress Tolerance_2'] - df_['Self-Control&Stress Tolerance_1']) ]
    df_['Collaboration With The Hierarchy_3/6_Diff'] = df_['Collaboration With The Hierarchy_2'] - df_['Collaboration With The Hierarchy_1']

def remove_unwanted_columns(df_):
    df_=df_.drop(['Efficiency&Autonomy_1', 'Analytical Skills_1',
       'Communication_1', 'Technical Skills_1', 'Discipline&Quality Of Work_1',
       'Initiative_1', 'Cooperation&Teamwork_1', 'Availability_1',
       'Punctuality&Regular Attendance_1', 'Self-Control&Stress Tolerance_1',
       'Collaboration With The Hierarchy_1', 'Efficiency&Autonomy_2',
       'Analytical Skills_2', 'Communication_2', 'Technical Skills_2',
       'Discipline&Quality Of Work_2', 'Initiative_2',
       'Cooperation&Teamwork_2', 'Availability_2',
       'Punctuality&Regular Attendance_2', 'Self-Control&Stress Tolerance_2',
       'Collaboration With The Hierarchy_2'],axis=1)

#Calculate the average of each skill (Yearly Data)
def calculate_average_skills_yearly(df_):
    for skill in yearly.drop(["EMP ID","EXPERT ID", "Ttler's Target Points"],axis=1).columns:
        df_[skill]=[round(item, 2) for item in yearly.groupby("EMP ID")[skill].mean() if item is not None]

def get_number_missing_values_column():
    df_ = load_pickle('df_.pkl', pickle_directory)
    skills_columns = df_.drop('EMP ID', axis=1)
    # Count null and not null values for each employee
    null_values = skills_columns.isnull().sum(axis=1)
    not_null_values = skills_columns.notnull().sum(axis=1)
    labels = df_['EMP ID'].tolist()
    null_values_data = null_values.tolist()
    not_null_values_data = not_null_values.tolist()
    return jsonify({'labels': labels, 'data': [{'data': null_values_data, 'label': 'Missing Values'}, {'data': not_null_values_data, 'label': 'Present Values'}]})

def get_missing_columns_value_name(): 
    df_ = load_pickle('df_.pkl', pickle_directory)
    null_columns=list()
    total_columns=df_.columns
    i=-1
    for emp in df_["EMP ID"].unique():
        i=i+1
        test1=df_[df_["EMP ID"]==emp].isnull().sum().sum() >=1
        if (test1==True) :
            for col in total_columns:
                test2=df_[df_["EMP ID"]==emp][col].isnull().values[0]
                if (test2==True):
                    expert_dict = {
                    'column': col,
                    'emp_id': str(emp)
                }
                    null_columns.append(expert_dict)
    return jsonify(null_columns)
        


def remove_employees_columns_missing_values():
    df_ = load_pickle('df_.pkl', pickle_directory)
    rows_to_drop=list() #to store the ID of the removed Employees
    index_to_drop=list()
    i=-1
    for emp in df_["EMP ID"].unique():
        i=i+1
        test1=df_[df_["EMP ID"]==emp].isnull().sum().sum() >=1
        if (test1==True) :
            rows_to_drop.append(emp)
            index_to_drop.append(i)
    df_=df_[~df_['EMP ID'].isin(rows_to_drop)]
    tfidf_matrix_2 = load_pickle('tfidf_matrix.pkl', pickle_directory)
    tfidf_matrix_2_filtered=tfidf_matrix_2[np.delete(np.arange(tfidf_matrix_2.shape[0]), index_to_drop)]

    #Normalize Data:
    scaler = StandardScaler()
    num_feautures = scaler.fit_transform(df_)
    feature_matrix = pd.concat([pd.DataFrame(tfidf_matrix_2_filtered.toarray()),pd.DataFrame(num_feautures)],axis=1)
    save_pickle('feature_matrix.pkl', feature_matrix, pickle_directory)
    save_pickle('df_without_missing_columns.pkl', df_, pickle_directory)

#needed in retrain    
def generate_soft_skills_dendrograms():
    dendrogram_path = "./images/soft_skills_dendrogram.png"
    feature_matrix = load_pickle('feature_matrix.pkl', pickle_directory)
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(feature_matrix, method='ward'))
    # Save the figure to a file
    plt.savefig(dendrogram_path)
    plt.close()

#needed in retrain 
def generate_optimal_number_cluster(): 
    elbow_path = "./images/soft_skills_elbow.png"
    print('Elbow Method to determine the number of clusters to be formed:')
    Elbow_M = KElbowVisualizer(AgglomerativeClustering(affinity='euclidean', linkage='ward'), k=10)
    feature_matrix = load_pickle('feature_matrix.pkl', pickle_directory)
    Elbow_M.fit(feature_matrix)
    Elbow_M.show()
    plt.savefig(elbow_path)
    plt.close()
    save_pickle('Elbow_M.pkl', Elbow_M, pickle_directory)
    return Elbow_M.elbow_value_


def get_soft_skills_dendrograms(filepath):
    dendrogram_path = filepath + "/soft_skills_dendrogram.png"
    return send_file(dendrogram_path, mimetype='image/png') 

def get_optimal_number_cluster(filepath): 
    elbow_path = filepath + "/soft_skills_elbow.png"
    return send_file(elbow_path, mimetype='image/png') 

def clustering_agglomerative_algorithm():
    df_ = load_pickle('df_without_missing_columns.pkl', pickle_directory)
    feature_matrix = load_pickle('feature_matrix.pkl', pickle_directory)
    generate_soft_skills_dendrograms()
    optimal_k = generate_optimal_number_cluster()
    cluster = AgglomerativeClustering(n_clusters=optimal_k , affinity='euclidean', linkage='ward')
    df_["cluster_3"]=cluster.fit_predict(feature_matrix)
    save_pickle('df_clustering_agglomerative.pkl', df_,  pickle_directory)
    save_pickle('clustering_agglomerative.pkl', df_["cluster_3"], pickle_directory)


def reduce_data_repartition():
    df_ = load_pickle('df_clustering_agglomerative.pkl', pickle_directory)
    feature_matrix = load_pickle('feature_matrix.pkl', pickle_directory)
    #Apply PCA: Pricpal Component Anlysis
    ### PCA is used to reduce the dimensions (number of columns)
    pca_num_components = 2
    reduced_data = PCA(n_components=pca_num_components).fit_transform(feature_matrix)
    results = pd.DataFrame(reduced_data,columns=['Dim1','Dim2'])
    sns.scatterplot(x="Dim1", y="Dim2", hue=np.array(df_["cluster_3"]), data=results)
    # Save the plot as an image
    scatterplot_path = './images/soft_skills_scatterplot.png'
    plt.savefig(scatterplot_path)
    plt.close()
    return send_file(scatterplot_path, mimetype='image/png')

def get_soft_skills_scatterplot(filepath): 
    elbow_path = filepath + "/soft_skills_scatterplot.png"
    return send_file(elbow_path, mimetype='image/png') 

#needed in retrain
def generate_cluster_distribution(filepath):
    cluster_path= filepath + '/clusters.png'
    df_ = load_pickle('df_clustering_agglomerative.pkl', pickle_directory)
    # cluster_3 = load_pickle('clustering_agglomerative.pkl', pickle_directory)
    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
    pl = sns.countplot(x=df_["cluster_3"], palette= pal)
    pl.set_title("Distribution Of The Clusters")
    plt.show()
    plt.savefig(cluster_path)
    plt.close()

def get_cluster_distribution(filepath):
    cluster_path= filepath + '/clusters.png'
    return send_file(cluster_path, mimetype='image/png') 

#needed in retrain
def generate_cluster_categories(): 
    df_ = load_pickle('df_clustering_agglomerative.pkl', pickle_directory)
    # cluster_3 = load_pickle('clustering_agglomerative.pkl', pickle_directory)
    Elbow_M = load_pickle('Elbow_M.pkl', pickle_directory)
    cluster_skill_average_1={} #dictionnary
    skills=yearly.drop(["EMP ID","EXPERT ID", "Ttler's Target Points" ],axis=1).columns

    for cluster in df_["cluster_3"].unique():
        for skill in skills:
            if cluster not in cluster_skill_average_1:
                cluster_skill_average_1[cluster] = []
            cluster_skill_average_1[cluster].append(df_[df_["cluster_3"]==cluster][skill].mean())

    average_rating={}
    for cluster, ratings_list in cluster_skill_average_1.items():
        mean_ratings = np.mean(ratings_list, axis=0)
        average_rating[cluster] = mean_ratings
    f = pd.DataFrame(average_rating.items(), columns=["cluster_3","rating_average"])
    f=f.sort_values(by="rating_average")

    cluster_categories={}
    for i in range(Elbow_M.elbow_value_):
        cluster_categories[i]="cluster_"+str(i)

    f['cluster_3'] = f['cluster_3'].map(cluster_categories)
    save_pickle('cluster_skill_average_1.pkl', cluster_skill_average_1, pickle_directory)
    save_pickle('categorical_cluster.pkl', f, pickle_directory)

def generate_cluster_categories(filepath): 
    categorical_cluster_path = filepath + '/categorical_cluster.png'
    f= load_pickle('categorical_cluster.pkl', pickle_directory)
    plt.figure(figsize=(5, 5))
    sns.barplot(x="cluster_3", y="rating_average", data=f, color='g')
    plt.title("Average of all the ratings for each cluster")
    plt.xlabel("Clusters")
    plt.ylabel("Average of ratings")
    plt.tight_layout()
    plt.show()
    plt.savefig(categorical_cluster_path)
    plt.close()

def get_cluster_categories(filepath): 
    categorical_cluster_path = filepath + '/categorical_cluster.png'
    return send_file(categorical_cluster_path, mimetype='image/png') 

def generate_improvement_level_cluster():  
    df_ = load_pickle('df_clustering_agglomerative.pkl', pickle_directory)
    # cluster_3 = load_pickle('clustering_agglomerative.pkl', pickle_directory)
    optimal_k = load_pickle('Elbow_M.pkl', pickle_directory).elbow_value_
    cluster_diff_average={} #dictionnary
    diffs=['Efficiency&Autonomy_3/6_Diff', 'Analytical Skills_3/6_Diff',
        'Communication_3/6_Diff', 'Technical Skills_3/6_Diff',
        'Discipline&Quality Of Work_3/6_Diff', 'Initiative_3/6_Diff',
        'Cooperation&Teamwork_3/6_Diff', 'Availability_3/6_Diff',
        'Punctuality&Regular Attendance_3/6_Diff',
        'Self-Control&Stress Tolerance_3/6_Diff',
        'Collaboration With The Hierarchy_3/6_Diff']

    for cluster in df_["cluster_3"].unique():
        for diff in diffs:
            if cluster not in cluster_diff_average:
                cluster_diff_average[cluster] = []
            cluster_diff_average[cluster].append(df_[df_["cluster_3"]==cluster][diff].mean())

    average_diff={}
    for cluster, ratings_list in cluster_diff_average.items():
        mean_ratings = np.mean(ratings_list, axis=0)
        average_diff[cluster] = mean_ratings
    t = pd.DataFrame(average_diff.items(), columns=["cluster_3","diff_average"])
    t=t.sort_values(by="diff_average")

    cluster_categories={}
    for i in range(optimal_k):
        cluster_categories[i]="cluster_"+str(i)
    # Convert numerical rating to categorical
    t['cluster_3'] = t['cluster_3'].map(cluster_categories)
    save_pickle('average_diff.pkl', average_diff, pickle_directory)
    save_pickle('improvement_level_cluster.pkl', t, pickle_directory)

def generate_improvement_level_cluster(filepath):
    improvement_cluster_path = filepath + '/improvement_level_cluster.png'
    t= load_pickle('improvement_level_cluster.pkl', pickle_directory)
    plt.figure(figsize=(5, 5))
    sns.barplot(x="cluster_3", y="diff_average", data=t, color='b')

    plt.title("Improvement level average for each cluster")
    plt.xlabel("Clusters")
    plt.ylabel("Average")
    plt.tight_layout()
    plt.show()
    plt.savefig(improvement_cluster_path)
    plt.close()

def get_improvement_level_cluster(filepath):
    improvement_cluster_path = filepath + '/improvement_level_cluster.png'
    return send_file(improvement_cluster_path, mimetype='image/png') 

def get_soft_skills_outputs():
    df_ = load_pickle('df_clustering_agglomerative.pkl', pickle_directory)
    emp_ids=df_["EMP ID"].unique()
    clusters=np.array(df_["cluster_3"])
    l=len(np.unique(clusters))

    #Top Soft Skills + Target Points 
    top_soft_skills={}
    target_points_1={}
    skills=yearly.drop(["EMP ID","EXPERT ID", "Ttler's Target Points"],axis=1).columns
    cluster_skill_average_1 = load_pickle('cluster_skill_average_1.pkl', pickle_directory)
    for emp,cluster in zip(emp_ids,clusters):
        j=0
        while (j<l):
            if cluster==j:
                for i in range(len(skills)):
                    if cluster_skill_average_1[cluster][i]>=7:
                        if emp not in top_soft_skills:
                            top_soft_skills[emp] = []
                            top_soft_skills[emp].append(skills[i])

                    if cluster_skill_average_1[cluster][i]<7:
                        if emp not in target_points_1:
                            target_points_1[emp] = []
                            target_points_1[emp].append(skills[i])

            j=j+1

    for emp,cluster in zip(emp_ids,clusters):
        if emp not in top_soft_skills:
            top_soft_skills[emp] = ["_"]
        if emp not in target_points_1:
            target_points_1[emp] = ["_"]
    save_pickle('top_soft_skills.pkl', top_soft_skills, pickle_directory)
    save_pickle('target_points_1.pkl', target_points_1, pickle_directory)

    # Level of improvement
    skills_improvement_level={}
    average_diff = load_pickle('average_diff.pkl', pickle_directory)

    for emp,cluster in zip(emp_ids,clusters):
        j=0
        while (j<l):
            if cluster==j:
                if average_diff[cluster]>1:
                    skills_improvement_level[emp]="High level of improvement"
                if (average_diff[cluster]<=1) & (average_diff[cluster]>0.5):
                    skills_improvement_level[emp]="Good level of improvement"
                if (average_diff[cluster]>=-0.5) & (average_diff[cluster]<=0.5):
                    skills_improvement_level[emp]="Stable level of improvement"
                if average_diff[cluster]<-0.5:
                    skills_improvement_level[emp]="Level of skills performance decreased"
            j=j+1
    save_pickle('skills_improvement_level.pkl', skills_improvement_level, pickle_directory)

# Call the function to display the results of a specific employee
def display_employee_results_1(employee_id, top_soft_skills, target_points,level_of_improvement):
    data = []
    if employee_id in top_soft_skills and employee_id in target_points :
        employee_top_skills = top_soft_skills[employee_id]
        employee_targets = target_points[employee_id]

        data.append({'employee_id': int(employee_id),
                'employee_top_skills': [', '.join(employee_top_skills)],
                'employee_targets': [', '.join(employee_targets)],
                 'level_of_improvement' : [level_of_improvement[employee_id]]})

        # v = pd.DataFrame(data)
        # v = (v.style.set_properties(**{'text-align': 'center'})
        #              .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]))

        # display(HTML(v.render()))
        print('data', data)
    return data
 
# Display the result of all the Employees:
def display_all_employee_results_1(top_soft_skills, target_points, level_of_improvement):
    data = []

    for employee_id in top_soft_skills.keys():
        if employee_id in target_points and employee_id in level_of_improvement  :
            employee_skills = ', '.join(top_soft_skills[employee_id])
            employee_targets = ', '.join(target_points[employee_id])
            employee_level=level_of_improvement[employee_id]
            data.append({'employee_id': int(employee_id), 'employee_top_skills': employee_skills, 'employee_targets': employee_targets, 'level_of_improvement':employee_level})

    # if data:
        # x = pd.DataFrame(data)
        # pd.set_option('display.max_colwidth', 30)
        # x = x.style.set_properties(**{'text-align': 'left'})
        # print("\nEmployees Results:")
        # return(x)
    return data

    # else:
    #     print("No employee data found.")

def display_employees_result(employee_id = None ):
    top_soft_skills = load_pickle('top_soft_skills.pkl', pickle_directory)
    target_points_1 = load_pickle('target_points_1.pkl', pickle_directory)
    skills_improvement_level = load_pickle('skills_improvement_level.pkl', pickle_directory)
    employee_data = display_all_employee_results_1(top_soft_skills, target_points_1,skills_improvement_level)
    if employee_id is not None:
        print(type(employee_id))
        employee_data = display_employee_results_1(employee_id, top_soft_skills, target_points_1, skills_improvement_level)
    return jsonify(employee_data)


def generate_general_soft_skills_Result(filepath):
    general_result_path = filepath + '/general_result_soft_skills.png'

    top_soft_skills = load_pickle('top_soft_skills.pkl', pickle_directory)
    target_points_1 = load_pickle('target_points_1.pkl', pickle_directory)

    top_skill_counts = {}
    targets_counts = {}
    print('sfs', top_soft_skills.values())

    for emp in top_soft_skills.values():
        for skill in emp:
            top_skill_counts[skill] = top_skill_counts.get(skill, 0) + 1

    for emp in target_points_1.values():
        for target in emp:
            targets_counts[target] = targets_counts.get(target, 0) + 1

    # Create lists of skills and target along with their corresponding counts
    skills = list(top_skill_counts.keys())
    top_skill_counts = list(top_skill_counts.values())

    targets = list(targets_counts.keys())
    targets_counts = list(targets_counts.values())

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Bar chart for top tech skills
    ax1.barh(skills, top_skill_counts, color='blue')
    ax1.set_xlabel('Number of Employees')
    ax1.set_ylabel('Skill')
    ax1.set_title('Number of Employees with Each Skill as a Top Skill')

    # Bar chart for target points
    ax2.barh(targets, targets_counts, color='red')
    ax2.set_xlabel('Number of Employees')
    ax2.set_ylabel('Target')
    ax2.set_title('Number of Employees with Each skill as a Target Point')

    plt.tight_layout()
    plt.show()
    plt.savefig(general_result_path)
    plt.close()


def general_display_soft_skills_Result(filepath):
    general_result_path = filepath + '/general_result_soft_skills.png'
    return send_file(general_result_path, mimetype='image/png') 

def data_preprocessing_tech_skills():
    return data_preprocessing_files(tech_skills)

def get_tech_skills_missing_values():
    my_list = []
    list_tech_skills=[name for name in tech_skills["Skill Name"].unique()]

    for emp in tech_skills["EMP ID"].unique():
        for skill in list_tech_skills:
            filtered_data = tech_skills[(tech_skills["EMP ID"] == emp) & (tech_skills["Skill Name"] == skill)].drop(['EMP ID', 'Category'], axis=1)
            expert_dict = {
                'emp_id': str(emp),
                **{
                    column: 'Missing' if tech_skills[(tech_skills["Skill Name"]==skill) & (tech_skills["EMP ID"]==emp)]["Expert Evaluation"].isnull().values[0] else 'Present'
                    for column in filtered_data.columns
                },
                'skill': skill,
            }
            my_list.append(expert_dict)
    return  jsonify(my_list)


#generate df tech skills:
def generate_df_tech_skills(): 
    df__=pd.DataFrame()
    df__["EMP ID"]=[i for i in tech_skills["EMP ID"].unique()]
    df__.head()
    list_tech_skills=[name for name in tech_skills["Skill Name"].unique()]
    for skill in list_tech_skills:
        df__[skill] = [evaluation for evaluation in tech_skills[tech_skills["Skill Name"]==skill]['Expert Evaluation'] ]

    save_pickle('df_tech_skills.pkl', df__, pickle_directory)

#Merge the technical skills Data of each employee in a new DataFrame:
def get_number_missing_tech_skills_values_column(): 
    df__ = load_pickle('df_tech_skills.pkl', pickle_directory)
    #Create a graph that displays the number of columns with missing values for each employee:
    tech_skills_columns = df__.drop('EMP ID', axis=1)
    # Count null and not null values for each employee
    null_values = tech_skills_columns.isnull().sum(axis=1)
    not_null_values = tech_skills_columns.notnull().sum(axis=1)
    labels = df__['EMP ID'].tolist()
    null_values_data = null_values.tolist()
    not_null_values_data = not_null_values.tolist()
    return jsonify({'labels': labels, 'data': [{'data': null_values_data, 'label': 'Missing Values'}, {'data': not_null_values_data, 'label': 'Present Values'}]})


def get_missing_columns_tech_skills_value_name(): 
    df_ = load_pickle('df_tech_skills.pkl', pickle_directory)
    null_columns=list()
    total_columns=df_.columns
    i=-1
    for emp in df_["EMP ID"].unique():
        i=i+1
        test1=df_[df_["EMP ID"]==emp].isnull().sum().sum() >=1
        if (test1==True) :
            for col in total_columns:
                test2=df_[df_["EMP ID"]==emp][col].isnull().values[0]
                if (test2==True):
                    expert_dict = {
                    'column': col,
                    'emp_id': str(emp)
                }
                    null_columns.append(expert_dict)
    return jsonify(null_columns)
     

def remove_employees_columns_missing_tech_skills_values():
    df__ = load_pickle('df_tech_skills.pkl', pickle_directory)
    rows_to_drop=list() #A list to store the ID of the removed Employees
    index_to_drop=list()
    i=-1
    for emp in df__["EMP ID"].unique():
        i=i+1
        test1=df__[df__["EMP ID"]==emp].isnull().sum().sum() >=1
        if (test1==True) :
            rows_to_drop.append(emp)
            index_to_drop.append(i)
    df__=df__[~df__['EMP ID'].isin(rows_to_drop)]
    save_pickle('df_tech_skills_without_missing_columns.pkl', df__, pickle_directory)

#needed in retrain    
def generate_tech_skills_dendrograms():
    df__ = load_pickle('df_tech_skills_without_missing_columns.pkl', pickle_directory)
    scaler_2 = StandardScaler()
    num_feautures_2 = scaler_2.fit_transform(df__)
    save_pickle('num_feautures_2.pkl', num_feautures_2, pickle_directory)
    dendrogram_path = "./images/tech_skills_dendrogram.png"
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(num_feautures_2, method='ward'))
    # Save the figure to a file
    plt.savefig(dendrogram_path)
    plt.close()

def get_tech_skills_dendrograms(filepath):
    dendrogram_path = filepath + "/tech_skills_dendrogram.png"
    return send_file(dendrogram_path, mimetype='image/png') 

#needed in retrain 
def generate_optimal_number_cluster_tech_skills(): 
    num_feautures_2 = load_pickle('num_feautures_2.pkl', pickle_directory)
    elbow_path = "./images/tech_skills_elbow.png"
    print('Elbow Method to determine the number of clusters to be formed:')
    Elbow_M_1 = KElbowVisualizer(KMeans(random_state=42), k=10)
    Elbow_M_1.fit(num_feautures_2)
    Elbow_M_1.show()
    plt.savefig(elbow_path)
    plt.close()
    save_pickle('Elbow_M_tech_skills.pkl', Elbow_M_1, pickle_directory)
    # return Elbow_M_1.elbow_value_

    
def get_optimal_number_tech_skills_cluster(filepath): 
    elbow_path = filepath + "/tech_skills_elbow.png"
    return send_file(elbow_path, mimetype='image/png') 

#needed in retrain 
def clustering_kmeans_algorithm():
    Elbow_M_1 = load_pickle('Elbow_M_tech_skills.pkl',  pickle_directory)
    df__ = load_pickle('df_tech_skills_without_missing_columns.pkl', pickle_directory)
    num_feautures_2 = load_pickle('num_feautures_2.pkl', pickle_directory)
    optimal_k__=Elbow_M_1.elbow_value_ #Optimal number of clusters

    kmeans = KMeans(n_clusters=optimal_k__, random_state=42)
    df__['cluster_1'] = kmeans.fit_predict(num_feautures_2)
    save_pickle('df_clustering_kmeans.pkl', df__,  pickle_directory)
    save_pickle('clustering_kmeans.pkl', df__["cluster_1"], pickle_directory)

def reduce_data_tech_skills_repartition():
    df__ = load_pickle('df_clustering_kmeans.pkl', pickle_directory)
    num_feautures_2 = load_pickle('num_feautures_2.pkl', pickle_directory)
    #Reduce dimensions
    pca_num_components = 2
    reduced_data_2 = PCA(n_components=pca_num_components).fit_transform(num_feautures_2)
    results_2 = pd.DataFrame(reduced_data_2,columns=['Dim1','Dim2'])
    #Display Data repartition
    sns.scatterplot(x="Dim1", y="Dim2", hue=np.array(df__["cluster_1"]), data=results_2)
    scatterplot_path = './images/tech_skills_scatterplot.png'
    plt.savefig(scatterplot_path)
    plt.close()
    return send_file(scatterplot_path, mimetype='image/png')

def get_tech_skills_scatterplot(filepath): 
    elbow_path = filepath + "/tech_skills_scatterplot.png"
    return send_file(elbow_path, mimetype='image/png') 

#needed in retrain
def generate_cluster_distribution_tech_skills(filepath):
    cluster_path= filepath + '/clusters_tech_skills.png'
    df__ = load_pickle('df_clustering_kmeans.pkl', pickle_directory)
    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60" , "#F3AB93"]
    pl = sns.countplot(x=df__["cluster_1"], palette= pal)
    pl.set_title("Distribution Of The Clusters")
    plt.show()
    plt.savefig(cluster_path)
    plt.close()

def get_cluster_distribution_tech_skills(filepath):
    cluster_path= filepath + '/clusters_tech_skills.png'
    return send_file(cluster_path, mimetype='image/png') 


#needed in retrain
def generate_cluster_categories_tech_skills(): 
    df__ = load_pickle('df_clustering_kmeans.pkl', pickle_directory)
    list_tech_skills=[name for name in tech_skills["Skill Name"].unique()]
    Elbow_M_1 = load_pickle('Elbow_M_tech_skills.pkl',  pickle_directory)
    optimal_k__=Elbow_M_1.elbow_value_ #Optimal number of clusters
    cluster_skill_average_2={} #dictionnary
    average_rating_2={}

    for cluster in df__["cluster_1"].unique():
        for skill in list_tech_skills:
            if cluster not in cluster_skill_average_2:
                cluster_skill_average_2[cluster] = []
            cluster_skill_average_2[cluster].append(df__[df__["cluster_1"]==cluster][skill].mean())

    for cluster, ratings_list in cluster_skill_average_2.items():
        mean_ratings = np.mean(ratings_list, axis=0)
        average_rating_2[cluster] = mean_ratings
    #Sort values

    h = pd.DataFrame(average_rating_2.items(), columns=["cluster_1","rating_average"])
    h=h.sort_values(by="rating_average")

    # Convert clusters identifiers to categorical
    cluster_categories={}
    for i in range(optimal_k__):
        cluster_categories[i]="cluster_"+str(i)

    h['cluster_1'] = h['cluster_1'].map(cluster_categories)
    save_pickle('cluster_tech_skill_average_2.pkl', cluster_skill_average_2, pickle_directory)
    save_pickle('categorical_cluster_tech_skills.pkl', h, pickle_directory)


def generate_cluster_categories_tech_skills(filepath): 
    h = load_pickle('categorical_cluster_tech_skills.pkl', pickle_directory)
    categorical_cluster_path = filepath + '/categorical_cluster_tech_skills.png'
    plt.figure(figsize=(5, 5))
    sns.barplot(x="cluster_1", y="rating_average", data=h, color='g')

    plt.title("Average of all the ratings for each cluster")
    plt.xlabel("Clusters")
    plt.ylabel("Average of ratings")
    plt.tight_layout()
    plt.show()
    plt.savefig(categorical_cluster_path)
    plt.close()


def get_cluster_categories_tech_skills(filepath): 
    categorical_cluster_path = filepath + '/categorical_cluster_tech_skills.png'
    return send_file(categorical_cluster_path, mimetype='image/png') 

def generate_skills_average_cluster(filepath):
    cluster_tech_skills_average_path = filepath +'cluster_tech_skills_average.png'
    #Display the skills averages for every Cluster
    cluster_skill_average_2 = load_pickle('cluster_tech_skill_average_2.pkl', pickle_directory)
    list_tech_skills=[name for name in tech_skills["Skill Name"].unique()]

    clusters = list(cluster_skill_average_2.keys())
    skills = list(range(len(list_tech_skills)))
    averages = list(cluster_skill_average_2.values())


    fig, ax = plt.subplots()
    for i in range(len(clusters)):
        ax.bar([x + i * 0.15 for x in skills], averages[i], width=0.15, align='center', label=f'Cluster {clusters[i]}')

    ax.set_xlabel('Technical skills')
    ax.set_ylabel('Average')
    ax.set_title('Skills averages for all Clusters')
    ax.set_xticks([x + 0.3 for x in skills])
    ax.set_xticklabels([skill for skill in list_tech_skills])
    ax.legend()
    plt.show()
    plt.savefig(cluster_tech_skills_average_path)
    plt.close()

def get_skills_average_cluster(filepath):
    cluster_tech_skills_average_path = filepath +'cluster_tech_skills_average.png'
    return send_file(cluster_tech_skills_average_path, mimetype='image/png') 


def get_tech_skills_outputs():
    df__ = load_pickle('df_clustering_kmeans.pkl', pickle_directory)
    cluster_skill_average_2 = load_pickle('cluster_tech_skill_average_2.pkl', pickle_directory)
    
    list_tech_skills=[name for name in tech_skills["Skill Name"].unique()]

    emp_ids=df__["EMP ID"].unique()
    clusters=np.array(df__["cluster_1"])
    l=len(np.unique(clusters))

    #Top Tech Skills + Target Points 
    top_tech_skills={}
    target_points_2={}

    for emp,cluster in zip(emp_ids,clusters):
        j=0
        while (j<l):
            if cluster==j:
                for i in range(len(list_tech_skills)):
                    if cluster_skill_average_2[cluster][i]>=7:
                        if emp not in top_tech_skills:
                            top_tech_skills[emp] = []
                            top_tech_skills[emp].append(list_tech_skills[i])

                    if cluster_skill_average_2[cluster][i]<7:
                        if emp not in target_points_2:
                            target_points_2[emp] = []
                            target_points_2[emp].append(list_tech_skills[i])

            j=j+1
    for emp,cluster in zip(emp_ids,clusters):
        if emp not in top_tech_skills:
            top_tech_skills[emp] = ["_"]
        if emp not in target_points_2:
            target_points_2[emp] = ["_"]

    save_pickle('top_tech_skills.pkl', top_tech_skills, pickle_directory)
    save_pickle('target_points_2_tech_skills.pkl', target_points_2, pickle_directory)


# Call the function to display the results of a specific employee
def display_employee_tech_skills_results_1(employee_id, top_tech_skills, target_points):
    data = []
    if employee_id in top_tech_skills and employee_id in target_points:
        employee_top_skills = top_tech_skills[employee_id]
        employee_targets = target_points[employee_id]

        data.append({'employee_id': int(employee_id),
                'employee_top_tech_skills': [', '.join(employee_top_skills)],
                'employee_targets': [', '.join(employee_targets)]})
        print('data', data)
    return data

# Display the result of all the Employees:
def display_all_employee_tech_skills_results_1(top_tech_skills, target_points):
    data = []
    for employee_id in top_tech_skills.keys():
        if employee_id in target_points :
            employee_skills = ', '.join(top_tech_skills[employee_id])
            employee_targets = ', '.join(target_points[employee_id])
            data.append({'employee_id': int(employee_id), 'employee_top_tech_skills': employee_skills, 'employee_targets': employee_targets})
    return data


def display_employees_tech_skills_result(employee_id = None ):
    top_tech_skills = load_pickle('top_tech_skills.pkl', pickle_directory)
    target_points_2 = load_pickle('target_points_2_tech_skills.pkl', pickle_directory)
    employee_data = display_all_employee_tech_skills_results_1(top_tech_skills, target_points_2)
    if employee_id is not None:
        print(type(employee_id))
        employee_data = display_employee_tech_skills_results_1(employee_id, top_tech_skills, target_points_2)
    return jsonify(employee_data)

def generate_general_tech_skills_Result(filepath):
    general_result_path = filepath + '/general_result_tech_skills.png'

    top_tech_skills = load_pickle('top_tech_skills.pkl', pickle_directory)
    target_points_2 = load_pickle('target_points_2_tech_skills.pkl', pickle_directory)

    top_skill_counts = {}
    targets_counts = {}

    for emp in top_tech_skills.values():
        for skill in emp:
            top_skill_counts[skill] = top_skill_counts.get(skill, 0) + 1

    for emp in target_points_2.values():
        for target in emp:
            targets_counts[target] = targets_counts.get(target, 0) + 1

    # Create lists of skills and target along with their corresponding counts
    skills = list(top_skill_counts.keys())
    top_skill_counts = list(top_skill_counts.values())

    targets = list(targets_counts.keys())
    targets_counts = list(targets_counts.values())

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Bar chart for top tech skills
    ax1.barh(skills, top_skill_counts, color='blue')
    ax1.set_xlabel('Number of Employees')
    ax1.set_ylabel('Skill')
    ax1.set_title('Number of Employees with Each Skill as a Top Skill')

    # Bar chart for target points
    ax2.barh(targets, targets_counts, color='red')
    ax2.set_xlabel('Number of Employees')
    ax2.set_ylabel('Target')
    ax2.set_title('Number of Employees with Each skill as a Target Point')

    plt.tight_layout()
    plt.show()
    plt.savefig(general_result_path)
    plt.close()


def general_display_tech_skills_Result(filepath):
    general_result_path = filepath + '/general_result_tech_skills.png'
    return send_file(general_result_path, mimetype='image/png') 
