from flask import Flask
import nltk
from flask_cors import CORS
from flask import Flask, send_from_directory, jsonify, request
from utils import dataset_manage, eda_model_keys_words, model_predict_position

# declare constants
# HOST = '0.0.0.0'
# PORT = 8081

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES_FOLDER'] = './images'
app.config['GLOBAL_VARIABLE'] = 'This is a global value'
CORS(app, origins = "*")

# nltk.data.path.append('path_to_your_nltk_data_directory')

@app.route("/")
def home():
    model_predict_position.enter_employee_info()
    return "Hello World!"



@app.route('/upload/<filename>', methods=['POST'])
def upload_file(filename):
    return dataset_manage.upload_file(filename, app.config['UPLOAD_FOLDER'])



@app.route('/download_excel/<filename>', methods=['GET'])
def download_excel_file(filename):
    return dataset_manage.download_excel_file(app.config["UPLOAD_FOLDER"], filename)

#bbo
@app.route('/download_csv/<filename>', methods=['GET'])
def download_csv(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename+'.csv')


@app.route('/download_excel_structure/<filename>', methods=['GET'])
def download_excel_structure(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename+'.xlsx')




@app.route('/get_chart_data/<filename>', methods=['GET'])
def get_chart_data(filename):
    return eda_model_keys_words.data_preprocessing(app.config["UPLOAD_FOLDER"]+ '/'+ filename)




#################################### PICKLE ###################################################

@app.route('/apply_tfidf', methods=['GET'])
def train_tfidf():
    try:
        eda_model_keys_words.apply_tf_idf()
        return jsonify({'message': 'TF-IDF model trained and saved.'})
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/get_key_words', methods=['GET'])
def get_key_words():
    try:
        return jsonify(eda_model_keys_words.get_key_words())
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/preprocessing_soft_skills/<filename>', methods=['GET'])
def data_preprocessing_soft_skills(filename):
    try:
        return eda_model_keys_words.data_preprocessing_soft_skills(filename)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_missing_values/<filename>', methods=['GET'])
def get_missing_values(filename):
    try:
        return eda_model_keys_words.get_missing_values(filename)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_number_missing_values_column', methods=['GET'])
def get_number_missing_values_column():
    try:
        return eda_model_keys_words.get_number_missing_values_column()
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_missing_columns_value_name', methods=['GET'])
def get_missing_columns_value_name():
    try:
        return eda_model_keys_words.get_missing_columns_value_name()
    except Exception as e:
        return jsonify({'error': str(e)})

# needed in retrain     
@app.route('/remove_employees_columns_missing_values', methods=['GET'])
def remove_employees_columns_missing_values():
    try:
        eda_model_keys_words.remove_employees_columns_missing_values()
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_soft_skills_dendrograms', methods=['GET'])
def get_soft_skills_dendrograms():
    try:
        return eda_model_keys_words.get_soft_skills_dendrograms(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_optimal_number_cluster', methods=['GET'])
def get_optimal_number_cluster():
    try:
        return eda_model_keys_words.get_optimal_number_cluster(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_soft_skills_scatterplot', methods=['GET'])
def get_soft_skills_scatterplot():
    try:
        return eda_model_keys_words.get_soft_skills_scatterplot(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_cluster_distribution', methods=['GET'])
def get_cluster_distribution():
    try:
        return eda_model_keys_words.get_cluster_distribution(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_cluster_categories', methods=['GET'])
def get_cluster_categories():
    try:
        return eda_model_keys_words.get_cluster_categories(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_improvement_level_cluster', methods=['GET'])
def get_improvement_level_cluster():
    try:
        return eda_model_keys_words.get_improvement_level_cluster(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/display_employees_result/', defaults={'emp_id': None}, methods=['GET'])
@app.route('/display_employees_result/<int:emp_id>', methods=['GET'])
def display_employees_result(emp_id):
    try:
        # emp_id = request.args.get('emp_id')  # Get the 'id' parameter from the query string
        return eda_model_keys_words.display_employees_result(emp_id)
    except Exception as e:
        return jsonify({'error': str(e)})

  
@app.route('/general_display_soft_skills_Result', methods=['GET'])
def general_display_soft_skills_Result():
    try:
        return eda_model_keys_words.general_display_soft_skills_Result(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/preprocessing_tech_skills', methods=['GET'])
def data_preprocessing_tech_skills():
    try:
        return eda_model_keys_words.data_preprocessing_tech_skills()
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/get_tech_skills_missing_values', methods=['GET'])
def get_tech_skills_missing_values():
    try:
        return eda_model_keys_words.get_tech_skills_missing_values()
    except Exception as e:
        return jsonify({'error': str(e)})
    
    
@app.route('/get_number_missing_tech_skills_values_column', methods=['GET'])
def get_number_missing_tech_skills_values_column():
    try:
        return eda_model_keys_words.get_number_missing_tech_skills_values_column()
    except Exception as e:
        return jsonify({'error': str(e)})
    

    
@app.route('/get_missing_columns_tech_skills_value_name', methods=['GET'])
def get_missing_columns_tech_skills_value_name():
    try:
        return eda_model_keys_words.get_missing_columns_tech_skills_value_name()
    except Exception as e:
        return jsonify({'error': str(e)})


# needed in retrain     
@app.route('/remove_employees_columns_missing_tech_skills_values', methods=['GET'])
def remove_employees_columns_missing_tech_skills_values():
    try:
        eda_model_keys_words.remove_employees_columns_missing_tech_skills_values()
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_tech_skills_dendrograms', methods=['GET'])
def get_tech_skills_dendrograms():
    try:
        return eda_model_keys_words.get_tech_skills_dendrograms(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    

    
@app.route('/get_optimal_number_tech_skills_cluster', methods=['GET'])
def get_optimal_number_tech_skills_cluster():
    try:
        return eda_model_keys_words.get_optimal_number_tech_skills_cluster(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_tech_skills_scatterplot', methods=['GET'])
def get_teck_skills_scatterplot():
    try:
        return eda_model_keys_words.get_tech_skills_scatterplot(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    

    
@app.route('/get_cluster_distribution_tech_skills', methods=['GET'])
def get_cluster_distribution_tech_skills():
    try:
        return eda_model_keys_words.get_cluster_distribution_tech_skills(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    
   
@app.route('/get_cluster_categories_tech_skills', methods=['GET'])
def get_cluster_categories_tech_skills():
    try:
        return eda_model_keys_words.get_cluster_categories_tech_skills(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
   
@app.route('/get_skills_average_cluster', methods=['GET'])
def get_skills_average_cluster():
    try:
        return eda_model_keys_words.get_skills_average_cluster(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/display_employees_tech_skills_result/', defaults={'emp_id': None}, methods=['GET'])
@app.route('/display_employees_tech_skills_result/<int:emp_id>', methods=['GET'])
def display_employees_tech_skills_result(emp_id):
    try:
        return eda_model_keys_words.display_employees_tech_skills_result(emp_id)
    except Exception as e:
        return jsonify({'error': str(e)})

  
@app.route('/general_display_tech_skills_Result', methods=['GET'])
def general_display_tech_skills_Result():
    try:
        return eda_model_keys_words.general_display_tech_skills_Result(app.config['IMAGES_FOLDER'])
    except Exception as e:
        return jsonify({'error': str(e)})
    


####################################### Model 3 ########################################

@app.route('/get_employee_info_chart', methods=['GET'])
def get_employee_info_chart():
    return model_predict_position.get_employee_info_chart()

@app.route('/get_employee_msno_chart', methods=['GET'])
def get_employee_msno_chart():
    return model_predict_position.get_employee_msno_chart(app.config['IMAGES_FOLDER'])

@app.route('/get_all_positions', methods=['GET'])
def get_all_positions():
    return model_predict_position.get_all_positions()


@app.route('/get_bar_features_distribution/<featurename>', methods=['GET'])
def get_bar_features_distribution(featurename):
    return model_predict_position.get_bar_features_distribution(app.config['IMAGES_FOLDER'], featurename)



@app.route('/get_pie_features_distribution/<featurename>', methods=['GET'])
def get_pie_features_distribution(featurename):
    return model_predict_position.get_pie_features_distribution(app.config['IMAGES_FOLDER'], featurename)



@app.route('/get_numeric_features', methods=['GET'])
def get_numeric_features():
    return model_predict_position.get_numeric_features(app.config['IMAGES_FOLDER'])



@app.route('/get_target_dependency/<dependencyname>', methods=['GET'])
def get_target_dependency(dependencyname):
    return model_predict_position.get_target_dependency(app.config['IMAGES_FOLDER'], dependencyname)




@app.route('/get_preprocessing_employee_position_data', methods=['GET'])
def get_preprocessing_employee_position_data():
    return model_predict_position.get_preprocessing_employee_position_data(app.config['IMAGES_FOLDER'])

    
@app.route('/get_position_missing_values', methods=['GET'])
def get_position_missing_values():
    try:
        return model_predict_position.get_position_missing_values()
    except Exception as e:
        return jsonify({'error': str(e)})


    
@app.route('/get_missing_columns_position_value_name', methods=['GET'])
def get_missing_columns_position_value_name():
    try:
        return model_predict_position.get_missing_columns_position_value_name()
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/get_correlation', methods=['GET'])
def get_correlation():
    return model_predict_position.get_correlation(app.config['IMAGES_FOLDER'])

@app.route('/get_model_evaluation_data', methods=['GET'])
def get_model_evaluation_data():
    return model_predict_position.get_model_evaluation_data(app.config['IMAGES_FOLDER'])

@app.route('/get_model_performance', methods=['GET'])
def get_model_performance():
    return model_predict_position.get_model_performance(app.config['IMAGES_FOLDER'])

@app.route('/get_classification_report', methods=['GET'])
def get_classification_report():
    return model_predict_position.get_classification_report(app.config['IMAGES_FOLDER'])

@app.route('/get_classification_prediction_error', methods=['GET'])
def get_classification_prediction_error():
    return model_predict_position.get_classification_prediction_error(app.config['IMAGES_FOLDER'])

@app.route('/get_model_importance', methods=['GET'])
def get_model_importance():
    return model_predict_position.get_model_importance(app.config['IMAGES_FOLDER'])

@app.route('/get_positions_missing_values', methods=['GET'])
def get_positions_missing_values():
    return model_predict_position.get_positions_missing_values()


@app.route('/get_new_prediction', methods=['POST'])
def get_new_prediction():
    employee_data = request.get_json()
    print('data ...', employee_data)
    return model_predict_position.get_new_prediction(employee_data)


    # return jsonify({"message": "Pie chart generated", "chart_image_path": image_path})
# if __name__ == '__main__':
#     # run web server
#     app.run(host=HOST,
#             debug=True,  # automatic reloading enabled
#             port=PORT)