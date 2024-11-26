import os
import csv
import pandas as pd
from flask import jsonify, request, send_file

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




def upload_file(filename, filePath):
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file and allowed_file(file.filename):
        return convert_file_to_csv(filename, file, filePath)
    return jsonify({"message": "Invalid file format"}), 400




def convert_file_to_csv(filename, file, filePath):
    try:
        df = pd.read_excel(file) if file.filename.lower().endswith('.xlsx') else pd.read_csv(file)
        # csv_filename = secure_filename(os.path.splitext(file.filename)[0] + ".csv")
        csv_filepath = os.path.join(filePath, filename)
        df.to_csv(csv_filepath, index = False, header = True)
        return jsonify({"message": "File converted to CSV", "csv_filename": filename}), 200
    except Exception as e:
        return jsonify({"message": f"Error converting file: {str(e)}"}), 500




def download_excel_file(filePath, filename):
    return convert_file_to_Excel(filePath, filename)




def convert_file_to_Excel(filePath, filename):
    try:
        csv_filepath = os.path.join(filePath, filename +'.csv')
        excel_filepath = filename + '.xlsx'
        # Detect the delimiter used in the CSV file
        with open(csv_filepath, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            delimiter = dialect.delimiter
        df = pd.read_csv(csv_filepath, delimiter)
        df.to_excel(excel_filepath, index=False)
          # Send the Excel file for downloading
        return send_file(excel_filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"message": f"Error converting file: {str(e)}"}), 500
