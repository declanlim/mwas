"""server for other users to run mwas remotely"""
from flask import Flask, request, jsonify
import os
import subprocess
from tempfile import NamedTemporaryFile

app = Flask(__name__)


@app.route('/run', methods=['POST'])
def run():
    """run mwas"""
    try:
        # Receive CSV data from request
        csv_data = request.data.decode('utf-8')

        # flags for mwas
        flags = request.args.getlist('flag')

        # Create a temporary file to store the CSV data
        with NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as temp_csv_file:
            temp_csv_file.write(csv_data)
            # Get the filepath of the temporary CSV file
            temp_csv_filepath = temp_csv_file.name

        log_file_path = os.path.join(os.path.dirname(temp_csv_filepath), 'log.txt')

        # Prepare the subprocess command
        command = ['python3', 'mwas_general.py', temp_csv_filepath] + flags

        # Execute the subprocess command with the temporary CSV file as an argument
        with open(log_file_path, 'w') as log_file:
            # Execute the subprocess command with the temporary CSV file as an argument
            process = subprocess.Popen(command, cwd=os.path.dirname(temp_csv_filepath), stdout=log_file, stderr=log_file)
            process.communicate()  # Wait for the process to complete

        # After processing, remove the temporary CSV file
        os.remove(temp_csv_filepath)

        return jsonify({"message": "MWAS processed successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
