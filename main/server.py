"""server for other users to run mwas remotely"""
import sys

from flask import Flask, request, jsonify
import os
import subprocess
from tempfile import NamedTemporaryFile
import mwas_general

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

        # Prepare the subprocess command
        status = mwas_general.main([temp_csv_filepath] + flags, False)

        # After processing, remove the temporary CSV file
        os.remove(temp_csv_filepath)

        return jsonify({"message": f"MWAS processed successfully, with exit code {status}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Server starting...")
    app.run(host='0.0.0.0', port=5000)
