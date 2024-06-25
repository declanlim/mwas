"""server for other users to run mwas remotely"""
from flask import Flask, request, jsonify
import os
import logging
import mwas_general

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/run_mwas', methods=['POST'])
def run():
    """run mwas"""
    try:
        # Receive CSV data and flags from the request
        csv_data = request.data.decode('utf-8')
        flags = request.args.getlist('flag')
        length = len(csv_data)
        logging.info("Received request with data: %s ... (and %s more chars)", request.data[:20], length - 20)
        logging.info("Received flags: %s", flags)

        # Create a temporary file to store the CSV data
        with open('temp_input_from_request.csv', mode='w') as temp_csv_file:
            temp_csv_file.write(csv_data)
            temp_csv_filepath = temp_csv_file.name

        logging.info("Temporary file created: %s", temp_csv_filepath)

        # execute mwas
        try:
            status = mwas_general.main([temp_csv_filepath] + flags, False)
        except Exception as e:
            status = 1
            logging.error("Error running MWAS: %s", e)

        # Remove the temporary file
        os.remove(temp_csv_filepath)

        return jsonify({"message": f"MWAS process finished with exit code {status}"})

    except Exception as e:
        logging.error("Error processing request: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Server starting...")
    app.run(host='0.0.0.0', port=5000)
