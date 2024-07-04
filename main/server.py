"""server for other users to run mwas remotely"""
import csv

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
        flags = request.json.get('flags')
        destination = request.json.get('dest')
        length = len(request.data)
        logging.info("Received request with data: %s ... (and %s more chars)", request.data[:20], length - 20)
        logging.info("Received flags: %s", flags)
        logging.info("s3_bucket: %s", destination)

        # Create a temporary file to store the CSV data
        with open('temp_input_from_request.csv', mode='w') as temp_csv_file:
            try:
                json_data = request.json.get('data')
            except Exception as e:
                logging.error("Error decoding data: %s", e)
                return jsonify({"error": str(e)}), 500

            writer = csv.writer(temp_csv_file)
            writer.writerow(json_data[0].keys())

            for row in json_data:
                writer.writerow(row.values())

        temp_csv_filepath = os.path.abspath(temp_csv_file.name)

        logging.info("Temporary file created: %s", temp_csv_filepath)

        # execute mwas
        try:
            # turn flags into a list
            flags = flags.split(" ")
            status = mwas_general.main(['server.py', temp_csv_filepath, "--s3-storing", destination] + flags, True)
        except Exception as e:
            status = 1
            logging.error("Error running MWAS: %s", e)

        logging.info("MWAS process finished with exit code %s", status)
        # Remove the temporary file
        os.remove(temp_csv_filepath)
        logging.info("Temporary file removed: %s", temp_csv_filepath)

        return jsonify({"message": f"MWAS process finished with exit code {status}"})

    except Exception as e:
        logging.error("Error processing request: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Server starting...")
    app.run(host='0.0.0.0', port=5000)
