"""server for other users to run mwas remotely"""
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)


@app.route('/run_script', methods=['POST'])
def run_script():
    """Run a Python script with the given arguments and return the output."""
    # Get JSON data from the request
    data = request.json
    args = data.get('args', [])

    # Construct the command
    command = ['nohup', 'python3', 'mwas_general.py'] + args + ['>', 'log.txt', '2>&1', '&']

    try:
        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({"message": "Script is running in the background", "pid": process.pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
