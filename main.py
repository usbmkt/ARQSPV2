from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio

from corinthians_prediction_agent import corinthians_agent

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    opponent = data.get('opponent')
    if not opponent:
        return jsonify({"error": "Opponent is required"}), 400

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(corinthians_agent.collect_and_analyze_match_data(opponent))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
