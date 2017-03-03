from app.api import api
from app.api.orchestrate import prepare_trainer, summarise_reviews
from flask import jsonify, request
import json


@api.route("/")
def healthcheck():
    return "Congratulations! You have successfully launched your project"


@api.route("/score")
def sentiment_score():
    classifier = request.args.get('classification', '')
    trainer = prepare_trainer(classifier)
    return jsonify(trainer.score())


@api.route("/predict", methods=['POST'])
def predict_reviews():
    data = request.get_json()
    summary = json.dumps(summarise_reviews(data))
    return summary
