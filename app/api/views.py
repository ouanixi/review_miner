"""API views."""

from app.api import api
from app.api.orchestrate import _prepare_trainer, _summarise_reviews
from flask import jsonify, request
import json


@api.route("/")
def healthcheck():
    """Route to check api is up and running."""
    return "Congratulations! You have successfully launched your project"


@api.route("/score")
def score():
    """Endpoint that returns the score of the classifier specified in the parameters."""
    classifier = request.args.get('classification', '')
    trainer = _prepare_trainer(classifier)
    return jsonify(trainer.score())


@api.route("/predict", methods=['POST'])
def predict_reviews():
    """Return a summary of the review in json file."""
    data = request.get_json()
    summary = json.dumps(_summarise_reviews(data))
    return summary
