
import logging

from manage import app


@api.route("/")
def congratulations():
    return "Congratulations! You have successfully launched your project"

