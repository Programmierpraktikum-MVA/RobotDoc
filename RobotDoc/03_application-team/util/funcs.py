import requests
from util.exceptions import *
from util.db_model import *
from enum import Enum

import matplotlib

# huggingface api (reference: https://huggingface.co/d4data/biomedical-ner-all)
API_URL = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
headers = {"Authorization": "Bearer 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"}


def validate_username(username):
    """
    Username has to be alphanumeric and at least on character long.
    Raises InvalidUsername Exception.
    :param username: to validate
    :return: None if valid
    """
    if not str(username).isalnum():
        raise InvalidUsernameError
    return


def validate_password(password):
    """
    Password has to be at least 8 characters long.
    Raises InvalidPassword Exception
    :param password: to validate
    :return: None if valid
    """
    if len(password) < 8:
        raise InvalidPasswordError
    return

