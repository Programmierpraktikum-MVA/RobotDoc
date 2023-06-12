from flask_sqlalchemy import SQLAlchemy
from util.db_model import *
from util.exceptions import *
from util.funcs import validate_username, validate_password
import bcrypt as bcr


def register_user(username_cand, password):
    """
    Registers a new user with given username and password.
    :param username_cand: chosen username
    :param password: chosen password
    :return: 0 on success
    :exception OccupiedUsername: if username is in use
    :exception InvalidUsername: if username is invalid (from validate_username())
    :exception InvalidPassword: if password is invalid (from validate_password())
    """
    validate_username(username_cand)
    validate_password(password)

    username_db = db.session.scalars(
        db.select(Accounts.username).filter_by(username=username_cand))
    if username_db == "":
        raise OccupiedUsernameError

    pw = password.encode('UTF-8')
    salt = bcr.gensalt()
    pw = bcr.hashpw(pw, salt)

    new_user = Accounts()
    new_user.username = str(username_cand).strip()
    new_user.hash = pw

    db.session.add(new_user)
    db.session.commit()
    return 0
