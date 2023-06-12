from flask_sqlalchemy import SQLAlchemy
from util.db_model import *
import bcrypt as bcr


def register_user(username_cand, password):
    """
    Registers a new user with given username and password.
    Raises exception if username is already in use.
    :param username_cand: chosen username
    :param password: chosen password
    :return: 0 on success
    """
    username_db = db.session.scalars(
        db.select(Accounts.username).filter_by(username=username_cand))
    if username_db == "":
        raise Exception

    pw = password.encode('UTF-8')
    salt = bcr.gensalt()
    pw = bcr.hashpw(pw, salt)

    new_user = Accounts()
    new_user.username = username_cand
    new_user.hash = pw

    db.session.add(new_user)
    db.session.commit()
    return 0
