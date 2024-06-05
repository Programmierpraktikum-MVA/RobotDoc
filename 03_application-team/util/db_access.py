from flask_sqlalchemy import SQLAlchemy
from util.db_model import *
from util.exceptions import *
from util.funcs import validate_username, validate_password
import bcrypt as bcr
from flask import request, jsonify


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
    username_db = None
    rows = db.session.scalars(
        db.select(Accounts.username).filter_by(username=username_cand))
    for row in rows:
        username_db = row
    if username_db == username_cand:
        raise OccupiedUsernameError

    validate_username(username_cand)
    validate_password(password)

    pw = password.encode('UTF-8')
    salt = bcr.gensalt()
    pw = bcr.hashpw(pw, salt)

    new_user = Accounts()
    new_user.username = str(username_cand).strip()
    new_user.hash = pw

    db.session.add(new_user)
    db.session.commit()
    return 0


def upload_image():
    """
    Receives the image data from the client and stores it in the database.
    :return: ID of the stored image
    """
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        new_image = Image(file=file.read())
        db.session.add(new_image)
        db.session.commit()
        return jsonify({'message': 'Image uploaded successfully'}), 201


# def storeImage(image_base64):
#     """
#     Stores the base64 encoded image data into the database.
#     :param image_base64: base64 encoded image data
#     :return: ID of the stored image
#     """
#     new_image = Images()
#     new_image.image_data = image_base64

#     db.session.add(new_image)
#     db.session.commit()

#     return new_image.id
    