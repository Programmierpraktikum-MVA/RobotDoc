from flask import request, jsonify, session
from flask_login import login_user
from util.db_model import db, Accounts
from util.exceptions import InvalidUsernameError, InvalidPasswordError
import bcrypt as bcr


def validate_username(username):
    if not str(username).isalnum():
        raise InvalidUsernameError
    return


def validate_password(password):
    if len(password) < 8:
        raise InvalidPasswordError
    return

def login_route():
    data = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '').encode('UTF-8')

    user = db.session.execute(
        db.select(Accounts).filter_by(username=username)
    ).scalar_one_or_none()

    if not user or not bcr.checkpw(password, user.hash):
        return jsonify({'error': 'Invalid credentials'}), 401

    login_user(user)

    return jsonify({'message': 'Login successful'}), 200


def register_route():
    data = request.get_json()
    username = str(data.get('username', '')).strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    existing = db.session.scalar(
        db.select(Accounts).filter_by(username=username)
    )
    if existing:
        return jsonify({'error': 'Username already taken'}), 400

    try:
        validate_username(username)
        validate_password(password)
    except (InvalidUsernameError, InvalidPasswordError) as e:
        return jsonify({'error': str(e)}), 400

    hashed_pw = bcr.hashpw(password.encode('utf-8'), bcr.gensalt())

    new_user = Accounts(username=username, hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201


def logout():
    session.clear()
    return jsonify({'message': 'Logged out'}), 200
