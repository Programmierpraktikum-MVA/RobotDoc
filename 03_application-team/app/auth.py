from flask import Blueprint
from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from modules.db_model import *

bp = Blueprint('auth', __name__, url_prefix='/')

login_manager = LoginManager()


# user config
class User(UserMixin):
    pass


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/home")
    if request.method == "POST":
        username = request.form["username"]
        remember_me = request.form.get("remember_me")
        username_db = None
        user_data = db.session.scalars(
            db.select(Accounts).filter_by(username=username))
        for row in user_data:
            username_db = row.username
            password_db = row.password
        if username_db is None:
            return render_template("index.html", loginFailed=True)
        password_cand = request.form["password"]
        if username == username_db and password_cand == password_db:
            user = User()
            user.id = username
            if remember_me:
                login_user(user, remember=True)
            else:
                login_user(user)
            return redirect("/home")
        return render_template("index.html", loginFailed=True)
    else:
        return render_template("index.html")


@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect("/login")


@login_manager.user_loader
def load_user(username):
    username_db = db.session.scalars(
        db.select(Accounts.username).filter_by(username=username))
    if username_db == None:
        return
    user = User()
    user.id = username
    return user
