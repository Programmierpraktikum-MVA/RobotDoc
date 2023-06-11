from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify, Blueprint
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.db_model import *
from util.db_access import *
import bcrypt as bcr

auth = Blueprint('auth', __name__)

# flask-login config
login_manager = LoginManager()


class User (UserMixin):
    pass


@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/home")
    if request.method == "POST":
        username = request.form["username"]
        remember_me = request.form.get("remember_me")
        username_db = None
        pw = None
        user_data = db.session.scalars(
            db.select(Accounts).filter_by(username=username))
        for row in user_data:
            username_db = row.username
            pw = row.hash
        if username_db is None:
            return render_template("index.html", loginFailed=True)
        pw_cand = request.form["password"].encode('UTF-8')
        if username == username_db and bcr.checkpw(pw_cand, pw):
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


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        pw = request.form["password"]
        try:
            register_user(username, pw)
        except:
            print("Didnt work")
        return render_template("signup.html")
    else:
        return render_template("signup.html")


@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect("/login")


@login_manager.user_loader
def load_user(username):
    username_db = db.session.scalars(
        db.select(Accounts.username).filter_by(username=username))
    if username_db is None:
        return
    user = User()
    user.id = username
    return user
