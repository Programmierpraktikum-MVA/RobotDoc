from flask import Flask, Response, url_for, request, session, abort, render_template, redirect
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
import psycopg2
from flask_sqlalchemy import SQLAlchemy


# default config
app = Flask(__name__)
app.secret_key = "~((<SH,jM_YU9_x3$2f!_x2"

# postgreSQL DB config

users = {"Doc1": {"password": "mva2023"}}


# flask-login config
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
# user config


class User (UserMixin):
    pass


@ app.route("/")
@login_required
def start():
    if current_user.is_authenticated:
        return redirect("/home")


@ app.route("/home")
@login_required
def home():
    return render_template("home.html", user=str(current_user.id))


@ app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/home")
    if request.method == "POST":
        username = request.form["username"]
        remember_me = request.form.get("remember_me")
        if username in users and request.form["password"] == users[username]["password"]:
            user = User()
            user.id = username
            if (remember_me):
                login_user(user, remember=True)
            else:
                login_user(user)
            return redirect("/home")
        return render_template("index.html", loginFailed=True)
    else:
        return render_template("index.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect("/login")


@ login_manager.user_loader
def load_user(username):
    if username not in users:
        return
    user = User()
    user.id = username
    return user
