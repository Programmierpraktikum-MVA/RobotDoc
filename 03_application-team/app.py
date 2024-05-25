from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.funcs import *
from util.db_model import *
from modules.auth.auth import *
from modules.patients.patients import *

# default config
app = Flask(__name__)
app.secret_key = "~((<SH,jM_YU9_x3$2f!_x2"

# URI of the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://xqrornwg:QjZBbShdIqvLjHohXXMfvIXoSuc3lnZr@horton.db.elephantsql.com/xqrornwg'

db.init_app(app)

app.register_blueprint(auth)
app.register_blueprint(patients)

login_manager.init_app(app)
login_manager.login_view = "login"


@app.route("/")
@login_required
def start():
    if current_user.is_authenticated:
        return redirect("/home")


@app.route("/home")
@login_required
def home():
    return render_template("home.html", user=str(current_user.id))
