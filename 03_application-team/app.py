from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from funcs import *
from db_model import *

# default config
app = Flask(__name__)
app.secret_key = "~((<SH,jM_YU9_x3$2f!_x2"

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://xqrornwg:QjZBbShdIqvLjHohXXMfvIXoSuc3lnZr@horton.db.elephantsql.com/xqrornwg'
db.init_app(app)

# flask-login config
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# user config
class User(UserMixin):
    pass


@app.route("/")
@login_required
def start():
    if current_user.is_authenticated:
        return redirect("/home")


@app.route("/home")
@login_required
def home():
    return render_template("home.html", user=str(current_user.id))


@app.route("/patients")
@login_required
def patients():
    return render_template("patients.html", patients=patientData)


@app.route("/assignTokens/<int:id>", methods=["POST"])
@login_required
def assignTokens(id):
    try:
        textToconvert = request.form.get("textToConvert")
        output = query({
            "inputs": textToconvert
        })
        parsedOutput = parseString(output)
    except:
        parsedOutput = "Error"
    if "Sign_symptom" in parsedOutput:
        print(parsedOutput["Sign_symptom"])
        patientData[id - 1]["symptoms"].append(parsedOutput["Sign_symptom"])
    return render_template("patientSpec.html", patientData=patientData[id - 1])


@app.route("/patients/<int:id>")
@login_required
def patients_route(id):
    print("You pressed on: " + str(id))
    return render_template("patientSpec.html", patientData=patientData[id - 1])


@app.route("/sendInput", methods=["POST"])
@login_required
def convertText():
    textToconvert = request.form.get("textToConvert")
    try:
        output = query({
            "inputs": textToconvert
        })
        cleanOutput = convertString(output)
    except:
        cleanOutput = "Error"
    print(cleanOutput)
    return render_template("home.html", user=str(current_user.id), output=cleanOutput, initialText=textToconvert)


@app.route("/login", methods=["GET", "POST"])
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
        if username_db == None:
            return render_template("index.html", loginFailed=True)
        password_cand = request.form["password"]
        if username == username_db and password_cand == password_db:
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


@app.route("/signup")
def signup():
    return render_template("index.html", signUp=True)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


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
