from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from modules.funcs import *
from modules.db_model import *
from .auth import *

# default config
app = Flask(__name__)
app.secret_key = "~((<SH,jM_YU9_x3$2f!_x2"

app.register_blueprint(auth.bp)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://xqrornwg:QjZBbShdIqvLjHohXXMfvIXoSuc3lnZr@horton.db.elephantsql.com/xqrornwg'
db.init_app(app)

# flask-login config
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


@app.route("/signup")
def signup():
    return render_template("index.html", signUp=True)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")
