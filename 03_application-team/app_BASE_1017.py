from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.funcs import *
from util.db_model import *
from modules.auth.auth import *
from modules.patients.patients import *

from util.cache_config import cache
from sshtunnel import SSHTunnelForwarder

# Konfigurationsparameter für den SSH-Tunnel und die Datenbank
SSH_HOST = 'gpu.adastruct.com'
SSH_PORT = 22
SSH_USER = 'pp'
SSH_PASSWORD = "Z7.b'NV9i$n6"  
DB_HOST = 'localhost'


# Aufbau des SSH-Tunnels
server = SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_pkey=None,
    ssh_username=SSH_USER,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=(DB_HOST, 5432),
    local_bind_address=('localhost', 5432)
)
#ssh tunnel wird autom. beendee, wenn das Programm beendet wird
server.start()

# default config
app = Flask(__name__)
app.secret_key = "~((<SH,jM_YU9_x3$2f!_x2"

# URI of the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://admin:0gKtt43obCX7@localhost:5432/robotdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

cache.init_app(app)


db.init_app(app)

app.register_blueprint(auth)
app.register_blueprint(patients)

login_manager.init_app(app)
login_manager.login_view = "login"




@app.route("/sendImage", methods=['POST'])
def uploadHelper():
    return upload_image()
    # response = upload_image()
    # if response.status_code == 201:
    #     return redirect("/home")
    # else:
    #     return Response(status=500)

@app.route("/")
@login_required
def start():
    if current_user.is_authenticated:
        return redirect("/home")
    
# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Dies stellt sicher, dass die Tabelle 'images' erstellt wird
#     app.run(debug=True)


@app.route("/home")
@login_required
def home():
    return render_template("home.html", user=str(current_user.id))

@app.route('/add_patient')
def add_patient():
    return render_template('add_patient.html')

@app.route("/createPatient", methods=["GET", "POST"])
@login_required
def createPatient():
    name = request.form["name"]
    age = int(request.form["age"])
    weight = float(request.form["weight"])
    sex = request.form["sex"]
    symptoms = request.form['symptoms'].split(',')

    register_patient(name, age, weight,sex,symptoms)
    return render_template("patients.html", patients=Patients.query.all())
