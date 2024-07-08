from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.db_model import *
from modules.auth.auth import *
from modules.patients.patients import *
from modules.newmodel.Llava.llava_inference import *
import os
import io
from dotenv import load_dotenv

from util.cache_config import cache
from sshtunnel import SSHTunnelForwarder

load_dotenv()

# Konfigurationsparameter für den SSH-Tunnel und die Datenbank
SSH_HOST = 'newgpu.adastruct.com'
SSH_PORT = 22
SSH_USER = 'pp'
SSH_PASSWORD = os.getenv('SSH_PWD')  
DB_HOST = 'localhost'


#Aufbau des SSH-Tunnels
#server = SSHTunnelForwarder(
#    (SSH_HOST, SSH_PORT),
#    ssh_pkey=None,
#    ssh_username=SSH_USER,
#    ssh_password=SSH_PASSWORD,
#    remote_bind_address=(DB_HOST, 5432),
#    local_bind_address=('localhost', 5432)
# )
#ssh tunnel wird autom. beendee, wenn das Programm beendet wird
#server.start()

# default config
app = Flask(__name__)
app.secret_key = os.getenv("APP_SCRT_KEY")

# URI of the database
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
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


@app.route('/uploadImageForPatient/<int:patient_id>', methods=['POST'])
def uploadHelperPatient(patient_id):
    with current_app.app_context():
        try:
            upload_image_for_patient(patient_id)

            file = request.files['image']
            img = load_image_from_bytes(file)

         
            message = request.form['imgcontext']

            patient = Patients.query.get(patient_id)
            patientInfo = patient.to_dict()

            llava_ouput = image_captioning_with_robodoc(img)
       

            reply = subgraphExtractor.processMessage(patient_id, patientInfo, message, imgCaptioning=llava_ouput)
            return jsonify({"reply": reply, "type": "message"})
        except Exception as e:
            return jsonify({"reply": str(e), "type": "message"})  
  



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
    user_id = current_user.intid

    
    register_patient(name, age, weight,sex,symptoms,user_id)
    cache.delete_memoized(getAllPatients)
    return render_template("patients.html", patients=Patients.query.all())
