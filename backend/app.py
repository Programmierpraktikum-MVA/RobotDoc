from flask import Flask, request, jsonify
from flask_login import LoginManager, login_required, current_user
from flask_cors import CORS
from util.db_model import db, Accounts
from util.db_access import (
    get_patient, get_patient_amount, get_all_patients,
    update_patient_by_id, create_patient, delete_patient_by_id #, respond_to_message
)
from util.auth import login_route, register_route, logout
import os

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:8080", "http://localhost:5173"])
login_manager = LoginManager()
login_manager.init_app(app)
app.secret_key = os.getenv("APP_SCRT_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


#######
## AUTH
#######


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Accounts, int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return jsonify({'error': 'unauthenticated'}), 401

@app.route('/api/check_session')
def check_session():
    if current_user.is_authenticated:
        return jsonify({'status': 'authenticated'}), 200
    return jsonify({'error': 'unauthenticated'}), 401


app.route('/api/login', methods=['POST'])(login_route)
app.route('/api/logout', methods=['POST'])(logout)
app.route('/api/register', methods=['POST'])(register_route)






######
# Patient
######

@app.route('/api/createPatient', methods=['POST'])
@login_required
def create_patient_route():
    data = request.get_json()
    return create_patient(data, user_id=0)

@app.route('/api/patient/<int:patient_id>', methods=['DELETE'])
@login_required
def delete_patient_route(patient_id):
    return delete_patient_by_id(patient_id)

@app.route('/api/patient/count', methods=['GET'])
def get_patient_amount_route():
    return get_patient_amount()

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient_route(patient_id):
    return get_patient(patient_id)

@app.route('/api/patients', methods=['GET'])
def get_all_patients_route():
    return get_all_patients()

@app.route('/api/patient/<int:patient_id>', methods=['PUT'])
def update_patient_route(patient_id):
    return update_patient_by_id(patient_id, request.get_json())




########
### AI
########



#@app.route('/api/respond/<int:patient_id>', methods=['POST'])
#@login_required
#def respond_to_message_route(patient_id):
#    data = request.get_json()
#    return respond_to_message(patient_id, data)
    