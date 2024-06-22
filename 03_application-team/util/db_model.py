from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

"""
This represents the table structure of the database.
The classes' names need to be the exact same as the table name in the database with a capital letter.    
"""


class Accounts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String)
    hash = db.Column(db.LargeBinary)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file = db.Column(db.LargeBinary, nullable=False)
    def __repr__(self):
        return f'<Image {self.id}>'

class Patients(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    sex = db.Column(db.String)
    symptoms = db.Column(db.ARRAY(db.String))
    #user_id = db.Column(db.Integer, db.ForeignKey('Accounts.id'))

    def __repr__(self):
        return f'<Patient {self.name}>'




# # dummy data, database model to implemented
# patientData = [
#     {"id": 1, "name": "John", "age": 35, "weight": 75.5, "sex": "", "symptoms": ["Headache", "Fever", "Coughing"]},
#     {"id": 2, "name": "Sarah", "age": 42, "weight": 68.2, "sex": "", "symptoms": []},
#     {"id": 3, "name": "Tamer", "age": 22, "weight": 85.2, "sex": "", "symptoms": []},
#     {"id": 4, "name": "Noah", "age": 23, "weight": 70.2, "sex": "", "symptoms": []},
#     {"id": 5, "name": "Mike", "age": 54, "weight": 80.1, "sex": "", "symptoms": []}
# ]
