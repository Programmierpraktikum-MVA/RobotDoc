from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

"""
This represents the table structure of the database.
The classes' names need to be the exact same as the table name in the database with a capital letter.    
"""


class Accounts(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String)
    hash = db.Column(db.LargeBinary)


# dummy data, database model to implemented
patientData = [
    {"id": 1, "name": "John", "age": 35, "weight": 75.5, "sex": "", "symptoms": ["Headache", "Fever", "Coughing"]},
    {"id": 2, "name": "Sarah", "age": 42, "weight": 68.2, "sex": "", "symptoms": []},
    {"id": 3, "name": "Tamer", "age": 22, "weight": 85.2, "sex": "", "symptoms": []},
    {"id": 4, "name": "Noah", "age": 23, "weight": 70.2, "sex": "", "symptoms": []},
    {"id": 5, "name": "Mike", "age": 54, "weight": 80.1, "sex": "", "symptoms": []}
]
