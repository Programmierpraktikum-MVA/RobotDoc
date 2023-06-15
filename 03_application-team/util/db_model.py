from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Accounts(db.Model):
    """
    represents the table structure of "accounts" at Database
    """
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String)
    hash = db.Column(db.LargeBinary)


# postgreSQL DB config coming soon
patientData = [
    {"id": 1, "name": "John", "age": 35, "weight": 75.5, "sex": "", "symptoms": []},
    {"id": 2, "name": "Sarah", "age": 42, "weight": 68.2, "sex": "", "symptoms": []},
    {"id": 3, "name": "Tamer", "age": 22, "weight": 85.2, "sex": "", "symptoms": []},
    {"id": 4, "name": "Noah", "age": 23, "weight": 70.2, "sex": "", "symptoms": []},
    {"id": 5, "name": "Mike", "age": 54, "weight": 80.1, "sex": "", "symptoms": []}
]
