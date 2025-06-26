from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class Accounts(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String)
    hash = db.Column(db.LargeBinary)


class Patients(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    #id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    sex = db.Column(db.String)
    symptoms = db.Column(db.ARRAY(db.String))
    user_id = db.Column(db.Integer, db.ForeignKey('accounts.id'))

    def __repr__(self):
        return f'<Patient {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'weight': self.weight,
            'sex': self.sex,
            'symptoms': self.symptoms,
        }




