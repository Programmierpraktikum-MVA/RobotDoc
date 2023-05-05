from flask import Flask, render_template, redirect, request
import flask_login

app = Flask(__name__)
app.secret_key = b'_6#y2L"F4Q8z\n\xec]/'

# PAGE: welcome
@app.route('/')
def index():
    return render_template('index.html')

# PAGE: start
@app.route('/start/')
@flask_login.login_required # requires login
def start():
    return render_template('start.html', user=str(flask_login.current_user.id))

# START: login (reference: https://pypi.org/project/Flask-Login/)

# login manager
login_manager = flask_login.LoginManager() # instantiation
login_manager.init_app(app) # add app

users = {'beta@robotdoc.de': {'password': 'mva'}} # user (mock) database

class User(flask_login.UserMixin): # user object
    pass
@login_manager.user_loader # callback
def user_loader(email):
    if email not in users:
        return
    user = User()
    user.id = email
    return user
@login_manager.request_loader # callback
def request_loader(request):
    email = request.form.get('email')
    if email not in users:
        return
    user = User()
    user.id = email
    return user

# PAGE: login
@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")

    email = request.form['email']
    if email in users and request.form['password'] == users[email]['password']:
        user = User()
        user.id = email
        flask_login.login_user(user)
        return redirect('/start/')

    return render_template("login.html", error='Error logging in!')

# logout
@app.route('/logout/')
def logout():
    flask_login.logout_user()
    return redirect('/login/')

# failure
@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect('/login/')

# END: login

if __name__ == '__main__':
    app.run(debug=True)