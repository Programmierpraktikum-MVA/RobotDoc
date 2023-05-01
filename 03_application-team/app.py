from flask import Flask, render_template

app = Flask(__name__)

# welcome page
@app.route('/')
def index():
    return render_template('index.html')

# start page
@app.route('/start/')
def start():
    return render_template('start.html')

if __name__ == '__main__':
    app.run(debug=True)