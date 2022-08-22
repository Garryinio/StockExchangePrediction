from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_object(Config)

from app import routes
