from open_data_app.route import app
# import config

from flask_debugtoolbar import DebugToolbarExtension

toolbar = DebugToolbarExtension(app)

app.run('0.0.0.0',port=7000)