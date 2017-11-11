from open_data_app.route import app
import config
app.run('0.0.0.0',port=7000,debug=config.DEBUG)