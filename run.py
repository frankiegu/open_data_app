from open_data_app.route import app
# import config
# from flask_debugtoolbar import DebugToolbarExtension

if __name__ == '__main__':
    # toolbar = DebugToolbarExtension(app)
    app.run('0.0.0.0',port=7000)