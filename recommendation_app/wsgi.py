from application import app
import auth

if __name__ == '__main__':
	auth.authenticate()


	app.run(debug=True)