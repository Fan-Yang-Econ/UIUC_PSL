# How to Test the Movie Rating App in Local?

### Start the backend server

We require `Python 3.7+` to run the backend server.
```
cd MovieAppBackendServer
pip3 install -r requirements.txt
python3 manage.py runserver 7000 --settings MovieAppBackendServer.settings_dev
```

### Start the frontend server

We require `npm` installed to run the frontend server.

```
cd MovieAppFrontend
npm install -g npx
npm install
npx next dev
```

Then you can go to http://localhost:3000/ in your browser to play with the app.