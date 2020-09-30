# Importing necessary libraries
import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin

# Declaring the flask app
app = Flask(__name__)

#Loading the model from pickle file
# model = pickle.load(open('rf.pkl', 'rb'))



@app.route('/', methods=['POST','GET'])
def main():
    return 'Deploy Model Tutorial'


if __name__ == '__main__':
    app.run()