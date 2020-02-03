from flask import Blueprint
from flask import Flask, render_template

home = Blueprint('main', __name__)

@main.route("/")
def home():
    return render_template("home.html")