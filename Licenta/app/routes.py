import csv
import os
import time

import pandas as pd
import wget
from flask import render_template, redirect, url_for, request
from werkzeug.utils import secure_filename

from app import Loading
from app import app, ALLOWED_EXTENSIONS
from app.forms import DataForm

# from fastai.tabular.all import *
global lrC, knnC, aaC, lstmC


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    Loading.loading = False
    return redirect(url_for('main'))


@app.route('/main', methods=['GET', 'POST'])
def main():
    def algAles():
        global lrC
        lrC = request.form.get('LR')
        global knnC
        knnC = request.form.get('KNN')
        global aaC
        aaC = request.form.get('AA')
        global lstmC
        lstmC = request.form.get('LSTM')

        if request.form.get('LR') or request.form.get('KNN') or request.form.get('AA') or request.form.get('LSTM'):
            return True
        else:
            return False

    form = DataForm()

    if Loading.loading:
        return redirect(url_for('inside'))

    if form.validate_on_submit():

        try:
            file = request.files['file']

            if file and allowed_file(file.filename):
                file.filename = 'dataSet.csv'
                filename = secure_filename(file.filename)
                if os.path.exists("dataSet.csv"):
                    os.remove("dataSet.csv")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                file = open(filename)

                reader = csv.reader(file)
                lines = len(list(reader))
            else:
                print("ajunge aici")
                if os.path.exists("dataSet.csv"):
                    os.remove("dataSet.csv")
                path = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1=1491609600&period2' \
                       '=1649376000&interval=1d&events=history&includeAdjustedClose=true'.format(
                    str(form.dataSet.data))
                wget.download(path, 'dataSet.csv')
        except:
            print('3')
            try:
                print("ajunge aici")
                if os.path.exists("dataSet.csv"):
                    os.remove("dataSet.csv")
                path = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1=1491609600&period2' \
                       '=1649376000&interval=1d&events=history&includeAdjustedClose=true'.format(
                    str(form.dataSet.data))
                wget.download(path, 'dataSet.csv')
                print("Dataset-ul a fost inregistrat!!!!")
            except:

                return render_template('test.html')

        if algAles():
            Loading.loading = True

            return render_template('index.html', form=form, loading=Loading.loading)
    # return render_template('main.html')
    else:
        if request.method == 'POST' and algAles():
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.filename = 'dataSet.csv'
                filename = secure_filename(file.filename)
                if os.path.exists("dataSet.csv"):
                    os.remove("dataSet.csv")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                file = open(filename)

                reader = csv.reader(file)
                lines = len(list(reader))

                if lines != 1260:
                    return render_template('test.html')
            else:
                return render_template('test.html')

            # return render_template('inside.html')

            Loading.loading = True
            return render_template('index.html', form=form, loading=Loading.loading)

    return render_template('index.html', form=form, loading=Loading.loading)


@app.route('/inside', methods=['GET', 'POST'])
def inside():
    time.sleep(5)
    if lrC == '1' or knnC == '2':
        exec(open("LR_KNN.py").read())

    df = pd.read_csv('dataSet.csv')

    mini = df['Close'].min()
    maxi = df['Close'].max()
    avreage = df['Close'].mean()
    amplitude = (maxi - mini) / 2
    standardDeviation = df['Close'].std()
    coeficientAplatizare = df['Close'].kurtosis()
    coeficientSimetrie = df['Close'].skew()

    f = open("dataSet_stats.txt", "w")

    # print("Minim = " + str(mini))
    # print("Maxi = " + str(maxi))
    # print("Amplitudine = " + str(amplitude))
    # print("Media = " + str(avreage))
    # print("Deviatie standard = " + str(standardDeviation))
    # print("Coeficient de aplatizare = " + str(coeficientAplatizare))
    # print("Coeficient de simetrie = " + str(coeficientSimetrie))
    f.write("Minim = " + str(mini) + '\n')
    f.write("Maxim = " + str(maxi) + '\n')
    f.write("Amplitudine = " + str(amplitude) + '\n')
    f.write("Deviatie standard = " + str(standardDeviation) + '\n')
    f.write("Coeficient de aplatizare = " + str(coeficientAplatizare) + '\n')
    f.write("Coeficient de simetrie = " + str(coeficientSimetrie) + '\n')
    f.close()
    if aaC == '3':
        exec(open("Auto_ARIMA.py").read())

    if lstmC == '4':
        exec(open("LSTM.py").read())

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    #change here
    lr = [row for row in list(open(r'F:\Programare\Licenta\LinearRegression_stats.txt'))]
    knn = [row for row in list(open(r'F:\Programare\Licenta\KNN_stats.txt'))]
    arima = [row for row in list(open(r"F:\Programare\Licenta\Auto_ARIMA_stats.txt"))]
    lstm = [row for row in list(open(r'F:\Programare\Licenta\LSTM_stats.txt'))]
    dataSet = [row for row in list(open(r'F:\Programare\Licenta\dataSet_stats.txt'))]
    return render_template('main.html', lr=lr, knn=knn, arima=arima,
                           lstm=lstm, lrC=lrC, knnC=knnC, aaC=aaC, lstmC=lstmC, dataSet=dataSet)

# @app.route('/loading', methods=['GET', 'POST'])
# def exc():
