from flask import Flask, render_template, request, session, redirect, send_file
import json, keras, math, os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tensorflow as tf
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
import pymysql
from board import Board
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'This is secret key'

@app.route('/')
def viewchart():
    return render_template('kdhpf.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/subplot')
def subplot():
    b = path.exists('static/subplot.png')
    if not b:
        t = np.arange(0, 5, 0.1)
        y1 = np.sin(2 * np.pi * t)
        y2 = np.sin(2 * np.pi * t + np.pi)
        plt.subplot(211)
        plt.plot(t, y1, 'b-.')
        plt.legend([r'$sin(x)$'])
        plt.subplot(212)
        plt.plot(t, y2, 'r--')
        plt.legend([r'$sin(x+1)$'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('static/subplot.png')
    plt.close()
    img = {'chartname': 'static/subplot.png'}
    return json.dumps(img)

@app.route('/triplot')
def triplot():
    b = path.exists('static/triplot.png')
    if not b:
        rad = np.arange(0, np.pi * 2, 0.01)
        sv = np.sin(rad)
        cv = np.cos(rad)
        tv = np.tan(rad)
        plt.title('sin/cos/tan')
        plt.axis([0, np.pi * 2, -1.5, 1.5])
        plt.plot(rad, sv, 'r.', rad, cv, 'g.', rad, tv, 'b.')
        plt.legend([r'$y=sin(x)$', r'$y=cos(x)$', r'$y=tan(x)$'])
        plt.xlabel('Radian angles')
        plt.ylabel('Return Value')
        plt.grid(True)
        plt.savefig('static/triplot.png')
    plt.close()
    img = {'chartname': 'static/triplot.png'}
    return json.dumps(img)

@app.route('/cumsum')
def cumsum():
    b = path.exists('static/cumsum.png')
    if not b:
        arr = np.random.randint(0, 50, 20)
        plt.plot(arr.cumsum())
        plt.savefig('static/cumsum.png')
    plt.close()
    img = {'chartname': 'static/cumsum.png'}
    return json.dumps(img)

@app.route('/barchart')
def barchart():
    df = pd.DataFrame(np.random.randint(1, 16, 16).reshape(4, 4))
    mean_axis_0 = df.mean()
    std = df.std()

    plt.bar(np.arange(4), mean_axis_0, yerr=std, error_kw={'ecolor': '0', 'capsize': 5})
    plt.savefig('static/barchart.png')
    plt.close()
    img = {'chartname': 'static/barchart.png'}
    return json.dumps(img)

@app.route('/bar')
def bar():
    b = path.exists('static/bar.png')
    if not b:
        idx = np.arange(5)
        y1 = np.random.random(5)
        y2 = np.random.random(5)
        y1 = np.round(y1, 2)
        y2 = np.round(y2, 2)

        plt.bar(idx, y1)
        plt.bar(idx, -y2)
        plt.ylim(-1.2, 1.2)

        for x, y in zip(idx, y1):
            plt.text(x, y + 0.05, f'{y}', ha='center', va='bottom')

        for x, y in zip(idx, y2):
            plt.text(x, -y - 0.05, f'{-y}', ha='center', va='top')
        plt.savefig('static/bar.png')
    plt.close()
    img = {'chartname': 'static/bar.png'}
    return json.dumps(img)

@app.route('/circle')
def circle():
    b = path.exists('static/circle.png')
    if not b:
        values = [10, 30, 20, 50]
        colors = ['blue', 'yellow', 'green', 'black']
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        explode = [0.2, 0, 0, 0]
        plt.pie(values, colors=colors, labels=labels, explode=explode)
        plt.axis('equal')
        plt.savefig('static/circle.png')
    plt.close()
    img = {'chartname': 'static/circle.png'}
    return json.dumps(img)

@app.route('/axes3d')
def Axes3Dplot():
    b = path.exists('static/axes3d.png')
    if not b:
        fig = plt.figure()
        ax = Axes3D(fig)
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        Z = np.random.randn(40 * 40).cumsum().reshape(40, 40)

        X, Y = np.meshgrid(X, Y)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
        plt.savefig('static/axes3d.png')
    plt.close()
    img = {'chartname': 'static/axes3d.png'}
    return json.dumps(img)

@app.route('/addsubplot')
def addsubplot():
    b = path.exists('static/addsubplot.png')
    if not b:
        gs = plt.GridSpec(3, 3)
        fig = plt.figure(figsize=(6, 6))
        s1 = fig.add_subplot(gs[1, :2])
        s2 = fig.add_subplot(gs[0, :2])
        s3 = fig.add_subplot(gs[2, 0])
        s4 = fig.add_subplot(gs[:2, 2])
        s5 = fig.add_subplot(gs[2, 1:])
        plt.savefig('static/addsubplot.png')
    plt.close()
    img = {'chartname': 'static/addsubplot.png'}
    return json.dumps(img)

@app.route('/machinelearning')
def machinelearning():
    return render_template('machinelearning.html')

@app.route('/hand', methods=['post', 'get'])
def hand():
    return render_template('handwrite.html')

@app.route('/handwrite', methods=['post','get'])
def handwrite():
    data = request.files.to_dict()
    data['filename'].save('static/'+f"{secure_filename(data['filename'].filename)}")
    img = load_img('static/'+f"{secure_filename(data['filename'].filename)}", color_mode='grayscale')

    model = tf.keras.models.load_model('handwrite.h5')

    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 784)

    img_array /= 255
    pred = model.predict(img_array)
    pred = np.argmax(pred, axis=1)
    return str(list(pred)[0])

@app.route('/lstm', methods=['post', 'get'])
def lstm():
    if not path.exists('static/LSTM.png'):
        dataframe = pd.read_csv('cansim-0800020-eng-6674700030567901031.csv', skiprows=6)['Unadjusted']
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        scaler = MinMaxScaler()
        dataset = scaler.fit_transform(dataset[:, np.newaxis])

        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:, :]

        look_back = 3
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], 3, 1))
        testX = np.reshape(testX, (testX.shape[0], 3, 1))

        model = tf.keras.models.load_model('LSTM.h5')

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (look_back+1):len(dataset)-2, :] = testPredict

        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)

        plt.savefig('static/LSTM.png')
        plt.close()
    img = 'LSTM.png'
    return render_template('lstm.html', img=img)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-3):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

@app.route('/kmeans')
def kmeans_home():
    return render_template('kmeans.html')

@app.route('/kmeans/<int:num>')
def kmeans(num):
    X, y_label = make_blobs(n_samples=500, centers=4, cluster_std=0.4, random_state=num)
    plt.scatter(X[:, 0], X[:, 1], c=y_label, s=20, alpha=0.5)
    plt.savefig(f'static/kmeans{num}.png')
    plt.close()
    imgname = f'kmeans{num}.png'
    img = {'img': imgname}
    return json.dumps(img)

@app.route('/meanshift')
def meanshift_home():
    return render_template('meanshift.html')

@app.route('/meanshift/<int:num>')
def meanshift(num):
    X, y_label = make_blobs(n_samples=500, centers=num, cluster_std=1)

    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    n_clusters_ = len(np.unique(labels))
    print("Estimated clusters:", n_clusters_)

    colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker="x", color='k', s=150, linewidths=5, zorder=10)

    plt.savefig(f'static/meanshift{num}.png')
    plt.close()

    imgname = f'meanshift{num}.png'
    img = {'img': imgname}
    return json.dumps(img)

@app.route('/bbs')
def bbs():
    return render_template('bbs/bbs_login.html')

@app.route('/bbs/login', methods=['post'])
def login():
    data = request.form
    session['uid'] = data['uid']
    return redirect('/bbs/list')

@app.route('/bbs/logout', methods=['post', 'get'])
def logout():
    session['uid'] = ''
    return redirect('/bbs/list')

@app.route('/bbs/page/<int:page>')
def page(page):
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        curs.execute("SELECT * FROM bbs")
        num = int(len(curs.fetchall())/15)+1
        curs.execute('SET @RN:=0')
        # sql = "SELECT * FROM (SELECT @RN:=@RN+1 RN, b.* FROM bbs b WHERE (@RN:=0)=0)t1 WHERE RN BETWEEN %s AND %s"
        sql = "SELECT * FROM (SELECT FLOOR((RN-1)/15+1) page, t1.* FROM (SELECT @RN:=@RN+1 RN, b.* FROM bbs b WHERE (@RN:=0)=0)t1)t2 WHERE page=%s"
        curs.execute(sql, page)
        lists = curs.fetchall()
        sql1 = "SELECT a.fname, a.fsize FROM bbs b LEFT OUTER JOIN attach a ON b.num=a.num"
        curs.execute(sql1)
        flists = curs.fetchall()
        if session.get('uid'):
            uid = session.get('uid')
        elif not session.get('uid'):
            uid = ''
        return render_template('bbs/bbs_list.html', lists=lists, uid=uid, flists=flists, num=num)
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/bbs/download/<int:fid>')  # 서버의 폴더구조와 다른 URL을 사용하여 요청한다
def download_attach(fid):
    print('다운로드 요청 :', fid)
    conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                           db='cloudheat', charset='utf8')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    curs.execute("SELECT * FROM attach WHERE fid=%s", fid)
    name = curs.fetchone()
    file_name = f"static/uploads/{secure_filename(name['fname'])}"   # 실제의 폴더 구조
    return send_file(file_name, mimetype=None, attachment_filename=name['or_fname'], as_attachment=True)

@app.route('/bbs/read/<int:num>')
def numcontent(num):
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = 'UPDATE bbs SET hitcnt = hitcnt + 1 WHERE num = %s'
        curs.execute(sql, num)
        sql1 = 'SELECT b. *, a. * FROM bbs b LEFT OUTER JOIN attach a on b.num = a.num WHERE b.num = %s'
        curs.execute(sql1, num)
        content = curs.fetchone()
        content_list = content['content'].split('\n')
        names = get_file(curs, num)
        conn.commit()
        if session.get('uid'):
            uid = session.get('uid')
        elif not session.get('uid'):
            uid = ''
        return render_template('bbs/bbs_content.html', content=content, uid=uid, content_list=content_list,
                               names=names)
    except Exception as e:
        print(e)
    finally:
        conn.close()

def get_file(curs, num):
    curs.execute('SELECT * FROM attach WHERE num=%s', num)
    names = curs.fetchall()
    if names:
        return names
    else:
        return ''

@app.route('/bbs/edit/<int:num>')
def edit(num):
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = "SELECT * FROM bbs WHERE num=%s"
        curs.execute(sql, num)
        content = curs.fetchone()
        content_list = content['content'].split('\n')
        names = get_file(curs, num)
        if session.get('uid'):
            uid = session.get('uid')
        elif not session.get('uid'):
            uid = ''
        return render_template('bbs/bbs_edit.html', content=content, uid=uid, content_list=content_list, names=names)
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/bbs/update', methods=['post'])
def update():
    try:
        data = request.form
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = "UPDATE bbs SET title=%s, content=%s where num=%s"
        res = curs.execute(sql, (data['title'], data['content'], data['num']))
        conn.commit()
        return redirect('/bbs/read/'+data['num'])
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/bbs/delete/<int:num>', methods=['post', 'get'])
def delete(num):
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = "DELETE FROM bbs WHERE num=%s"
        curs.execute(sql, num)
        conn.commit()
        return redirect('/bbs/list')
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/bbs/write')
def write():
    if session.get('uid'):
        uid = session.get('uid')
    elif not session.get('uid'):
        uid = ''
    return render_template('bbs/bbs_write.html', uid=uid)


@app.route('/bbs/write1', methods=['post', 'get'])
def write1():
    uid = session.get('uid')
    data = request.form
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql1 = "INSERT INTO bbs(title, author, wdate, content) VALUES(%s, %s, NOW(), %s)"
        curs.execute(sql1, (data['title'], uid, data['content']))
        if not request.files.to_dict():
            conn.commit()
        else:
            saved = file_handler(curs)
            if saved:
                conn.commit()
            else:
                conn.rollback()
        return redirect('/bbs/list')
    except Exception as e:
        print(e)
    finally:
        conn.close()

def file_handler(curs):
    fdic = request.files.to_dict()
    upload_cnt = len(fdic)
    oknum = 0
    try:
        for key in fdic:
            or_fname = fdic[key].filename
            while True:
                b = path.exists('static/uploads/'+fdic[key].filename)
                a = fdic[key].filename.split('.')
                name = a[0]
                ext = a[1]
                dt_obj = datetime.now()
                timest = dt_obj.timestamp()
                ts = math.floor(timest*10000000)
                fdic[key].filename = name+str(ts)+'.'+ext
                if not b:
                    break
            fdic[key].save('static/uploads/'+f"{secure_filename(fdic[key].filename)}")
            n = os.path.getsize('D:/PyCharmProjects/PythonWeb/static/uploads/'+secure_filename(fdic[key].filename))
            n = ("{0:.3f}".format(n / 1024))
            sql2 = "SELECT max(num) FROM bbs"
            curs.execute(sql2)
            num = curs.fetchone()
            sql3 = "INSERT INTO attach(num, fname, fsize, or_fname) VALUES(%s, %s, %s, %s)"
            nrow = curs.execute(sql3, (num['max(num)'], fdic[key].filename, str(n)+" KB", or_fname))
            oknum += nrow
        if oknum == upload_cnt:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

@app.route('/bbs/search')
def search():
    return render_template('bbs/bbs_search.html')

@app.route('/bbs/search1', methods=['post'])
def search1():
    data = request.form
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = 'SELECT * FROM bbs WHERE '+data['s_menu']+' LIKE \'%'+data['search']+'%\''
        curs.execute(sql)
        info = curs.fetchall()
        return render_template('bbs/bbs_s_result.html', content=info)
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/bbs/hitcnt/<int:num>', methods=['post'])
def hitcnt(num):
    try:
        conn = pymysql.connect(host='db4free.net', user='cloudheat', password='rlaehgus',
                               db='cloudheat', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        sql = "SELECT * FROM bbs WHERE num=%s"
        curs.execute(sql, num)
        data = curs.fetchone()
        sql = "UPDATE bbs SET hitcnt=%s WHERE num=%s"
        curs.execute(sql, (data['hitcnt']+1, num))
        conn.commit()
        return redirect('/bbs/read/'+str(data['num']))
    except Exception as e:
        print(e)
    finally:
        conn.close()

@app.route('/reinforcement')
def reinforcement_home():
    return render_template('reinforcement.html')
