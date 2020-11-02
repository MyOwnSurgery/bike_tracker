from flask import Flask, render_template, url_for, redirect, request, session, send_file
from werkzeug.utils import secure_filename
import os, os.path
import sys
import shutil

# full path to project
project_path = os.path.dirname(os.path.realpath(__file__))+"/"

#importing text detection
sys.path.append(project_path+"text-detection/")
import start  

#importing text recognition
sys.path.append(project_path+"text-recognition/")  
import demo

import moviepy.editor as mp
import pandas as pd
import openpyxl
import psycopg2
from sqlalchemy import create_engine, text


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.urandom(24)
app.config['TEMPLATES_AUTO_RELOAD'] = True
#db url etc.
db_url = 'postgresql+psycopg2://mayorov:x3light99@localhost:5432/mydb'

@app.route("/")  # this sets the route to this page
def home():
    return render_template("index.html")

def convert(source,target):
    #avi => mp4
    clip = mp.VideoFileClip(source)
    #may not work so I leave it commented
    #clip.write_videofile(target)

def read_table(session,engine):
    #reading and sorting data
    ret = engine.dialect.has_table(engine, 'table_' + session)
    if ret is True:
      df = pd.read_sql_table('table_' + session, engine)
      df = df.sort_values(by='finish_time', ascending=True)
      return df
    else:
      return pd.DataFrame()


def write_html(target,df):
    html_file = open(target, 'w')
    html_text = str(df.to_html())
    html_text = html_text.replace('&lt;', '<')
    html_text = html_text.replace('&gt;', '>')

    #some magic to make our html page a dynamic page with edit
    html_text = html_text.replace('<thead>', '{% set count = 0 %}<thead>')
    html_text = html_text.replace('width="500" ></td>', 'width="500" ></td><td><form action = "/edit/{{count}}"><button type = "submit">Edit</button></form></td> {% set count = count + 1 %}')
    html_text = html_text.replace('</table>','</table> <body class="body"> <div class="container" align="left"> <a href="/return-files/" target="blank"><button>Download excel</button></a></div></body>')
   

    # writing dataframe to html
    html_file.write(html_text)
    html_file.close()

def setUserInSession():
    if 'visits' in session:
        session['visits'] = session.get('visits') + 1  # обновление данных сессии
    else:
        session['visits'] = 1  # запись данных в сессию

def getUserInSession():
    return session.get('visits').__str__()
    
#xlsx downloading
@app.route('/return-files/')
def return_file():
    session = getUserInSession()
    return send_file(project_path+'static/'+session+'/results.xlsx', attachment_filename='results.xlsx')
	
#editing plate number
@app.route('/edit/<int:id>')
def edit(id):
    session = getUserInSession()

    #getting old plate number to show it in input field (it`s updating after all)
    engine = create_engine(db_url)
    conn = engine.connect()
    sql = text('SELECT plate_number from table_' + session + ' where index = ' + id.__str__() + ';')
    result = conn.execute(sql).fetchone()
    return render_template("edit.html", id = id.__str__(), number = ''.join(x for x in result if x.isdigit()))

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    session = getUserInSession()

    #getting new plate number from input field and updating table
    number = request.form.get('number')
    engine = create_engine(db_url)
    sql = text('UPDATE table_' + session + ' SET PLATE_NUMBER = ' + number + ' WHERE INDEX = ' + id.__str__() +';')
    engine.execute(sql)

    #reading updated table and writing new data to html
    df = read_table(session, engine)
    write_html('templates/' + session + '.html', df)
    df.to_excel('static/'+ session + '/results.xlsx')
    return render_template(session+".html")

#uploading video
@app.route('/upload', methods = ['POST'])
def upload():
    #getting file from request
    file = request.files['inputFile']
    name = file.filename
    #getting extention of the file
    ext = name.split('.')[1]
    setUserInSession()
    session = getUserInSession()
    #preparing directory 
    target = os.path.join(APP_ROOT,"static/")
    if not os.path.isdir(target):
        os.mkdir(target)
    

    #video number = session number in order to not mix up users` videos
    destination = "/".join([target,session+ "." +ext])
    file.save(destination)

    #html can`t work with avi so we should convert video to mp4
    convert("static/"+session+"." +ext,"static/"+session+".mp4")
    clip = mp.VideoFileClip("static/"+session+"." +ext)
    return render_template("preview.html", video = session + ".mp4", time = 60 + clip.duration*3, ext = ext)
    
#video analysis
@app.route('/process', methods = ['POST'])
def process():
    session = getUserInSession()
    ext = request.form.get('ext')
    #dropping table with the same number as the current session has if it somehow exists
    engine = create_engine(db_url)
    sql = text('DROP TABLE IF EXISTS table_' + session + ';')
    engine.execute(sql)

    #calling tracker from cmd that may be not correct (perhaps having the entire tracker code as an imported module would be better)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    os.system(
        "conda activate project & python object_tracker.py --video static/" + session + "." + ext + " --output data/video/output.mp4")
        
    #calling text detector
    start.main(project_path+"static/" + session + "/", project_path+"static/" + session + "/")
    
    #calling text recogniser, 2nd parameter helps us to get db name etc
    demo.main(project_path+"static/"+session+"/", len(project_path)+7)
    
    #reading table and writing data to html
    df = read_table(session,engine)
    write_html('templates/'+session+'.html',df)
    df.to_excel('static/'+ session + '/results.xlsx')
    return render_template(session+".html")

if __name__ == "__main__":
    shutil.rmtree(project_path+"/static/")
    os.mkdir(project_path+"/static/")
    app.run(host='0.0.0.0', port=5000)