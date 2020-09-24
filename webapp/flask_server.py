import os
import shutil
from pipeline import add_sound_effect
from flask import Flask
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/static/'
ALLOWED_EXTENSIONS = set(['mp4','srt','wav','avi','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.secret_key = "123456789"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    author = "Rohit sroch"
    name = "ViEmo"
    return render_template('index.html', author=author, name=name)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_dir(folder):
   for the_file in os.listdir(folder):
      file_path = os.path.join(folder, the_file)
      try:
         if os.path.isfile(file_path):
            if file_path.split('.')[-1]=='mp4' or file_path.split('.')[-1]=='srt':
               os.remove(file_path)
      except Exception as e:
         print(e)

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      # check if the post request has the file part
      if 'file_video' not in request.files or 'file_srt' not in request.files :
            flash('No file part')
            return redirect(request.url)
      f_video = request.files['file_video']
      f_srt = request.files['file_srt']

      if f_video.filename == '' or f_srt.filename == '':
            flash('No selected file')
            return redirect(request.url)
      video_path=os.path.join(app.config['UPLOAD_FOLDER'], f_video.filename)
      srt_path=os.path.join(app.config['UPLOAD_FOLDER'], f_srt.filename)

      if f_video and allowed_file(f_video.filename):
         if f_srt and allowed_file(f_srt.filename):
            filename_video = secure_filename(f_video.filename)
            filename_srt = secure_filename(f_srt.filename)
            flash(filename_video)
            clean_dir(app.config['UPLOAD_FOLDER'])
            f_video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_video))
            f_srt.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_srt))
            flag, err = add_sound_effect(video_path, srt_path)
            if err!='':
               flash(err)
            return redirect(url_for('result'))
      return  """
         <!doctype html>
         <title>Upload new File</title>
         <h1>Upload new File</h1>
         <form action="" method=post enctype=multipart/form-data>
            <p><input type=file name=file1>
               <input type=file name=file2>
               <input type=submit value=Upload>
         </form>
         <p>%s</p>
         """ % "<br>".join(os.listdir(app.config['UPLOAD_FOLDER'],))

@app.route('/result')
def result():
   author = "Rohit sroch"
   return render_template('result.html', author=author)


if __name__ == '__main__':
    app.run(debug=True)
