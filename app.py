from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
from flask import send_file
import pandas as pd
from io import BytesIO
import random

app = Flask(__name__)

cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/nextCode24/Desktop/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 30
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (
                b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
    cap.release()
    cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/nextCode24/Desktop/FlaskOpencv_FaceRecognition/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Export Today Scan to excel >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/export_to_excel')
def export_to_excel():
    query = "SELECT accs_prsn AS 'Person #', prs_name AS 'Name', prs_skill AS 'Class', accs_added AS 'Added' FROM accs_hist JOIN prs_mstr ON accs_hist.accs_prsn = prs_mstr.prs_nbr WHERE accs_date = CURDATE()"
    mycursor.execute(query)
    data = mycursor.fetchall()
    columns = ['Person #', 'Name', 'Class', 'Added']

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Create an in-memory bytes buffer for the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    # Set the cursor position to the beginning of the stream
    output.seek(0)

    # Send the file as a download
    return send_file(output,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="exported_data.xlsx")


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Data Dashboard >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_student_data(student_id):
    # Example data; replace with database query in a real app
    return {
        "id": student_id,
        "name": "Student " + student_id,
        "attendance": random.randint(80, 100),
        "grades": [random.randint(60, 100) for _ in range(5)]
    }


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/student-data/<student_id>')
def student_data(student_id):
    data = get_student_data(student_id)
    return jsonify(data)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< load Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Export Personnel Data to excel >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/export_personal_data')
def export_personal_data():
    query = "SELECT prs_nbr AS 'Person Id', prs_name AS 'Name', prs_skill AS 'Class', prs_active AS 'Active', prs_added AS 'Added' FROM prs_mstr"
    mycursor.execute(query)
    data = mycursor.fetchall()
    columns = ['Person Id', 'Name', 'Class', 'Active', 'Added']

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Create an in-memory bytes buffer for the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    # Set the cursor position to the beginning of the stream
    output.seek(0)

    # Send the file as a download
    return send_file(output,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="personal_data.xlsx")


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,
                      clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor,
                                               minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color,
                              2)
                cv2.rectangle(img, (x, y + h + 40),
                              (x + int(w_filled), y + h + 50), (153, 255, 255),
                              cv2.FILLED)

                mycursor.execute(
                    "select a.img_person, b.prs_name, b.prs_skill "
                    "  from img_dataset a "
                    "  left join prs_mstr b on a.img_person = b.prs_nbr "
                    " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    # Check if the person has already been scanned today
                    today_date = date.today()
                    mycursor.execute("SELECT COUNT(*) FROM accs_hist WHERE accs_date = %s AND accs_prsn = %s",
                                     (today_date, pnbr))
                    count = mycursor.fetchone()[0]

                    if count == 0:  # If the person has not been scanned today
                        mycursor.execute(
                            "INSERT INTO accs_hist (accs_date, accs_prsn) VALUES (%s, %s)",
                            (today_date, pnbr))
                        mydb.commit()

                        cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255),
                                    2, cv2.LINE_AA)
                        time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0),
                               "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "C:/Users/nextCode24/Desktop/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/')
def home():
    # Home page with project information
    return render_template('index.html')


@app.route('/personnel_data')
def personnel_data():
    mycursor.execute(
        "select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('personnel_data.html', data=data)


@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute(
        "select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
        "  from accs_hist a "
        "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
        " where a.accs_date = curdate() "
        " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():
    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mycursor.execute(
        "select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
        "  from accs_hist a "
        "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
        " where a.accs_date = curdate() "
        " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
