import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendancerealtime-334c0-default-rtdb.firebaseio.com/",
    'storageBucket':"faceattendancerealtime-334c0.appspot.com"
})

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentsIds = []
for path in pathList:
    studentsIds.append(cv2.imread(os.path.join(folderPath, path)))

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


    # print(path)
    # print(os.path.splitext(path)[0])
print(studentsIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

        return encodeList

print("Encoding Started ...")
encodeListknown = findEncodings(imgList)
encodeListknownWithIds = [encodeListknown, studentsIds]
print("Encoding Complete")


file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListknownWithIds, file)
file.close()
print("File Saved")
