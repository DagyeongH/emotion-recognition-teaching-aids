from tensorflow.keras.models import load_model
import cv2
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import av
import queue

# Dataset 만들기
shape_x = 48
shape_y = 48

model = load_model('./model/model.h5')


# 전체 이미지에서 얼굴을 찾아내는 함수
def detect_face(frame):

    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_alt.xml')

    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cascade 멀티스케일 분류
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1,
                                                   minNeighbors = 6,
                                                   minSize = (shape_x, shape_y),
                                                   flags = cv2.CASCADE_SCALE_IMAGE
                                                  )

    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w]
            coord.append([x, y, w, h])

    return gray, detected_faces, coord

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:

        # 얼굴로 감지된 영역
        x, y, w, h = det

        # 이미지 경계값 받기
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))

        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

        # 얼굴 이미지만 확대
        new_extracted_face = zoom(extracted_face, (shape_x/extracted_face.shape[0], shape_y/extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        new_face.append(new_extracted_face)

    return new_face

def pred_expression(face):
    # 얼굴 추출
    gray, detected_faces, coord = detect_face(face)
    face_zoom = extract_face_features(gray, detected_faces, coord)

    # 모델 추론
    input_data = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
    output_data = model.predict(input_data)
    result = np.argmax(output_data)

    # 결과 문자로 변환
    if result == 0:
        emotion = 'Anger'
    elif result == 1:
        emotion = 'Disgust'
    elif result == 2:
        emotion = 'Fear'
    elif result == 3:
        emotion = 'Happy'
    elif result == 4:
        emotion = 'Sad'
    elif result == 5:
        emotion = 'Surprise'
    elif result == 6:
        emotion = 'Neutral'

    return emotion

# 웹캠에서 얼굴 인식하는 간단한 모델
def webcam_expression(frame):
    try:
        image = frame.to_ndarray(format="bgr24")

        face_index = 0
        gray, detected_faces, coord = detect_face(image)
        
        face_zoom = extract_face_features(gray, detected_faces, coord)
        face_zoom = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
        x, y, w, h = coord[face_index]
        
        # 머리 둘레에 직사각형 그리기: (0, 255, 0)을 통해 녹색으로 선두께는 2
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 감정 예측
        pred = model.predict(face_zoom)
        pred_result = np.argmax(pred)
        
        # 예측값이 높은 라벨 하나만 프레임 옆에 표시
        if pred_result == 0:
            cv2.putText(image, "Angry ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        elif pred_result == 1:
            cv2.putText(image, "Disgust ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        elif pred_result == 2:
            cv2.putText(image, "Fear ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        elif pred_result == 3:
            cv2.putText(image, "Happy ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        elif pred_result == 4:
            cv2.putText(image, "Sad ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        elif pred_result == 5:
            cv2.putText(image, "Surprise ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        else:
            cv2.putText(image, "Neutral ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        return av.VideoFrame.from_ndarray(image, format="bgr24")
    except:
        pass