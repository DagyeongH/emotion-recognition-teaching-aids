### 감정 인식 교구

<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=Avengers%20with%20Citizen&fontSize=50" />
<div align="left">

### Face Classification Model😀
- 목적 : Face Classification Model을 통해서, 자폐아동을 대상으로 표정 인식을 도와주는 프로그램입니다.
- Dataset : image_to_csv.ipynb / 최종 이미지파일은 Google Drive 주소 참고
  - 데이터셋은 이미지의 pixel단위를 csv로 저장하여, train과 test로 나누었습니다.
- 사용 모델 : Face detection , keras CNN model & OpenCV
  - CNN : 이미지나 영상 데이터를 처리할 때 사용함. Convolution이라는 전처리 작업이 들어가는 Neural Network 모델
- 모델 파일 : face_classification_model.ipynb

### Emotion Detection Model Visualization


### Face Detection Webcam📷
- 사용 모델 : keras CNN model & openCV
- 모델 파일 : face_classification_webcam.ipynb

### Webcam Examples
(데모 사진 넣기/ 짧은 영상도 갠찮을듯)

### Streamlit 
- 목적 : Face classification Model과 Face detection webcam 모델을 이용하여, 실제로 자폐아동들이 학습할 수 있는 페이지를 구현하였습니다.
- 실행 파일 : app.py
- Dataset : image_file.py 
- 꾸미기 파일 : style.css

### Streamlit Demo
(실제 구현 gif 올리기)
