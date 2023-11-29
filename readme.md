### 감정 인식 교구

<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=Avengers%20with%20Citizen&fontSize=50" />
<div align="left">

### DataSet 수집
- 구글 이미지 크롤러, Aihub, Kaggel (기준이 애매한 사진 직접 제거)
- img -> csv 파일로 변환 (Label, Pixel, train&test 여부) : 저장 및 관리가 용이하기 위함  
  
![train_data](/figures/img_to_csv_train.PNG)
![test_data](/figures/img_to_csv_test.PNG)

### Face Classification Model😀
- 목적 : Face Classification Model을 통해서, 자폐아동을 대상으로 표정 인식을 도와주는 프로그램입니다.
- Dataset : image_to_csv.ipynb / 최종 이미지파일은 Google Drive 주소 참고
  - 데이터셋은 이미지의 pixel단위를 csv로 저장하여, train과 test로 나누었습니다.
- 사용 모델 : Face detection , keras CNN model & OpenCV
  - CNN : 이미지나 영상 데이터를 처리할 때 사용함. Convolution이라는 전처리 작업이 들어가는 Neural Network 모델
- 모델 파일 : face_classification_model.ipynb
- 학습 과정

### Emotion Detection Model Visualization
- CNN Model 구조 설명
![conv2d](/figures/1.PNG)
![pooling_layer](/figures/2.PNG)
![faltten](/figures/3.PNG)
![fully_connected_layer](/figures/4.PNG)
![conv2d_model](/figures/conv2d.PNG)
![dropout_vs](/figures/dropout.PNG)
![max_pooling2d_vs](/figures/max_pooling2d.PNG)
- Train data result
![훈련데이터](/figures/aa.PNG)
![testdata](/figures/bb.PNG)
- 학습결과
![train_result](/figures/100Epoch.PNG)
![accuracy_curve](/figures/a_curve.PNG)
![loss_curve](/figures/l_curve.PNG)

### Face Detection Webcam📷
- 사용 모델 : keras CNN model & openCV
- 모델 파일 : face_classification_webcam.ipynb

### Webcam Examples
![KakaoTalk_Photo_2023-11-29-09-22-02](https://github.com/DagyeongH/mini_project/assets/123550946/0bad79ec-8edb-4d5b-895f-8283488241fb)

### Streamlit 
- 목적 : Face classification Model과 Face detection webcam 모델을 이용하여, 실제로 자폐아동들이 학습할 수 있는 페이지를 구현하였습니다.
- 실행 파일 : app.py
- Dataset : image_file.py 
- 꾸미기 파일 : style.css

### Streamlit Demo
![train](https://github.com/DagyeongH/mini_project/assets/123550946/84821b78-1937-409c-90a0-6bfdee4b9d7f)

![test](https://github.com/DagyeongH/mini_project/assets/123550946/f9037892-2da8-4d57-a825-a5c2be7fdd94)

![self](https://github.com/DagyeongH/mini_project/assets/123550946/b6299bb6-335d-4490-a5b7-6466cc6bf3d5)
