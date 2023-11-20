import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer
webrtc_streamer(key="sample")
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase

# 페이지의 너비를 설정
st.set_page_config(layout="wide")
#st.set_page_config(layout="centered")  - 이 코드는 전체적으로 가운데 정렬을 해주는 코드.

# 프로그램 제목
st.title("감정 인식 교구")
st.write("이 프로그램은 자폐아동을 대상으로 감정 인식을 도와주기 위한 교구입니다.")

# 새로운 사이드바 (최종)
# Sidebar에 표시할 카테고리 목록
categories = ["훈련", "시험", "Self"]
# Sidebar에 카테고리 선택을 위한 라디오 버튼 추가
selected_category = st.sidebar.radio("하고싶은 거 고르기", categories)
# 선택된 카테고리에 따라 내용 표시

# 훈련 페이지를 표시하는 함수
def show_training_page():
    st.subheader("훈련-사진을 보고, 감정을 알 수 있습니다.")
    st.divider() # 구분선 코드
    # 훈련 페이지의 내용을 여기에 추가
    st.image('https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg')
    # '다음' 버튼과 '이전' 버튼을 만듭니다.
    if st.button("다음", key="right_button"):
        st.image('https://st.depositphotos.com/1008939/3933/i/950/depositphotos_39338631-stock-photo-sad-man.jpg')
    elif st.button("이전", key="left_button"):
        st.image('https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg')  # 이전 이미지의 링크를 넣으세요.

    #st.caption('감정명: Happy')
    st.markdown("<p style='font-size:18px;'>감정명: Happy</p>", unsafe_allow_html=True)

# 시험 페이지를 표시하는 함수
def show_exam_page():
    st.subheader("시험- 훈련페이지에서 배운 내용을 확실하게 익혔는지 확인 할 수 있습니다.")
    st.divider() # 구분선 코드
    # 시험 페이지의 내용을 여기에 추가
    st.image('https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg')
    # 이미지의 정답을 입력할 수 있는 칸 만들기
    # 사용자가 정답을 적어서 맞으면,

# Self 페이지를 표시하는 함수
def show_self_page():
    st.subheader("Self- Webcam과 연결해서 본인의 감정을 확인 할 수 있습니다.")
    st.divider() # 구분선 코드
    # Self 페이지의 내용을 여기에 추가
    from streamlit_webrtc import webrtc_streamer
    import av
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        flipped = img[::-1,:,:]
        return av.VideoFrame.from_ndarray(flipped, format="bgr24")
    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
# 선택된 카테고리에 따라 페이지 표시
if selected_category == "훈련":
    show_training_page()
elif selected_category == "시험":
    show_exam_page()
elif selected_category == "Self":
    show_self_page()

#st.divider() # 구분선 코드


def get_data():
    print("get_data")
    df = pd.DataFrame({"A": np.arange(0, 10, 1), "B": np.arange(0, 1, 0.1)})
    return df

#데이터 받아오는 부분
#data = get_data()
#st.write(data)
#st.dataframe(data)
#st.table(data)

# 컨테이너 생성
with st.container():
    # 여기에 블록 내용 추가
    #st.title("이곳에 블록 내용 추가")
    #st.write("블록 내의 텍스트 또는 다른 위젯들을 추가할 수 있습니다.")
    
    # 버튼을 오른쪽에 고정
    st.button("다음", key="right_button")
    #st.button("이전", key='left_button')

with st.container():
    st.button('이전', key='left_button')