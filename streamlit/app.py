from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import cv2
from pred import pred_expression, webcam_expression
from streamlit_webrtc import webrtc_streamer
import av
from urllib.request import Request, urlopen
from img_file import images_and_captions


# 페이지의 너비를 설정
# st.set_page_config(layout="wide")
st.set_page_config(layout="centered")  # - 이 코드는 전체적으로 가운데 정렬을 해주는 코드.

# 프로그램 제목
st.title("🌻 🌼 💐 감정 인식 교구 🌹 🌺 🌸")
st.write("이 프로그램은 자폐아동을 대상으로 감정 인식을 도와주기 위한 교구입니다.")
st.divider()
# 새로운 사이드바 (최종)
# Sidebar에 표시할 카테고리 목록
with open('./streamlit/style.css') as f:
    st.markdown(f'<style>{f.read()}', unsafe_allow_html=True) 

categories = ["T r a i n  ⛳️", "T e s t  💯", "T r y  🪞"]
# Sidebar에 카테고리 선택을 위한 라디오 버튼 추가
selected_category = st.sidebar.radio("감정 인식 교구", categories)


############################################################################################################################################################################

# 훈련 페이지를 표시하는 함수
def show_training_page():
    st.subheader("사진을 보고, 어떤 감정인지를 알 수 있습니다.")
    

    # 현재 이미지 및 캡션의 index
    current_index = st.session_state.get('current_index', 7)

    # 이미지와 버튼을 배치하는 열 생성하기
    col1, col2, col3 = st.columns([1, 15, 1])

    with col1:
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # "이전" 버튼 표시
        if st.button("⬅️", key="page1_left_button"):
            current_index = (current_index - 1) % len(images_and_captions)
        # st.write('    ')
        # # "다음" 버튼 표시
        # if st.button("다음", key="page1_right_button"):
        #     current_index = (current_index + 1) % len(images_and_captions)
        # 현재 인덱스를 세션 상태에 저장하기
        st.session_state.current_index = current_index

    with col3:
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # st.write('    ')
        # "이전" 버튼 표시
        # if st.button("이전", key="page1_left_button"):
        #     current_index = (current_index - 1) % len(images_and_captions)
        # st.write('    ')
        # "다음" 버튼 표시
        if st.button("➡️", key="page1_right_button"):
            current_index = (current_index + 1) % len(images_and_captions)
        # 현재 인덱스를 세션 상태에 저장하기
        st.session_state.current_index = current_index

    with col2:
        req = Request(images_and_captions[current_index]['image_url'], headers={'User-Agent': 'Mozilla/5.0'})
        pil_image = Image.open(urlopen(req))

        # PIL Image를 NumPy 배열로 변환
        image_np = np.array(pil_image)
        
        # OpenCV를 사용하여 이미지 읽기
        cv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # 결과 예측
        emotion = pred_expression(cv_image)

        pil_image_resize = pil_image.resize((600, 500))

        st.image(pil_image_resize, use_column_width=True)
        st.header(f'감정: {emotion}')

############################################################################################################################################################################

# 시험 페이지를 표시하는 함수

def generate_question_page(question_number):
    st.subheader(f"문제 {question_number}. 이 사진은 무슨 감정을 나타내고 있나요?")
    user_answer = st.text_input("답을 입력하세요:")
    return user_answer


def show_exam_page():
    
    st.subheader("사진을 보고, 어떤 감정인지를 맞출 수 있습니다.")

    # 현재 이미지 및 캡션의 index
    page2_current_index = st.session_state.get('current_index', 0)

    # 이미지와 버튼을 배치하는 열 생성하기
    p2_col1, p2_col2 = st.columns([9, 1])



    with p2_col2:
        
        if st.button("➡️", key="page2_right_button"):
            page2_current_index = np.random.randint(len(images_and_captions)) % (len(images_and_captions)-1)
            st.session_state.current_index = page2_current_index

    with p2_col1:

        req = Request(images_and_captions[page2_current_index]['image_url'], headers={'User-Agent': 'Mozilla/5.0'})
        pil_image = Image.open(urlopen(req))

        # PIL Image를 NumPy 배열로 변환
        image_np = np.array(pil_image)
        
        # OpenCV를 사용하여 이미지 읽기
        cv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # 결과 예측
        emotion = pred_expression(cv_image)

        pil_image_resize = pil_image.resize((600, 500))
        st.image(pil_image_resize, use_column_width=True)
    
    # 가로 줄 세우기
    s1 = ''
    a, p2_btn_col1, b, p2_btn_col2, c, p2_btn_col3, d = st.columns([1.5, 3, 1.5, 3, 1.5, 3, 1.5])
    with p2_btn_col1:
        if st.button('Happy'):
            if 'Happy'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    with p2_btn_col2:
        if st.button('Sad'):
            if 'Sad'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    with p2_btn_col3:
        if st.button('Neutral'):
            if 'Neutral'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    p2_btn_col4, f, p2_btn_col5, g,  p2_btn_col6, i,  p2_btn_col7 = st.columns([3, 1, 3, 1, 3, 1, 3])
    with p2_btn_col4:
        if st.button('Surprise'):
            if 'Surprise'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    with p2_btn_col5:
        if st.button('Angry'): 
            if 'Anger'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    with p2_btn_col6:
        if st.button('Fear'): 
            if 'Fear'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    with p2_btn_col7:
        if st.button('Disgust'): 
            if 'Disgust'==emotion:
                s1 = '🎉 🎉 🎉 정답입니다 🎉 🎉 🎉'
                st.balloons()
            else:
                s1 = '오답입니다 ! 다시 도전하세요 ~'

    if s1 != '':
        st.subheader(s1)
        # st.subheader(s2)

############################################################################################################################################################################

# Self 페이지를 표시하는 함수
def show_self_page():
    st.subheader("Webcam과 사진을 활용하여 감정을 확인할 수 있습니다.")

    # webcam 과 upload tab 나누기
    tab1, tab2 = st.tabs(['Webcam', 'Upload'])
    # webcam tab
    with tab1:
        # webrtc_ctx = \
        webrtc_streamer(key="example", 
                        video_frame_callback=webcam_expression,
                        async_processing=True)
        
    # upload tab
    with tab2:
        st.write("Image :camera:")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # 업로드된 이미지 표시
            pil_image = Image.open(uploaded_file)

            # PIL Image를 NumPy 배열로 변환
            image_np = np.array(pil_image)
            
            # OpenCV를 사용하여 이미지 읽기
            cv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # 결과 예측 후 출력
            emotion = pred_expression(cv_image)
            st.title(f'{emotion}')
            
            st.image(image_np, use_column_width=True)


# Streamlit 앱 실행
if __name__ == "__main__":
    # 선택된 카테고리에 따라 페이지 표시
    if selected_category == "T r a i n  ⛳️":
        show_training_page()
    elif selected_category == "T e s t  💯":
        show_exam_page()
    elif selected_category == "T r y  🪞":
        show_self_page()


############################################################################################################################################################################


# def get_data():
#     print("get_data")
#     df = pd.DataFrame({"A": np.arange(0, 10, 1), "B": np.arange(0, 1, 0.1)})
#     return df

