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


# 페이지의 너비를 설정
# st.set_page_config(layout="wide")
st.set_page_config(layout="centered")  # - 이 코드는 전체적으로 가운데 정렬을 해주는 코드.


# 프로그램 제목
st.title("감정 인식 교구")
st.write("이 프로그램은 자폐아동을 대상으로 감정 인식을 도와주기 위한 교구입니다.")

# 새로운 사이드바 (최종)
# Sidebar에 표시할 카테고리 목록
categories = ["훈련", "시험", "Self"]
# Sidebar에 카테고리 선택을 위한 라디오 버튼 추가
selected_category = st.sidebar.radio("하고싶은 거 고르기", categories)

############################################################################################################################################################################
# 이미지 URL과 설명(캡션)의 리스트
images_and_captions = [
    {'image_url': 'https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg', 'caption': '감정명: Happy'},
    {'image_url': 'https://st.depositphotos.com/1008939/3933/i/950/depositphotos_39338631-stock-photo-sad-man.jpg', 'caption': '감정명: Sad'},
    {'image_url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIT54WATGcxGT461g5MygimzKZmdxq8oUw6w&usqp=CAU', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvNZr8oEcxNniPWEZ7R3JxwGHF9tlAjeIu2g&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFK_tTzwAFz-KCtrqKiEH60Fp-GBRlfQ1f-g&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlEKhAJuRBogSWA69hcgDmdj14ZhDXsdEf-w&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8Tkb0R339-IocTzgOEmRSLSR0M9Z9a9jDnA&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTn7XMyQS9t44_rTLJFD9y-ViKiL31Ixp8iqg&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpVp0Jz5nei0RkDpH_KMj0dUiu9Wy_hNaApw&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ81rvG-MojfwgJEkyNX9I47zsc_UnihnirEg&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5hDvTt4clvm9CpJ8qb_cHVPaQsBbSdNcOrw&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-lwKMkkE6-3hJq1c7M-KN0vjXmjC3kE4Tag&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVDxq2j58XUoiRqZHUcqLcei1jH6s6-d3XKhvsh-rT3XB1ZAtUbmkT6Iwty84FkPix5Zc&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsgx2loM7gRGEkjURpDrb7MRC7RE5l2oCllA&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF_9kd0lzemmKOOY4aCLq55UzNV3IaVo8o1w&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRc9oX_xoTT5rheh8_UmwLt_DTQpE8QxoC54Q&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnaLpJe6xrCwHIah4HHEm70cki0AxFda_Yvg&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREAirqW8i3vl3HfVpgR1Tt9m6_QV-wjIanLg&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvn7OPsKm6U88-KzaLf1cZJMBoaRNuUFeGdg&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmtJVCFgg-QNEsNDZzoK121-bTvhabDFCg3A&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQj7pwmYnKJ1YW5Mnm_YkHo8Ihi18jRMCpH_g&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqE9opXPuUqV0XN-249tilxc2shktkmzczOQ&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTs6VCNWdMCvGduMo3Q--o0d7y0csw0H8zHJA&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJmrL5mkTNBg3ldRFXCpPyyKieNDsEynXqYA&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNETF5E6PYlWtxSgxe0FL0yKGeXd0Y4Og3BA&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvt12Lm1AzvshCASmkzXN0j-ehf0BZgQay_Q&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStOZNYETW4EjlS1caPRi7QvUmRrkuk5XimlbswszUCpAnvBnZqiqZn68yOC9UVJqRv-CA&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSttMrhZn_pFj8JEPGwPRZO7MFsggXTtNo2BA&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeD4A0h1V46VlH18-VZnXA0Hv-Ye4bIwOrVQ&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQuG7yIAj63-3L2axZspf-WDNKINvjgZqmX9Q&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeXLjjShFakdbkkd-Iv-kTQldrCGAe8itaGMOzlOPcTAqQlE4aWxKdXtOAkE5W56ZO62Y&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSicBqv28s5568A1iPN8h-nYgcJaIeAfsy7A&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfl3ayTsiRe4m7AUXR0vdsKVhNbjkyJxZsk3tdOXDHM1WHU7bwakiadoNevvGZ2uIZMBc&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3MJJww435Oh5lcgpDyeTRBRmG0xyQ0Zx4yA&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvXzdQHt5mCYaSm654EVDXC4hLchNg8Oa1yA&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRb33DeVA6MIcxg4PyfNQOcNaZ6qSzGjQKUJQ&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFcYmYx8SUSIYsj5oQlAKwvHy-t58TwDX-QQ&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMbASWlsz4xvFXyVdtrjhkJoSi9GXWVh_s-g&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbs8KYliETGPFAP7ImQqTs5V8yO06QiFBx1g&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgT1X49Xvko49iuGdlFgfWJ7jA2qeywSkCrg&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzPErWVORgZMQByMdekZQGCsvOg7TEx-PIzw&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQp0BT1v_0vr5A0f_TwNyXHhuDvIq5wZCeKJFAcSFJM8-lbMdPXznwYZIDn6UXB85rAjDg&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQq9LFVdbHjiaKWpmlTdRzi3ILma6XZvRKuuQ&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://thumbs.dreamstime.com/b/portrait-beautiful-young-woman-angry-face-looking-furious-human-expressions-emotions-close-up-attractive-caucasian-154265038.jpg>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzFhOj2Y6dxsGAxXWtCUafog12XMK-cZjUAfrSK6rLJ-HYTwerR--JAAuBdSSU10_oggk&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSqwzPPgYgFa9NdF9O1z2bTiGea4OP2uZRTpQ&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjsRO74hik_RSy9z6IvtHBrWMU17-Wssdzqw9whjRN4spk_WtCh1Owo6tKrqSbmYarks8&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdJvP0EL5SXGIYWKM5tvy5Ekkb0z9v2dxz6g&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWhoBdBOL52nucPhISuh1fFg1_cvKnHQT48w&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQuwMOsL3WvlRZiC1PdnkJFXw-fvYc4IAJHTe0bFV1mwL6zoAEBLRM_lPDaAYBBRNhO4Q8&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRiq6e41TKmW1iQ-PECbnvmhSh1PvAtnPrRl9mG9SeCAy2dnqZAfhRe26CV_0q48tSawro&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6HKq_QntIrama6mWXo0ZgA-m8YDM1Q_g7Tw&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQm6asmJ73YD1aiyQR737ZHMrBQF49TcDTcdw&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQIuah_UADrD9H6-lf0986U9r1DWucKaU4IQ&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWuIPX6fjbMUb5x7uuHkXt7YxUxbknbBLrOhwB6ioLzXlF-l8c4LNgzx5pJAxU5ysIdew&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOz5cYsqqjBxYIxi0iw6MLTOz4IanlWvbU0A&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOIXoViUfIF8cD4xc2M0oq8YghQ89731xyUA&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTou8L5bizDT54c0SBsLFwgLvOpDe25IRCO7w&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsEId82h_W53M0cyXhbOBt4eHuVMpPmu9R_g&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmtJVCFgg-QNEsNDZzoK121-bTvhabDFCg3A&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT3U3IKA4ZTP4r4wIxL431zuc5MUM9IyyzlfA&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8rtH-3BsXOPuyuTeTpP4wjYbZvRY2ltfTOA&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhi4VaQhquGi1S6HcFZzmr1oTNuwt7qRi5sQ&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5I0WJ4gU6a_LKIZFr5stqbfalLGLYPe28Lg&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThh1DeUlXP37g1N4H8kaFkDegB4n5wYmjvRg&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSb6UYrF-p1YfWfF-julJME1EhFIZhHKxZ8Ng&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSV4xhY9XZHt5snGdl5MamSCAh8IXreCTaz8Q&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgqNUb8_7NbHlplizd5g7N9FnjHwUPFQBE9A&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoW1hBAMFZl_6t7rbUr5dObQoaCPOexB4cKg&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvTW9b9HB9Oo0VpTxNpPGbxn-pqHQbnuLxAw&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_Mffjf-4Fo4y-Xk29AQri2qXf49jUtP7rSQ&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSnU74093rURxqZ-nvth1_6JUGT6j4kOK4L3w&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfgP5k6u9pv9AR8XOL97AmOym_QHZ_U0NE1A&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQq2P-WtSlM6hDYHnM9hah7Wd7Wex6CDGMHAQ&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTkec0GLiT1rBazmLYT6gnfkB0OATj3KSrwtQ&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdMO1CybR1FObqDaSVVcdqcE__m_z9dh5kcw&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT3cirSofmbqfXg-8P2v4z6d7P6_xCbLLA-NA&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXE9_PIdQWNS2BotpLQizlOxAfMYnUmkUW68Pf4FkD6wgtOnS-kOhP-gdMBuGleKbopwM&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPfpF4GPcVHKk3Xbg1NquEWdO_tsu3ZVU3Vw&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAunxR4klKVU36swI1JKGpzsjOes2HQItFeg&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEE7Fcwmiw6S1RS2_FZ2W9XCxXm3VyNde-EQ&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOsLOriS2KY_NhjB0alajBOCPFZmvPOBNtEQ&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzzGz73u6nzHWerITazjA8mYYvK09LeIQJs5OFthDbMUBzoInwIQi062KQ_0dCPsFekaw&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOdV-fg8arpgQ59TCRI-DJC0NaJF3CW9--7g&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6-zfJFxSKT2lAjSZl_8Tb3Hse-6HhtVJ-KA&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2Y2w8505wMOkiuZezeUFyahDvZ6OI-j9vAQ&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoo0GweFg0pxTuFd-pmOhwA2xFTY3np2BcPw&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgnAn8YZQMyRfcRj50uc9KL0weZYyl0IAqXg&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsXkGNrAoTzZptvv20q9gSMYE3EeHdb17hOg&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGcvLcomfjn2vZKhNaLwlawehrowZv9O63MFVUZvoy0vog-dEXqpKkMJuDHmJwVWLgJpk&usqp=CAU>', 'caption': '감정명: Anger'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9vcfAnQ7L1KFaHRwPGIVwZOAC65VQpDGujnaTiU0v17i3YMKsg0axv-p16XWSNLwXQfk&usqp=CAU>', 'caption': '감정명: Surprise'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoQ58c6NGAeKRwZCzfT1b2qIqFi9oWh-3EFw&usqp=CAU>', 'caption': '감정명: Happy'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSN1oSwUcOnlxobNrNhPLcIwOpSCx6C_5oRgmqIG97zvVtxdG6BckEj8g4jlcpVyDLwZ5E&usqp=CAU>', 'caption': '감정명: Sad'},
    {'image_url': '<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTY7hZRzih4d_hrAIowlfLHy6NBN0V-GGNiAA&usqp=CAU>', 'caption': '감정명: Anger'},
]

############################################################################################################################################################################


# 훈련 페이지를 표시하는 함수
def show_training_page():
    st.subheader("훈련-사진을 보고, 어떤 감정인지를 알 수 있습니다.")
    st.divider() # 구분선 코드

    # 현재 이미지 및 캡션의 index
    current_index = st.session_state.get('current_index', 0)

    # 이미지와 버튼을 배치하는 열 생성하기
    col1, col2 = st.columns([9, 1])

    with col2:
        st.write('    ')
        st.write('    ')
        st.write('    ')
        st.write('    ')
        st.write('    ')
        # "이전" 버튼 표시
        if st.button("이전", key="page1_left_button"):
            current_index = (current_index - 1) % len(images_and_captions)
        st.write('    ')
        # "다음" 버튼 표시
        if st.button("다음", key="page1_right_button"):
            current_index = (current_index + 1) % len(images_and_captions)
    
        # 현재 인덱스를 세션 상태에 저장하기
        st.session_state.current_index = current_index
    with col1:
        st.header(images_and_captions[current_index]['caption'])
        st.image(images_and_captions[current_index]['image_url'], use_column_width=True)

############################################################################################################################################################################

# 시험 페이지를 표시하는 함수

def generate_question_page(question_number):
    st.subheader(f"문제 {question_number}. 이 사진은 무슨 감정을 나타내고 있나요?")
    user_answer = st.text_input("답을 입력하세요:")
    return user_answer


def show_exam_page():
    
    st.subheader("시험-사진을 보고, 어떤 감정인지를 맞출 수 있습니다.")
    st.divider() # 구분선 코드

    # 현재 이미지 및 캡션의 index
    page2_current_index = st.session_state.get('current_index', 0)

    # 이미지와 버튼을 배치하는 열 생성하기
    p2_col1, p2_col2 = st.columns([9, 1])



    with p2_col2:
        
        if st.button("다음", key="page2_right_button"):
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

        st.image(image_np, use_column_width=True)
    
    # 가로 줄 세우기
    s1 = ''
    s2 = ''
    p2_btn_col1, p2_btn_col2, p2_btn_col3, p2_btn_col4, p2_btn_col5, p2_btn_col6, p2_btn_col7 = st.columns([1,1,1,1,1,1,1])
    with p2_btn_col1:
        if st.button('Happy'):
            if 'Happy'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Happy이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Happy이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col2:
        if st.button('Sad'):
            if 'Sad'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Sad이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Sad이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col3:
        if st.button('Neutral'):
            if 'Neutral'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Neutral이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Neutral이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col4:
        if st.button('Surprise'):
            if 'Surprise'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Surprise이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Surprise이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col5:
        if st.button('Anger'): 
            if 'Anger'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Anger이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Anger이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col6:
        if st.button('Fear'): 
            if 'Fear'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Fear이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Fear이고, 모델이 예측한 감정은 {emotion}입니다.'

    with p2_btn_col7:
        if st.button('Disgust'): 
            if 'Disgust'==emotion:
                s1 = '정답입니다 !'
                s2 = f'당신이 선택한 감정은 Disgust이고, 모델이 예측한 감정은 {emotion}입니다.'
            else:
                s1 = '오답입니다 !'
                s2 = f'당신이 선택한 감정은 Disgust이고, 모델이 예측한 감정은 {emotion}입니다.'

    if s1 != '':
        st.write(s1)
        st.write(s2)

############################################################################################################################################################################

# Self 페이지를 표시하는 함수
def show_self_page():
    st.subheader("Self- Webcam을 연결하거나 사진을 업로드하여 감정을 확인 할 수 있습니다.")
    st.divider() 

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
    if selected_category == "훈련":
        show_training_page()
    elif selected_category == "시험":
        show_exam_page()
    elif selected_category == "Self":
        show_self_page()


############################################################################################################################################################################


# def get_data():
#     print("get_data")
#     df = pd.DataFrame({"A": np.arange(0, 10, 1), "B": np.arange(0, 1, 0.1)})
#     return df

