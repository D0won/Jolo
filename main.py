from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
from collections import deque

# fastapi 앱 생성 
app = FastAPI()

# React 서버 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Yolo 모델 설정
model = YOLO("best.pt", device = "mps")

# 최종 예측 결과 Json 형식으로 프론트엔드에 전송
@app.get("/prediction")
async def get_prediction(detection_list) :
    # 최종 결과 변수
    classification = {"classification" : None}
    # 대분류 카운트 변수 
    counts =  {1 : 0, 2 : 0} # 1 : 졸음운전_긴급, 2 : 졸음운전_의심 

    # detection_list의 각 항목에 대한 소분류 클래스 확인
    for detection in detection_list :
          for cls in detection :
               if cls > 0 : # 소분류가 0(미확인)이 아닌 경우만 처리 
                    if cls < 3 : # 1(운전하다), 2(꾸벅꾸벅졸다) -> 졸음운전_긴급 대분류에 카운트
                        counts[1] += 1
                    # 3(하품), 4(박수치다), 5(뺨을 떄리다), 6(목을 만지다), 7(어깨를 두드리다), 8(무언가를 쥐다)
                    # 9(무언가를 보다), 10(허벅지두드리기), 11(팔주무르기), 12(눈비비기), 13(눈깜빡이기), 14(고개를 좌우로 흔들다)
                    # -> 졸음운전_의심 대분류에 카운트
                    else :
                    # 3(하품), 4(박수치다), 5(뺨을 떄리다), 6(목을 만지다), 7(어깨를 두드리다), 8(무언가를 쥐다)
                    # 9(무언가를 보다), 10(허벅지두드리기), 11(팔주무르기), 12(눈비비기), 13(눈깜빡이기), 14(고개를 좌우로 흔들다)
                    # -> 졸음운전_의심 대분류에 카운트
                        counts[2] += 1

    # 대분류를 판단할 수 있는 기준: 각 소분류 클래스가 2번 이상 등장하면 해당 대분류를 반환
    if counts[1] >= 2:
        classification = {"classification" : 1} # 졸음운전_긴급
    elif counts[2] >= 2:
        classification = {"classification" : 2}  # 졸음운전_의심
    
    # 프론트엔드에 JSON 파일 보내기 
    return JSONResponse(content=classification)
                    
def predict(frame) :
    # 대분류 클래스 변수 초기화
    detection_queue = deque(maxlen= 5)
    results = model(frame)

    class_ids = []
    for result in results :
        if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])  # 클래스 번호
                    class_ids.append(class_id)
        else :
                class_ids.append(0)  # 감지되지 않은 경우 클래스 번호 0
    
    detection_queue.append(class_ids)


@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # 프레임을 캡처하고 예측 수행
                predict(frame)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()
    
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")