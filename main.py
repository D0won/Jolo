from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np
# FastAPI 앱 생성
app = FastAPI()

# React 서버 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 설정
model = YOLO("best.pt")

# Classification 초기값 설정
classification = {"classification": 0}  # 0: 정상, 1: 긴급, 2: 의심

# Detection 결과 저장용 큐
detection_queue = deque(maxlen=5)

# Classification 대분류 설정 함수
def get_classification(detection_list):
    global classification
    counts = {1: 0, 2: 0}  # 1: 긴급, 2: 의심

    # Detection 결과 기반으로 카운트
    for detection in detection_list:
        for cls in detection:
            if cls > 0:  # 소분류가 0(미확인)이 아닌 경우만 처리
                if cls < 3:  # 1(운전하다), 2(꾸벅꾸벅 졸다): 긴급
                    counts[1] += 1
                else:  # 나머지는 의심
                    counts[2] += 1

    # 대분류 판단
    if counts[1] >= 2:
        classification = {"classification": 1}  # 긴급
    elif counts[2] >= 2:
        classification = {"classification": 2}  # 의심
    else:
        classification = {"classification": 0}  # 정상
    print("Updated Classification:", classification)

# YOLO 예측 수행 함수
def predict(frame):
    global detection_queue

    # 이미지 전처리: 1280x720 해상도로 리사이즈
    resized_frame = cv2.resize(frame, (1280, 720))

    # 모델 예측 수행
    results = model(resized_frame)

    class_ids = []
    for result in results:
        if result.boxes:
            # 클래스 및 신뢰도를 저장할 리스트
            boxes = []
            for box in result.boxes:
                class_id = int(box.cls[0])  # 클래스 번호
                confidence = float(box.conf[0])  # 감지 확률

                # 최소 신뢰도 조건 확인
                if confidence >= 0.7:
                    boxes.append((class_id, confidence))

            # boxes 리스트에서 가장 높은 confidence에 해당하는 클래스 선택
            if boxes:
                best_box = max(boxes, key=lambda x: x[1])  # confidence 기준으로 정렬
                class_ids.append(best_box[0])  # 가장 높은 confidence의 class_id 추가
                print(f"Selected Class: {best_box[0]}, Confidence: {best_box[1]:.2f}")
            else:
                class_ids.append(0)  # 신뢰도가 낮아 검출이 무시된 경우
        else:
            class_ids.append(0)  # 감지되지 않은 경우 클래스 번호 0
            print("No detection, Class: 0, Confidence: 0.0")
            
    detection_queue.append(class_ids)
    print("Detection Queue:", list(detection_queue))

    # 큐에 5개의 감지 결과가 쌓이면 classification 업데이트
    if len(detection_queue) == 5:
        get_classification(list(detection_queue))
        detection_queue.popleft()  # 가장 오래된 항목 제거

# 프론트엔드에 JSON 데이터 전달
@app.get("/prediction")
async def get_classification_endpoint():
    # camera = cv2.VideoCapture(0) <- 만약 비디오 화면을 프론트에서 받는다면? 
    # while True :
    #     success, frame = camera.read()
    #     if not success :
    #         break
    #     else :
    #         predict(frame)
    print("Returning Classification:", classification)
    return JSONResponse(content=classification)
    
# 실시간 비디오 스트리밍
@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)  # 웹캠 연결
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # 프레임 예측 수행
                predict(frame)

                # 프레임을 JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
