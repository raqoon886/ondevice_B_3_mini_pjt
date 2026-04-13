"""
RPS Rhythm - 데이터 수집 스크립트
카메라로 가위/바위/보 손 이미지를 촬영하여 학습 데이터를 수집합니다.

사용법:
  python collect_data.py

조작:
  's' → scissors(가위) 폴더에 저장
  'r' → rock(바위) 폴더에 저장
  'p' → paper(보) 폴더에 저장
  'q' → 종료
"""

import cv2
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector

# ── 설정 ──
IMG_SIZE = 224
OFFSET = 30
CAM_W, CAM_H = 320, 240
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

CLASSES = {
    ord('s'): 'scissors',
    ord('r'): 'rock',
    ord('p'): 'paper',
}

# ── 손 감지기 ──
hd = HandDetector(maxHands=1, detectionCon=0.7)

def make_square_img(img):
    """손 이미지를 224x224 정사각형으로 변환 (흰색 배경)"""
    ho, wo = img.shape[0], img.shape[1]
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if ho / wo > 1:
        k = IMG_SIZE / ho
        wk = int(wo * k)
        img = cv2.resize(img, (wk, IMG_SIZE))
        d = (IMG_SIZE - img.shape[1]) // 2
        wbg[:img.shape[0], d:img.shape[1] + d] = img
    else:
        k = IMG_SIZE / wo
        hk = int(ho * k)
        img = cv2.resize(img, (IMG_SIZE, hk))
        d = (IMG_SIZE - img.shape[0]) // 2
        wbg[d:img.shape[0] + d, :img.shape[1]] = img
    return wbg

def extract_hand(frame):
    """프레임에서 손 영역을 추출하여 정사각형 이미지로 반환"""
    hands, _ = hd.findHands(frame, draw=False)
    if not hands:
        return None, None

    x, y, w, h = hands[0]['bbox']
    if x < OFFSET or y < OFFSET or x + w + OFFSET > CAM_W or y + h > CAM_H:
        return None, None

    x1, y1 = x - OFFSET, y - OFFSET
    x2, y2 = x + w + OFFSET, y + h
    hand_img = frame[y1:y2, x1:x2]
    square_img = make_square_img(hand_img)
    return square_img, (x1, y1, x2, y2)

def count_images():
    """각 클래스별 이미지 수 반환"""
    counts = {}
    for cls in CLASSES.values():
        path = os.path.join(DATA_DIR, cls)
        os.makedirs(path, exist_ok=True)
        counts[cls] = len([f for f in os.listdir(path) if f.endswith('.jpg')])
    return counts

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('collect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('collect', 640, 520)

    counts = count_images()
    print("=== RPS 데이터 수집 ===")
    print("  's': 가위  'r': 바위  'p': 보  'q': 종료")
    print(f"  현재: {counts}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_img, bbox = extract_hand(frame)

        # 손 감지 표시
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 안내 텍스트
        display = cv2.resize(frame, (640, 480))
        y_off = 30
        for cls_key, cls_name in CLASSES.items():
            label = f"{chr(cls_key)}: {cls_name} ({counts.get(cls_name, 0)})"
            cv2.putText(display, label, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_off += 30

        if hand_img is not None:
            cv2.putText(display, "Hand OK!", (10, y_off + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # 추출된 손 이미지 미리보기 (우상단)
            preview = cv2.resize(hand_img, (120, 120))
            display[10:130, 510:630] = preview
        else:
            cv2.putText(display, "No hand", (10, y_off + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('collect', display)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key in CLASSES and hand_img is not None:
            cls_name = CLASSES[key]
            save_dir = os.path.join(DATA_DIR, cls_name)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{cls_name}_{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, hand_img)
            counts[cls_name] = counts.get(cls_name, 0) + 1
            print(f"  저장: {filepath} (총 {counts[cls_name]}장)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n최종 수집 결과: {counts}")

if __name__ == '__main__':
    main()
