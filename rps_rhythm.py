"""
RPS Rhythm - 가위바위보 리듬 게임
화면에 표시되는 가위/바위/보 시퀀스를 기억했다가, 박자에 맞춰 카메라에 보여주세요!

사용법:
  python rps_rhythm.py                          (기존 TFLite 모델 사용)
  python rps_rhythm.py --model models/rps_mobilenetv2.tflite  (직접 학습한 TFLite 모델)
  python rps_rhythm.py --model models/best.onnx            (ONNX YOLO 모델 사용)

조작:
  SPACE → 게임 시작 / 재시작
  q     → 종료
"""

import sys
import os
import argparse
import time
import random
import math
import threading
import termios
import tty
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from ai_edge_litert.interpreter import Interpreter

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ══════════════════════════════════════════════
#  상수 & 설정
# ══════════════════════════════════════════════

# 화면
SCREEN_W = 640
SCREEN_H = 520
CAM_W, CAM_H = 320, 240
IMG_SIZE = 224
OFFSET = 30

# 제스처
GESTURES = ['scissors', 'rock', 'paper']
GESTURE_EMOJI = {'scissors': 'V', 'rock': 'O', 'paper': 'W'}
GESTURE_KR = {'scissors': 'SCISSORS', 'rock': 'ROCK', 'paper': 'PAPER'}
GESTURE_COLOR = {
    'scissors': (255, 80, 80),    # 파랑
    'rock':     (80, 255, 80),    # 초록
    'paper':    (80, 80, 255),    # 빨강
}

# 판정
SCORE_CORRECT = 200
SCORE_BONUS_PER_COMBO = 30
CONFIRM_FRAMES = 3       # 같은 제스처 연속 N프레임 감지 시 확정
TIME_LIMIT_PER_NOTE = 8  # 노트당 제한시간 (초), 0이면 무제한

# 스테이지 설정
STAGES = [
    # (시퀀스 길이, 미리보기 시간(초), 노트 제한시간(초), 설명)
    (3,  6.0, 0,  "Tutorial"),
    (4,  5.5, 0,  "Easy 1"),
    (4,  5.0, 0,  "Easy 2"),
    (5,  5.0, 10, "Normal 1"),
    (5,  4.5, 10, "Normal 2"),
    (6,  4.5, 8,  "Normal 3"),
    (7,  4.0, 8,  "Hard 1"),
    (8,  4.0, 7,  "Hard 2"),
    (9,  3.5, 6,  "Hard 3"),
    (10, 3.0, 5,  "Expert"),
]

# ══════════════════════════════════════════════
#  손 감지 & 모델 추론
# ══════════════════════════════════════════════

class RPSDetector:
    def __init__(self, model_path):
        self.hd = HandDetector(maxHands=1, detectionCon=0.7)
        self.model_path = model_path
        self.is_onnx = model_path.lower().endswith('.onnx')

        if self.is_onnx:
            if YOLO is None:
                raise ImportError('Ultralytics YOLO is required for ONNX models. Install with "pip install ultralytics".')
            self.model = YOLO(model_path, task='detect')
            self.names = {int(k): v.lower() for k, v in self.model.names.items()}
        else:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_dtype = self.input_details[0]['dtype']

    def make_square_img(self, img):
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

    def detect(self, frame):
        """프레임에서 손을 감지하고 RPS 분류 결과 반환
        Returns: (gesture_name, confidence, bbox) or (None, 0, None)
        """
        if self.is_onnx:
            results = self.model(frame, verbose=False)
            if not results or len(results) == 0:
                return None, 0, None
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                return None, 0, None

            confs = res.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
            classes = res.boxes.cls.cpu().numpy().astype(int)
            class_id = int(classes[idx])
            gesture = self.names.get(class_id, None)
            if gesture not in GESTURES:
                return None, 0, None

            xyxy = res.boxes.xyxy.cpu().numpy()[idx]
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            return gesture, float(confs[idx]), (x1, y1, x2, y2)

        hands, _ = self.hd.findHands(frame, draw=False)
        if not hands:
            return None, 0, None

        x, y, w, h = hands[0]['bbox']
        if x < OFFSET or y < OFFSET or x + w + OFFSET > CAM_W or y + h > CAM_H:
            return None, 0, None

        x1, y1 = x - OFFSET, y - OFFSET
        x2, y2 = x + w + OFFSET, y + h
        hand_img = frame[y1:y2, x1:x2]
        square = self.make_square_img(hand_img)

        # 추론
        inp = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, 0).astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # softmax 결과 처리 (uint8 양자화 모델 대응)
        if self.input_dtype == np.uint8:
            output = output.astype(np.float32)
        if output.max() > 1.0:
            exp = np.exp(output - output.max())
            output = exp / exp.sum()

        ans = int(np.argmax(output))
        conf = float(output[ans])
        return GESTURES[ans], conf, (x1, y1, x2, y2)


# ══════════════════════════════════════════════
#  게임 상태 관리
# ══════════════════════════════════════════════

class GameState:
    # 게임 상태 상수
    TITLE = 'title'
    MEMORIZE = 'memorize'
    COUNTDOWN = 'countdown'
    PLAY = 'play'
    JUDGE = 'judge'
    STAGE_CLEAR = 'stage_clear'
    GAME_OVER = 'game_over'
    ALL_CLEAR = 'all_clear'

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = self.TITLE
        self.stage_idx = 0
        self.score = 0
        self.total_combo = 0
        self.max_combo = 0
        self.lives = 5
        self.sequence = []
        self.current_note = 0
        self.combo = 0
        self.results = []        # ('correct'/'wrong'/'timeout', gesture)
        self.memorize_start = 0
        self.countdown_start = 0
        self.last_detection = None
        self.stage_score = 0
        self.confirm_count = 0   # 같은 제스처 연속 감지 프레임 수
        self.confirm_gesture = None
        self.note_start_time = 0 # 현재 노트 시작 시점
        self.prev_confirmed = None  # 이전에 확정된 제스처 (연속 같은 제스처 방지)

    def get_stage(self):
        if self.stage_idx < len(STAGES):
            return STAGES[self.stage_idx]
        return STAGES[-1]

    def generate_sequence(self):
        length, _, _, _ = self.get_stage()
        self.sequence = [random.choice(GESTURES) for _ in range(length)]
        self.current_note = 0
        self.results = []
        self.stage_score = 0
        self.confirm_count = 0
        self.confirm_gesture = None
        self.prev_confirmed = None

    def note_time_limit(self):
        return self.get_stage()[2]

    def preview_time(self):
        return self.get_stage()[1]


# ══════════════════════════════════════════════
#  UI 그리기
# ══════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """모서리가 둥근 사각형"""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_gesture_icon(img, gesture, cx, cy, size, alpha=1.0, highlight=False):
    """제스처 아이콘 (원 + 글자)"""
    color = GESTURE_COLOR.get(gesture, (200, 200, 200))
    if alpha < 1.0:
        color = tuple(int(c * alpha) for c in color)

    if highlight:
        cv2.circle(img, (cx, cy), size + 6, (255, 255, 255), 3)
    cv2.circle(img, (cx, cy), size, color, -1)
    cv2.circle(img, (cx, cy), size, (255, 255, 255), 2)

    text = GESTURE_EMOJI[gesture]
    font_scale = size / 30.0
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)


def draw_sequence_bar(img, game, now):
    """시퀀스 진행 바 — 완료/현재/남은 노트 표시"""
    bar_y = 60
    bar_h = 80
    # 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (0, bar_y), (SCREEN_W, bar_y + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    n = len(game.sequence)
    if n == 0:
        return

    # 노트들을 균등 간격으로 배치
    margin = 60
    spacing = (SCREEN_W - margin * 2) // max(n - 1, 1)
    cy = bar_y + bar_h // 2

    if game.state == game.PLAY:
        for i, gesture in enumerate(game.sequence):
            cx = margin + i * spacing
            if i < game.current_note:
                # 완료된 노트
                if i < len(game.results):
                    result = game.results[i][0]
                    if result == 'correct':
                        col = (0, 255, 0)
                    else:
                        col = (0, 0, 200)
                    cv2.circle(img, (cx, cy), 18, col, -1)
                    cv2.circle(img, (cx, cy), 18, (255, 255, 255), 1)
                    mark = 'O' if result == 'correct' else 'X'
                    cv2.putText(img, mark, (cx - 8, cy + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif i == game.current_note:
                # 현재 노트 — 크게 + 깜빡임
                pulse = int(abs(math.sin(now * 4)) * 8)
                draw_gesture_icon(img, gesture, cx, cy, 28 + pulse, highlight=True)
            else:
                # 남은 노트 — 물음표
                cv2.circle(img, (cx, cy), 20, (60, 60, 60), 2)
                cv2.putText(img, '?', (cx - 7, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        # 진행률 바
        progress = game.current_note / n
        bar_x1, bar_x2 = 20, SCREEN_W - 20
        bar_bottom = bar_y + bar_h + 5
        cv2.rectangle(img, (bar_x1, bar_bottom), (bar_x2, bar_bottom + 4), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x1, bar_bottom),
                      (bar_x1 + int((bar_x2 - bar_x1) * progress), bar_bottom + 4),
                      (0, 255, 200), -1)

    elif game.state == game.MEMORIZE:
        # 암기 화면: 순차적으로 나타남
        for i, gesture in enumerate(game.sequence):
            cx = margin + i * spacing
            progress = (now - game.memorize_start) / game.preview_time()
            note_appear = i / n
            if progress >= note_appear:
                draw_gesture_icon(img, gesture, cx, cy, 25, highlight=True)
                cv2.putText(img, str(i + 1), (cx - 5, cy + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            else:
                cv2.circle(img, (cx, cy), 20, (50, 50, 50), 2)
                cv2.putText(img, '?', (cx - 7, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)


def draw_hud(img, game):
    """상단 HUD — 점수, 콤보, 라이프, 스테이지"""
    # 배경
    cv2.rectangle(img, (0, 0), (SCREEN_W, 52), (20, 20, 20), -1)

    seq_len, _, time_limit, stage_name = game.get_stage()

    # 스테이지
    cv2.putText(img, f"Stage {game.stage_idx + 1}: {stage_name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    limit_text = f"Limit:{time_limit}s" if time_limit > 0 else "No Limit"
    cv2.putText(img, limit_text,
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # 점수
    score_text = f"Score: {game.score}"
    cv2.putText(img, score_text, (SCREEN_W // 2 - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 콤보
    if game.combo > 1:
        cv2.putText(img, f"{game.combo}x COMBO", (SCREEN_W // 2 - 50, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # 라이프
    for i in range(5):
        hx = SCREEN_W - 30 * (5 - i)
        color = (0, 0, 255) if i < game.lives else (60, 60, 60)
        cv2.putText(img, "<3", (hx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_camera_feed(img, frame, game, detection_text):
    """카메라 피드 영역 (하단)"""
    cam_x = SCREEN_W // 2 - CAM_W // 2
    cam_y = 160
    # 카메라 프레임 리사이즈
    cam_display = cv2.resize(frame, (CAM_W, CAM_H))
    img[cam_y:cam_y + CAM_H, cam_x:cam_x + CAM_W] = cam_display
    # 테두리
    cv2.rectangle(img, (cam_x - 2, cam_y - 2),
                  (cam_x + CAM_W + 2, cam_y + CAM_H + 2), (100, 100, 100), 2)

    # 감지 결과 표시
    if detection_text:
        cv2.putText(img, detection_text, (cam_x + 10, cam_y + CAM_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_judgment_effect(img, result, now, judge_time):
    """판정 이펙트 (CORRECT! / WRONG / TIMEOUT)"""
    elapsed = now - judge_time
    if elapsed > 1.0:
        return

    alpha = max(0, 1.0 - elapsed)
    scale = 1.0 + elapsed * 0.5

    if result == 'correct':
        text, color = "CORRECT!", (0, 255, 100)
    elif result == 'wrong':
        text, color = "WRONG!", (0, 0, 255)
    else:
        text, color = "TIME OUT", (0, 100, 255)

    color = tuple(int(c * alpha) for c in color)
    font_scale = 1.2 * scale
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
    tx = SCREEN_W // 2 - text_size[0] // 2
    ty = 430 + int(elapsed * -30)

    cv2.putText(img, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)


def draw_title_screen(img):
    """타이틀 화면"""
    cv2.rectangle(img, (0, 0), (SCREEN_W, SCREEN_H), (20, 20, 40), -1)

    # 타이틀
    title = "RPS RHYTHM"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
    tx = SCREEN_W // 2 - text_size[0] // 2
    cv2.putText(img, title, (tx, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

    # 부제
    sub = "Remember the Sequence, Show Your Hands!"
    text_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    tx = SCREEN_W // 2 - text_size[0] // 2
    cv2.putText(img, sub, (tx, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 아이콘 표시
    icons_y = 290
    for i, g in enumerate(GESTURES):
        cx = SCREEN_W // 2 + (i - 1) * 100
        draw_gesture_icon(img, g, cx, icons_y, 35)
        cv2.putText(img, GESTURE_KR[g],
                    (cx - 35, icons_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 시작 안내
    blink = int(time.time() * 2) % 2
    if blink:
        start_text = "Press SPACE to Start"
        text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = SCREEN_W // 2 - text_size[0] // 2
        cv2.putText(img, start_text, (tx, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_memorize_screen(img, game, now):
    """시퀀스 암기 화면"""
    elapsed = now - game.memorize_start
    remaining = game.preview_time() - elapsed

    # 제목
    cv2.putText(img, "MEMORIZE!", (SCREEN_W // 2 - 80, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

    # 남은 시간 바
    bar_w = 300
    bar_x = SCREEN_W // 2 - bar_w // 2
    progress = max(0, remaining / game.preview_time())
    cv2.rectangle(img, (bar_x, 485), (bar_x + bar_w, 500), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, 485), (bar_x + int(bar_w * progress), 500), (0, 200, 255), -1)
    cv2.putText(img, f"{remaining:.1f}s", (bar_x + bar_w + 10, 498),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 시퀀스 크게 표시 (가운데)
    n = len(game.sequence)
    total_w = n * 70
    start_x = SCREEN_W // 2 - total_w // 2 + 35
    for i, gesture in enumerate(game.sequence):
        cx = start_x + i * 70
        cy = 300
        note_appear = i / n
        progress_ratio = elapsed / game.preview_time()
        if progress_ratio >= note_appear:
            # 등장 애니메이션
            age = (progress_ratio - note_appear) * game.preview_time()
            scale = min(1.0, age * 3)
            r = int(30 * scale)
            draw_gesture_icon(img, gesture, cx, cy, r, highlight=True)
            # 순서 번호
            cv2.putText(img, str(i + 1), (cx - 5, cy + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            cv2.circle(img, (cx, cy), 30, (50, 50, 50), 2)
            cv2.putText(img, "?", (cx - 7, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)


def draw_countdown(img, game, now):
    """카운트다운 화면"""
    elapsed = now - game.countdown_start
    count = 3 - int(elapsed)
    if count < 1:
        count = 1

    text = str(count)
    scale = 3.0 - (elapsed % 1.0) * 1.0
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)[0]
    tx = SCREEN_W // 2 - text_size[0] // 2
    ty = SCREEN_H // 2 + text_size[1] // 2

    alpha = max(0.3, 1.0 - (elapsed % 1.0) * 0.7)
    color = tuple(int(c * alpha) for c in (0, 255, 255))
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 4)

    cv2.putText(img, "GET READY!", (SCREEN_W // 2 - 75, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)


def draw_stage_clear(img, game):
    """스테이지 클리어 화면"""
    cv2.putText(img, "STAGE CLEAR!", (SCREEN_W // 2 - 140, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    # 결과 요약
    correct = sum(1 for r, _ in game.results if r == 'correct')
    wrong = sum(1 for r, _ in game.results if r in ('wrong', 'timeout'))

    y = 260
    cv2.putText(img, f"CORRECT: {correct}", (SCREEN_W // 2 - 80, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    cv2.putText(img, f"WRONG:   {wrong}", (SCREEN_W // 2 - 80, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"MAX COMBO: {game.max_combo}", (SCREEN_W // 2 - 80, y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    cv2.putText(img, f"STAGE SCORE: {game.stage_score}", (SCREEN_W // 2 - 100, y + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    blink = int(time.time() * 2) % 2
    if blink:
        cv2.putText(img, "Press SPACE for Next Stage", (SCREEN_W // 2 - 155, y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


def draw_game_over(img, game):
    """게임 오버 화면"""
    cv2.putText(img, "GAME OVER", (SCREEN_W // 2 - 120, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(img, f"FINAL SCORE: {game.score}", (SCREEN_W // 2 - 120, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Stage {game.stage_idx + 1}  |  Max Combo: {game.max_combo}",
                (SCREEN_W // 2 - 140, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    blink = int(time.time() * 2) % 2
    if blink:
        cv2.putText(img, "Press SPACE to Retry", (SCREEN_W // 2 - 120, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


def draw_all_clear(img, game):
    """올 클리어 화면"""
    cv2.putText(img, "ALL CLEAR!!", (SCREEN_W // 2 - 140, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(img, f"TOTAL SCORE: {game.score}", (SCREEN_W // 2 - 120, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(img, f"Max Combo: {game.max_combo}", (SCREEN_W // 2 - 85, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(img, "Congratulations!", (SCREEN_W // 2 - 100, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 1)

    blink = int(time.time() * 2) % 2
    if blink:
        cv2.putText(img, "Press SPACE to Restart", (SCREEN_W // 2 - 125, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


# ══════════════════════════════════════════════
#  비프음 생성 (pygame 없이 간단한 비프)
# ══════════════════════════════════════════════

def try_beep():
    """박자마다 비프음 시도 (없으면 무시)"""
    try:
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception:
        pass


# ══════════════════════════════════════════════
#  터미널 키보드 입력 (별도 스레드)
# ══════════════════════════════════════════════

class KeyboardReader:
    """터미널에서 직접 키 입력을 받는 non-blocking 리더.
    OpenCV 윈도우에 포커스가 없어도 키 입력이 동작합니다."""

    def __init__(self):
        self._key_buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._old_settings = None
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()

    def stop(self):
        self._running = False
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _reader_loop(self):
        while self._running:
            try:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._key_buffer.append(ch)
            except Exception:
                break

    def get_key(self):
        """버퍼에서 키 하나를 꺼냄. 없으면 None."""
        with self._lock:
            if self._key_buffer:
                return self._key_buffer.pop(0)
        return None


# ══════════════════════════════════════════════
#  메인 게임 루프
# ══════════════════════════════════════════════

def find_default_model():
    """기본 모델 경로 탐색"""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, 'models', 'rps_mobilenetv2.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_qat.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_ptq_int8.tflite'),
        # 기존 예제 모델 fallback
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2.tflite'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description='RPS Rhythm Game')
    parser.add_argument('--model', type=str, default=None,
                        help='TFLite 또는 ONNX 모델 경로')
    args = parser.parse_args()

    # 모델 찾기
    model_path = args.model or find_default_model()
    if not model_path or not os.path.exists(model_path):
        print("[오류] 모델을 찾을 수 없습니다.")
        print("  --model 옵션으로 TFLite 또는 ONNX 모델 경로를 지정하거나,")
        print("  train_model.py로 TFLite 모델을 먼저 학습하세요.")
        print("  또는 examples/03_CNN_Based_On-Device_AI/ 의 TFLite 모델을 복사하세요.")
        return

    print(f"모델 로드: {model_path}")
    detector = RPSDetector(model_path)
    game = GameState()

    # 카메라
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('RPS Rhythm', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RPS Rhythm', SCREEN_W, SCREEN_H)

    # 터미널 키보드 리더 시작
    kb = KeyboardReader()
    kb.start()

    last_judge_result = None
    last_judge_time = 0
    fps_time = time.time()

    print("\n=== RPS RHYTHM ===")
    print("이 터미널에서 키 입력:  SPACE=시작  q=종료\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # ── 손 감지 (플레이 중에만) ──
        detection_text = ""
        if game.state == game.PLAY:
            gesture, conf, bbox = detector.detect(frame)
            if gesture and conf > 0.5:
                detection_text = f"{GESTURE_KR[gesture]} ({conf:.0%})"
                game.last_detection = gesture
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  GESTURE_COLOR[gesture], 2)
            else:
                game.last_detection = None

        # ── 상태 머신 ──
        if game.state == game.TITLE:
            draw_title_screen(screen)

        elif game.state == game.MEMORIZE:
            draw_hud(screen, game)
            draw_memorize_screen(screen, game, now)
            if now - game.memorize_start >= game.preview_time():
                game.state = game.COUNTDOWN
                game.countdown_start = now

        elif game.state == game.COUNTDOWN:
            draw_hud(screen, game)
            draw_camera_feed(screen, frame, game, "")
            draw_countdown(screen, game, now)
            if now - game.countdown_start >= 3.0:
                game.state = game.PLAY
                game.note_start_time = now
                game.confirm_count = 0
                game.confirm_gesture = None
                game.prev_confirmed = None

        elif game.state == game.PLAY:
            draw_hud(screen, game)
            draw_camera_feed(screen, frame, game, detection_text)
            draw_sequence_bar(screen, game, now)

            if game.current_note < len(game.sequence):
                target = game.sequence[game.current_note]
                time_limit = game.note_time_limit()
                elapsed_note = now - game.note_start_time

                # 현재 내야 할 제스처 크게 표시 (카메라 옆)
                cam_x = SCREEN_W // 2 - CAM_W // 2
                draw_gesture_icon(screen, target,
                                  cam_x + CAM_W + 40, 250, 35, highlight=True)
                cv2.putText(screen, GESTURE_KR[target],
                            (cam_x + CAM_W + 5, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(screen, f"#{game.current_note + 1}/{len(game.sequence)}",
                            (cam_x + CAM_W + 10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # 제한시간 표시
                if time_limit > 0:
                    remaining = max(0, time_limit - elapsed_note)
                    bar_w = 100
                    bar_x = cam_x + CAM_W + 5
                    bar_y_pos = 340
                    progress = remaining / time_limit
                    bar_color = (0, 255, 0) if progress > 0.3 else (0, 0, 255)
                    cv2.rectangle(screen, (bar_x, bar_y_pos),
                                  (bar_x + bar_w, bar_y_pos + 8), (50, 50, 50), -1)
                    cv2.rectangle(screen, (bar_x, bar_y_pos),
                                  (bar_x + int(bar_w * progress), bar_y_pos + 8),
                                  bar_color, -1)
                    cv2.putText(screen, f"{remaining:.1f}s",
                                (bar_x + bar_w + 5, bar_y_pos + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                # ── 제스처 확정 로직 (연속 N프레임 같은 제스처 감지) ──
                confirmed = False
                if game.last_detection:
                    # 이전에 확정된 것과 같으면 무시 (다른 제스처를 해야 넘어감)
                    if game.last_detection == game.prev_confirmed:
                        pass
                    elif game.last_detection == game.confirm_gesture:
                        game.confirm_count += 1
                    else:
                        game.confirm_gesture = game.last_detection
                        game.confirm_count = 1

                    if game.confirm_count >= CONFIRM_FRAMES:
                        confirmed = True
                        detected = game.confirm_gesture
                else:
                    # 손이 안 보이면 이전 확정 초기화 (다음 제스처 준비)
                    game.prev_confirmed = None
                    game.confirm_count = 0
                    game.confirm_gesture = None

                if confirmed:
                    if detected == target:
                        # 정답!
                        game.combo += 1
                        game.max_combo = max(game.max_combo, game.combo)
                        bonus = min(game.combo, 10)
                        earned = SCORE_CORRECT + bonus * SCORE_BONUS_PER_COMBO
                        game.score += earned
                        game.stage_score += earned
                        game.results.append(('correct', target))
                        last_judge_result = 'correct'
                    else:
                        # 오답
                        game.combo = 0
                        game.results.append(('wrong', target))
                        game.lives -= 1
                        last_judge_result = 'wrong'
                    last_judge_time = now
                    game.prev_confirmed = detected
                    game.confirm_count = 0
                    game.confirm_gesture = None
                    game.current_note += 1
                    game.note_start_time = now

                elif time_limit > 0 and elapsed_note > time_limit:
                    # 시간 초과
                    game.combo = 0
                    game.results.append(('timeout', target))
                    game.lives -= 1
                    last_judge_result = 'timeout'
                    last_judge_time = now
                    game.confirm_count = 0
                    game.confirm_gesture = None
                    game.prev_confirmed = None
                    game.current_note += 1
                    game.note_start_time = now

                # 라이프 체크
                if game.lives <= 0:
                    game.state = game.GAME_OVER
            else:
                # 시퀀스 완료 → 스테이지 클리어
                game.state = game.STAGE_CLEAR

            # 판정 이펙트
            if last_judge_result:
                draw_judgment_effect(screen, last_judge_result, now, last_judge_time)

        elif game.state == game.STAGE_CLEAR:
            draw_hud(screen, game)
            draw_stage_clear(screen, game)

        elif game.state == game.GAME_OVER:
            draw_game_over(screen, game)

        elif game.state == game.ALL_CLEAR:
            draw_all_clear(screen, game)

        # FPS
        cur_time = time.time()
        fps = 1.0 / max(cur_time - fps_time, 0.001)
        fps_time = cur_time
        cv2.putText(screen, f"FPS:{fps:.0f}", (SCREEN_W - 80, SCREEN_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow('RPS Rhythm', screen)
        cv2.waitKey(1)  # 화면 갱신용 (최소 대기)

        # ── 키 입력 (터미널에서 수신) ──
        key_ch = kb.get_key()
        if key_ch == 'q':
            break
        elif key_ch == ' ':
            if game.state == game.TITLE:
                game.reset()
                game.state = game.MEMORIZE
                game.generate_sequence()
                game.memorize_start = time.time()
                print(f"[Stage {game.stage_idx + 1}] 시퀀스: {game.sequence}")

            elif game.state == game.STAGE_CLEAR:
                game.stage_idx += 1
                if game.stage_idx >= len(STAGES):
                    game.state = game.ALL_CLEAR
                else:
                    game.state = game.MEMORIZE
                    game.generate_sequence()
                    game.memorize_start = time.time()
                    print(f"[Stage {game.stage_idx + 1}] 시퀀스: {game.sequence}")

            elif game.state in (game.GAME_OVER, game.ALL_CLEAR):
                game.reset()
                game.state = game.MEMORIZE
                game.generate_sequence()
                game.memorize_start = time.time()
                print(f"[Stage 1] 시퀀스: {game.sequence}")

    kb.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n최종 점수: {game.score} | 최대 콤보: {game.max_combo}")


if __name__ == '__main__':
    main()
