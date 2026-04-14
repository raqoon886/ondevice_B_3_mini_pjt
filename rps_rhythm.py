"""
RPS Rhythm - 가위바위보 리듬 게임 (성능/인식 개선판)

v1 대비 변경점:
  1. 카메라 캡처 스레드 분리         → cap.read() 블로킹 제거
  2. 추론(손검출+분류) 워커 스레드 분리 → 메인 루프가 추론 대기 안 함
  3. Frame skip (INFERENCE_EVERY_N)   → 매 프레임 추론 X
  4. TFLite num_threads=4              → 멀티코어 활용
  5. 경계 clip (OFFSET 초과해도 손이 잘리지 않으면 인식)
  6. uint8 양자화 모델 입출력 자동 처리
  7. PredictionSmoother (다수결+conf 합산) → 안정화
  8. HandDetector 임계치 완화 (0.7→0.6)
  9. 시간 기반 HOLD + 쿨다운 (연속 같은 제스처 OK)
  10. ONNX YOLO 모델 지원 (HandDetector 우회, 한 번에 검출+분류)

사용법:
  python rps_rhythm.py                                       (자동 탐색)
  python rps_rhythm.py --model models/rps_mobilenetv2.tflite (TFLite)
  python rps_rhythm.py --model models/best.onnx              (ONNX/YOLO)

조작:
  SPACE → 게임 시작 / 다음 스테이지 / 재시작
  q     → 종료
"""

import sys
import os
import argparse
import time
import random
import math
import threading
import queue
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
SCREEN_W = 640
SCREEN_H = 520
CAM_W, CAM_H = 320, 240
IMG_SIZE = 224
OFFSET = 30

# ── 성능 튜닝 파라미터 ──
INFERENCE_EVERY_N = 2     # N 프레임 중 1번만 추론 (1=매번, 2=절반, 3=1/3)
TFLITE_THREADS = 4         # TFLite 인터프리터 스레드 수 (보드 코어 수)
HAND_DETECT_CON = 0.6      # 손 감지 임계치 (낮을수록 잘 잡힘)
SMOOTHER_WINDOW = 5        # 다수결 윈도우 (프레임)
SMOOTHER_CONF_TH = 0.65    # 분류 confidence 하한
CAM_FPS_TARGET = 30

GESTURES = ['scissors', 'rock', 'paper']
GESTURE_EMOJI = {'scissors': 'V', 'rock': 'O', 'paper': 'W'}
GESTURE_KR = {'scissors': 'SCISSORS', 'rock': 'ROCK', 'paper': 'PAPER'}
GESTURE_COLOR = {
    'scissors': (255, 80, 80),
    'rock':     (80, 255, 80),
    'paper':    (80, 80, 255),
}

SCORE_CORRECT = 200
SCORE_BONUS_PER_COMBO = 30
HOLD_TIME = 0.7           # 같은 제스처를 N초 이상 유지해야 확정 (시간 기반)
COOLDOWN_TIME = 0.25      # 확정 후 다음 입력 받기까지 대기 (같은 제스처 연속 입력 가능)
TIME_LIMIT_PER_NOTE = 8

STAGES = [
    # (시퀀스 길이, 미리보기 시간(초), 노트 제한시간(초), 설명)
    (3,  6.0, 0,  "Tutorial"),
    (5,  5.0, 0,  "Easy 1"),
    (5,  4.5, 8,  "Easy 2"),
    (5,  4.5, 10, "Normal 1"),
    (5,  4.0, 10, "Normal 2"),
    (6,  4.0, 8,  "Normal 3"),
    (7,  3.8, 8,  "Hard 1"),
    (8,  3.5, 7,  "Hard 2"),
    (9,  3.5, 6,  "Hard 3"),
    (10, 3.0, 5,  "Expert"),
]


# ══════════════════════════════════════════════
#  카메라 캡처 스레드
#  - cap.read()를 백그라운드에서 계속 호출
#  - 메인은 항상 "최신 프레임"만 가져감 (blocking 없음)
# ══════════════════════════════════════════════
class ThreadedCamera:
    def __init__(self, src=0, w=CAM_W, h=CAM_H, fps=CAM_FPS_TARGET):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        # 첫 프레임이 들어올 때까지 잠깐 대기
        for _ in range(20):
            if self._frame is not None:
                break
            time.sleep(0.05)
        return self

    def _loop(self):
        while self._running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ══════════════════════════════════════════════
#  손 검출 + RPS 분류 (워커 스레드용)
# ══════════════════════════════════════════════
class RPSDetector:
    def __init__(self, model_path, num_threads=TFLITE_THREADS):
        self.model_path = model_path
        self.is_onnx = model_path.lower().endswith('.onnx')

        if self.is_onnx:
            # ONNX/YOLO 모드: HandDetector 불필요 (YOLO가 위치+클래스 동시 검출)
            if YOLO is None:
                raise ImportError(
                    'ONNX 모델 사용을 위해 ultralytics가 필요합니다. '
                    '"pip install ultralytics" 로 설치하세요.')
            self.hd = None
            self.model = YOLO(model_path, task='detect')
            self.names = {int(k): v.lower() for k, v in self.model.names.items()}
            self.interpreter = None
            self.input_dtype = None
            self.is_quantized = False
            self.in_q = (0.0, 0)
            self.out_q = (0.0, 0)
        else:
            # TFLite 모드: cvzone HandDetector + MobileNetV2 분류
            self.hd = HandDetector(maxHands=1, detectionCon=HAND_DETECT_CON)
            self.model = None
            self.names = {}
            try:
                self.interpreter = Interpreter(model_path=model_path,
                                               num_threads=num_threads)
            except TypeError:
                self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_dtype = self.input_details[0]['dtype']
            self.in_q = self.input_details[0].get('quantization', (0.0, 0))
            self.out_q = self.output_details[0].get('quantization', (0.0, 0))
            self.is_quantized = self.input_dtype in (np.uint8, np.int8)

    # ────────────────────────────────────────────
    def make_square_img(self, img):
        ho, wo = img.shape[0], img.shape[1]
        if ho == 0 or wo == 0:
            return None
        wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        if ho / wo > 1:
            k = IMG_SIZE / ho
            wk = max(1, int(wo * k))
            img = cv2.resize(img, (wk, IMG_SIZE))
            d = (IMG_SIZE - wk) // 2
            wbg[:, d:d + wk] = img
        else:
            k = IMG_SIZE / wo
            hk = max(1, int(ho * k))
            img = cv2.resize(img, (IMG_SIZE, hk))
            d = (IMG_SIZE - hk) // 2
            wbg[d:d + hk, :] = img
        return wbg

    # ────────────────────────────────────────────
    def _prepare_input(self, square_bgr):
        rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
        if self.is_quantized:
            scale, zero = self.in_q
            if scale and scale > 0:
                # uint8/int8 양자화 모델: (img/255 / scale) + zero 형태가 일반적
                # 다만 학습 시 정규화 안 했다면 그대로 캐스팅도 가능
                # 안전하게 둘 다 시도: 우선 그대로 넣고 (대부분의 RPS 모델 입력)
                arr = rgb.astype(self.input_dtype)
            else:
                arr = rgb.astype(self.input_dtype)
        else:
            arr = rgb.astype(self.input_dtype)
        return np.expand_dims(arr, 0)

    def _decode_output(self, raw):
        if self.is_quantized:
            scale, zero = self.out_q
            if scale and scale > 0:
                out = (raw.astype(np.float32) - zero) * scale
            else:
                out = raw.astype(np.float32)
        else:
            out = raw.astype(np.float32)
        # softmax 정규화 (이미 확률이면 영향 적음)
        if out.max() > 1.0 or out.min() < 0.0:
            e = np.exp(out - out.max())
            out = e / e.sum()
        s = out.sum()
        if s > 0 and abs(s - 1.0) > 1e-3:
            out = out / s
        return out

    # ────────────────────────────────────────────
    def detect(self, frame):
        """프레임 → (gesture_name, confidence, bbox) or (None, 0, None)
        v1 대비 OFFSET 초과 시 None 반환하지 않고 clip 처리.
        ONNX/YOLO 모드면 HandDetector 우회.
        """
        # ── ONNX/YOLO 분기: 한 번의 추론으로 검출+분류 ──
        if self.is_onnx:
            results = self.model(frame, verbose=False)
            if not results or len(results) == 0:
                return None, 0.0, None
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                return None, 0.0, None
            confs = res.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
            classes = res.boxes.cls.cpu().numpy().astype(int)
            class_id = int(classes[idx])
            gesture = self.names.get(class_id, None)
            if gesture not in GESTURES:
                return None, 0.0, None
            xyxy = res.boxes.xyxy.cpu().numpy()[idx]
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            return gesture, float(confs[idx]), (x1, y1, x2, y2)

        # ── TFLite 분기 ──
        hands, _ = self.hd.findHands(frame, draw=False)
        if not hands:
            return None, 0.0, None

        x, y, w, h = hands[0]['bbox']
        # 경계 clip (반환 None 안 함)
        H, W = frame.shape[:2]
        x1 = max(0, x - OFFSET)
        y1 = max(0, y - OFFSET)
        x2 = min(W, x + w + OFFSET)
        y2 = min(H, y + h + OFFSET)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None, 0.0, None

        square = self.make_square_img(frame[y1:y2, x1:x2])
        if square is None:
            return None, 0.0, None

        inp = self._prepare_input(square)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prob = self._decode_output(raw)

        ans = int(np.argmax(prob))
        return GESTURES[ans], float(prob[ans]), (x1, y1, x2, y2)


# ══════════════════════════════════════════════
#  추론 워커 스레드
#  - 메인 루프와 독립적으로 손검출+추론 수행
#  - 결과는 latest_result에 보관
# ══════════════════════════════════════════════
class InferenceWorker:
    def __init__(self, detector: RPSDetector):
        self.detector = detector
        self._frame_q = queue.Queue(maxsize=1)  # 항상 최신 프레임 1장만
        self._result_lock = threading.Lock()
        # (gesture, conf, bbox, latency_ms, seq) - seq는 결과가 갱신될 때마다 +1
        self._latest = (None, 0.0, None, 0.0, 0)
        self._seq = 0
        self._running = True
        self._enabled = False                   # PLAY 상태에서만 True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

    def set_enabled(self, on: bool):
        self._enabled = on
        if not on:
            with self._result_lock:
                self._latest = (None, 0.0, None, 0.0, self._seq)
            # 큐 비우기
            try:
                while True:
                    self._frame_q.get_nowait()
            except queue.Empty:
                pass

    def submit(self, frame):
        """최신 프레임을 워커에 전달 (이전 프레임이 남아있으면 덮어씀)"""
        if not self._enabled:
            return
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            pass

    def get_latest(self):
        with self._result_lock:
            return self._latest

    def _loop(self):
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self._enabled:
                continue
            t0 = time.time()
            g, c, bbox = self.detector.detect(frame)
            dt_ms = (time.time() - t0) * 1000.0
            with self._result_lock:
                self._seq += 1
                self._latest = (g, c, bbox, dt_ms, self._seq)


# ══════════════════════════════════════════════
#  예측 안정화 (다수결 + conf 합산)
# ══════════════════════════════════════════════
class PredictionSmoother:
    def __init__(self, window=SMOOTHER_WINDOW, conf_th=SMOOTHER_CONF_TH):
        self.window = window
        self.conf_th = conf_th
        self.history = []  # [(gesture, conf), ...]

    def update(self, gesture, conf):
        if gesture is not None and conf >= self.conf_th:
            self.history.append((gesture, conf))
        else:
            self.history.append((None, 0.0))
        if len(self.history) > self.window:
            self.history.pop(0)

        votes, conf_sum = {}, {}
        for g, c in self.history:
            if g is None:
                continue
            votes[g] = votes.get(g, 0) + 1
            conf_sum[g] = conf_sum.get(g, 0) + c
        if not votes:
            return None, 0.0
        best = max(votes, key=lambda k: (votes[k], conf_sum[k]))
        return best, conf_sum[best] / votes[best]

    def reset(self):
        self.history = []


# ══════════════════════════════════════════════
#  게임 상태 (v1과 동일)
# ══════════════════════════════════════════════
class GameState:
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
        self.results = []
        self.memorize_start = 0
        self.countdown_start = 0
        self.last_detection = None
        self.stage_score = 0
        self.confirm_count = 0       # (호환 유지용, 실제론 사용 안함)
        self.confirm_gesture = None  # 현재 hold 중인 제스처
        self.hold_start_time = 0     # 현재 제스처 hold 시작 시각
        self.cooldown_until = 0      # 이 시각까지는 입력 무시 (직전 확정 후 안정화)
        self.note_start_time = 0

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
        self.hold_start_time = 0
        self.cooldown_until = 0

    def note_time_limit(self):
        return self.get_stage()[2]

    def preview_time(self):
        return self.get_stage()[1]


# ══════════════════════════════════════════════
#  UI (v1과 동일)
# ══════════════════════════════════════════════
def draw_gesture_icon(img, gesture, cx, cy, size, alpha=1.0, highlight=False):
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
    cv2.putText(img, text, (cx - text_size[0]//2, cy + text_size[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)


def draw_sequence_bar(img, game, now):
    bar_y = 60
    bar_h = 80
    overlay = img.copy()
    cv2.rectangle(overlay, (0, bar_y), (SCREEN_W, bar_y + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    n = len(game.sequence)
    if n == 0:
        return
    margin = 60
    spacing = (SCREEN_W - margin * 2) // max(n - 1, 1)
    cy = bar_y + bar_h // 2
    if game.state == game.PLAY:
        for i, _ in enumerate(game.sequence):
            cx = margin + i * spacing
            label = str(i + 1)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            if i < game.current_note:
                if i < len(game.results):
                    result = game.results[i][0]
                    col = (0, 255, 0) if result == 'correct' else (0, 0, 200)
                    cv2.circle(img, (cx, cy), 20, col, -1)
                    cv2.circle(img, (cx, cy), 20, (255, 255, 255), 1)
                    cv2.putText(img, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif i == game.current_note:
                pulse = int(abs(math.sin(now * 4)) * 8)
                r = 26 + pulse
                cv2.circle(img, (cx, cy), r + 5, (255, 255, 255), 2)
                cv2.circle(img, (cx, cy), r, (0, 200, 255), -1)
                cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
                cv2.putText(img, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.circle(img, (cx, cy), 20, (60, 60, 60), 2)
                cv2.putText(img, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 2)
        progress = game.current_note / n
        bar_x1, bar_x2 = 20, SCREEN_W - 20
        bar_bottom = bar_y + bar_h + 5
        cv2.rectangle(img, (bar_x1, bar_bottom), (bar_x2, bar_bottom + 4), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x1, bar_bottom),
                      (bar_x1 + int((bar_x2 - bar_x1) * progress), bar_bottom + 4),
                      (0, 255, 200), -1)
    elif game.state == game.MEMORIZE:
        for i, gesture in enumerate(game.sequence):
            cx = margin + i * spacing
            progress = (now - game.memorize_start) / game.preview_time()
            if progress >= i / n:
                draw_gesture_icon(img, gesture, cx, cy, 25, highlight=True)
                cv2.putText(img, str(i + 1), (cx - 5, cy + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            else:
                cv2.circle(img, (cx, cy), 20, (50, 50, 50), 2)
                cv2.putText(img, '?', (cx - 7, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)


def draw_hud(img, game):
    cv2.rectangle(img, (0, 0), (SCREEN_W, 52), (20, 20, 20), -1)
    seq_len, _, time_limit, stage_name = game.get_stage()
    cv2.putText(img, f"Stage {game.stage_idx + 1}: {stage_name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    limit_text = f"Limit:{time_limit}s" if time_limit > 0 else "No Limit"
    cv2.putText(img, limit_text, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(img, f"Score: {game.score}", (SCREEN_W // 2 - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if game.combo > 1:
        cv2.putText(img, f"{game.combo}x COMBO", (SCREEN_W // 2 - 50, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    for i in range(5):
        hx = SCREEN_W - 30 * (5 - i)
        color = (0, 0, 255) if i < game.lives else (60, 60, 60)
        cv2.putText(img, "<3", (hx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_camera_feed(img, frame, game, detection_text):
    cam_x = SCREEN_W // 2 - CAM_W // 2
    cam_y = 160
    cam_display = cv2.resize(frame, (CAM_W, CAM_H))
    img[cam_y:cam_y + CAM_H, cam_x:cam_x + CAM_W] = cam_display
    cv2.rectangle(img, (cam_x - 2, cam_y - 2),
                  (cam_x + CAM_W + 2, cam_y + CAM_H + 2), (100, 100, 100), 2)
    if detection_text:
        cv2.putText(img, detection_text, (cam_x + 10, cam_y + CAM_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_judgment_effect(img, result, now, judge_time):
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
    cv2.rectangle(img, (0, 0), (SCREEN_W, SCREEN_H), (20, 20, 40), -1)
    title = "RPS RHYTHM"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
    cv2.putText(img, title, (SCREEN_W//2 - text_size[0]//2, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
    sub = "Faster Detection, Smoother Gameplay"
    text_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(img, sub, (SCREEN_W//2 - text_size[0]//2, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    icons_y = 290
    for i, g in enumerate(GESTURES):
        cx = SCREEN_W // 2 + (i - 1) * 100
        draw_gesture_icon(img, g, cx, icons_y, 35)
        cv2.putText(img, GESTURE_KR[g], (cx - 35, icons_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    blink = int(time.time() * 2) % 2
    if blink:
        start_text = "Press SPACE to Start"
        text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(img, start_text, (SCREEN_W//2 - text_size[0]//2, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_memorize_screen(img, game, now):
    elapsed = now - game.memorize_start
    remaining = game.preview_time() - elapsed
    cv2.putText(img, "MEMORIZE!", (SCREEN_W // 2 - 80, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
    bar_w = 300
    bar_x = SCREEN_W // 2 - bar_w // 2
    progress = max(0, remaining / game.preview_time())
    cv2.rectangle(img, (bar_x, 485), (bar_x + bar_w, 500), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, 485), (bar_x + int(bar_w * progress), 500), (0, 200, 255), -1)
    cv2.putText(img, f"{remaining:.1f}s", (bar_x + bar_w + 10, 498),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    n = len(game.sequence)
    total_w = n * 70
    start_x = SCREEN_W // 2 - total_w // 2 + 35
    for i, gesture in enumerate(game.sequence):
        cx = start_x + i * 70
        cy = 300
        progress_ratio = elapsed / game.preview_time()
        if progress_ratio >= i / n:
            age = (progress_ratio - i / n) * game.preview_time()
            scale = min(1.0, age * 3)
            r = int(30 * scale)
            draw_gesture_icon(img, gesture, cx, cy, r, highlight=True)
            cv2.putText(img, str(i + 1), (cx - 5, cy + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            cv2.circle(img, (cx, cy), 30, (50, 50, 50), 2)
            cv2.putText(img, "?", (cx - 7, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)


def draw_countdown(img, game, now):
    elapsed = now - game.countdown_start
    count = max(1, 3 - int(elapsed))
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
    cv2.putText(img, "STAGE CLEAR!", (SCREEN_W // 2 - 140, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
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
#  터미널 키보드 (v1과 동일)
# ══════════════════════════════════════════════
class KeyboardReader:
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
        with self._lock:
            if self._key_buffer:
                return self._key_buffer.pop(0)
        return None


# ══════════════════════════════════════════════
#  모델 경로 탐색
# ══════════════════════════════════════════════
def find_default_model():
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, 'models', 'rps_mobilenetv2.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_qat.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_ptq_int8.tflite'),
        os.path.join(base, 'models', 'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(base, 'models', 'best.onnx'),
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2.tflite'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ══════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='RPS Rhythm Game')
    parser.add_argument('--model', type=str, default=None,
                        help='TFLite 또는 ONNX 모델 경로')
    parser.add_argument('--no-thread-cam', action='store_true',
                        help='카메라 스레드 끄기 (디버그용)')
    parser.add_argument('--inference-every', type=int, default=INFERENCE_EVERY_N,
                        help='몇 프레임마다 추론할지 (기본 2)')
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if not model_path or not os.path.exists(model_path):
        print("[오류] 모델을 찾을 수 없습니다.")
        print("  --model 옵션으로 TFLite 또는 ONNX 모델 경로를 지정하거나,")
        print("  train_model.py 로 모델을 먼저 학습하세요.")
        return

    is_onnx = model_path.lower().endswith('.onnx')
    print(f"[모델] {model_path}  ({'ONNX/YOLO' if is_onnx else 'TFLite'})")
    print(f"[설정] inference_every={args.inference_every}, "
          f"tflite_threads={TFLITE_THREADS}, hand_conf={HAND_DETECT_CON}, "
          f"hold={HOLD_TIME}s, cooldown={COOLDOWN_TIME}s")

    detector = RPSDetector(model_path, num_threads=TFLITE_THREADS)
    game = GameState()
    smoother = PredictionSmoother()

    # 카메라
    if args.no_thread_cam:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        def grab():
            ok, f = cam.read(); return f if ok else None
        def release(): cam.release()
    else:
        tcam = ThreadedCamera().start()
        def grab(): return tcam.read()
        def release(): tcam.release()

    # 추론 워커
    worker = InferenceWorker(detector)
    worker.start()

    cv2.namedWindow('RPS Rhythm', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RPS Rhythm', SCREEN_W, SCREEN_H)

    kb = KeyboardReader()
    kb.start()

    last_judge_result = None
    last_judge_time = 0
    fps_time = time.time()
    frame_idx = 0
    inf_latency_ms = 0.0

    # 안정화된 결과 (새 추론 결과가 올 때만 갱신, 그 외 메인루프에서는 유지)
    stable_g, stable_c = None, 0.0
    last_seen_seq = 0

    print("\n=== RPS RHYTHM ===")
    print("터미널에서 키 입력:  SPACE=시작  q=종료\n")

    try:
        while True:
            frame = grab()
            if frame is None:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)

            now = time.time()
            screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            frame_idx += 1

            # ── 추론 워커 제어 ──
            in_play = (game.state == game.PLAY)
            worker.set_enabled(in_play)

            # PLAY 중이고 N프레임마다 frame을 워커에 push
            if in_play and (frame_idx % max(args.inference_every, 1) == 0):
                worker.submit(frame)

            # 워커가 가지고 있는 최신 결과 가져오기
            # ※ 같은 추론 결과를 메인루프가 N번 보더라도 smoother는 1번만 업데이트
            detection_text = ""
            bbox = None
            if in_play:
                g, c, bbox, dt_ms, seq = worker.get_latest()
                if dt_ms > 0:
                    inf_latency_ms = 0.85 * inf_latency_ms + 0.15 * dt_ms
                # 새 추론 결과일 때만 smoother 업데이트
                if seq != last_seen_seq:
                    last_seen_seq = seq
                    stable_g, stable_c = smoother.update(g, c)
                # stable_g 는 새 결과 없을 때 이전 값 유지 (메인루프 60Hz가 영향 X)
                if stable_g:
                    detection_text = f"{GESTURE_KR[stable_g]} ({stable_c:.0%})"
                    game.last_detection = stable_g
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      GESTURE_COLOR[stable_g], 2)
                else:
                    game.last_detection = None
            else:
                smoother.reset()
                stable_g, stable_c = None, 0.0
                last_seen_seq = 0
                game.last_detection = None

            # ── 상태 머신 (v1과 동일) ──
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
                    game.confirm_gesture = None
                    game.hold_start_time = 0
                    game.cooldown_until = 0
                    smoother.reset()
                    stable_g, stable_c = None, 0.0

            elif game.state == game.PLAY:
                draw_hud(screen, game)
                draw_camera_feed(screen, frame, game, "")
                draw_sequence_bar(screen, game, now)

                if game.current_note < len(game.sequence):
                    target = game.sequence[game.current_note]
                    time_limit = game.note_time_limit()
                    elapsed_note = now - game.note_start_time

                    cam_x = SCREEN_W // 2 - CAM_W // 2

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

                    # ── 제스처 확정 (시간 기반 HOLD + 쿨다운) ──
                    # 같은 제스처를 HOLD_TIME 초 유지 → 확정.
                    # 확정 직후 COOLDOWN_TIME 초간 입력 무시 → 그 후엔 손이 그대로여도
                    # hold 타이머가 새로 시작 (같은 제스처 연속 입력 OK).
                    confirmed = False
                    detected = None
                    cur_g = game.last_detection
                    in_cooldown = now < game.cooldown_until

                    if in_cooldown:
                        # 쿨다운: hold 타이머 리셋 (시각 피드백을 위해 유지하지 않음)
                        game.confirm_gesture = None
                        game.hold_start_time = 0
                    elif cur_g:
                        if cur_g == game.confirm_gesture:
                            # 같은 제스처 유지 중 → hold 시간 누적
                            held = now - game.hold_start_time
                            if held >= HOLD_TIME:
                                confirmed = True
                                detected = cur_g
                        else:
                            # 새 제스처 시작 (또는 손 새로 잡힘) → hold 타이머 시작
                            game.confirm_gesture = cur_g
                            game.hold_start_time = now
                    else:
                        # 손 안 보임 → hold 리셋
                        game.confirm_gesture = None
                        game.hold_start_time = 0

                    # ── 진행 바 표시 ──
                    hbar_w = 200
                    hbar_x = SCREEN_W // 2 - hbar_w // 2
                    hbar_y = 145
                    if in_cooldown:
                        # 쿨다운 진행 바 (회색)
                        cd_progress = 1.0 - (game.cooldown_until - now) / COOLDOWN_TIME
                        cv2.rectangle(screen, (hbar_x, hbar_y),
                                      (hbar_x + hbar_w, hbar_y + 10),
                                      (50, 50, 50), -1)
                        cv2.rectangle(screen, (hbar_x, hbar_y),
                                      (hbar_x + int(hbar_w * cd_progress), hbar_y + 10),
                                      (120, 120, 120), -1)
                        cv2.putText(screen, "WAIT...", (hbar_x, hbar_y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                    elif game.confirm_gesture and game.hold_start_time > 0:
                        # HOLD 진행 바 (제스처 색상)
                        held = min(now - game.hold_start_time, HOLD_TIME)
                        hold_progress = held / HOLD_TIME
                        cv2.rectangle(screen, (hbar_x, hbar_y),
                                      (hbar_x + hbar_w, hbar_y + 10),
                                      (50, 50, 50), -1)
                        bar_color = GESTURE_COLOR.get(game.confirm_gesture, (200, 200, 200))
                        cv2.rectangle(screen, (hbar_x, hbar_y),
                                      (hbar_x + int(hbar_w * hold_progress), hbar_y + 10),
                                      bar_color, -1)
                        cv2.putText(screen, f"HOLD {GESTURE_KR[game.confirm_gesture]}",
                                    (hbar_x, hbar_y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                    if confirmed:
                        if detected == target:
                            game.combo += 1
                            game.max_combo = max(game.max_combo, game.combo)
                            bonus = min(game.combo, 10)
                            earned = SCORE_CORRECT + bonus * SCORE_BONUS_PER_COMBO
                            game.score += earned
                            game.stage_score += earned
                            game.results.append(('correct', target))
                            last_judge_result = 'correct'
                        else:
                            game.combo = 0
                            game.results.append(('wrong', target))
                            game.lives -= 1
                            last_judge_result = 'wrong'
                        last_judge_time = now
                        game.confirm_gesture = None
                        game.hold_start_time = 0
                        game.cooldown_until = now + COOLDOWN_TIME
                        game.current_note += 1
                        game.note_start_time = now
                        smoother.reset()
                        stable_g, stable_c = None, 0.0

                    elif time_limit > 0 and elapsed_note > time_limit:
                        game.combo = 0
                        game.results.append(('timeout', target))
                        game.lives -= 1
                        last_judge_result = 'timeout'
                        last_judge_time = now
                        game.confirm_gesture = None
                        game.hold_start_time = 0
                        game.cooldown_until = now + COOLDOWN_TIME
                        game.current_note += 1
                        game.note_start_time = now
                        smoother.reset()
                        stable_g, stable_c = None, 0.0

                    if game.lives <= 0:
                        game.state = game.GAME_OVER
                else:
                    game.state = game.STAGE_CLEAR

                if last_judge_result:
                    draw_judgment_effect(screen, last_judge_result, now, last_judge_time)

            elif game.state == game.STAGE_CLEAR:
                draw_hud(screen, game)
                draw_stage_clear(screen, game)

            elif game.state == game.GAME_OVER:
                draw_game_over(screen, game)

            elif game.state == game.ALL_CLEAR:
                draw_all_clear(screen, game)

            # ── FPS / 추론 latency 표시 ──
            cur_time = time.time()
            fps = 1.0 / max(cur_time - fps_time, 0.001)
            fps_time = cur_time
            cv2.putText(screen, f"FPS:{fps:.0f}  INF:{inf_latency_ms:.0f}ms",
                        (SCREEN_W - 180, SCREEN_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.imshow('RPS Rhythm', screen)
            cv2.waitKey(1)

            # ── 키 입력 ──
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
    finally:
        kb.stop()
        worker.stop()
        release()
        cv2.destroyAllWindows()
        print(f"\n최종 점수: {game.score} | 최대 콤보: {game.max_combo}")


if __name__ == '__main__':
    main()