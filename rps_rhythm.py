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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
GESTURE_IMAGE = {}
for gesture_name, filename in {
        'scissors': 'scissors.png',
        'rock':     'rock.png',
        'paper':    'paper.png',
    }.items():
    image_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            GESTURE_IMAGE[gesture_name] = image

SCORE_BONUS_PER_COMBO = 20
COOLDOWN_TIME = 0.15      # 노트 판정 후 짧은 대기

# ── 레인/판정 관련 상수 ──
LANE_X_POSITIONS = {
    'scissors': SCREEN_W // 2 - 120,
    'rock':     SCREEN_W // 2,
    'paper':    SCREEN_W // 2 + 120,
}
LANE_TOP    = 60
LANE_BOTTOM = 450
JUDGE_Y     = LANE_BOTTOM
NOTE_SPEED  = 180          # 픽셀/초

# 노트 바 판정 구간 시간 (짧게)
NOTE_ACTIVE_TIME_MIN = 0.4
NOTE_ACTIVE_TIME_MAX = 1.5

# 쉐이크 노트: 짧은 시간 안에 제스처 변화 횟수
SHAKE_THRESHOLD = 3        # N번 이상 제스처 변화 → 성공

# PERFECT/GOOD/MISS 비율 기준
JUDGE_PERFECT_RATIO = 0.80
JUDGE_GOOD_RATIO    = 0.45

# 판정별 점수
SCORE_PERFECT = 300
SCORE_GOOD    = 150
SCORE_MISS    = 0

# 노트 타입
NOTE_NORMAL = 'normal'   # 해당 제스처 유지
NOTE_SHAKE  = 'shake'    # 빠르게 제스처 변화 (어떤 제스처든)
NOTE_AVOID  = 'avoid'    # 손을 치워야 함 (아무것도 인식 안 되면 OK)

# 스테이지: (노트 수, 노트 간격(초), BPM 힌트, 설명, shake비율, avoid비율)
STAGES = [
    (15, 1.2, 60,  "Intro",    0.0,  0.0),
    (20, 1.0, 70,  "Easy 1",   0.0,  0.05),
    (25, 0.9, 80,  "Easy 2",   0.1,  0.05),
    (30, 0.8, 90,  "Normal 1", 0.15, 0.1),
    (30, 0.7, 100, "Normal 2", 0.15, 0.1),
    (35, 0.6, 110, "Hard 1",   0.2,  0.1),
    (35, 0.55,115, "Hard 2",   0.2,  0.15),
    (40, 0.5, 120, "Hard 3",   0.25, 0.15),
    (40, 0.4, 130, "Expert 1", 0.25, 0.2),
    (40, 0.3, 140, "Expert 2", 0.3,  0.2),
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
    TITLE       = 'title'
    COUNTDOWN   = 'countdown'
    PLAY        = 'play'
    STAGE_CLEAR = 'stage_clear'
    GAME_OVER   = 'game_over'
    ALL_CLEAR   = 'all_clear'

    def __init__(self):
        self.reset()

    def reset(self):
        self.state        = self.TITLE
        self.stage_idx    = 0
        self.score        = 0
        self.max_combo    = 0
        self.lives        = 5
        self.combo        = 0
        self.miss_streak  = 0      # 연속 MISS 카운터
        self.results      = []     # [('perfect'/'good'/'miss', gesture)]
        self.countdown_start = 0
        self.last_detection  = None
        self.stage_score  = 0
        # 레인
        self.lane_notes   = []
        self.particles    = []
        self.note_queue   = []     # 스폰 대기 중인 노트 목록
        self.next_spawn_t = 0.0
        self.total_notes  = 0
        self.judged_count = 0
        # 쉐이크 추적
        self.prev_gesture = None
        self.shake_changes = 0
        self.shake_last_t  = 0.0

    def get_stage(self):
        return STAGES[min(self.stage_idx, len(STAGES) - 1)]

    def note_interval(self):
        return self.get_stage()[1]

    def generate_notes(self):
        """스테이지 노트 큐를 생성합니다."""
        n_notes, interval, bpm, name, shake_r, avoid_r = self.get_stage()
        notes = []
        for i in range(n_notes):
            r = random.random()
            if r < avoid_r:
                ntype = NOTE_AVOID
                gesture = random.choice(GESTURES)
            elif r < avoid_r + shake_r:
                ntype = NOTE_SHAKE
                gesture = random.choice(GESTURES)
            else:
                ntype = NOTE_NORMAL
                gesture = random.choice(GESTURES)
            notes.append((ntype, gesture))
        self.note_queue   = notes
        self.total_notes  = n_notes
        self.judged_count = 0
        self.results      = []
        self.stage_score  = 0
        self.lane_notes   = []
        self.particles    = []
        self.next_spawn_t = 0.0


# ══════════════════════════════════════════════
#  UI 헬퍼
# ══════════════════════════════════════════════
def draw_gesture_icon(img, gesture, cx, cy, size, alpha=1.0, highlight=False):
    icon = GESTURE_IMAGE.get(gesture)
    if icon is not None:
        icon_size = max(1, int(size * 2))
        icon_resized = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
        x1 = cx - icon_size // 2
        y1 = cy - icon_size // 2
        x2 = x1 + icon_size
        y2 = y1 + icon_size
        x1i, y1i = max(0, x1), max(0, y1)
        x2i, y2i = min(img.shape[1], x2), min(img.shape[0], y2)
        ix1 = x1i - x1
        iy1 = y1i - y1
        ix2 = ix1 + (x2i - x1i)
        iy2 = iy1 + (y2i - y1i)
        roi = img[y1i:y2i, x1i:x2i]
        icon_crop = icon_resized[iy1:iy2, ix1:ix2]
        if icon_crop.shape[2] == 4:
            alpha_mask = icon_crop[:, :, 3:] / 255.0
            icon_rgb = icon_crop[:, :, :3]
            roi[:] = (icon_rgb * alpha_mask + roi * (1 - alpha_mask)).astype(np.uint8)
        else:
            roi[:] = icon_crop
        return

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


def draw_hud(img, game, now):
    """상단 HUD: 배경 그라데이션 + 스코어 + 콤보 멀티플라이어 + 목숨"""
    # 반투명 HUD 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, 55), (10, 10, 25), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # 스테이지 정보
    _, _, bpm, stage_name, _, _ = game.get_stage()
    cv2.putText(img, f"Stage {game.stage_idx + 1}  {stage_name}  BPM {bpm}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 200), 1)

    # 진행률 바
    progress = game.judged_count / max(game.total_notes, 1)
    bar_x1, bar_x2 = 10, SCREEN_W - 10
    cv2.rectangle(img, (bar_x1, 28), (bar_x2, 34), (40, 40, 60), -1)
    prog_color = (0, 200, 255) if progress < 0.7 else (0, 255, 150)
    cv2.rectangle(img, (bar_x1, 28),
                  (bar_x1 + int((bar_x2 - bar_x1) * progress), 34), prog_color, -1)
    cv2.putText(img, f"{game.judged_count}/{game.total_notes}",
                (bar_x2 - 50, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 180), 1)

    # 점수 (중앙, 크게)
    score_text = f"{game.score:,}"
    tw = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
    cv2.putText(img, score_text, (SCREEN_W // 2 - tw // 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 콤보 멀티플라이어
    if game.combo >= 20:
        mult, mcol = "x3.0", (255, 100, 0)
    elif game.combo >= 10:
        mult, mcol = "x2.0", (255, 200, 0)
    elif game.combo >= 5:
        mult, mcol = "x1.5", (100, 255, 100)
    else:
        mult, mcol = None, None
    if mult:
        pulse = 0.8 + 0.2 * abs(math.sin(now * 4))
        mcol_p = tuple(int(c * pulse) for c in mcol)
        cv2.putText(img, f"{game.combo} COMBO  {mult}",
                    (SCREEN_W // 2 - 70, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mcol_p, 2)
    elif game.combo > 1:
        cv2.putText(img, f"{game.combo} COMBO",
                    (SCREEN_W // 2 - 40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 1)

    # 목숨 (하트, 우측)
    for i in range(5):
        hx = SCREEN_W - 22 * (5 - i)
        col = (80, 80, 220) if i < game.lives else (40, 40, 60)
        cv2.putText(img, "♥", (hx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # MISS 경고
    if game.miss_streak >= 2:
        alpha_w = 0.4 + 0.3 * abs(math.sin(now * 6))
        col_w = tuple(int(c * alpha_w) for c in (0, 0, 255))
        cv2.putText(img, f"!! MISS x{game.miss_streak} !!",
                    (SCREEN_W - 130, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_w, 1)


def draw_sequence_bar(img, game, now):
    pass  # 암기 단계 제거로 사용 안 함


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
    # 하위 호환용 - 실제 판정은 draw_judgment_texts 에서 처리
    pass


def draw_glow_rect(img, x1, y1, x2, y2, color, thickness=2, glow_size=4):
    """네온 글로우 사각형"""
    for g in range(glow_size, 0, -1):
        alpha = 0.15 * (glow_size - g + 1) / glow_size
        gcol = tuple(int(c * alpha) for c in color)
        cv2.rectangle(img, (x1 - g, y1 - g), (x2 + g, y2 + g), gcol, 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


# ── 레인 배경 (Fancy) ──
def draw_lane_background(img, now):
    lane_w = 64
    for gesture, lx in LANE_X_POSITIONS.items():
        color = GESTURE_COLOR[gesture]
        x1, x2 = lx - lane_w // 2, lx + lane_w // 2
        # 레인 배경 그라데이션 느낌 (어두운 컬러)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, LANE_TOP), (x2, LANE_BOTTOM + 40), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        # 레인 사이드 라인 (네온)
        side_col = tuple(int(c * 0.4) for c in color)
        cv2.line(img, (x1, LANE_TOP), (x1, LANE_BOTTOM + 40), side_col, 1)
        cv2.line(img, (x2, LANE_TOP), (x2, LANE_BOTTOM + 40), side_col, 1)
        # 레인 하단 제스처 아이콘 + 글로우 원
        pulse = 0.6 + 0.4 * abs(math.sin(now * 2 + lx))
        gcol = tuple(int(c * pulse * 0.5) for c in color)
        cv2.circle(img, (lx, LANE_BOTTOM + 22), 22, gcol, -1)
        cv2.circle(img, (lx, LANE_BOTTOM + 22), 22, color, 2)
        draw_gesture_icon(img, gesture, lx, LANE_BOTTOM + 22, 18)

    # 판정선 (글로우 효과)
    lx0 = LANE_X_POSITIONS['scissors'] - 40
    lx1 = LANE_X_POSITIONS['paper'] + 40
    for g in range(5, 0, -1):
        alpha = 0.08 * g
        gcol = tuple(int(255 * alpha) for _ in range(3))
        cv2.line(img, (lx0, JUDGE_Y - g), (lx1, JUDGE_Y - g), gcol, 1)
        cv2.line(img, (lx0, JUDGE_Y + g), (lx1, JUDGE_Y + g), gcol, 1)
    cv2.line(img, (lx0, JUDGE_Y), (lx1, JUDGE_Y), (255, 255, 255), 2)


# ── 노트 스폰 ──
def spawn_lane_note(game, ntype, gesture):
    active_time = random.uniform(NOTE_ACTIVE_TIME_MIN, NOTE_ACTIVE_TIME_MAX)
    bar_h = int(active_time * NOTE_SPEED)
    game.lane_notes.append({
        'type':          ntype,
        'gesture':       gesture,
        'y':             float(LANE_TOP),
        'bar_h':         bar_h,
        'speed':         NOTE_SPEED,
        'active_time':   active_time,
        'judged':        False,
        'result_applied': False,
        'judge_started': False,
        'judge_start_t': 0.0,
        'detect_time':   0.0,
        'avoid_ok_time': 0.0,    # AVOID 노트: 손 없는 시간 누적
        'shake_count':   0,      # SHAKE 노트: 변화 횟수
        'last_gesture':  None,
        'realtime_judge': None,  # 실시간 판정 텍스트 (진행 중)
        'note_idx':      len(game.lane_notes),
    })


# ── 노트 업데이트 ──
def update_lane_notes(game, dt, cur_g, now):
    for note in game.lane_notes:
        if note['judged']:
            note['y'] += note['speed'] * dt
            continue
        note['y'] += note['speed'] * dt

        # 판정선에 닿으면 판정 시작
        if not note['judge_started'] and note['y'] >= JUDGE_Y:
            note['judge_started'] = True
            note['judge_start_t'] = now

        if note['judge_started']:
            elapsed = now - note['judge_start_t']

            if note['type'] == NOTE_NORMAL:
                if cur_g == note['gesture']:
                    note['detect_time'] += dt
                # 실시간 판정 업데이트
                ratio = note['detect_time'] / max(note['active_time'], 0.01)
                if ratio >= JUDGE_PERFECT_RATIO:
                    note['realtime_judge'] = 'PERFECT'
                elif ratio >= JUDGE_GOOD_RATIO:
                    note['realtime_judge'] = 'GOOD'
                else:
                    note['realtime_judge'] = 'MISS'

            elif note['type'] == NOTE_SHAKE:
                # 제스처가 바뀔 때마다 카운트
                if cur_g is not None and cur_g != note['last_gesture']:
                    note['shake_count'] += 1
                    note['last_gesture'] = cur_g
                cnt = note['shake_count']
                note['realtime_judge'] = 'PERFECT' if cnt >= SHAKE_THRESHOLD + 1 \
                    else 'GOOD' if cnt >= SHAKE_THRESHOLD \
                    else f'SHAKE {cnt}/{SHAKE_THRESHOLD}'

            elif note['type'] == NOTE_AVOID:
                if cur_g is None:
                    note['avoid_ok_time'] += dt
                ratio = note['avoid_ok_time'] / max(note['active_time'], 0.01)
                note['realtime_judge'] = 'PERFECT' if ratio >= JUDGE_PERFECT_RATIO \
                    else 'GOOD' if ratio >= JUDGE_GOOD_RATIO else 'MISS'

            # 판정 구간 종료
            if elapsed >= note['active_time']:
                note['judged'] = True

    game.lane_notes = [n for n in game.lane_notes
                       if n['y'] - n['bar_h'] < SCREEN_H + 30]


# ── 판정 계산 ──
def calc_judgment(note):
    if note['type'] == NOTE_NORMAL:
        ratio = note['detect_time'] / max(note['active_time'], 0.01)
        if ratio >= JUDGE_PERFECT_RATIO:   return 'perfect'
        elif ratio >= JUDGE_GOOD_RATIO:    return 'good'
        else:                              return 'miss'
    elif note['type'] == NOTE_SHAKE:
        if note['shake_count'] >= SHAKE_THRESHOLD + 1:  return 'perfect'
        elif note['shake_count'] >= SHAKE_THRESHOLD:     return 'good'
        else:                                            return 'miss'
    elif note['type'] == NOTE_AVOID:
        ratio = note['avoid_ok_time'] / max(note['active_time'], 0.01)
        if ratio >= JUDGE_PERFECT_RATIO:   return 'perfect'
        elif ratio >= JUDGE_GOOD_RATIO:    return 'good'
        else:                              return 'miss'
    return 'miss'


# ── 노트 그리기 ──
def draw_lane_notes(img, game, now):
    lane_w = 58
    for note in game.lane_notes:
        gesture = note['gesture']
        lx = LANE_X_POSITIONS[gesture]
        base_color = GESTURE_COLOR[gesture]
        x1, x2 = lx - lane_w // 2, lx + lane_w // 2
        y_top = int(note['y'] - note['bar_h'])
        y_bot = int(note['y'])
        yt = max(0, y_top)
        yb = min(SCREEN_H, y_bot)
        if yt >= yb:
            continue

        # 노트 타입별 색상
        if note['type'] == NOTE_AVOID:
            base_color = (60, 60, 220)   # 빨간 계열 (피해야 함)
            label = 'AVOID'
        elif note['type'] == NOTE_SHAKE:
            base_color = (200, 180, 0)   # 노란 계열
            label = 'SHAKE'
        else:
            label = GESTURE_EMOJI.get(gesture, '?')

        if note['judged']:
            ov = img.copy()
            cv2.rectangle(ov, (x1, yt), (x2, yb),
                          tuple(int(c * 0.12) for c in base_color), -1)
            cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
            continue

        if note['judge_started']:
            # 실시간 판정 색상
            rj = note.get('realtime_judge', '')
            if 'PERFECT' in str(rj):
                draw_color = (0, 255, 220)
            elif 'GOOD' in str(rj):
                draw_color = (0, 220, 80)
            else:
                draw_color = (80, 60, 200)
            pulse = 0.75 + 0.25 * abs(math.sin(now * 10))
            dc = tuple(int(c * pulse) for c in draw_color)
            # 바 채우기
            cv2.rectangle(img, (x1, yt), (x2, yb), dc, -1)
            # 인식 진행률 흰색 채움
            if note['type'] == NOTE_NORMAL:
                ratio = note['detect_time'] / max(note['active_time'], 0.01)
            elif note['type'] == NOTE_AVOID:
                ratio = note['avoid_ok_time'] / max(note['active_time'], 0.01)
            else:
                ratio = min(note['shake_count'] / max(SHAKE_THRESHOLD, 1), 1.0)
            fill_h = int((yb - yt) * min(ratio, 1.0))
            if fill_h > 2:
                cv2.rectangle(img, (x1 + 4, yb - fill_h), (x2 - 4, yb - 2),
                              (255, 255, 255), -1)
            draw_glow_rect(img, x1, yt, x2, yb, draw_color, thickness=2, glow_size=5)
            # 실시간 판정 텍스트 (바 위에)
            if rj:
                rj_str = str(rj)
                rj_col = (0, 255, 255) if 'PERFECT' in rj_str \
                    else (0, 255, 100) if 'GOOD' in rj_str else (180, 100, 255)
                fs = 0.38
                tw = cv2.getTextSize(rj_str, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
                cv2.putText(img, rj_str, (lx - tw // 2, max(LANE_TOP + 12, yt - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, rj_col, 1)
        else:
            # 접근 중인 노트: 일반 색
            cv2.rectangle(img, (x1, yt), (x2, yb), base_color, -1)
            draw_glow_rect(img, x1, yt, x2, yb, base_color, thickness=1, glow_size=3)

        # 노트 중앙 라벨
        cy = max(LANE_TOP + 12, min(SCREEN_H - 12, (yt + yb) // 2))
        if note['type'] in (NOTE_SHAKE, NOTE_AVOID):
            fs = 0.38
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            cv2.putText(img, label, (lx - tw // 2, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
        else:
            draw_gesture_icon(img, gesture, lx, cy, 15)


# ── 파티클 생성 ──
def spawn_particles(game, x, y, color, count=20):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(60, 220)
        game.particles.append({
            'x': float(x), 'y': float(y),
            'vx': math.cos(angle) * speed,
            'vy': math.sin(angle) * speed,
            'color': color,
            'life': 1.0,
            'max_life': random.uniform(0.4, 0.9),
            'size': random.randint(3, 8),
        })


# ── 파티클 업데이트 & 그리기 ──
def update_draw_particles(img, game, dt):
    alive = []
    for p in game.particles:
        p['life'] -= dt / p['max_life']
        if p['life'] <= 0:
            continue
        p['x'] += p['vx'] * dt
        p['y'] += p['vy'] * dt
        p['vy'] += 300 * dt  # 중력
        alpha = max(0.0, p['life'])
        color = tuple(int(c * alpha) for c in p['color'])
        cx, cy = int(p['x']), int(p['y'])
        if 0 <= cx < SCREEN_W and 0 <= cy < SCREEN_H:
            cv2.circle(img, (cx, cy), p['size'], color, -1)
        alive.append(p)
    game.particles = alive


# ── 판정 텍스트 이펙트 (PERFECT/GOOD/MISS) ──
JUDGMENT_DISPLAY = {}   # gesture → {'text', 'color', 'time', 'scale_anim'}

def show_judgment_text(gesture, judgment):
    config = {
        'perfect': ("PERFECT!", (0, 255, 255)),
        'good':    ("GOOD!",    (0, 255, 80)),
        'miss':    ("MISS",     (80, 80, 255)),
    }
    text, color = config.get(judgment, ("?", (255, 255, 255)))
    JUDGMENT_DISPLAY[gesture] = {
        'text': text, 'color': color,
        'time': time.time(),
    }


def draw_judgment_texts(img, now):
    for gesture, info in list(JUDGMENT_DISPLAY.items()):
        elapsed = now - info['time']
        if elapsed > 1.0:
            del JUDGMENT_DISPLAY[gesture]
            continue
        alpha = max(0.0, 1.0 - elapsed)
        scale = 0.9 + elapsed * 0.5
        lx = LANE_X_POSITIONS[gesture]
        y = JUDGE_Y - 30 - int(elapsed * 40)
        color = tuple(int(c * alpha) for c in info['color'])
        text = info['text']
        fs = scale * 0.75
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        cv2.putText(img, text, (lx - tw // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)


def draw_title_screen(img, now):
    # 배경 그라데이션
    for y in range(SCREEN_H):
        ratio = y / SCREEN_H
        r = int(5 + 15 * ratio)
        g = int(5 + 10 * ratio)
        b = int(20 + 40 * ratio)
        cv2.line(img, (0, y), (SCREEN_W, y), (b, g, r), 1)

    # 타이틀
    title = "RPS RHYTHM"
    pulse = 0.85 + 0.15 * abs(math.sin(now * 1.5))
    col = tuple(int(c * pulse) for c in (0, 255, 255))
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0]
    # 글로우
    for g in range(8, 0, -2):
        gcol = tuple(int(c * 0.05 * g) for c in (0, 255, 255))
        cv2.putText(img, title, (SCREEN_W // 2 - tw // 2, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, gcol, 4 + g)
    cv2.putText(img, title, (SCREEN_W // 2 - tw // 2, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 4)

    sub = "Rock  Paper  Scissors  Rhythm Game"
    tw2 = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(img, sub, (SCREEN_W // 2 - tw2 // 2, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 200), 1)

    # 아이콘
    for i, g in enumerate(GESTURES):
        cx = SCREEN_W // 2 + (i - 1) * 110
        cy = 290
        pulse2 = 0.7 + 0.3 * abs(math.sin(now * 2 + i))
        col2 = tuple(int(c * pulse2) for c in GESTURE_COLOR[g])
        cv2.circle(img, (cx, cy), 32, col2, -1)
        cv2.circle(img, (cx, cy), 32, (255, 255, 255), 2)
        draw_gesture_icon(img, g, cx, cy, 25)
        cv2.putText(img, GESTURE_KR[g], (cx - 35, cy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1)

    # 조작 안내
    lines = [
        "Normal : Hold gesture while bar passes judge line",
        "SHAKE  : Change gesture rapidly",
        "AVOID  : Hide your hand (no detection)",
    ]
    for i, line in enumerate(lines):
        tw3 = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(img, line, (SCREEN_W // 2 - tw3 // 2, 365 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 160), 1)

    blink = int(now * 2) % 2
    if blink:
        start_text = "Press SPACE to Start"
        tw4 = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        cv2.putText(img, start_text, (SCREEN_W // 2 - tw4 // 2, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_memorize_screen(img, game, now):
    pass  # 암기 단계 제거


def draw_countdown(img, game, now):
    elapsed = now - game.countdown_start
    count = max(1, 3 - int(elapsed))
    text = str(count)
    scale = 3.5 - (elapsed % 1.0) * 1.5
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 5)[0]
    tx = SCREEN_W // 2 - tw[0] // 2
    ty = SCREEN_H // 2 + tw[1] // 2
    alpha = max(0.2, 1.0 - (elapsed % 1.0) * 0.8)
    col = tuple(int(c * alpha) for c in (0, 255, 255))
    for g in range(6, 0, -2):
        gcol = tuple(int(c * 0.06 * g) for c in (0, 255, 255))
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, gcol, 5 + g)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 5)
    _, _, bpm, stage_name, _, _ = game.get_stage()
    info = f"Stage {game.stage_idx + 1}: {stage_name}  |  BPM {bpm}"
    tw2 = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    cv2.putText(img, info, (SCREEN_W // 2 - tw2 // 2, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)


def draw_stage_clear(img, game, now):
    # 배경 어둡게
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, SCREEN_H), (5, 15, 5), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    pulse = 0.8 + 0.2 * abs(math.sin(now * 2))
    col = tuple(int(c * pulse) for c in (0, 255, 200))
    tw = cv2.getTextSize("STAGE CLEAR!", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0][0]
    cv2.putText(img, "STAGE CLEAR!", (SCREEN_W // 2 - tw // 2, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)

    perfect = sum(1 for r, _ in game.results if r == 'perfect')
    good    = sum(1 for r, _ in game.results if r == 'good')
    miss    = sum(1 for r, _ in game.results if r == 'miss')
    y = 220
    stats = [
        (f"PERFECT : {perfect}", (0, 255, 220)),
        (f"GOOD    : {good}",    (0, 220, 100)),
        (f"MISS    : {miss}",    (100, 80, 220)),
        (f"MAX COMBO : {game.max_combo}", (255, 200, 0)),
        (f"STAGE SCORE : {game.stage_score:,}", (255, 255, 255)),
    ]
    for i, (text, col2) in enumerate(stats):
        tw2 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        cv2.putText(img, text, (SCREEN_W // 2 - tw2 // 2, y + i * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col2, 2)

    blink = int(now * 2) % 2
    if blink:
        msg = "Press SPACE for Next Stage"
        tw3 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        cv2.putText(img, msg, (SCREEN_W // 2 - tw3 // 2, y + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1)


def draw_game_over(img, game, now):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, SCREEN_H), (20, 5, 5), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    pulse = 0.7 + 0.3 * abs(math.sin(now * 3))
    col = tuple(int(c * pulse) for c in (80, 80, 255))
    tw = cv2.getTextSize("GAME OVER", cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0][0]
    cv2.putText(img, "GAME OVER", (SCREEN_W // 2 - tw // 2, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)

    tw2 = cv2.getTextSize(f"{game.score:,}", cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0][0]
    cv2.putText(img, f"{game.score:,}", (SCREEN_W // 2 - tw2 // 2, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    info = f"Stage {game.stage_idx + 1}  |  Max Combo: {game.max_combo}"
    tw3 = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(img, info, (SCREEN_W // 2 - tw3 // 2, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 180), 1)
    blink = int(now * 2) % 2
    if blink:
        msg = "Press SPACE to Retry"
        tw4 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        cv2.putText(img, msg, (SCREEN_W // 2 - tw4 // 2, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def draw_all_clear(img, game, now):
    for y in range(SCREEN_H):
        ratio = y / SCREEN_H
        cv2.line(img, (0, y), (SCREEN_W, y),
                 (int(5 + 20 * ratio), int(20 + 30 * ratio), int(30 + 20 * ratio)), 1)

    pulse = 0.8 + 0.2 * abs(math.sin(now * 2))
    col = tuple(int(c * pulse) for c in (0, 255, 200))
    tw = cv2.getTextSize("ALL CLEAR!!", cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0][0]
    for g in range(8, 0, -2):
        gcol = tuple(int(c * 0.05 * g) for c in (0, 255, 200))
        cv2.putText(img, "ALL CLEAR!!", (SCREEN_W // 2 - tw // 2, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, gcol, 4 + g)
    cv2.putText(img, "ALL CLEAR!!", (SCREEN_W // 2 - tw // 2, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)

    tw2 = cv2.getTextSize(f"{game.score:,}", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
    cv2.putText(img, f"TOTAL  {game.score:,}", (SCREEN_W // 2 - tw2 // 2 - 40, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    tw3 = cv2.getTextSize(f"Max Combo: {game.max_combo}", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
    cv2.putText(img, f"Max Combo: {game.max_combo}", (SCREEN_W // 2 - tw3 // 2, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    blink = int(now * 2) % 2
    if blink:
        msg = "Press SPACE to Restart"
        tw4 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        cv2.putText(img, msg, (SCREEN_W // 2 - tw4 // 2, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


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
        os.path.join(base, 'models', 'best.onnx'),
        os.path.join(base, 'models', 'rps_mobilenetv2.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_qat.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_ptq_int8.tflite'),
        os.path.join(base, 'models', 'RPS_MobileNetV2_Augmentation.tflite'),
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
    parser = argparse.ArgumentParser(description='RPS Rhythm Game v2')
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
          f"cooldown={COOLDOWN_TIME}s, "
          f"note_active={NOTE_ACTIVE_TIME_MIN}~{NOTE_ACTIVE_TIME_MAX}s")

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

            # ── 상태 머신 ──
            if game.state == game.TITLE:
                draw_title_screen(screen, now)

            elif game.state == game.COUNTDOWN:
                # 카메라 피드 (화면 중앙)
                cam_display = cv2.resize(frame, (CAM_W, CAM_H))
                cx0 = SCREEN_W // 2 - CAM_W // 2
                cy0 = 150
                screen[cy0:cy0 + CAM_H, cx0:cx0 + CAM_W] = cam_display
                draw_hud(screen, game, now)
                draw_countdown(screen, game, now)
                if now - game.countdown_start >= 3.0:
                    game.state = game.PLAY
                    game.next_spawn_t = now
                    game._prev_t = now
                    smoother.reset()
                    stable_g, stable_c = None, 0.0

            elif game.state == game.PLAY:
                # ── dt 계산 ──
                if not hasattr(game, '_prev_t'):
                    game._prev_t = now
                dt = min(now - game._prev_t, 0.05)
                game._prev_t = now

                cur_g = game.last_detection

                # ── 쉐이크 추적 (게임 레벨) ──
                if cur_g is not None and cur_g != game.prev_gesture:
                    game.shake_changes += 1
                    game.shake_last_t = now
                game.prev_gesture = cur_g
                # 1초 이상 지나면 쉐이크 카운터 리셋
                if now - game.shake_last_t > 1.0:
                    game.shake_changes = 0

                # ── 노트 스폰 (노트 큐에서 꺼냄) ──
                if game.note_queue and now >= game.next_spawn_t:
                    ntype, gesture = game.note_queue.pop(0)
                    spawn_lane_note(game, ntype, gesture)
                    game.next_spawn_t = now + game.note_interval()

                # ── 노트 업데이트 ──
                update_lane_notes(game, dt, cur_g, now)

                # ── 판정 완료 노트 처리 ──
                for note in game.lane_notes:
                    if note['judged'] and not note['result_applied']:
                        note['result_applied'] = True
                        judgment = calc_judgment(note)
                        lx = LANE_X_POSITIONS[note['gesture']]
                        game.judged_count += 1

                        if judgment == 'perfect':
                            earned = SCORE_PERFECT
                            pcol, pcnt = (0, 255, 255), 35
                        elif judgment == 'good':
                            earned = SCORE_GOOD
                            pcol, pcnt = (0, 220, 100), 20
                        else:
                            earned = SCORE_MISS
                            pcol, pcnt = (80, 60, 200), 10

                        if judgment != 'miss':
                            game.combo += 1
                            game.max_combo = max(game.max_combo, game.combo)
                            # 콤보 멀티플라이어
                            if game.combo >= 20:   mult = 3.0
                            elif game.combo >= 10: mult = 2.0
                            elif game.combo >= 5:  mult = 1.5
                            else:                  mult = 1.0
                            total_earned = int(earned * mult) + game.combo * SCORE_BONUS_PER_COMBO
                            game.score += total_earned
                            game.stage_score += total_earned
                            game.results.append((judgment, note['gesture']))
                            game.miss_streak = 0
                        else:
                            game.combo = 0
                            game.miss_streak += 1
                            game.results.append(('miss', note['gesture']))
                            # 연속 MISS 3번마다 목숨 감소
                            if game.miss_streak % 3 == 0:
                                game.lives -= 1

                        spawn_particles(game, lx, JUDGE_Y, pcol, pcnt)
                        show_judgment_text(note['gesture'], judgment)

                        if game.lives <= 0:
                            game.state = game.GAME_OVER

                # ── 레인 그리기 ──
                draw_lane_background(screen, now)
                draw_lane_notes(screen, game, now)
                draw_judgment_texts(screen, now)
                update_draw_particles(screen, game, dt)
                draw_hud(screen, game, now)

                # ── 카메라 소형 (우하단) ──
                cam_h_s = 120
                cam_w_s = int(cam_h_s * CAM_W / CAM_H)
                cam_xs = SCREEN_W - cam_w_s - 5
                cam_ys = SCREEN_H - cam_h_s - 5
                cam_small = cv2.resize(frame, (cam_w_s, cam_h_s))
                # bbox 그리기
                if bbox:
                    bx1, by1, bx2, by2 = bbox
                    sw = cam_w_s / CAM_W; sh = cam_h_s / CAM_H
                    cv2.rectangle(cam_small,
                                  (int(bx1*sw), int(by1*sh)),
                                  (int(bx2*sw), int(by2*sh)),
                                  GESTURE_COLOR.get(stable_g, (255,255,255)), 1)
                screen[cam_ys:cam_ys + cam_h_s, cam_xs:cam_xs + cam_w_s] = cam_small
                cv2.rectangle(screen, (cam_xs - 1, cam_ys - 1),
                              (cam_xs + cam_w_s + 1, cam_ys + cam_h_s + 1),
                              (80, 80, 100), 1)
                if detection_text:
                    cv2.putText(screen, detection_text, (cam_xs, cam_ys - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 100), 1)

                # ── 다음 노트 예고 (레인 상단) ──
                if game.note_queue:
                    nxt_type, nxt_g = game.note_queue[0]
                    nxt_lx = LANE_X_POSITIONS[nxt_g]
                    nxt_label = 'SHAKE' if nxt_type == NOTE_SHAKE \
                        else 'AVOID' if nxt_type == NOTE_AVOID \
                        else GESTURE_EMOJI.get(nxt_g, '?')
                    cv2.putText(screen, f"▼{nxt_label}",
                                (nxt_lx - 15, LANE_TOP - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)

                # ── 스테이지 종료 판정 ──
                if not game.note_queue and \
                        all(n['result_applied'] for n in game.lane_notes):
                    game.state = game.STAGE_CLEAR

            elif game.state == game.STAGE_CLEAR:
                draw_stage_clear(screen, game, now)

            elif game.state == game.GAME_OVER:
                draw_game_over(screen, game, now)

            elif game.state == game.ALL_CLEAR:
                draw_all_clear(screen, game, now)

            # ── FPS 표시 ──
            cur_time = time.time()
            fps = 1.0 / max(cur_time - fps_time, 0.001)
            fps_time = cur_time
            cv2.putText(screen, f"FPS:{fps:.0f} INF:{inf_latency_ms:.0f}ms",
                        (SCREEN_W - 170, SCREEN_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 90), 1)

            cv2.imshow('RPS Rhythm', screen)
            cv2.waitKey(1)

            # ── 키 입력 ──
            key_ch = kb.get_key()
            if key_ch == 'q':
                break
            elif key_ch == ' ':
                if game.state == game.TITLE:
                    game.reset()
                    game.state = game.COUNTDOWN
                    game.generate_notes()
                    game.countdown_start = time.time()
                    print(f"[Stage {game.stage_idx + 1}] 노트 수: {game.total_notes}")
                elif game.state == game.STAGE_CLEAR:
                    game.stage_idx += 1
                    if game.stage_idx >= len(STAGES):
                        game.state = game.ALL_CLEAR
                    else:
                        game.state = game.COUNTDOWN
                        game.generate_notes()
                        game.countdown_start = time.time()
                        print(f"[Stage {game.stage_idx + 1}] 노트 수: {game.total_notes}")
                elif game.state in (game.GAME_OVER, game.ALL_CLEAR):
                    game.reset()
                    game.state = game.COUNTDOWN
                    game.generate_notes()
                    game.countdown_start = time.time()
                    print(f"[Stage 1] 노트 수: {game.total_notes}")
    finally:
        kb.stop()
        worker.stop()
        release()
        cv2.destroyAllWindows()
        print(f"\n최종 점수: {game.score:,} | 최대 콤보: {game.max_combo}")


if __name__ == '__main__':
    main()
