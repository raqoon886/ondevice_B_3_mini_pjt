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
  B/ESC → 선택 화면으로 돌아가기
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
    import tkinter as tk
except ImportError:
    tk = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ══════════════════════════════════════════════
#  상수 & 설정
# ══════════════════════════════════════════════
SCREEN_W = 820
SCREEN_H = 620
CAM_W, CAM_H = 440, 330
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


def load_image_with_transparency(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.dstack([image, alpha])
    if image.shape[2] == 4 and image[:, :, 3].min() == 255:
        rgb = image[:, :, :3].astype(np.int16)
        gray_mask = (np.abs(rgb[:, :, 0] - rgb[:, :, 1]) < 20) & \
                    (np.abs(rgb[:, :, 0] - rgb[:, :, 2]) < 20)
        bright_mask = (rgb[:, :, 0] > 140) & (rgb[:, :, 1] > 140) & (rgb[:, :, 2] > 140)
        image[gray_mask & bright_mask, 3] = 0
    return image

GESTURE_IMAGE = {}
for gesture_name, filename in {
        'scissors': 'scissors.png',
        'rock':     'rock.png',
        'paper':    'paper.png',
    }.items():
    image_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(image_path):
        image = load_image_with_transparency(image_path)
        if image is not None:
            GESTURE_IMAGE[gesture_name] = image

HEART_IMAGE = None
HEART_CACHE = {}  # pre-resized heart icons: {size: {'active': (rgb, alpha), 'inactive': (rgb, alpha)}}
heart_path = os.path.join(DATA_DIR, 'heart.png')
if os.path.exists(heart_path):
    heart_img = load_image_with_transparency(heart_path)
    if heart_img is not None:
        HEART_IMAGE = heart_img

def _cache_heart(size):
    if size in HEART_CACHE or HEART_IMAGE is None:
        return
    resized = cv2.resize(HEART_IMAGE, (size, size), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, :3]
    a = resized[:, :, 3:] / 255.0 if resized.shape[2] == 4 else np.ones_like(rgb[:, :, :1])
    dim_rgb = (rgb * 0.35).astype(np.uint8)
    dim_a = a * 0.35
    HEART_CACHE[size] = {
        'active':   (rgb, a),
        'inactive': (dim_rgb, dim_a),
    }

_cache_heart(24)

SCORE_CORRECT = 200
SCORE_BONUS_PER_COMBO = 30
HOLD_TIME = 1.0           # 같은 제스처를 N초 이상 유지해야 확정 (시간 기반)
COOLDOWN_TIME = 0.8     # 확정 후 다음 입력 받기까지 대기 (같은 제스처 연속 입력 가능)
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

# ── 게임 모드 ──
WINS_AGAINST = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
BEATEN_BY    = {v: k for k, v in WINS_AGAINST.items()}
GAME_MODES   = ['Tutorial', 'Same', 'Win', 'Lose', 'Rhythm']
MODE_DESC    = [
    'Tutorial',
    'Match gesture',
    'Win to each gesture',
    'Lose to each gesture',
    'Rhythm lane game',
]

# ── 리듬 모드 전용 상수 ──
LANE_W      = 100  # 레인 폭
LANE_GAP    = 20   # 레인 간격
_LANE_TOTAL = LANE_W * 3 + LANE_GAP * 2
_LANE_START = SCREEN_W // 2 - _LANE_TOTAL // 2 + LANE_W // 2
LANE_X_POSITIONS = {
    'scissors': _LANE_START,
    'rock':     _LANE_START + LANE_W + LANE_GAP,
    'paper':    _LANE_START + (LANE_W + LANE_GAP) * 2,
}
LANE_TOP    = 60
LANE_BOTTOM = 510
JUDGE_Y     = LANE_BOTTOM
NOTE_SPEED  = 180

NOTE_ACTIVE_TIME_MIN = 0.4
NOTE_ACTIVE_TIME_MAX = 1.5
SHAKE_THRESHOLD = 3

JUDGE_PERFECT_RATIO = 0.80
JUDGE_GOOD_RATIO    = 0.45
SCORE_PERFECT = 300
SCORE_GOOD    = 150

NOTE_NORMAL = 'normal'
NOTE_SHAKE  = 'shake'
NOTE_AVOID  = 'avoid'

# ── 3가지 난이도 스테이지 ──
# 각 난이도는 여러 서브 스테이지로 구성
# (노트수, 간격(초), BPM, 설명, shake비율, avoid비율)
RHYTHM_DIFFICULTY = [
    {   # 0: Easy
        'name': 'EASY',
        'color': (100, 255, 100),
        'desc': 'Slow tempo, NORMAL notes only',
        'icon': 'star1',
        'stages': [
            (12, 1.4, 55,  "Warm Up",   0.0,  0.0),
            (18, 1.2, 65,  "Easy Flow", 0.0,  0.0),
            (22, 1.0, 75,  "Cruise",    0.05, 0.0),
        ],
    },
    {   # 1: Normal
        'name': 'NORMAL',
        'color': (0, 200, 255),
        'desc': 'Mid tempo with SHAKE notes',
        'icon': 'star2',
        'stages': [
            (25, 0.9, 85,  "Steady",     0.1,  0.0),
            (30, 0.8, 95,  "Pick It Up", 0.15, 0.05),
            (30, 0.7, 105, "Momentum",   0.15, 0.1),
            (35, 0.65,110, "Push",       0.2,  0.1),
        ],
    },
    {   # 2: Hard
        'name': 'HARD',
        'color': (80, 80, 255),
        'desc': 'Fast tempo, SHAKE + AVOID mixed',
        'icon': 'star3',
        'stages': [
            (35, 0.55, 120, "Overdrive",  0.2,  0.15),
            (40, 0.45, 130, "Frenzy",     0.25, 0.15),
            (40, 0.35, 140, "Inferno",    0.3,  0.2),
        ],
    },
]

# 하위호환: get_rhythm_stage()에서 사용
STAGES_RHYTHM = RHYTHM_DIFFICULTY[0]['stages']  # 기본값, 런타임에 교체됨


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
        if self._enabled == on:
            return
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
    STAGE_SELECT = 'stage_select'
    MEMORIZE = 'memorize'
    COUNTDOWN = 'countdown'
    PLAY = 'play'
    JUDGE = 'judge'
    STAGE_CLEAR = 'stage_clear'
    GAME_OVER = 'game_over'
    ALL_CLEAR = 'all_clear'

    def __init__(self):
        self.selected_mode = 1  # 타이틀에서 선택한 모드 (persistent)
        self.mode = 1           # 현재 플레이 중인 모드 (reset해도 유지)
        self.rhythm_difficulty = 0  # 리듬 모드 난이도 (0=Easy, 1=Normal, 2=Hard)
        self.selected_difficulty = 0  # 스테이지 선택 화면 커서
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
        # ── 리듬 모드 전용 ──
        self.lane_notes   = []
        self.particles    = []
        self.note_queue   = []
        self.next_spawn_t = 0.0
        self.total_notes  = 0
        self.judged_count = 0
        self.miss_streak  = 0
        self.prev_gesture = None
        self.shake_changes = 0
        self.shake_last_t  = 0.0
        self._prev_t = 0.0

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

    def get_rhythm_stage(self):
        stages = RHYTHM_DIFFICULTY[self.rhythm_difficulty]['stages']
        return stages[min(self.stage_idx, len(stages) - 1)]

    def rhythm_stages_count(self):
        return len(RHYTHM_DIFFICULTY[self.rhythm_difficulty]['stages'])

    def generate_notes(self):
        n_notes, interval, bpm, name, shake_r, avoid_r = self.get_rhythm_stage()
        notes = []
        for _ in range(n_notes):
            r = random.random()
            if r < avoid_r:
                ntype = NOTE_AVOID
            elif r < avoid_r + shake_r:
                ntype = NOTE_SHAKE
            else:
                ntype = NOTE_NORMAL
            notes.append((ntype, random.choice(GESTURES)))
        self.note_queue   = notes
        self.total_notes  = n_notes
        self.judged_count = 0
        self.results      = []
        self.stage_score  = 0
        self.lane_notes   = []
        self.particles    = []
        self.next_spawn_t = 0.0
        self.miss_streak  = 0


# ══════════════════════════════════════════════
#  게임 모드 헬퍼
# ══════════════════════════════════════════════
def get_required_gesture(mode, target):
    """명시된 모드에 따라 플레이어가 내야 하는 제스첸 반환"""
    if mode == 2:
        return BEATEN_BY[target]      # 이기기: target을 이기는 제스첸
    if mode == 3:
        return WINS_AGAINST[target]   # 지기: target이 이기는 제스첸 (플레이어가 짐)
    return target                     # Tutorial / Same: 똑같이


# ══════════════════════════════════════════════
#  리듬 모드 헬퍼
# ══════════════════════════════════════════════
JUDGMENT_COLORS = {
    'perfect': (255, 255, 0),
    'good':    (0, 255, 100),
    'miss':    (0, 0, 255),
}
NOTE_TYPE_COLORS = {
    NOTE_NORMAL: (0, 220, 255),
    NOTE_SHAKE:  (255, 100, 255),
    NOTE_AVOID:  (80, 80, 255),
}
NOTE_TYPE_LABELS = {
    NOTE_NORMAL: '',
    NOTE_SHAKE:  'SHAKE',
    NOTE_AVOID:  'AVOID',
}


def spawn_lane_note(game, ntype, gesture):
    active_t = random.uniform(NOTE_ACTIVE_TIME_MIN, NOTE_ACTIVE_TIME_MAX)
    note = {
        'gesture': gesture, 'type': ntype,
        'y': float(LANE_TOP), 'speed': NOTE_SPEED,
        'state': 'falling', 'judgment': None,
        'detect_time': 0.0, 'max_active': active_t,
        'active_start': 0.0, 'judge_time': 0.0,
    }
    game.lane_notes.append(note)


def calc_judgment(note):
    if note['max_active'] <= 0:
        return 'miss'
    ratio = note['detect_time'] / note['max_active']
    if ratio >= JUDGE_PERFECT_RATIO:
        return 'perfect'
    if ratio >= JUDGE_GOOD_RATIO:
        return 'good'
    return 'miss'


def _apply_judgment(game, note, now):
    note['judge_time'] = now
    j = note['judgment']
    if j == 'perfect':
        earned = SCORE_PERFECT
        game.combo += 1
    elif j == 'good':
        earned = SCORE_GOOD
        game.combo += 1
    else:
        earned = 0
        game.combo = 0
        game.miss_streak += 1
        game.lives -= 1
    if game.combo >= 30:
        earned = int(earned * 3.0)
    elif game.combo >= 15:
        earned = int(earned * 2.0)
    elif game.combo >= 5:
        earned = int(earned * 1.5)
    game.max_combo = max(game.max_combo, game.combo)
    game.score += earned
    game.stage_score += earned
    game.results.append((j, note['gesture']))
    game.judged_count += 1
    cx = LANE_X_POSITIONS[note['gesture']]
    color = JUDGMENT_COLORS.get(j, (200, 200, 200))
    spawn_particles(game, cx, JUDGE_Y, color, 8 if j == 'perfect' else 4)


def update_lane_notes(game, dt, cur_gesture, now):
    to_remove = []
    for note in game.lane_notes:
        if note['state'] == 'falling':
            note['y'] += note['speed'] * dt
            if note['y'] >= JUDGE_Y:
                note['y'] = JUDGE_Y
                note['state'] = 'active'
                note['active_start'] = now
                if note['type'] == NOTE_SHAKE:
                    game.shake_changes = 0
        elif note['state'] == 'active':
            elapsed = now - note['active_start']
            if note['type'] == NOTE_NORMAL:
                if cur_gesture == note['gesture']:
                    note['detect_time'] += dt
                if elapsed >= note['max_active']:
                    note['state'] = 'judged'
                    note['judgment'] = calc_judgment(note)
                    _apply_judgment(game, note, now)
            elif note['type'] == NOTE_SHAKE:
                if (cur_gesture and game.prev_gesture is not None
                        and cur_gesture != game.prev_gesture):
                    game.shake_changes += 1
                if elapsed >= note['max_active']:
                    note['state'] = 'judged'
                    if game.shake_changes >= SHAKE_THRESHOLD:
                        note['judgment'] = 'perfect'
                    elif game.shake_changes >= max(1, SHAKE_THRESHOLD // 2):
                        note['judgment'] = 'good'
                    else:
                        note['judgment'] = 'miss'
                    _apply_judgment(game, note, now)
            elif note['type'] == NOTE_AVOID:
                if cur_gesture == note['gesture']:
                    note['state'] = 'judged'
                    note['judgment'] = 'miss'
                    _apply_judgment(game, note, now)
                elif elapsed >= note['max_active']:
                    note['state'] = 'judged'
                    note['judgment'] = 'perfect'
                    _apply_judgment(game, note, now)
        elif note['state'] == 'judged':
            if now - note['judge_time'] > 0.8:
                to_remove.append(note)
    for note in to_remove:
        game.lane_notes.remove(note)
    game.prev_gesture = cur_gesture


def spawn_particles(game, x, y, color, count=8):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(60, 180)
        game.particles.append({
            'x': float(x), 'y': float(y),
            'vx': math.cos(angle) * speed,
            'vy': math.sin(angle) * speed - 50,
            'life': 0.6, 'color': color,
            'size': random.randint(2, 5),
        })


def update_draw_particles(img, game, dt):
    alive = []
    for p in game.particles:
        p['life'] -= dt
        if p['life'] <= 0:
            continue
        p['x'] += p['vx'] * dt
        p['y'] += p['vy'] * dt
        p['vy'] += 300 * dt
        alpha = max(0, p['life'] / 0.6)
        color = tuple(int(c * alpha) for c in p['color'])
        cv2.circle(img, (int(p['x']), int(p['y'])), p['size'], color, -1)
        alive.append(p)
    game.particles = alive


def draw_glow_rect(img, x1, y1, x2, y2, color, thickness=2):
    for i in range(4, 0, -1):
        alpha = 0.12 * i
        glow = tuple(int(c * alpha) for c in color)
        cv2.rectangle(img, (x1 - i * 2, y1 - i), (x2 + i * 2, y2 + i), glow, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_lane_gradient(img, lx1, lx2, color):
    """Draw a subtle vertical gradient inside a lane."""
    h = LANE_BOTTOM - LANE_TOP
    for row in range(h):
        y = LANE_TOP + row
        t = row / h
        r = int(15 + color[0] * 0.04 * t)
        g = int(15 + color[1] * 0.04 * t)
        b = int(15 + color[2] * 0.04 * t)
        cv2.line(img, (lx1 + 1, y), (lx2 - 1, y), (b, g, r), 1)


def draw_lane_background(img, now):
    lw = LANE_W
    for gesture, cx in LANE_X_POSITIONS.items():
        lx1, lx2 = cx - lw // 2, cx + lw // 2
        color = GESTURE_COLOR[gesture]
        # lane fill with gradient
        _draw_lane_gradient(img, lx1, lx2, color)
        # lane border
        cv2.rectangle(img, (lx1, LANE_TOP), (lx2, LANE_BOTTOM), color, 1)
        # divider lines every 60px
        for dy in range(LANE_TOP + 60, LANE_BOTTOM, 60):
            cv2.line(img, (lx1 + 4, dy), (lx2 - 4, dy),
                     tuple(int(c * 0.15) for c in color), 1)
    # judge zone: bright horizontal bar spanning all lanes
    all_lx1 = min(LANE_X_POSITIONS.values()) - lw // 2
    all_lx2 = max(LANE_X_POSITIONS.values()) + lw // 2
    cv2.rectangle(img, (all_lx1, JUDGE_Y - 4), (all_lx2, JUDGE_Y + 4),
                  (60, 60, 60), -1)
    for gesture, cx in LANE_X_POSITIONS.items():
        color = GESTURE_COLOR[gesture]
        lx1, lx2 = cx - lw // 2, cx + lw // 2
        draw_glow_rect(img, lx1, JUDGE_Y - 4, lx2, JUDGE_Y + 4, color)
    # gesture icons below lanes
    for gesture, cx in LANE_X_POSITIONS.items():
        draw_gesture_icon(img, gesture, cx, LANE_BOTTOM + 40, 22)
        label = GESTURE_KR[gesture]
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(img, label, (cx - ts[0] // 2, LANE_BOTTOM + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GESTURE_COLOR[gesture], 1)


def draw_lane_notes(img, game, now):
    lw = LANE_W
    bar_h = 36
    corner_r = 8
    for note in game.lane_notes:
        cx = LANE_X_POSITIONS[note['gesture']]
        x1, x2 = cx - lw // 2 + 6, cx + lw // 2 - 6
        nc = NOTE_TYPE_COLORS[note['type']]
        if note['state'] == 'falling':
            ny = int(note['y'])
            y1 = ny - bar_h // 2
            y2 = ny + bar_h // 2
            # rounded filled rectangle
            cv2.rectangle(img, (x1 + corner_r, y1), (x2 - corner_r, y2), nc, -1)
            cv2.rectangle(img, (x1, y1 + corner_r), (x2, y2 - corner_r), nc, -1)
            cv2.circle(img, (x1 + corner_r, y1 + corner_r), corner_r, nc, -1)
            cv2.circle(img, (x2 - corner_r, y1 + corner_r), corner_r, nc, -1)
            cv2.circle(img, (x1 + corner_r, y2 - corner_r), corner_r, nc, -1)
            cv2.circle(img, (x2 - corner_r, y2 - corner_r), corner_r, nc, -1)
            # icon inside note
            draw_gesture_icon(img, note['gesture'], cx, ny, 12)
            # type label above
            label = NOTE_TYPE_LABELS.get(note['type'], '')
            if label:
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                cv2.putText(img, label, (cx - ts[0] // 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, nc, 1)
        elif note['state'] == 'active':
            elapsed = now - note['active_start']
            progress = min(1.0, elapsed / note['max_active']) if note['max_active'] > 0 else 1.0
            pulse = int(abs(math.sin(now * 8)) * 5)
            y1 = JUDGE_Y - bar_h // 2 - pulse
            y2 = JUDGE_Y + bar_h // 2 + pulse
            # dark bg
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
            # progress fill
            fill_w = int((x2 - x1) * progress)
            cv2.rectangle(img, (x1, y1), (x1 + fill_w, y2), nc, -1)
            # border glow
            draw_glow_rect(img, x1, y1, x2, y2, nc)
            # circular progress arc
            arc_cx, arc_cy = cx, y1 - 16
            arc_r = 10
            cv2.ellipse(img, (arc_cx, arc_cy), (arc_r, arc_r), -90, 0, 360, (40, 40, 40), 2)
            cv2.ellipse(img, (arc_cx, arc_cy), (arc_r, arc_r), -90, 0,
                        int(360 * progress),
                        nc, 2)
            # type label
            if note['type'] == NOTE_SHAKE:
                cv2.putText(img, "SHAKE!", (cx - 28, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1)
            elif note['type'] == NOTE_AVOID:
                cv2.putText(img, "AVOID!", (cx - 28, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
        elif note['state'] == 'judged':
            fade = max(0, 1.0 - (now - note['judge_time']) / 0.8)
            if fade > 0:
                j = note['judgment']
                color = tuple(int(c * fade) for c in JUDGMENT_COLORS.get(j, (200, 200, 200)))
                label = j.upper() if j else '?'
                scale = 0.7 + (1.0 - fade) * 0.3
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
                ty = JUDGE_Y - 20 - int((1.0 - fade) * 40)
                cv2.putText(img, label, (cx - ts[0] // 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_rhythm_hud(img, game):
    # background bar
    overlay = img[0:55, :].copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, 55), (15, 15, 25), -1)
    img[0:55, :] = cv2.addWeighted(overlay, 0.85, img[0:55, :], 0.15, 0)
    # separator line
    diff_color = RHYTHM_DIFFICULTY[game.rhythm_difficulty]['color']
    cv2.line(img, (0, 55), (SCREEN_W, 55), diff_color, 1)

    _, _, bpm, stage_name, _, _ = game.get_rhythm_stage()
    diff_name = RHYTHM_DIFFICULTY[game.rhythm_difficulty]['name']
    cv2.putText(img, f"{diff_name} - {stage_name}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, diff_color, 2)
    cv2.putText(img, f"BPM {bpm}  |  Stage {game.stage_idx + 1}/{game.rhythm_stages_count()}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # score (center)
    score_text = f"{game.score}"
    ts = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(img, score_text, (SCREEN_W // 2 - ts[0] // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    label_ts = cv2.getTextSize("SCORE", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    cv2.putText(img, "SCORE", (SCREEN_W // 2 - label_ts[0] // 2, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    # combo
    if game.combo > 1:
        if game.combo >= 30:
            ct, cc = f"{game.combo}x  x3.0", (255, 100, 100)
        elif game.combo >= 15:
            ct, cc = f"{game.combo}x  x2.0", (255, 200, 0)
        elif game.combo >= 5:
            ct, cc = f"{game.combo}x  x1.5", (255, 255, 100)
        else:
            ct, cc = f"{game.combo}x", (200, 200, 200)
        cts = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(img, ct, (SCREEN_W // 2 - cts[0] // 2, LANE_TOP - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, 1)

    # progress  judged/total
    prog_text = f"{game.judged_count}/{game.total_notes}"
    pts = cv2.getTextSize(prog_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(img, prog_text, (SCREEN_W - pts[0] - 10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # hearts (top right)
    for i in range(5):
        hx = SCREEN_W - 28 * (5 - i) - 5
        heart_size = 22
        cached = HEART_CACHE.get(heart_size)
        if cached is None:
            _cache_heart(heart_size)
            cached = HEART_CACHE.get(heart_size)
        if cached is not None:
            key = 'active' if i < game.lives else 'inactive'
            icon_rgb, alpha_mask = cached[key]
            hx1, hy1 = hx - heart_size // 2, 14 - heart_size // 2
            hx1i, hy1i = max(0, hx1), max(0, hy1)
            hx2i = min(img.shape[1], hx1 + heart_size)
            hy2i = min(img.shape[0], hy1 + heart_size)
            ix1, iy1 = hx1i - hx1, hy1i - hy1
            roi = img[hy1i:hy2i, hx1i:hx2i]
            sh = (hy2i - hy1i, hx2i - hx1i)
            a = alpha_mask[iy1:iy1 + sh[0], ix1:ix1 + sh[1]]
            r = icon_rgb[iy1:iy1 + sh[0], ix1:ix1 + sh[1]]
            roi[:] = (r * a + roi * (1 - a)).astype(np.uint8)
        else:
            color = (0, 0, 255) if i < game.lives else (60, 60, 60)
            cv2.putText(img, "<3", (hx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


def draw_rhythm_stage_clear(img, game):
    diff = RHYTHM_DIFFICULTY[game.rhythm_difficulty]
    cv2.putText(img, "STAGE CLEAR!", (SCREEN_W // 2 - 140, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    cv2.putText(img, f"{diff['name']} - Stage {game.stage_idx + 1}",
                (SCREEN_W // 2 - 80, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, diff['color'], 1)
    perfect = sum(1 for r, _ in game.results if r == 'perfect')
    good = sum(1 for r, _ in game.results if r == 'good')
    miss = sum(1 for r, _ in game.results if r == 'miss')
    total = max(1, perfect + good + miss)
    y = 230
    bx = SCREEN_W // 2 - 120
    # result bars
    for label, count, color in [("PERFECT", perfect, (255, 255, 0)),
                                 ("GOOD", good, (0, 255, 100)),
                                 ("MISS", miss, (0, 0, 255))]:
        cv2.putText(img, f"{label}:", (bx, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        bar_x = bx + 110
        bar_w = 120
        cv2.rectangle(img, (bar_x, y - 10), (bar_x + bar_w, y + 5), (40, 40, 40), -1)
        fill = int(bar_w * count / total) if total > 0 else 0
        cv2.rectangle(img, (bar_x, y - 10), (bar_x + fill, y + 5), color, -1)
        cv2.putText(img, str(count), (bar_x + bar_w + 8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 35
    cv2.putText(img, f"MAX COMBO: {game.max_combo}", (bx, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(img, f"STAGE SCORE: {game.stage_score}", (bx, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    blink = int(time.time() * 2) % 2
    if blink:
        cv2.putText(img, "Press SPACE for Next Stage", (SCREEN_W // 2 - 155, y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(img, "Press B for Mode Select", (SCREEN_W // 2 - 125, y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)


def _rhythm_play_tick(screen, frame, game, now, stable_g):
    dt = now - game._prev_t if game._prev_t > 0 else 0.016
    game._prev_t = now
    # spawn notes from queue
    if game.note_queue and now >= game.next_spawn_t:
        ntype, gesture = game.note_queue.pop(0)
        spawn_lane_note(game, ntype, gesture)
        _, interval, _, _, _, _ = game.get_rhythm_stage()
        game.next_spawn_t = now + interval
    update_lane_notes(game, dt, game.last_detection, now)
    # draw
    draw_rhythm_hud(screen, game)
    draw_lane_background(screen, now)
    draw_lane_notes(screen, game, now)
    update_draw_particles(screen, game, dt)
    # camera feed (left side, below lanes)
    cam_w, cam_h = 160, 120
    cam_small = cv2.resize(frame, (cam_w, cam_h))
    sx, sy = 8, SCREEN_H - cam_h - 8
    screen[sy:sy + cam_h, sx:sx + cam_w] = cam_small
    cv2.rectangle(screen, (sx - 1, sy - 1),
                  (sx + cam_w + 1, sy + cam_h + 1), (60, 60, 60), 1)
    # detection label & highlight on camera
    if stable_g:
        gc = GESTURE_COLOR.get(stable_g, (200, 200, 200))
        cv2.rectangle(screen, (sx - 2, sy - 2),
                      (sx + cam_w + 2, sy + cam_h + 2), gc, 2)
        draw_gesture_icon(screen, stable_g, sx + cam_w + 28, sy + cam_h // 2, 20)
    # end conditions
    if (game.judged_count >= game.total_notes
            and not game.note_queue and not game.lane_notes):
        game.state = game.STAGE_CLEAR
    elif game.lives <= 0:
        game.state = game.GAME_OVER


# ══════════════════════════════════════════════
#  UI (v1과 동일)
# ══════════════════════════════════════════════
_gesture_icon_cache = {}  # (gesture, icon_size) → (rgb, alpha_mask)

def _get_cached_icon(gesture, icon_size):
    key = (gesture, icon_size)
    cached = _gesture_icon_cache.get(key)
    if cached is not None:
        return cached
    icon = GESTURE_IMAGE.get(gesture)
    if icon is None:
        return None
    resized = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
    if resized.shape[2] == 4:
        rgb = resized[:, :, :3]
        a = resized[:, :, 3:] / 255.0
    else:
        rgb = resized
        a = np.ones((icon_size, icon_size, 1), dtype=np.float32)
    _gesture_icon_cache[key] = (rgb, a)
    return (rgb, a)


def draw_gesture_icon(img, gesture, cx, cy, size, alpha=1.0, highlight=False):
    icon_size = max(1, int(size * 2))
    cached = _get_cached_icon(gesture, icon_size)
    if cached is not None:
        icon_rgb, alpha_mask = cached
        x1 = cx - icon_size // 2
        y1 = cy - icon_size // 2
        x1i, y1i = max(0, x1), max(0, y1)
        x2i = min(img.shape[1], x1 + icon_size)
        y2i = min(img.shape[0], y1 + icon_size)
        ix1, iy1 = x1i - x1, y1i - y1
        ix2 = ix1 + (x2i - x1i)
        iy2 = iy1 + (y2i - y1i)
        roi = img[y1i:y2i, x1i:x2i]
        a = alpha_mask[iy1:iy2, ix1:ix2]
        roi[:] = (icon_rgb[iy1:iy2, ix1:ix2] * a + roi * (1 - a)).astype(np.uint8)
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
        for i, gesture in enumerate(game.sequence):
            cx = margin + i * spacing
            if i < game.current_note:
                if i < len(game.results):
                    result = game.results[i][0]
                    col = (0, 255, 0) if result == 'correct' else (0, 0, 200)
                    cv2.circle(img, (cx, cy), 18, col, -1)
                    cv2.circle(img, (cx, cy), 18, (255, 255, 255), 1)
                    mark = 'O' if result == 'correct' else 'X'
                    cv2.putText(img, mark, (cx - 8, cy + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif i == game.current_note:
                pulse = int(abs(math.sin(now * 4)) * 8)
                draw_gesture_icon(img, gesture, cx, cy, 28 + pulse, highlight=True)
            else:
                cv2.circle(img, (cx, cy), 20, (60, 60, 60), 2)
                cv2.putText(img, '?', (cx - 7, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
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
    mode_hud_colors = [(180, 180, 180), (0, 200, 200), (100, 255, 100), (120, 120, 255), (255, 100, 255)]
    cv2.putText(img, GAME_MODES[game.mode], (SCREEN_W - 100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_hud_colors[game.mode], 2)
    if game.combo > 1:
        cv2.putText(img, f"{game.combo}x COMBO", (SCREEN_W // 2 - 50, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    for i in range(5):
        hx = SCREEN_W - 30 * (5 - i)
        color = (0, 0, 255) if i < game.lives else (60, 60, 60)
        heart_size = 24
        cached = HEART_CACHE.get(heart_size)
        if cached is not None:
            key = 'active' if i < game.lives else 'inactive'
            icon_rgb, alpha_mask = cached[key]
            x1 = hx - heart_size // 2
            y1 = 30 - heart_size // 2
            x1i, y1i = max(0, x1), max(0, y1)
            x2i = min(img.shape[1], x1 + heart_size)
            y2i = min(img.shape[0], y1 + heart_size)
            ix1, iy1 = x1i - x1, y1i - y1
            ix2 = ix1 + (x2i - x1i)
            iy2 = iy1 + (y2i - y1i)
            roi = img[y1i:y2i, x1i:x2i]
            roi[:] = (icon_rgb[iy1:iy2, ix1:ix2] * alpha_mask[iy1:iy2, ix1:ix2]
                      + roi * (1 - alpha_mask[iy1:iy2, ix1:ix2])).astype(np.uint8)
        else:
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


def draw_title_screen(img, game):
    cv2.rectangle(img, (0, 0), (SCREEN_W, SCREEN_H), (20, 20, 40), -1)
    title = "RPS RHYTHM"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
    cv2.putText(img, title, (SCREEN_W // 2 - text_size[0] // 2, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    sub = "Select Game Mode  (press 0 ~ 4)"
    text_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
    cv2.putText(img, sub, (SCREEN_W // 2 - text_size[0] // 2, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    box_x = SCREEN_W // 2 - 310
    box_w = 620
    for i in range(5):
        is_sel = (i == game.selected_mode)
        by = 155 + i * 85
        bg = (40, 60, 60) if is_sel else (22, 22, 38)
        cv2.rectangle(img, (box_x, by), (box_x + box_w, by + 72), bg, -1)
        if is_sel:
            cv2.rectangle(img, (box_x, by), (box_x + box_w, by + 72), (0, 200, 200), 2)
        num_col  = (0, 255, 255)   if is_sel else (80, 80, 80)
        name_col = (255, 255, 255) if is_sel else (140, 140, 140)
        desc_col = (160, 220, 160) if is_sel else (70, 100, 70)
        cv2.putText(img, f"{i}.", (box_x + 18, by + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, num_col, 2)
        cv2.putText(img, MODE_DESC[i], (box_x + 58, by + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_col, 2)
    blink = int(time.time() * 2) % 2
    if blink:
        start_text = "SPACE to Start"
        text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(img, start_text, (SCREEN_W // 2 - text_size[0] // 2, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def draw_stage_select_screen(img, game):
    cv2.rectangle(img, (0, 0), (SCREEN_W, SCREEN_H), (15, 15, 30), -1)
    # title
    title = "SELECT DIFFICULTY"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
    cv2.putText(img, title, (SCREEN_W // 2 - ts[0] // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 100, 255), 3)
    sub = "UP/DOWN to select, SPACE to start, B to go back"
    sts = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(img, sub, (SCREEN_W // 2 - sts[0] // 2, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    # difficulty cards
    card_w = 580
    card_h = 130
    card_x = SCREEN_W // 2 - card_w // 2
    for i, diff in enumerate(RHYTHM_DIFFICULTY):
        is_sel = (i == game.selected_difficulty)
        cy = 150 + i * (card_h + 15)
        # card background
        bg = (30, 40, 55) if is_sel else (18, 18, 30)
        cv2.rectangle(img, (card_x, cy), (card_x + card_w, cy + card_h), bg, -1)
        if is_sel:
            cv2.rectangle(img, (card_x, cy), (card_x + card_w, cy + card_h),
                          diff['color'], 2)
            # selection indicator
            cv2.circle(img, (card_x - 15, cy + card_h // 2), 6, diff['color'], -1)
        # difficulty name
        name_col = diff['color'] if is_sel else tuple(int(c * 0.5) for c in diff['color'])
        cv2.putText(img, diff['name'], (card_x + 20, cy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, name_col, 2)
        # stars
        n_stars = i + 1
        for s in range(3):
            sx = card_x + 200 + s * 25
            star_col = diff['color'] if (s < n_stars and is_sel) else (50, 50, 50)
            cv2.putText(img, "*", (sx, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, star_col, 2)
        # description
        desc_col = (180, 180, 180) if is_sel else (80, 80, 80)
        cv2.putText(img, diff['desc'], (card_x + 20, cy + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, desc_col, 1)
        # stage count & BPM range
        stages = diff['stages']
        bpm_min = min(s[2] for s in stages)
        bpm_max = max(s[2] for s in stages)
        info = f"{len(stages)} stages  |  BPM {bpm_min}-{bpm_max}  |  {sum(s[0] for s in stages)} notes"
        info_col = (140, 140, 140) if is_sel else (60, 60, 60)
        cv2.putText(img, info, (card_x + 20, cy + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, info_col, 1)
        # note type preview
        if is_sel:
            has_shake = any(s[4] > 0 for s in stages)
            has_avoid = any(s[5] > 0 for s in stages)
            types_y = cy + 115
            tx = card_x + 20
            cv2.circle(img, (tx + 6, types_y - 4), 5, NOTE_TYPE_COLORS[NOTE_NORMAL], -1)
            cv2.putText(img, "NORMAL", (tx + 16, types_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            tx += 90
            if has_shake:
                cv2.circle(img, (tx + 6, types_y - 4), 5, NOTE_TYPE_COLORS[NOTE_SHAKE], -1)
                cv2.putText(img, "SHAKE", (tx + 16, types_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                tx += 80
            if has_avoid:
                cv2.circle(img, (tx + 6, types_y - 4), 5, NOTE_TYPE_COLORS[NOTE_AVOID], -1)
                cv2.putText(img, "AVOID", (tx + 16, types_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    # blink prompt
    blink = int(time.time() * 2) % 2
    if blink:
        prompt = "SPACE to Start"
        pts = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(img, prompt, (SCREEN_W // 2 - pts[0] // 2, 595),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def draw_memorize_screen(img, game, now):
    elapsed = now - game.memorize_start
    remaining = game.preview_time() - elapsed
    if game.mode == 2:
        memo_title, title_color = "MEMORIZE! > BEAT EACH", (100, 255, 100)
    elif game.mode == 3:
        memo_title, title_color = "MEMORIZE! > LOSE TO EACH", (120, 120, 255)
    else:
        memo_title, title_color = "MEMORIZE!", (255, 200, 0)
    text_size = cv2.getTextSize(memo_title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    cv2.putText(img, memo_title, (SCREEN_W // 2 - text_size[0] // 2, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, title_color, 2)
    bar_w = 380
    bar_x = SCREEN_W // 2 - bar_w // 2
    progress = max(0, remaining / game.preview_time())
    cv2.rectangle(img, (bar_x, 520), (bar_x + bar_w, 540), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, 520), (bar_x + int(bar_w * progress), 540), (0, 200, 255), -1)
    cv2.putText(img, f"{remaining:.1f}s", (bar_x + bar_w + 15, 535),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    n = len(game.sequence)
    if game.mode == 0:
        # Tutorial: 순차적으로 나타남
        spacing = 100
        total_w = n * spacing
        start_x = SCREEN_W // 2 - total_w // 2 + spacing // 2
        for i, gesture in enumerate(game.sequence):
            cx = start_x + i * spacing
            cy = 360
            progress_ratio = elapsed / game.preview_time()
            if progress_ratio >= i / n:
                age = (progress_ratio - i / n) * game.preview_time()
                scale = min(1.0, age * 3)
                r = int(45 * scale)
                draw_gesture_icon(img, gesture, cx, cy, r, highlight=True)
                cv2.putText(img, str(i + 1), (cx - 7, cy + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            else:
                cv2.circle(img, (cx, cy), 40, (50, 50, 50), 2)
                cv2.putText(img, "?", (cx - 10, cy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    else:
        # Same / Win / Lose: 한 번에 모두 표출
        spacing = min(100, (SCREEN_W - 60) // max(n, 1))
        total_w = n * spacing
        start_x = SCREEN_W // 2 - total_w // 2 + spacing // 2
        for i, gesture in enumerate(game.sequence):
            cx = start_x + i * spacing
            cy = 360
            draw_gesture_icon(img, gesture, cx, cy, 45, highlight=True)
            cv2.putText(img, str(i + 1), (cx - 7, cy + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)


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
        cv2.putText(img, "Press B for Mode Select", (SCREEN_W // 2 - 125, y + 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)


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
        cv2.putText(img, "Press B for Mode Select", (SCREEN_W // 2 - 125, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)


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
        cv2.putText(img, "Press B for Mode Select", (SCREEN_W // 2 - 125, 485),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)


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
                if not ch:
                    continue
                if ch == '\x1b':
                    # 방향키 이스케이프 시퀀스 처리 (ESC [ A/B/C/D)
                    try:
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            ch = {'A': 'UP', 'B': 'DOWN',
                                  'C': 'RIGHT', 'D': 'LEFT'}.get(ch3, '\x1b')
                        else:
                            ch = '\x1b'
                    except Exception:
                        ch = '\x1b'
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
          f"hold={HOLD_TIME}s, cooldown={COOLDOWN_TIME}s")

    detector = RPSDetector(model_path, num_threads=TFLITE_THREADS)
    game = GameState()
    # ONNX(YOLO)는 conf가 항상 높아 히스토리를 빠르게 교체해야 클래스 전환이 빠름
    if is_onnx:
        smoother = PredictionSmoother(window=2, conf_th=0.4)
    else:
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

    # ── 마우스/터치 콜백 ──
    _mouse_clicks = []
    _mouse_lock = threading.Lock()

    def _mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with _mouse_lock:
                _mouse_clicks.append((x, y))

    cv2.setMouseCallback('RPS Rhythm', _mouse_cb)

    if tk is not None:
        try:
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
            x = max(0, (screen_w - SCREEN_W) // 2)
            y = max(0, (screen_h - SCREEN_H) // 2)
            cv2.moveWindow('RPS Rhythm', x, y)
        except Exception:
            pass

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

    screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

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
            screen[:] = 0
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
                    prev_stable = stable_g
                    # 감지 클래스가 바뀌면 smoother를 즉시 리셋해 이전 히스토리 제거
                    if g is not None and prev_stable is not None and g != prev_stable:
                        smoother.reset()
                    stable_g, stable_c = smoother.update(g, c)
                    # ── 즉시 hold 리셋 ──
                    # 추론 지연(inference latency) 중에 old stable_g가 유지되어
                    # hold 타이머가 계속 누적되는 문제 방지.
                    # raw g 가 현재 hold 제스처와 다를 때 smoother 결과를 기다리지 않고
                    # confirm_gesture 와 hold_start_time 을 즉시 갱신한다.
                    if (g is not None
                            and game.confirm_gesture is not None
                            and g != game.confirm_gesture
                            and not (now < game.cooldown_until)):
                        game.confirm_gesture = g
                        game.hold_start_time = now
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
                draw_title_screen(screen, game)

            elif game.state == game.STAGE_SELECT:
                draw_stage_select_screen(screen, game)

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
                    if game.mode == 4:
                        game._prev_t = now
                        game.next_spawn_t = now
                    else:
                        game.note_start_time = now
                        game.confirm_gesture = None
                        game.hold_start_time = 0
                        game.cooldown_until = 0
                    smoother.reset()
                    stable_g, stable_c = None, 0.0

            elif game.state == game.PLAY and game.mode == 4:
                _rhythm_play_tick(screen, frame, game, now, stable_g)

            elif game.state == game.PLAY:
                draw_hud(screen, game)
                draw_camera_feed(screen, frame, game, "")

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
                        cv2.putText(screen, f"#{game.current_note + 1}/{len(game.sequence)}",
                            (cam_x + CAM_W + 10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

                    if confirmed:
                        required = get_required_gesture(game.mode, target)
                        if detected == required:
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

            elif game.state == game.STAGE_CLEAR and game.mode == 4:
                draw_rhythm_hud(screen, game)
                draw_rhythm_stage_clear(screen, game)

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

            # ── 마우스/터치 클릭 처리 ──
            with _mouse_lock:
                clicks = _mouse_clicks[:]
                _mouse_clicks.clear()
            for (mx, my) in clicks:
                if game.state == game.TITLE:
                    _box_x = SCREEN_W // 2 - 310
                    _box_w = 620
                    for _i in range(5):
                        _by = 155 + _i * 85
                        if _box_x <= mx <= _box_x + _box_w and _by <= my <= _by + 72:
                            game.selected_mode = _i
                            game.mode = _i
                            if _i == 4:
                                game.state = game.STAGE_SELECT
                            else:
                                game.reset()
                                game.state = game.MEMORIZE
                                game.generate_sequence()
                                game.memorize_start = time.time()
                                print(f"[Mode: {GAME_MODES[game.mode]}] [Stage {game.stage_idx + 1}]")
                            break
                elif game.state == game.STAGE_SELECT:
                    _card_w = 580
                    _card_h = 130
                    _card_x = SCREEN_W // 2 - _card_w // 2
                    for _i in range(3):
                        _cy = 150 + _i * (_card_h + 15)
                        if _card_x <= mx <= _card_x + _card_w and _cy <= my <= _cy + _card_h:
                            game.selected_difficulty = _i
                            game.rhythm_difficulty = _i
                            game.reset()
                            game.generate_notes()
                            game.state = game.COUNTDOWN
                            game.countdown_start = time.time()
                            print(f"[Rhythm {RHYTHM_DIFFICULTY[_i]['name']}] [Stage 1]")
                            break

            # ── 키 입력 ──
            key_ch = kb.get_key()
            if key_ch == 'q':
                break
            elif key_ch in ('0', '1', '2', '3', '4'):
                if game.state == game.TITLE:
                    game.selected_mode = int(key_ch)
                elif game.state == game.STAGE_SELECT:
                    if key_ch in ('0', '1', '2'):
                        game.selected_difficulty = int(key_ch)
            elif key_ch in ('UP', 'LEFT'):
                if game.state == game.TITLE:
                    game.selected_mode = (game.selected_mode - 1) % 5
                elif game.state == game.STAGE_SELECT:
                    game.selected_difficulty = (game.selected_difficulty - 1) % 3
            elif key_ch in ('DOWN', 'RIGHT'):
                if game.state == game.TITLE:
                    game.selected_mode = (game.selected_mode + 1) % 5
                elif game.state == game.STAGE_SELECT:
                    game.selected_difficulty = (game.selected_difficulty + 1) % 3
            elif key_ch == '\x1b' or key_ch == 'b':
                if game.state == game.STAGE_SELECT:
                    game.state = game.TITLE
                elif game.state in (game.STAGE_CLEAR, game.GAME_OVER, game.ALL_CLEAR):
                    game.reset()
                    print("[Mode Select]")
            elif key_ch == ' ':
                if game.state == game.TITLE:
                    game.mode = game.selected_mode
                    if game.mode == 4:
                        game.state = game.STAGE_SELECT
                    else:
                        game.reset()
                        game.state = game.MEMORIZE
                        game.generate_sequence()
                        game.memorize_start = time.time()
                        print(f"[Mode: {GAME_MODES[game.mode]}] [Stage {game.stage_idx + 1}]")
                elif game.state == game.STAGE_SELECT:
                    game.rhythm_difficulty = game.selected_difficulty
                    game.reset()
                    game.generate_notes()
                    game.state = game.COUNTDOWN
                    game.countdown_start = time.time()
                    print(f"[Rhythm {RHYTHM_DIFFICULTY[game.rhythm_difficulty]['name']}] [Stage 1]")
                elif game.state == game.STAGE_CLEAR:
                    game.stage_idx += 1
                    stages_len = game.rhythm_stages_count() if game.mode == 4 else len(STAGES)
                    if game.stage_idx >= stages_len:
                        game.state = game.ALL_CLEAR
                    elif game.mode == 4:
                        game.generate_notes()
                        game.state = game.COUNTDOWN
                        game.countdown_start = time.time()
                        print(f"[Stage {game.stage_idx + 1}]")
                    else:
                        game.state = game.MEMORIZE
                        game.generate_sequence()
                        game.memorize_start = time.time()
                        print(f"[Stage {game.stage_idx + 1}] 시퀀스: {game.sequence}")
                elif game.state in (game.GAME_OVER, game.ALL_CLEAR):
                    if game.mode == 4:
                        game.state = game.STAGE_SELECT
                    else:
                        game.reset()
                        game.state = game.MEMORIZE
                        game.generate_sequence()
                        game.memorize_start = time.time()
                    print(f"[Restart]")
    finally:
        kb.stop()
        worker.stop()
        release()
        cv2.destroyAllWindows()
        print(f"\n최종 점수: {game.score} | 최대 콤보: {game.max_combo}")


if __name__ == '__main__':
    main()
