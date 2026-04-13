import sys
import os
import argparse
import glob
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


SCREEN_W = 640
SCREEN_H = 520
CAM_W, CAM_H = 320, 240
IMG_SIZE = 224
OFFSET = 30

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
MIN_CONFIDENCE = 0.65
CONFIRM_HOLD_SECONDS = 0.25
NOTE_INPUT_GRACE_SECONDS = 0.45
TIME_LIMIT_PER_NOTE = 8

STAGES = [
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


class RPSDetector:
    MODE_LANDMARK = 'landmark'
    MODE_MODEL = 'model'

    def __init__(self, mode=MODE_LANDMARK, model_path=None):
        self.mode = mode
        self.hd = HandDetector(
            maxHands=1,
            modelComplexity=0,
            detectionCon=0.55,
            minTrackCon=0.5
        )

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_dtype = None
        if self.mode == self.MODE_MODEL:
            if not model_path:
                raise ValueError("model mode requires model_path")
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
        hands, _ = self.hd.findHands(frame, draw=False)
        if not hands:
            return None, 0, None

        hand = hands[0]
        x, y, w, h = hand['bbox']
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x - OFFSET)
        y1 = max(0, y - OFFSET)
        x2 = min(frame_w, x + w + OFFSET)
        y2 = min(frame_h, y + h + OFFSET)
        bbox = (x1, y1, x2, y2)

        if self.mode == self.MODE_LANDMARK:
            gesture, conf = self.classify_landmarks(hand)
            return gesture, conf, bbox

        if x2 <= x1 or y2 <= y1:
            return None, 0, None

        hand_img = frame[y1:y2, x1:x2]
        square = self.make_square_img(hand_img)

        inp = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        if self.input_dtype == np.float32:
            inp = inp.astype(np.float32) / 255.0
        else:
            inp = inp.astype(self.input_dtype)
        inp = np.expand_dims(inp, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        if self.input_dtype == np.uint8:
            output = output.astype(np.float32)
        if output.max() > 1.0:
            exp = np.exp(output - output.max())
            output = exp / exp.sum()

        ans = int(np.argmax(output))
        conf = float(output[ans])
        return GESTURES[ans], conf, bbox

    def classify_landmarks(self, hand):
        fingers = self.hd.fingersUp(hand)
        if len(fingers) != 5:
            return None, 0

        thumb, index, middle, ring, pinky = fingers
        open_count = sum(fingers)

        if not index and not middle and not ring and not pinky:
            return 'rock', 0.95 if not thumb else 0.85
        if index and middle and not ring and not pinky:
            return 'scissors', 0.90 if not thumb else 0.80
        if open_count >= 4 or (index and middle and ring and pinky):
            return 'paper', 0.95 if thumb else 0.85

        return None, 0



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
        self.confirm_gesture = None
        self.confirm_started_at = 0
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
        self.confirm_gesture = None
        self.confirm_started_at = 0

    def note_time_limit(self):
        return self.get_stage()[2]

    def preview_time(self):
        return self.get_stage()[1]



def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


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
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)


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
    cv2.rectangle(img, (0, 0), (SCREEN_W, 52), (20, 20, 20), -1)

    seq_len, _, time_limit, stage_name = game.get_stage()

    cv2.putText(img, f"Stage {game.stage_idx + 1}: {stage_name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    limit_text = f"Limit:{time_limit}s" if time_limit > 0 else "No Limit"
    cv2.putText(img, limit_text,
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    score_text = f"Score: {game.score}"
    cv2.putText(img, score_text, (SCREEN_W // 2 - 60, 22),
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
    tx = SCREEN_W // 2 - text_size[0] // 2
    cv2.putText(img, title, (tx, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

    sub = "Remember the Sequence, Show Your Hands!"
    text_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    tx = SCREEN_W // 2 - text_size[0] // 2
    cv2.putText(img, sub, (tx, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    icons_y = 290
    for i, g in enumerate(GESTURES):
        cx = SCREEN_W // 2 + (i - 1) * 100
        draw_gesture_icon(img, g, cx, icons_y, 35)
        cv2.putText(img, GESTURE_KR[g],
                    (cx - 35, icons_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    blink = int(time.time() * 2) % 2
    if blink:
        start_text = "Press SPACE to Start"
        text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = SCREEN_W // 2 - text_size[0] // 2
        cv2.putText(img, start_text, (tx, 420),
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
        note_appear = i / n
        progress_ratio = elapsed / game.preview_time()
        if progress_ratio >= note_appear:
            age = (progress_ratio - note_appear) * game.preview_time()
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



def try_beep():
    try:
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception:
        pass



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



def find_default_model():
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, 'models', 'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_qat.tflite'),
        os.path.join(base, 'models', 'rps_mobilenetv2_ptq_int8.tflite'),
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(base, '..', 'examples', '03_CNN_Based_On-Device_AI',
                     'RPS_MobileNetV2.tflite'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def has_gui_display():
    if sys.platform.startswith('linux'):
        return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    return True


def list_video_devices():
    if not sys.platform.startswith('linux'):
        return []
    return sorted(glob.glob('/dev/video*'))


def main():
    parser = argparse.ArgumentParser(description='RPS Rhythm Game')
    parser.add_argument('--detector', choices=['landmark', 'model'], default='landmark',
                        help='hand detector mode: landmark=fast rule based, model=TFLite based')
    parser.add_argument('--camera', type=int, default=0,
                        help='camera index')
    parser.add_argument('--model', type=str, default=None,
                        help='TFLite model path for model detector')
    args = parser.parse_args()

    if not has_gui_display():
        print("[Error] No GUI display is available for the OpenCV window.")
        print("  DISPLAY or WAYLAND_DISPLAY is not set.")
        print("  Run this from a local desktop terminal, or connect X forwarding/GUI for a remote session.")
        return

    model_path = None
    if args.detector == RPSDetector.MODE_MODEL:
        model_path = args.model or find_default_model()
        if not model_path or not os.path.exists(model_path):
            print("[Error] Model file was not found.")
            print("  Pass a model path with --model,")
            print("  train a model with train_model.py,")
            print("  or copy a model from examples/03_CNN_Based_On-Device_AI/.")
            return
        print(f"Model loaded: {model_path}")
    else:
        print("Hand detector: landmark mode (fast rule based)")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera: index {args.camera}")
        devices = list_video_devices()
        if devices:
            print(f"  Detected device candidates: {', '.join(devices)}")
            print("  Example: python rps_rhythm.py --camera 1")
        else:
            print("  No /dev/video* device was found.")
            print("  Check webcam connection, OS camera permission, and device passthrough in Docker/WSL/remote environments.")
        return

    detector = RPSDetector(mode=args.detector, model_path=model_path)
    game = GameState()

    cv2.namedWindow('RPS Rhythm', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RPS Rhythm', SCREEN_W, SCREEN_H)

    kb = KeyboardReader()
    kb.start()

    last_judge_result = None
    last_judge_time = 0
    fps_time = time.time()

    print("\n=== RPS RHYTHM ===")
    print("Keyboard input in this terminal: SPACE=start  q=quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        detection_text = ""
        if game.state == game.PLAY:
            gesture, conf, bbox = detector.detect(frame)
            if gesture and conf >= MIN_CONFIDENCE:
                detection_text = f"{GESTURE_KR[gesture]} ({conf:.0%})"
                game.last_detection = gesture
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  GESTURE_COLOR[gesture], 2)
            else:
                game.last_detection = None

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
                game.confirm_started_at = 0

        elif game.state == game.PLAY:
            draw_hud(screen, game)
            draw_camera_feed(screen, frame, game, detection_text)
            draw_sequence_bar(screen, game, now)

            if game.current_note < len(game.sequence):
                target = game.sequence[game.current_note]
                time_limit = game.note_time_limit()
                elapsed_note = now - game.note_start_time

                cam_x = SCREEN_W // 2 - CAM_W // 2
                draw_gesture_icon(screen, target,
                                  cam_x + CAM_W + 40, 250, 35, highlight=True)
                cv2.putText(screen, GESTURE_KR[target],
                            (cam_x + CAM_W + 5, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(screen, f"#{game.current_note + 1}/{len(game.sequence)}",
                            (cam_x + CAM_W + 10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

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

                confirmed = False
                if elapsed_note < NOTE_INPUT_GRACE_SECONDS:
                    game.confirm_gesture = None
                    game.confirm_started_at = 0
                elif game.last_detection:
                    if game.last_detection != game.confirm_gesture:
                        game.confirm_gesture = game.last_detection
                        game.confirm_started_at = now

                    stable_for = now - game.confirm_started_at
                    if stable_for >= CONFIRM_HOLD_SECONDS:
                        confirmed = True
                        detected = game.confirm_gesture
                else:
                    game.confirm_gesture = None
                    game.confirm_started_at = 0

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
                    game.confirm_started_at = 0
                    game.current_note += 1
                    game.note_start_time = now

                elif time_limit > 0 and elapsed_note > time_limit:
                    game.combo = 0
                    game.results.append(('timeout', target))
                    game.lives -= 1
                    last_judge_result = 'timeout'
                    last_judge_time = now
                    game.confirm_gesture = None
                    game.confirm_started_at = 0
                    game.current_note += 1
                    game.note_start_time = now

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

        cur_time = time.time()
        fps = 1.0 / max(cur_time - fps_time, 0.001)
        fps_time = cur_time
        cv2.putText(screen, f"FPS:{fps:.0f}", (SCREEN_W - 80, SCREEN_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow('RPS Rhythm', screen)
        cv2.waitKey(1)

        key_ch = kb.get_key()
        if key_ch == 'q':
            break
        elif key_ch == ' ':
            if game.state == game.TITLE:
                game.reset()
                game.state = game.MEMORIZE
                game.generate_sequence()
                game.memorize_start = time.time()
                print(f"[Stage {game.stage_idx + 1}] sequence: {game.sequence}")

            elif game.state == game.STAGE_CLEAR:
                game.stage_idx += 1
                if game.stage_idx >= len(STAGES):
                    game.state = game.ALL_CLEAR
                else:
                    game.state = game.MEMORIZE
                    game.generate_sequence()
                    game.memorize_start = time.time()
                    print(f"[Stage {game.stage_idx + 1}] sequence: {game.sequence}")

            elif game.state in (game.GAME_OVER, game.ALL_CLEAR):
                game.reset()
                game.state = game.MEMORIZE
                game.generate_sequence()
                game.memorize_start = time.time()
                print(f"[Stage 1] sequence: {game.sequence}")

    kb.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal score: {game.score} | Max combo: {game.max_combo}")


if __name__ == '__main__':
    main()
