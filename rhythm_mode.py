"""
리듬 레인 모드 — rps_rhythm.py 에서 분리된 리듬 게임 전용 모듈

게임방식 (instant-hit):
  - 3개 레인(가위/바위/보)에 노트가 떨어짐
  - 노트가 판정선(JUDGE_Y) 근처에 도달했을 때 해당 제스처가 감지되면 즉시 판정
  - PERFECT / GOOD / MISS 판정 (판정선과의 픽셀 거리 기반)
  - SHAKE: 빠르게 제스처 변경  |  AVOID: 해당 제스처 피하기
"""

import time
import random
import math
import cv2
import numpy as np

# ══════════════════════════════════════════════
#  공유 상수 (rps_rhythm.py 와 동일 값, 순환 import 방지)
# ══════════════════════════════════════════════
SCREEN_W = 820
SCREEN_H = 620
GESTURES = ['scissors', 'rock', 'paper']
GESTURE_KR = {'scissors': 'SCISSORS', 'rock': 'ROCK', 'paper': 'PAPER'}
GESTURE_COLOR = {
    'scissors': (255, 80, 80),
    'rock':     (80, 255, 80),
    'paper':    (80, 80, 255),
}

# ══════════════════════════════════════════════
#  리듬 모드 상수
# ══════════════════════════════════════════════
LANE_W      = 100
LANE_GAP    = 20
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

# ── 판정 범위 (판정선으로부터 ±px) ──
JUDGE_PERFECT_PX = 18
JUDGE_GOOD_PX    = 40
JUDGE_MISS_PX    = 60

SCORE_PERFECT = 300
SCORE_GOOD    = 150

NOTE_NORMAL = 'normal'
NOTE_SHAKE  = 'shake'
NOTE_AVOID  = 'avoid'

SHAKE_THRESHOLD = 3

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

# ── 난이도별 노트 속도 ──
DIFFICULTY_NOTE_SPEED = {
    0: 180,   # Easy
    1: 260,   # Normal
    2: 350,   # Hard
}

# ── 3가지 난이도 스테이지 ──
RHYTHM_DIFFICULTY = [
    {
        'name': 'EASY',
        'color': (100, 255, 100),
        'desc': 'Slow tempo',
        'icon': 'star1',
        'stages': [
            (12, 1.4, 55,  "Warm Up",   0.0, 0.0),
            (18, 1.2, 65,  "Easy Flow", 0.0, 0.0),
            (22, 1.0, 75,  "Cruise",    0.0, 0.0),
        ],
    },
    {
        'name': 'NORMAL',
        'color': (0, 200, 255),
        'desc': 'Mid tempo',
        'icon': 'star2',
        'stages': [
            (25, 0.9, 85,  "Steady",     0.0, 0.0),
            (30, 0.8, 95,  "Pick It Up", 0.0, 0.0),
            (30, 0.7, 105, "Momentum",   0.0, 0.0),
            (35, 0.65,110, "Push",       0.0, 0.0),
        ],
    },
    {
        'name': 'HARD',
        'color': (80, 80, 255),
        'desc': 'Fast tempo',
        'icon': 'star3',
        'stages': [
            (35, 0.55, 120, "Overdrive", 0.0, 0.0),
            (40, 0.45, 130, "Frenzy",    0.0, 0.0),
            (40, 0.35, 140, "Inferno",   0.0, 0.0),
        ],
    },
]


# ══════════════════════════════════════════════
#  GameState 리듬 확장
# ══════════════════════════════════════════════
def reset_rhythm_fields(game):
    game.lane_notes   = []
    game.particles    = []
    game.note_queue   = []
    game.next_spawn_t = 0.0
    game.total_notes  = 0
    game.judged_count = 0
    game.miss_streak  = 0
    game.prev_gesture = None
    game.shake_changes = 0
    game.shake_last_t  = 0.0
    game._prev_t = 0.0


def get_rhythm_stage(game):
    stages = RHYTHM_DIFFICULTY[game.rhythm_difficulty]['stages']
    return stages[min(game.stage_idx, len(stages) - 1)]


def rhythm_stages_count(game):
    return len(RHYTHM_DIFFICULTY[game.rhythm_difficulty]['stages'])


def generate_notes(game):
    n_notes, interval, bpm, name, shake_r, avoid_r = get_rhythm_stage(game)
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
    game.note_queue   = notes
    game.total_notes  = n_notes
    game.judged_count = 0
    game.results      = []
    game.stage_score  = 0
    game.lane_notes   = []
    game.particles    = []
    game.next_spawn_t = 0.0
    game.miss_streak  = 0


# ══════════════════════════════════════════════
#  노트 로직 (instant-hit 방식)
# ══════════════════════════════════════════════
def _note_speed(game):
    return DIFFICULTY_NOTE_SPEED.get(game.rhythm_difficulty, 180)


def spawn_lane_note(game, ntype, gesture):
    note = {
        'gesture': gesture, 'type': ntype,
        'y': float(LANE_TOP), 'speed': _note_speed(game),
        'state': 'falling', 'judgment': None,
        'judge_time': 0.0,
    }
    game.lane_notes.append(note)


def _apply_judgment(game, note, judgment, now):
    note['state'] = 'judged'
    note['judgment'] = judgment
    note['judge_time'] = now
    j = judgment
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

            if note['type'] == NOTE_NORMAL:
                dist = abs(note['y'] - JUDGE_Y)
                if dist <= JUDGE_GOOD_PX and cur_gesture == note['gesture']:
                    if dist <= JUDGE_PERFECT_PX:
                        _apply_judgment(game, note, 'perfect', now)
                    else:
                        _apply_judgment(game, note, 'good', now)
                elif note['y'] > JUDGE_Y + JUDGE_MISS_PX:
                    _apply_judgment(game, note, 'miss', now)

            elif note['type'] == NOTE_SHAKE:
                dist = abs(note['y'] - JUDGE_Y)
                if dist <= JUDGE_GOOD_PX:
                    if (cur_gesture and game.prev_gesture is not None
                            and cur_gesture != game.prev_gesture):
                        game.shake_changes += 1
                    if game.shake_changes >= SHAKE_THRESHOLD:
                        _apply_judgment(game, note, 'perfect', now)
                        game.shake_changes = 0
                elif note['y'] > JUDGE_Y + JUDGE_MISS_PX:
                    if game.shake_changes >= max(1, SHAKE_THRESHOLD // 2):
                        _apply_judgment(game, note, 'good', now)
                    else:
                        _apply_judgment(game, note, 'miss', now)
                    game.shake_changes = 0

            elif note['type'] == NOTE_AVOID:
                dist = abs(note['y'] - JUDGE_Y)
                if dist <= JUDGE_GOOD_PX and cur_gesture == note['gesture']:
                    _apply_judgment(game, note, 'miss', now)
                elif note['y'] > JUDGE_Y + JUDGE_MISS_PX:
                    _apply_judgment(game, note, 'perfect', now)

        elif note['state'] == 'judged':
            if now - note['judge_time'] > 0.8:
                to_remove.append(note)

    for note in to_remove:
        game.lane_notes.remove(note)
    game.prev_gesture = cur_gesture


# ══════════════════════════════════════════════
#  파티클
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
#  그리기
# ══════════════════════════════════════════════
def draw_glow_rect(img, x1, y1, x2, y2, color, thickness=2):
    for i in range(4, 0, -1):
        alpha = 0.12 * i
        glow = tuple(int(c * alpha) for c in color)
        cv2.rectangle(img, (x1 - i * 2, y1 - i), (x2 + i * 2, y2 + i), glow, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_lane_gradient(img, lx1, lx2, color):
    h = LANE_BOTTOM - LANE_TOP
    for row in range(h):
        y = LANE_TOP + row
        t = row / h
        r = int(15 + color[0] * 0.04 * t)
        g = int(15 + color[1] * 0.04 * t)
        b = int(15 + color[2] * 0.04 * t)
        cv2.line(img, (lx1 + 1, y), (lx2 - 1, y), (b, g, r), 1)


def draw_lane_background(img, now, draw_icon_fn):
    lw = LANE_W
    for gesture, cx in LANE_X_POSITIONS.items():
        lx1, lx2 = cx - lw // 2, cx + lw // 2
        color = GESTURE_COLOR[gesture]
        _draw_lane_gradient(img, lx1, lx2, color)
        cv2.rectangle(img, (lx1, LANE_TOP), (lx2, LANE_BOTTOM), color, 1)
        for dy in range(LANE_TOP + 60, LANE_BOTTOM, 60):
            cv2.line(img, (lx1 + 4, dy), (lx2 - 4, dy),
                     tuple(int(c * 0.15) for c in color), 1)
    all_lx1 = min(LANE_X_POSITIONS.values()) - lw // 2
    all_lx2 = max(LANE_X_POSITIONS.values()) + lw // 2
    cv2.rectangle(img, (all_lx1, JUDGE_Y - 4), (all_lx2, JUDGE_Y + 4),
                  (60, 60, 60), -1)
    for gesture, cx in LANE_X_POSITIONS.items():
        color = GESTURE_COLOR[gesture]
        lx1, lx2 = cx - lw // 2, cx + lw // 2
        draw_glow_rect(img, lx1, JUDGE_Y - 4, lx2, JUDGE_Y + 4, color)
    for gesture, cx in LANE_X_POSITIONS.items():
        draw_icon_fn(img, gesture, cx, LANE_BOTTOM + 40, 22)
        label = GESTURE_KR[gesture]
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(img, label, (cx - ts[0] // 2, LANE_BOTTOM + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GESTURE_COLOR[gesture], 1)


def draw_lane_notes(img, game, now, draw_icon_fn):
    lw = LANE_W
    bar_h = 36
    corner_r = 8
    for note in game.lane_notes:
        cx = LANE_X_POSITIONS[note['gesture']]
        x1, x2 = cx - lw // 2 + 6, cx + lw // 2 - 6
        nc = NOTE_TYPE_COLORS[note['type']]
        if note['state'] == 'falling':
            ny = int(note['y'])
            y1c = ny - bar_h // 2
            y2c = ny + bar_h // 2
            # 판정선 근처 밝기 증가
            dist = abs(ny - JUDGE_Y)
            if dist <= JUDGE_GOOD_PX:
                bright = 1.0 + (1.0 - dist / JUDGE_GOOD_PX) * 0.6
                nc_draw = tuple(min(255, int(c * bright)) for c in nc)
            else:
                nc_draw = nc
            # rounded filled rectangle
            cv2.rectangle(img, (x1 + corner_r, y1c), (x2 - corner_r, y2c), nc_draw, -1)
            cv2.rectangle(img, (x1, y1c + corner_r), (x2, y2c - corner_r), nc_draw, -1)
            cv2.circle(img, (x1 + corner_r, y1c + corner_r), corner_r, nc_draw, -1)
            cv2.circle(img, (x2 - corner_r, y1c + corner_r), corner_r, nc_draw, -1)
            cv2.circle(img, (x1 + corner_r, y2c - corner_r), corner_r, nc_draw, -1)
            cv2.circle(img, (x2 - corner_r, y2c - corner_r), corner_r, nc_draw, -1)
            # icon inside note
            draw_icon_fn(img, note['gesture'], cx, ny, 12)
            # type label above
            label = NOTE_TYPE_LABELS.get(note['type'], '')
            if label:
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                cv2.putText(img, label, (cx - ts[0] // 2, y1c - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, nc, 1)
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


def draw_rhythm_hud(img, game, heart_cache, cache_heart_fn):
    overlay = img[0:55, :].copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, 55), (15, 15, 25), -1)
    img[0:55, :] = cv2.addWeighted(overlay, 0.85, img[0:55, :], 0.15, 0)
    diff_color = RHYTHM_DIFFICULTY[game.rhythm_difficulty]['color']
    cv2.line(img, (0, 55), (SCREEN_W, 55), diff_color, 1)

    _, _, bpm, stage_name, _, _ = get_rhythm_stage(game)
    diff_name = RHYTHM_DIFFICULTY[game.rhythm_difficulty]['name']
    cv2.putText(img, f"{diff_name} - {stage_name}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, diff_color, 2)
    cv2.putText(img, f"BPM {bpm}  |  Stage {game.stage_idx + 1}/{rhythm_stages_count(game)}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    score_text = f"{game.score}"
    ts = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(img, score_text, (SCREEN_W // 2 - ts[0] // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    label_ts = cv2.getTextSize("SCORE", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    cv2.putText(img, "SCORE", (SCREEN_W // 2 - label_ts[0] // 2, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

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

    prog_text = f"{game.judged_count}/{game.total_notes}"
    pts = cv2.getTextSize(prog_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(img, prog_text, (SCREEN_W - pts[0] - 10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    for i in range(5):
        hx = SCREEN_W - 28 * (5 - i) - 5
        heart_size = 22
        cached = heart_cache.get(heart_size)
        if cached is None:
            cache_heart_fn(heart_size)
            cached = heart_cache.get(heart_size)
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


def draw_stage_select_screen(img, game):
    cv2.rectangle(img, (0, 0), (SCREEN_W, SCREEN_H), (15, 15, 30), -1)
    title = "SELECT DIFFICULTY"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
    cv2.putText(img, title, (SCREEN_W // 2 - ts[0] // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 100, 255), 3)
    sub = "UP/DOWN to select, SPACE to start, B to go back"
    sts = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(img, sub, (SCREEN_W // 2 - sts[0] // 2, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    card_w = 580
    card_h = 130
    card_x = SCREEN_W // 2 - card_w // 2
    for i, diff in enumerate(RHYTHM_DIFFICULTY):
        is_sel = (i == game.selected_difficulty)
        cy = 150 + i * (card_h + 15)
        bg = (30, 40, 55) if is_sel else (18, 18, 30)
        cv2.rectangle(img, (card_x, cy), (card_x + card_w, cy + card_h), bg, -1)
        if is_sel:
            cv2.rectangle(img, (card_x, cy), (card_x + card_w, cy + card_h),
                          diff['color'], 2)
            cv2.circle(img, (card_x - 15, cy + card_h // 2), 6, diff['color'], -1)
        name_col = diff['color'] if is_sel else tuple(int(c * 0.5) for c in diff['color'])
        cv2.putText(img, diff['name'], (card_x + 20, cy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, name_col, 2)
        n_stars = i + 1
        for s in range(3):
            sx = card_x + 200 + s * 25
            star_col = diff['color'] if (s < n_stars and is_sel) else (50, 50, 50)
            cv2.putText(img, "*", (sx, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, star_col, 2)
        desc_col = (180, 180, 180) if is_sel else (80, 80, 80)
        cv2.putText(img, diff['desc'], (card_x + 20, cy + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, desc_col, 1)
        spd = DIFFICULTY_NOTE_SPEED.get(i, 180)
        stages = diff['stages']
        bpm_min = min(s[2] for s in stages)
        bpm_max = max(s[2] for s in stages)
        info = f"{len(stages)} stages  |  BPM {bpm_min}-{bpm_max}  |  Speed {spd}"
        info_col = (140, 140, 140) if is_sel else (60, 60, 60)
        cv2.putText(img, info, (card_x + 20, cy + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, info_col, 1)
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
    blink = int(time.time() * 2) % 2
    if blink:
        prompt = "SPACE to Start"
        pts = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(img, prompt, (SCREEN_W // 2 - pts[0] // 2, 595),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


# ══════════════════════════════════════════════
#  메인 틱
# ══════════════════════════════════════════════
def rhythm_play_tick(screen, frame, game, now, stable_g,
                     draw_icon_fn, heart_cache, cache_heart_fn):
    dt = now - game._prev_t if game._prev_t > 0 else 0.016
    game._prev_t = now
    if game.note_queue and now >= game.next_spawn_t:
        ntype, gesture = game.note_queue.pop(0)
        spawn_lane_note(game, ntype, gesture)
        _, interval, _, _, _, _ = get_rhythm_stage(game)
        game.next_spawn_t = now + interval
    update_lane_notes(game, dt, game.last_detection, now)
    draw_rhythm_hud(screen, game, heart_cache, cache_heart_fn)
    draw_lane_background(screen, now, draw_icon_fn)
    draw_lane_notes(screen, game, now, draw_icon_fn)
    update_draw_particles(screen, game, dt)
    cam_w, cam_h = 160, 120
    cam_small = cv2.resize(frame, (cam_w, cam_h))
    sx, sy = 8, SCREEN_H - cam_h - 8
    screen[sy:sy + cam_h, sx:sx + cam_w] = cam_small
    cv2.rectangle(screen, (sx - 1, sy - 1),
                  (sx + cam_w + 1, sy + cam_h + 1), (60, 60, 60), 1)
    if stable_g:
        gc = GESTURE_COLOR.get(stable_g, (200, 200, 200))
        cv2.rectangle(screen, (sx - 2, sy - 2),
                      (sx + cam_w + 2, sy + cam_h + 2), gc, 2)
        draw_icon_fn(screen, stable_g, sx + cam_w + 28, sy + cam_h // 2, 20)
    if (game.judged_count >= game.total_notes
            and not game.note_queue and not game.lane_notes):
        game.state = game.STAGE_CLEAR
    elif game.lives <= 0:
        game.state = game.GAME_OVER
