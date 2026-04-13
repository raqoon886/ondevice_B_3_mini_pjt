"""
RPS Rhythm - 학습 + 데이터 증강 + 양자화 스크립트
MobileNetV2 전이학습으로 가위/바위/보 분류 모델을 만들고,
PTQ(INT8), QAT 양자화를 적용합니다.

사용법:
  python train_model.py

출력:
  models/rps_mobilenetv2.tflite              (Float32 기본)
  models/rps_mobilenetv2_ptq_int8.tflite     (PTQ INT8)
  models/rps_mobilenetv2_qat.tflite          (QAT INT8)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── 경로 설정 ──
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 하이퍼파라미터 ──
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_TRANSFER = 10   # 전이학습 (헤드만)
EPOCHS_FINETUNE = 5    # 미세조정 (전체)
EPOCHS_QAT = 3         # QAT 추가 학습
NUM_CLASSES = 3
CLASS_NAMES = ['scissors', 'rock', 'paper']


def check_data():
    """데이터 존재 여부 확인"""
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"[오류] '{cls_dir}' 폴더가 없습니다. collect_data.py로 먼저 데이터를 수집하세요.")
            return False
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))])
        print(f"  {cls}: {count}장")
        if count < 10:
            print(f"  [경고] {cls} 데이터가 10장 미만입니다. 최소 50장 이상 권장.")
    return True


def create_data_generators():
    """데이터 증강이 적용된 학습/검증 데이터 생성기"""

    # 학습용: 다양한 증강 기법 적용
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,           # ±20도 회전
        width_shift_range=0.15,      # 좌우 이동
        height_shift_range=0.15,     # 상하 이동
        shear_range=0.1,             # 전단 변환
        zoom_range=0.2,              # 확대/축소
        horizontal_flip=True,        # 좌우 반전
        brightness_range=[0.7, 1.3], # 밝기 변화
        fill_mode='nearest',
        validation_split=0.2         # 20% 검증용
    )

    # 검증용: 정규화만 적용
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation',
        shuffle=False
    )

    print(f"\n학습 데이터: {train_gen.samples}장 (증강 적용)")
    print(f"검증 데이터: {val_gen.samples}장")
    print(f"클래스 매핑: {train_gen.class_indices}")
    return train_gen, val_gen


def build_model():
    """MobileNetV2 기반 전이학습 모델 구축"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # 기본 모델 동결

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model, base_model


def train_transfer(model, train_gen, val_gen):
    """1단계: 전이학습 (헤드만 학습)"""
    print("\n" + "=" * 50)
    print("1단계: 전이학습 (분류 헤드만 학습)")
    print("=" * 50)

    history = model.fit(
        train_gen,
        epochs=EPOCHS_TRANSFER,
        validation_data=val_gen,
        verbose=1
    )
    return history


def train_finetune(model, base_model, train_gen, val_gen):
    """2단계: 미세조정 (전체 모델 학습, 낮은 학습률)"""
    print("\n" + "=" * 50)
    print("2단계: 미세조정 (전체 모델, lr=1e-5)")
    print("=" * 50)

    base_model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        verbose=1
    )
    return history


def convert_to_tflite(model, name):
    """Keras 모델을 TFLite (Float32)로 변환"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(path) / 1024
    print(f"  저장: {path} ({size_kb:.1f} KB)")
    return path


def convert_ptq_int8(model, val_gen, name):
    """Post-Training Quantization (INT8) 변환"""
    print("\n--- PTQ INT8 양자화 ---")

    def representative_dataset():
        for images, _ in val_gen:
            for img in images:
                yield [np.expand_dims(img.astype(np.float32), axis=0)]
            break  # 하나의 배치만 사용

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(path) / 1024
    print(f"  저장: {path} ({size_kb:.1f} KB)")
    return path


def convert_qat(model, train_gen, val_gen, name):
    """Quantization-Aware Training (QAT)"""
    print("\n--- QAT 양자화 인식 학습 ---")

    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("  [경고] tensorflow_model_optimization이 없습니다. QAT를 건너뜁니다.")
        print("  설치: pip install tensorflow-model-optimization")
        return None

    # QAT 적용
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    qat_model.fit(
        train_gen,
        epochs=EPOCHS_QAT,
        validation_data=val_gen,
        verbose=1
    )

    # TFLite 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(path) / 1024
    print(f"  저장: {path} ({size_kb:.1f} KB)")
    return path


def print_summary(paths):
    """모델 비교 요약"""
    print("\n" + "=" * 60)
    print("모델 비교 요약")
    print("=" * 60)
    print(f"{'모델':<40} {'크기(KB)':>10}")
    print("-" * 52)
    for label, path in paths:
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"{label:<40} {size_kb:>10.1f}")
    print("=" * 60)


def main():
    print("=== RPS Rhythm - 모델 학습 ===\n")

    # 데이터 확인
    if not check_data():
        return

    # 데이터 생성기 (증강 포함)
    train_gen, val_gen = create_data_generators()

    # 모델 구축
    model, base_model = build_model()

    # 1단계: 전이학습
    train_transfer(model, train_gen, val_gen)

    # 2단계: 미세조정
    train_finetune(model, base_model, train_gen, val_gen)

    # 최종 평가
    print("\n--- 최종 평가 ---")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"  검증 정확도: {acc:.4f}")
    print(f"  검증 손실:   {loss:.4f}")

    # TFLite 변환
    paths = []

    print("\n--- Float32 변환 ---")
    p1 = convert_to_tflite(model, 'rps_mobilenetv2.tflite')
    paths.append(('Float32 (기본)', p1))

    # PTQ INT8
    p2 = convert_ptq_int8(model, val_gen, 'rps_mobilenetv2_ptq_int8.tflite')
    paths.append(('PTQ INT8', p2))

    # QAT
    p3 = convert_qat(model, train_gen, val_gen, 'rps_mobilenetv2_qat.tflite')
    paths.append(('QAT INT8', p3))

    # 요약
    print_summary(paths)
    print("\n완료! models/ 폴더에서 tflite 파일을 확인하세요.")
    print("게임 실행: python rps_rhythm.py")


if __name__ == '__main__':
    main()
