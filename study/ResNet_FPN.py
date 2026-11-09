
import tensorflow as tf
import numpy as np

from keras.applications import ResNet101
from keras.layers import (
    Conv2D,
    UpSampling2D,
    Add,
    MaxPooling2D,
    Input
)
from tensorflow.keras.models import Model


# =========================================================
# 1. 환경 설정
# =========================================================

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ResNet101 + FPN (Keras)")
print("=" * 60)


# =========================================================
# 2. 32 x 32 랜덤 정수 행렬 생성
# =========================================================

random_matrix = np.random.randint(
    0, 256,
    size=(32, 32),
    dtype=np.int32
)

print("\n[Input Matrix]")
print("Shape:", random_matrix.shape)
print(random_matrix)

# Grayscale 행렬을 RGB 3채널로 변환
input_image = np.stack(
    [random_matrix] * 3,
    axis=-1
)

# Batch 차원 추가
input_image = np.expand_dims(
    input_image, axis=0
)

# Float32 변환 및 정규화
input_image = input_image.astype(np.float32) / 255.0

print("\n[Input Tensor]")
print("Shape:", input_image.shape)
# (1, 32, 32, 3)


# =========================================================
# 3. ResNet101 Backbone
# =========================================================

resnet = ResNet101(
    include_top=False,
    weights=None,
    input_shape=(32, 32, 3)
)

# ResNet의 각 Stage 출력 Layer 지정
resnet_outputs = [
    resnet.get_layer("conv2_block3_out").output,
    resnet.get_layer("conv3_block4_out").output,
    resnet.get_layer("conv4_block23_out").output,
    resnet.get_layer("conv5_block3_out").output
]

backbone = Model(
    inputs=resnet.input,
    outputs=resnet_outputs,
    name="ResNet101_Backbone"
)

# C2, C3, C4, C5 추출
C2, C3, C4, C5 = backbone(
    input_image,
    training=False
)

print("\n" + "=" * 60)
print("[ResNet101 Outputs]")
print("=" * 60)

for name, feature in zip(
    ["C2", "C3", "C4", "C5"],
    [C2, C3, C4, C5]
):
    print(f"{name}: {feature.shape}")


# =========================================================
# 4. FPN 구현
# =========================================================

def build_fpn(C2, C3, C4, C5):

    print("\n" + "=" * 60)
    print("[FPN - Lateral Connections]")
    print("=" * 60)

    # -----------------------------------------------------
    # 1) 1x1 Convolution
    # 모든 Feature Map의 채널을 256으로 통일
    # -----------------------------------------------------

    L2 = Conv2D(
        256, kernel_size=1,
        name="lateral_conv2"
    )(C2)

    L3 = Conv2D(
        256, kernel_size=1,
        name="lateral_conv3"
    )(C3)

    L4 = Conv2D(
        256, kernel_size=1,
        name="lateral_conv4"
    )(C4)

    L5 = Conv2D(
        256, kernel_size=1,
        name="lateral_conv5"
    )(C5)

    for name, feature in zip(
        ["L2", "L3", "L4", "L5"],
        [L2, L3, L4, L5]
    ):
        print(f"{name}: {feature.shape}")

    # -----------------------------------------------------
    # 2) Top-down Pathway
    # Upsampling + Element-wise Addition
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("[FPN - Top-down + Addition]")
    print("=" * 60)

    # M5
    M5 = L5

    # M5 -> Upsampling -> L4와 합산
    up5 = UpSampling2D(
        size=(2, 2),
        interpolation="nearest"
    )(M5)

    M4 = Add()([L4, up5])

    print("Upsampled M5:", up5.shape)
    print("M4:", M4.shape)

    # M4 -> Upsampling -> L3와 합산
    up4 = UpSampling2D(
        size=(2, 2),
        interpolation="nearest"
    )(M4)

    M3 = Add()([L3, up4])

    print("Upsampled M4:", up4.shape)
    print("M3:", M3.shape)

    # M3 -> Upsampling -> L2와 합산
    up3 = UpSampling2D(
        size=(2, 2),
        interpolation="nearest"
    )(M3)

    M2 = Add()([L2, up3])

    print("Upsampled M3:", up3.shape)
    print("M2:", M2.shape)

    # -----------------------------------------------------
    # 3) 3x3 Smoothing Convolution
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("[FPN - 3x3 Convolution]")
    print("=" * 60)

    P2 = Conv2D(
        256, kernel_size=3,
        padding="same",
        name="smooth_p2"
    )(M2)

    P3 = Conv2D(
        256, kernel_size=3,
        padding="same",
        name="smooth_p3"
    )(M3)

    P4 = Conv2D(
        256, kernel_size=3,
        padding="same",
        name="smooth_p4"
    )(M4)

    P5 = Conv2D(
        256, kernel_size=3,
        padding="same",
        name="smooth_p5"
    )(M5)

    # -----------------------------------------------------
    # 4) P6 생성
    # -----------------------------------------------------

    P6 = MaxPooling2D(
        pool_size=1,
        strides=2,
        padding="valid",
        name="p6_pool"
    )(P5)

    return {
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "P6": P6
    }


# =========================================================
# 5. FPN 실행
# =========================================================

fpn_outputs = build_fpn(C2, C3, C4, C5)


# =========================================================
# 6. 최종 Feature Map 출력
# =========================================================

print("\n" + "=" * 60)
print("[Final FPN Outputs]")
print("=" * 60)

for name, feature in fpn_outputs.items():

    print(f"\n{name}")
    print("Shape:", feature.shape)

    feature_np = feature.numpy()

    print("Min:", np.min(feature_np))
    print("Max:", np.max(feature_np))
    print("Mean:", np.mean(feature_np))

    # 첫 번째 채널의 Feature Map 출력
    print("\nFirst Channel:")
    print(feature_np[0, :, :, 0])


print("\nExecution Completed!")
