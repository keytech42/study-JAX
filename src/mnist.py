# ruff: noqa: E402

# %% [1]
# from rich.traceback import install
# install(show_locals=False)

# %% [markdown]
# # JAX 기반 MNIST 다층 퍼셉트론 구현
#
# 이 문서는 순수 `.py` 파일로 작성되었으나, Jupytext 플러그인을 통해 **Jupyter Notebook** 형식으로 변환되어 실행된다.
# TensorFlow Datasets를 활용하여 MNIST 데이터를 불러오고 전처리한 뒤, JAX를 이용해 간단한 다층 퍼셉트론(MLP)의 가중치를 초기화하는 과정을 단계별로 설명한다.

# %% [markdown]
# ## 1. 라이브러리 임포트 및 데이터 로드
# 모델 학습에 필요한 데이터셋을 불러오기 위해 `tensorflow_datasets`를 사용한다.
# TensorFlow의 로깅 시스템이 출력하는 불필요한 C++ 및 Python 레벨의 시스템 경고를 사전에 차단하여 문서의 가독성을 높인다.

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # C++ 레벨의 Warning 억제

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Python 레벨의 Warning 억제
import tensorflow_datasets as tfds

data_dir = "/tmp/tfds"

data: dict[str, tf.data.Dataset]
info: tfds.core.DatasetInfo

data, info = tfds.load(  # type: ignore
    name="mnist",
    data_dir=data_dir,
    as_supervised=True,
    with_info=True,
)

data_train = data["train"]
data_test = data["test"]

# %% [markdown]
# ## 2. 데이터 시각화
# 다운로드한 MNIST 데이터의 형태와 클래스 라벨을 시각적으로 확인한다.
#
# `data_train.take()`를 순회할 때 발생하는 TensorFlow의 데이터셋 캐시 잘림 경고(Cache Truncation Warning)를 방지하기 위해, `.batch()`를 활용하여 필요한 만큼의 이미지를 한 번에 묶어서 메모리로 가져온다. Jupytext 환경에서는 코드 블록 끝에 `plt.show()`를 명시하지 않아도 생성된 Matplotlib 객체가 문서에 자동으로 렌더링된다.

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (10, 5)

ROWS = 3
COLS = 10

# 3행 10열의 서브플롯을 생성하여 샘플 이미지를 출력한다.
fig, ax = plt.subplots(ROWS, COLS)

# take() 반복 시 발생하는 TF 캐시 경고를 방지하기 위해 batch()로 한 번에 가져온다
images, labels = next(iter(data_train.batch(ROWS * COLS)))

for i in range(ROWS * COLS):
    ax[int(i / COLS), i % COLS].axis("off")
    ax[int(i / COLS), i % COLS].set_title(str(labels[i].numpy()))
    ax[int(i / COLS), i % COLS].imshow(np.reshape(images[i], (28, 28)), cmap="gray")

# %% [markdown]
# ## 3. 데이터 전처리 (Preprocessing)
# 신경망 모델이 데이터를 원활하게 학습할 수 있도록,
# 0~255 범위의 uint8 픽셀 값을 0.0~1.0 사이의 float32 타입으로 정규화(Normalization)한다.
#
# 데이터 파이프라인 구성 시 `prefetch(1)` 옵션을 적용한다.  \
# 이는 가속기(GPU/TPU)가 현재 배치를 연산하는 동안,
# CPU가 다음 배치의 데이터를 백그라운드에서 미리 준비(I/O Overlapping)하도록 지시하여
# 전체 학습 속도의 병목 현상을 줄이는 핵심 최적화 기법이다.

# %%
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
NUM_LABELS = info.features["label"].num_classes  # type: ignore


def preprocess(img, label):
    """
    이미지 픽셀을 [0, 1] 범위로 정규화한다.
    """
    return (tf.cast(img, tf.float32) / 255.0), label


train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
test_data = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))

# %% [markdown]
# ## 4. 네트워크 가중치 초기화 (JAX PRNG)
# JAX는 난수 생성 시 전역 상태(Global State)를 가지지 않는
# 결정론적 PRNG(Pseudo-Random Number Generator) 방식을 사용한다.  \
# `random.split` 함수를 통해 하나의 부모 키(Key)로부터 각 레이어마다
# 고유하고 독립적인 하위 키를 파생시키며, 이를 바탕으로 다층 퍼셉트론(MLP)의
# 가중치 행렬(Weight)과 편향 벡터(Bias)를 정규 분포로부터 안전하게 초기화한다.

# %%
from jax import Array, random

LAYER_SIZES = [28 * 28, 512, 10]
PARAM_SCALE = 0.01


def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2):
    """
    다층 퍼셉트론(MLP)의 레이어별 가중치(Weight)와 편향(Bias)을 무작위로 초기화한다.
    """

    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


# 초기화된 파라미터 구조 확인
params: list[tuple[Array, Array]] = init_network_params(
    LAYER_SIZES, random.PRNGKey(0), scale=PARAM_SCALE
)
print("네트워크 파라미터 구조:")
for i, (w, b) in enumerate(params):
    print(f"Layer {i} - Weight shape: {w.shape}, Bias shape: {b.shape}")


# %%
import jax.numpy as jnp
from jax.nn import swish


def predict(
    params: list[tuple[Array, Array]],  # list of layers(= `(weight, bias)` tuples)
    image: Array,
) -> Array:
    """Function for per-example predictions."""
    activations = image
    for w, b in params[:-1]:  # 출력츨 제외
        outputs = (
            jnp.dot(w, activations) + b
        )  # (1)! 가중치 행렬 곱 + 편향: 선형 변환 수행
        activations = swish(
            outputs
        )  # (2)! Swish 활성화 함수: 비선형성 도입 (ReLU보다 부드러움)

    final_w, final_b = params[-1]
    logits = (
        jnp.dot(final_w, activations) + final_b
    )  # (3)! 마지막 레이어는 활성화 함수 없이 로짓 출력
    return logits


# %%
random_flattened_image = random.normal(random.PRNGKey(1), (28 * 28 * 1,))
preds = predict(params, random_flattened_image)
print(preds.shape)
# %% [markdown]
# 위와 동일한 코드. 딱 하나만 바꿨다. 배치(batch) 형태의 구성.  \
# 하지만 이제 문제가 발생한다.

# %%
random_flattened_images_batch = random.normal(random.PRNGKey(1), (32, 28 * 28 * 1))
try:
    preds = predict(params, random_flattened_images_batch)
    # ↪ 에러 발생 예정
except Exception as e:
    print(e)
# %% [markdown]
# 위에서 일부로 문제를 일으킨 이유는 당연히 해결을 시도하기 위함이다.  \
# 세 가지 방법이 있다.
# 1. 배치 내 이미지들을 독립적인 이미지로 각각 분해한 뒤 `predict()` 함수에 전달  \
#    → <u>연산 자원을 효율적으로 사용 X</u>
# 2. 수동으로 `predict()` 함수 벡터화  \
#    → <u>함수를 벡터화함으로써 배치 데이터를 받아들일 수 있게 함수 코드 재작성</u>
# 3. **"자동 벡터화"**

# %%
from jax import vmap

batched_predict = vmap(predict, in_axes=(None, 0))
#                                  # (0)! 이건 ...

# %%
batched_preds = batched_predict(params, random_flattened_images_batch)
print(batched_preds.shape)

# %% [markdown]
# ## Step 5: Prepare Training - Loss Function
# 대망의 모델 학습(훈련)을 위한 단계에 진입했다. 우선은 손실함수부터 정의한다.

# %%
from jax.nn import logsumexp


def loss(params, images, targets):
    """
    Categorical Cross Entropy Loss
    """
    logits = batched_predict(params, images)
    log_preds = logits - logsumexp(logits)
    return -jnp.mean(targets * log_preds)


# %% [markdown]
# ## Step 6: Prepare Training - Obtaining Gradients
#

# %%
from jax import grad, value_and_grad

INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5


def update(
    params: list[tuple[Array, Array]],
    x,
    y,
    epoch_number: int,
) -> tuple[list[tuple[Array, Array]], float]:
    # grads = grad(loss)(params, x, y)
    loss_value, grads = value_and_grad(loss)(params, x, y)
    lr = INIT_LR * DECAY_RATE ** (epoch_number // DECAY_STEPS)
    # print(type(loss_value))
    return [
        (w - lr * dw, b - lr * db)  # (1)! 업데이트된 파라미터(가중치, 편향) 반환
        for (w, b), (dw, db) in zip(params, grads)
    ], loss_value


# %%
from jax.nn import one_hot

NUM_EPOCHS = 25


def _batch_accuracy(
    params: list[tuple],
    images,
    targets,
):
    images = jnp.reshape(images, (len(images), NUM_PIXELS))
    predicted_class = jnp.argmax(
        batched_predict(params, images),
        axis=1,  # (1)! axis=0은 배치 차원이므로
    )
    return jnp.mean(predicted_class == targets)


def accuracy(params, data):
    accs = []
    for images, targets in data:
        accs.append(_batch_accuracy(params, images, targets))
    return jnp.mean(jnp.array(accs))


import time

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    losses = []
    for x, y in train_data:
        x = jnp.reshape(x, (len(x), NUM_PIXELS))
        y = one_hot(y, NUM_LABELS)
        params, loss_value = update(params, x, y, epoch)
        losses.append(loss_value)
    epoch_time = time.time() - start_time

    start_time = time.time()
    train_acc = accuracy(params, train_data)
    test_acc = accuracy(params, test_data)
    eval_time = time.time() - start_time
    print(
        f"Epoch {epoch} in {epoch_time:.2f}s\n",
        f"Eval in {eval_time:.2f}s\n",
        f"  train loss: {jnp.mean(jnp.array(losses))}\n",
        f"  train acc: {train_acc}\n",
        f"  test acc: {test_acc}\n",
    )

# %%
print(*map(type, [images, labels]), sep="\n")
