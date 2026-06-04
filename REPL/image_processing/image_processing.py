# %% [markdown]
# 1. Load a photo file into a tensor in memory, by NumPy array.
# 2. Preprocess the image tensor.
# 3. Simulate a noisy version of the image.
# 4. Filter the image to reduce noise.
# 5. Save the processed image into a file.


# %% Cell 1
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage.io import imread, imsave
from skimage.util import img_as_float32, img_as_ubyte, random_noise

img = imread("data/photo.jpeg")

plt.figure(figsize=(6, 10))
plt.imshow(img)


# %% {"tags": ["remove-cell"]}
# fmt: off

# %% Cell 2
print(type(img))  # img 객체 자체의 파이썬 클래스 타입
print(img.ndim)   # 텐서의 축(axis) 개수 (수학적 의미의 텐서 랭크)
print(img.shape)  # 텐서의 형상(shape) (각 축에 배열된 원소의 개수를 나타내는 튜플)


# %% [markdown]
# 위에서 일부로 "차원(dimension)"이라는 표현을 사용하지 않았다.  \
# '텐서의 차원'을 '텐서의 랭크(축의 개수)'의 의미로 보통 많이 사용하지만,
# 이후 차원이라는 개념이 **'텐서가 가진 축의 개수'**(예: 3차원 배열)와
# **'특정 벡터 공간의 크기'**(예: 100차원 특징 벡터) 사이에서 혼용되며 의미가 충돌할 위험이 있기 때문이다.  \
# 따라서 축의 개수는 '랭크(Rank)'로, 각 축의 길이의 집합은 '형상(Shape)'으로 엄밀하게 표현을 분리해둔다.


# %% Cell 3
print(img.dtype)   # uint8 = 1 byte
print(img.size)    # 텐서 내 전체 원소 개수
print(img.nbytes)  # 텐서 내 전체 원소가 메모리에서 차지하는 바이트 수
# %% {"tags": ["remove-cell"]}
# fmt: on


# %% [markdown]
# ## 2 단계: 이미지 전처리


# %% Cell 4
plant_bright = img[545:875, 250:550, 1]
print(plant_bright.shape)

plt.imshow(plant_bright, cmap="gray")


# %% Cell 5
print(img.dtype)
img = img_as_float32(img)
print(img.dtype)


# %% [markdown]
# ## 3 단계: 노이즈 추가


# %% Cell 6
img_noisy = random_noise(img, mode="gaussian")

plt.figure(figsize=(6, 10))
plt.imshow(img_noisy)


# %% Cell 7
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 20)
crop_info = (slice(545, 875), slice(250, 550))
for i in range(2):
    ax[i].axis("off")
    ax[i].set_title("Noisy" if i == 1 else "Original")
    ax[i].imshow(img_noisy[crop_info] if i == 1 else img[crop_info])


# %% [markdown]
# ## 4 단계: 필터 커널 적용


# %% Cell 8
kernel_blur = np.ones((5, 5))
kernel_blur /= np.sum(kernel_blur)
print(kernel_blur)


# %% Cell 9
