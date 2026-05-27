# JAX & Haiku Deep Dive: Layer Normalization

이 문서는 트랜스포머 아키텍처의 필수 구성 요소인 **Layer Normalization(계층 정규화)**에 대해 다룬다. 코드가 왜 그렇게 작성되었는지(`axis=-1`, `scale`, `offset`)를 중심으로 멘탈 모델을 정립한다.

---

## 1. 정규화(Normalization)는 왜 필요한가?

딥러닝 모델, 특히 트랜스포머처럼 깊은 신경망을 학습시킬 때 **내부 공변량 변화(Internal Covariate Shift)**라는 문제가 발생한다.
입력 데이터가 레이어를 거칠 때마다 값들의 분포(분산과 평균)가 계속해서 널뛰기를 하게 되고, 이로 인해 그래디언트(기울기)가 폭발하거나 소실되어 학습이 매우 불안정해진다.

이를 막기 위해 각 레이어의 출력값을 **평균 0, 분산 1**이 되도록 예쁘게 모아주는 작업이 바로 정규화다.

---

## 2. 왜 Batch Norm이 아니라 Layer Norm인가?

비전(Computer Vision) 분야에서는 Batch Normalization이 표준이다. 하지만 자연어 처리(NLP)를 다루는 트랜스포머에서는 철저히 **Layer Normalization**을 사용한다. 왜 그럴까?

*   **Batch Norm의 한계:** 미니배치(Mini-batch) 단위로 정규화를 수행한다. 문장(Sequence) 데이터는 문장마다 길이가 제각각이어서 패딩(Padding)이 많이 들어가고, 배치 크기가 작을 경우 통계값(평균/분산)이 심하게 요동친다.
*   **Layer Norm의 장점 (`axis=-1`):** 배치나 문장 길이에 전혀 신경 쓰지 않는다. **오직 개별 토큰(단어) 하나가 가진 피처(Features) 차원 내부에서만 정규화를 수행한다.** 
    *   코드에서 `axis=-1`이라고 명시한 이유가 바로 이것이다. 형태가 `[Batch, Seq, Features]`일 때, 가장 마지막 차원(`Features`)에 대해서만 평균과 분산을 구하겠다는 아주 명확한 선언이다.

---

## 3. Scale($\gamma$)과 Offset($\beta$) 파라미터

```python
    return hk.LayerNorm(
        axis=-1,
        create_scale=True,   # Gamma (곱하기)
        create_offset=True,  # Beta (더하기)
        name=name,
    )(x)
```

정규화를 거치면 데이터는 무조건 "평균 0, 분산 1"의 종 모양 분포(정규 분포)를 가지게 된다.
하지만 이것은 딥러닝 모델 입장에서 **"데이터가 가진 원래의 뉘앙스(특징의 크기나 위치)를 강제로 잃어버리는 것"**일 수도 있다.

그래서 정규화를 끝낸 데이터에 학습 가능한 두 파라미터를 제공한다.
$$ y = \gamma \times \hat{x} + \beta $$

*   **`create_scale=True` ($\gamma$, 감마):** 분산을 네트워크가 원하는 만큼 다시 늘리거나 줄일 수 있게 해준다.
*   **`create_offset=True` ($\beta$, 베타):** 평균의 위치를 네트워크가 원하는 곳으로 다시 옮길 수 있게 해준다.

네트워크는 오차 역전파(Backpropagation)를 통해 "정규화된 상태 그대로 두는 것이 좋은지, 아니면 분포를 살짝 비트는 것이 좋은지" 스스로 $\gamma$와 $\beta$값을 학습하여 최적의 상태를 찾는다. Haiku에서 이 두 옵션을 켜두면, 내부적으로 이 두 파라미터가 자동으로 생성되어 관리된다.
