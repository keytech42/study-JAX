---
id: "001"
title: "JAX Shape 불일치 및 언패킹 에러 (Shape Mismatch & Silent Unpacking)"
difficulty: "Intermediate"
noise_level: "High"
tags: ["JAX", "shape", "broadcasting", "zip", "value_and_grad", "unpacking"]
created_at: "2026-05-27"
approved: true
---

# Debugging Problem: JAX Shape 불일치 및 언패킹 에러

## Context

이 프로젝트는 순수 JAX를 이용하여 MNIST 데이터셋을 훈련하는 다층 퍼셉트론(MLP) 모델입니다.
`init_network_params` 함수를 통해 `params`는 3개의 레이어를 가진 리스트 형태 `[(w1, b1), (w2, b2), (w3, b3)]`로 초기화되어 관리됩니다.
그런데 훈련 루프(Training Loop)를 실행하면 예상치 못한 에러들이 연속해서 발생하며 모델 학습이 중단됩니다.

## Goal

현재 이 코드에는 두 가지 치명적인 버그가 연쇄적으로 발생하도록 방치되어 있습니다.

1. **첫 번째 에러 (Shape Mismatch):** 경사하강법(Gradient Descent) 업데이트를 수행할 때 JAX 내부의 XLA 컴파일러가 차원(Shape)이 맞지 않는다며 에러를 발생시킵니다.
2. **두 번째 에러 (Unpacking Error):** 첫 번째 에러를 어찌어찌 고치고 나면, 전혀 엉뚱한 예측(`predict`) 함수 내부 혹은 파라미터 반환 과정에서 `too many values to unpack`이라는 파이썬 에러가 터집니다.

사고 실험을 하거나 디버거를 활용하여 두 가지 에러의 근본 원인을 파악하고, 코드가 정상적으로 학습 루프를 돌 수 있도록 코드를 수정하세요.

---

## Mentor's Guidance (Hint)

이 버그들은 실무에서 JAX를 다룰 때 가장 빈번하게 마주치는 구조적 실수입니다. 코드를 무작정 뜯어고치기 전에 아래 힌트들을 열어보며 스스로 사고를 전개해 보세요.

### Phase 1: Shape Mismatch 해결

<details>
<summary>💡 Hint 1: 스택 트레이스는 바텀-업(Bottom-up)으로 읽어라</summary>

JAX의 에러 로그는 매우 깁니다. 하지만 가장 중요한 정보는 항상 마지막 줄 근처에 있습니다.
에러 메시지에서 **`TypeError: unsupported operand type(s)`** 혹은 **Shape 불일치**를 지적하는 부분을 찾으셨나요? 
무엇과 무엇의 차원이 맞지 않는다고 불평하고 있나요?
</details>

<details>
<summary>💡 Hint 2: zip()의 언패킹 구조를 유심히 살펴보라</summary>

에러가 발생한 곳은 `update` 함수 안의 리스트 컴프리헨션입니다.
```python
[(w - lr * dw, b - lr * db) for (w, dw), (b, db) in zip(params, grads)]
```
`params` 리스트 안의 원소 하나는 `(w, b)` 형태입니다. 
그렇다면 `zip(params, grads)`가 한 번 순회할 때 던져주는 튜플은 어떤 형태일까요? 
`(w, dw)`에 과연 `w`와 `dw`가 제대로 들어갔을까요, 아니면 `(w, b)` 전체가 통째로 `w`에 들어갔을까요?
</details>

### Phase 2: Silent Unpacking Error 해결

첫 번째 문제를 해결했다면 훈련이 진행되는 듯하다가 갑자기 예측 루프 혹은 변수 할당에서 폭탄이 터질 것입니다.

<details>
<summary>💡 Hint 3: "침묵하는 버그(Silent Bug)"를 경계하라</summary>

에러는 두 번째 에포크나 다음 배치 스텝의 `batched_predict` 혹은 `params, loss_value = update(...)`에서 발생할 것입니다.
하지만 **폭탄은 그 이전 스텝에서 조용히 설치되었습니다.**
IDE의 디버거를 켜고 `params, loss_value = update(...)` 줄에 브레이크포인트를 걸어보세요.
이 줄을 실행한 직후, `params` 변수 안에는 과연 무엇이 들어있나요? 당신이 기대한 리스트 구조가 맞나요?
</details>

<details>
<summary>💡 Hint 4: JAX의 함수형 설계 철학</summary>

당신은 `update` 함수에서 `grads = grad(loss)(params, x, y)`를 통해 기울기만 계산하고 있습니다.
그런데 밖에서는 `params, loss_value = update(...)`로 손실값(loss)까지 언패킹하려고 합니다. 
현재 `update` 함수는 손실값을 리턴하고 있나요? 
JAX에서 값(value)과 기울기(grad)를 동시에 반환하는 아주 유용한 함수가 있습니다. 공식 문서를 찾아보세요!
</details>

---

## ✅ Solution (Spoiler)

<details>
<summary>해설 및 정답 코드 보기</summary>

이 문제는 JAX의 상태 비저장(Stateless) 특성과 파이썬의 언패킹 구조를 명확히 이해해야 풀 수 있습니다.

### 1. Shape Mismatch (zip 언패킹 오류)
`zip(params, grads)`는 각 순회마다 `((w, b), (dw, db))` 꼴의 튜플을 반환합니다.
하지만 작성된 코드는 `for (w, dw), (b, db)`로 언패킹을 시도했습니다. 이로 인해 파이썬은 구조를 강제로 맞추려다 엉뚱한 구조로 할당해 버렸고, 결과적으로 연산 시 차원이 꼬여버린 것입니다.
**수정:** `for (w, b), (dw, db) in zip(params, grads)`

### 2. Silent Unpacking Error (`value_and_grad`)
`update` 함수는 오직 파라미터 리스트만 반환하고 있었습니다. 그런데 호출부에서는 `params, loss_value = update(...)`로 두 개의 변수에 언패킹을 시도했습니다.
파이썬은 에러를 뿜는 대신, 반환된 길이 3짜리(레이어가 3개이므로) 리스트를 억지로 언패킹하려 들었고, 예상치 못한 구조 변형(State Corruption)을 일으킨 채 침묵했습니다. 이 오염된 상태가 다음 루프에 전달되어 엉뚱한 곳에서 에러가 터진 것입니다.

**수정:** JAX의 `value_and_grad`를 사용하여 손실값도 함께 반환하도록 `update` 함수를 수정해야 합니다.

```python
from jax import value_and_grad

def update(params: list[tuple], x, y, epoch_number: int) -> tuple[list[tuple], float]:
    loss_value, grads = value_and_grad(loss)(params, x, y)
    lr = INIT_LR * DECAY_RATE ** (epoch_number // DECAY_STEPS)
    updated_params = [
        (w - lr * dw, b - lr * db) 
        for (w, b), (dw, db) in zip(params, grads)
    ]
    return updated_params, loss_value
```
</details>
