# JAX Deep Dive: 기울기 메커니즘과 Optax 최적화 (순수 함수의 학습법)

이 문서는 트랜스포머 아키텍처의 설계를 넘어, 이를 실제로 학습시키기 위한 **JAX만의 독특한 최적화(Optimization) 철학**을 다룬다.

PyTorch의 `.backward()`와 `optimizer.step()`에 익숙한 개발자가 JAX의 `jax.grad`와 `optax` 라이브러리를 처음 접했을 때 겪는 "상태 관리의 혼란"을 해소하고, 왜 JAX가 이토록 명시적이고 파편화된(Decoupled) 방식을 고수하는지 그 기술적 이점을 해부한다.

---

## 1. 핵심 문제: 가변 상태(Mutable State)와 순수성의 충돌

딥러닝의 학습은 가중치(Weights)라는 **상태(State)**를 계속해서 업데이트하는 과정이다. 하지만 JAX의 핵심 엔진인 XLA 컴파일러는 가변 상태를 극도로 혐오하는 **순수 함수형** 구조 위에서 동작한다.

**문제의 본질:** "상태를 바꿀 수 없다(Immutable)"는 제약 조건 하에서, 어떻게 수억 개의 파라미터를 효율적으로 업데이트하고 기울기를 계산할 것인가?

---

## 2. 멘탈 모델과 비교 철학: 명령형 vs 선언적 미분

이 문제를 해결하기 위해 PyTorch와 JAX는 정반대의 길을 걷는다.

!!! question "PyTorch: 암시적 가변 상태 (Implicit Mutable State)"
    *   **방식:** 텐서(Tensor) 객체 자체가 자신의 기울기(`.grad`)를 내부에 품고 있다. `loss.backward()`를 호출하면 그래프를 거슬러 올라가며 텐서 내부의 가변 메모리를 직접 수정한다.
    *   **장점:** 매우 직관적이다. `optimizer.step()` 한 줄이면 모델 전체의 파라미터가 "마법처럼" 바뀐다.
    *   **단점:** "누가, 언제, 어디서" 파라미터를 바꿨는지 추적하기 어렵다. 특히 고차 미분(Hessian 등)을 계산하거나 병렬 처리를 할 때 상태가 꼬여버리는 버그가 발생하기 쉽다.

!!! abstract "JAX: 명시적 함수형 미분 (Explicit Functional Differentiation)"
    JAX는 파라미터를 "바꾸는(Update)" 것이 아니라, **"새로운 파라미터를 반환하는(Transform)"** 함수를 만든다.
    수학적으로 볼 때, 미분은 객체의 상태를 바꾸는 행위가 아니라 함수 $f(x)$를 $f'(x)$라는 새로운 함수로 변환하는 연산이다. JAX의 `jax.grad`는 이 수학적 본질에 철저히 충실하다.

---

## 3. 심층 기전: `jax.value_and_grad`의 마법

JAX에서 학습 루프의 핵심은 `jax.value_and_grad` 함수다.

```python
import jax

# (1)
def loss_fn(params, images, labels):
    logits = model.apply(params, images)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

# (2)
grad_fn = jax.value_and_grad(loss_fn)
loss_value, grads = grad_fn(params, x, y)
```

1. **순수 함수 (Pure Function):** `loss_fn`은 오직 입력 인자(`params`, `images`, `labels`)에만 의존하며, 외부의 어떤 상태도 건드리지 않는다.
2. **함수 변환 (Function Transformation):** `jax.value_and_grad`는 `loss_fn`이라는 함수를 인자로 받아, **"Loss 값과 그에 대한 기울기(Gradients)를 함께 반환하는 새로운 함수"**를 뱉어낸다.

!!! info "제약(Constraint) vs 관례(Convention): 왜 jax.grad 대신 value_and_grad인가?"
    `jax.grad`는 오직 기울기만 반환한다. 하지만 실제 학습 시에는 로깅(Logging)을 위해 현재 Loss 값이 얼마인지도 반드시 알아야 한다. 만약 `jax.grad`를 쓰고 따로 `loss_fn`을 호출하면, 동일한 순전파(Forward Pass) 연산을 두 번 수행하게 되어 연산 자원이 낭비된다. 따라서 **"한 번의 연산으로 값과 기울기를 모두 가져온다"**는 효율성을 위해 `value_and_grad`를 사용하는 것이 JAX 생태계의 절대적인 관례다.

---

## 4. Optax: 최적화 알고리즘의 모듈화

JAX 자체에는 `Adam`이나 `SGD` 같은 최적화 알고리즘이 포함되어 있지 않다. 대신 이를 전담하는 **Optax**라는 라이브러리를 사용한다. Optax의 핵심은 최적화 알고리즘조차 **"상태를 변환하는 순수 함수"**로 취급한다는 점이다.

### Optax의 3단계 워크플로우

1. **정의 (Definition):** `optimizer = optax.adam(learning_rate)`
   어떤 알고리즘을 쓸지 정의한다. 이 시점에서는 아무런 메모리 할당도 일어나지 않는다.
2. **초기화 (Initialization):** `opt_state = optimizer.init(params)`
   Adam 알고리즘이 필요로 하는 모멘텀(Momentum), 속도(Velocity) 등 내부 상태를 **별도의 딕셔너리(`opt_state`)**로 생성한다.
3. **업데이트 (Update):** `updates, new_opt_state = optimizer.update(grads, opt_state)`
   계산된 기울기(`grads`)와 이전 상태(`opt_state`)를 넣어, 어떻게 파라미터를 바꿀지에 대한 **"수정 사항(updates)"**과 **"다음 상태"**를 받아낸다.

!!! note "비교 분석: 왜 optimizer.step()이 없을까?"
    PyTorch의 `optimizer.step()`은 내부적으로 `params -= lr * grads`를 직접 실행(In-place Mutation)한다.
    반면 Optax의 `optimizer.update`는 단지 **"이렇게 바꾸세요"라는 제안서(Updates)**만 발행할 뿐이다. 실제 파라미터 업데이트는 개발자가 `optax.apply_updates(params, updates)`를 호출하여 **새로운 파라미터 객체**를 생성함으로써 완성된다. 

---

## 5. 최종 조립: 학습 루프 (The Update Function)

이제 모든 조각을 하나로 합쳐 JIT 컴파일이 가능한 가장 효율적인 학습 함수를 구성해 보자.

```python
@jax.jit # (1)
def update_step(params, opt_state, images, labels):
    # 1. 기울기 계산
    loss_value, grads = grad_fn(params, images, labels)
    
    # 2. 최적화 도구로부터 수정 사항과 다음 상태 획득
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    
    # 3. 파라미터에 수정 사항 적용하여 새로운 파라미터 생성
    next_params = optax.apply_updates(params, updates)
    
    return next_params, next_opt_state, loss_value
```

1. **JIT 컴파일의 위력:** 이 `update_step` 함수 전체가 XLA 컴파일러에 의해 하나의 거대한 기계어 블록으로 번역된다. 기울기 계산, Adam 상태 업데이트, 파라미터 갱신이 GPU 메모리 상에서 데이터 이동 없이 원자적으로(Atomics) 빠르게 일어난다.

---

## 6. 결론: 불편함이 주는 자유

JAX의 최적화 방식은 PyTorch보다 코드가 길고 번거로워 보일 수 있다. 파라미터, 옵티마이저 상태, 기울기를 모두 개발자가 손수 들고 다녀야 하기 때문이다.

하지만 이러한 **명시성(Explicitness)**은 복잡한 시스템에서 빛을 발한다.
*   **재현성(Reproducibility):** 모든 상태가 변수에 담겨 있으므로, 특정 시점의 학습 상태를 그대로 복제하거나 저장(Checkpointing)하기가 압도적으로 쉽다.
*   **병렬화(Parallelism):** 가변 상태가 없으므로 여러 GPU에 파라미터를 쪼개어 보내고 결과를 합칠 때(Multi-device Training) 데이터 경합(Race Condition)이 발생할 가능성이 원천 봉쇄된다.

결국 JAX에서의 학습이란, 거대한 상태의 흐름을 순수 함수의 사슬로 엮어내는 **데이터 파이프라인의 예술**이라고 할 수 있다.

---

*[jax.grad]: 자동 미분을 수행하는 JAX의 핵심 함수. 함수를 입력받아 그 도함수를 반환한다.
*[optax]: JAX용 최적화(Optimization) 라이브러리. DeepMind에서 개발하였다.
*[In-place Mutation]: 객체의 기존 메모리 주소에 있는 값을 직접 수정하는 행위. JAX에서는 부작용 방지를 위해 금지된다.
*[Hessian]: 함수의 2계 도함수(기울기의 기울기). JAX는 jax.jacfwd(jax.grad(f))와 같이 명시적인 함수 합성을 통해 이를 매우 쉽게 계산한다.
