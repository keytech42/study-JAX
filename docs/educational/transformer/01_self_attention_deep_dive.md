# JAX & Haiku Deep Dive: SelfAttention Module

이 문서는 JAX 및 Haiku 생태계에 대한 깊이 있는 이해를 돕기 위해 작성되었다. `main.py`에 구현된 `SelfAttention` 모듈을 줄 단위로 분석하여 프레임워크의 근본적인 멘탈 모델(Mental Model)을 설명한다.

## JAX의 핵심 철학

JAX 환경에서 코드를 작성할 때 가장 먼저 숙지해야 할 대원칙은 다음과 같다.

> **JAX는 상태(State)를 가지지 않으며, 모든 배열(Array)은 불변(Immutable)이고, 모든 함수는 순수 함수(Pure Function)여야 한다.**

이러한 철학은 자동 미분(Autograd)과 XLA 컴파일(JIT)을 통해 모델을 극도로 최적화하기 위한 필수 조건이다.

---

## 코드 분석 (`main.py`)

### 1. 모듈 임포트 및 생태계 이해

```python
import haiku as hk
import jax
import jax.numpy as jnp
```

*   **`jax`**: JAX 코어 엔진이다. XLA(Accelerated Linear Algebra) 컴파일러를 통해 코드를 최적화하고 GPU/TPU에서 실행하며, 자동 미분을 수행하는 기반 시스템이다.
*   **`jax.numpy as jnp`**: 표준 NumPy(`np`)와 API가 매우 유사하지만 내부 동작은 완전히 다르다. 
    *   `np`는 CPU 메모리에 데이터를 올리고 즉각적으로 연산(Eager execution)한다.
    *   `jnp`는 데이터를 Device(GPU/TPU) 메모리에 올린다(DeviceArray). 가장 중요한 특징은 값이 불변(Immutable)이라는 점이다. 데이터가 변하지 않아야 JAX가 부작용(Side Effect) 없이 미분 경로를 정확히 추적(Trace)할 수 있다.
*   **`haiku as hk`**: JAX는 순수 함수형 프로그래밍을 지향하므로 객체 내부에 상태(가중치 등)를 저장하는 것을 지양한다. Haiku는 PyTorch와 같은 객체 지향적 모듈 작성을 허용하되, 실행 시점에 이를 JAX가 요구하는 순수 함수(상태와 연산이 분리된 형태)로 변환해 주는 라이브러리다.

### 2. 클래스 정의

```python
class SelfAttention(hk.MultiHeadAttention):
```

*   Haiku의 `MultiHeadAttention`을 상속받는다. 이 모듈 내부에는 Query, Key, Value 연산에 필요한 가중치(Weights) 파라미터들이 정의되어 있다.
*   JAX의 특성상 이 클래스를 인스턴스화(`SelfAttention()`)한다고 해서 즉시 가중치가 메모리에 할당되지 않는다. 이후 `hk.transform`을 통해 변환(Transformation) 과정을 거쳐야 실제 파라미터가 생성된다.

### 3. 매직 메서드와 타입 힌트

```python
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array | None = None,
        value: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
```

*   **`__call__`**: 인스턴스를 함수처럼 호출할 때 실행되는 메서드로, 신경망의 순전파(Forward pass)를 정의한다.
*   **`jax.Array`**: JAX의 표준 데이터 타입이다. 변수가 CPU에 있는 일반 파이썬 객체가 아닌, JAX가 관리하는 디바이스 배열임을 명시한다. 이는 정적 타입 분석기와 XLA 컴파일러 모두에게 명확한 컨텍스트를 제공한다.

### 4. Self-Attention의 논리

```python
        key = key if key is not None else query
        value = value if value is not None else query
```

*   이 부분에서 'Self' Attention의 본질이 드러난다. 외부에서 별도의 `key`와 `value`가 제공되지 않으면, `query` 자체를 복사하여 `key`와 `value`로 사용한다. 즉, 입력 시퀀스 내의 요소들이 자기 자신(Self)의 다른 요소들과의 연관성을 계산하도록 한다.

### 5. 인과적 마스크 (Causal Mask) 생성

```python
        seq_len = query.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
```

*   **배열 차원 분석**: `query` 텐서의 형태는 통상적으로 `[Batch, Sequence_Length, Features]`이다. 인덱스 `1`을 통해 시퀀스 길이를 추출한다.
*   **자기회귀(Autoregressive) 제약**: 생성형 모델은 예측을 수행할 때 미래의 정보를 참조해서는 안 된다. 이를 방지하기 위해 인과적 마스크를 생성한다.
*   **`jnp.ones` & `jnp.tril`**: 디바이스 상에 1로 채워진 $N \times N$ 행렬을 생성한 뒤, `tril`(Lower Triangular) 함수를 적용하여 하삼각행렬을 만든다. 대각선을 포함한 아래쪽은 1(관측 가능)로 남고, 위쪽(미래 시점)은 0(관측 불가)으로 차단된다. 이 연산은 `jnp`를 사용하므로 JAX 컴파일러에 의해 최적화된다.

### 6. 마스크 병합 및 부모 모듈 호출

```python
        mask = mask * causal_mask if mask is not None else causal_mask
        
        return super().__call__(query, key, value, mask)
```

*   외부에서 전달된 마스크(예: 패딩 마스크)가 존재할 경우, 방금 생성한 인과적 마스크와 요소별 곱셈(Element-wise multiplication)을 수행하여 결합한다. 이를 통해 패딩된 토큰과 미래의 토큰을 모두 무시하는 완전한 마스크를 구성한다.
*   최종적으로 정제된 입력값들을 부모 클래스(`hk.MultiHeadAttention`)로 전달하여 $Q \times K^T$, 소프트맥스 연산 등의 핵심 어텐션 로직을 수행하도록 위임한다.
