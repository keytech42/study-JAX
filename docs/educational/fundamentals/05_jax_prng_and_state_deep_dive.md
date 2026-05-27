# JAX Deep Dive: 순수 함수, 상태(State), 그리고 PRNG의 철학

이 문서는 JAX 생태계에서 가장 이질적이고 진입 장벽이 높은 개념인 **난수 생성(PRNG)과 상태 관리**를 심층 해부한다. `src/mnist.py`에 등장하는 파라미터 초기화 코드를 바탕으로, **"왜 JAX는 NumPy나 PyTorch가 제공하는 편리한 난수 생성 방식을 거부하고 개발자를 괴롭히는가?"**라는 본질적인 질문에 대해 컴파일러 아키텍처 수준까지 파고들어 논리적 인과관계를 밝힌다.

---

## 1. 핵심 문제와 용어 정의 (The Core Problem & Contextual Definitions)

### 본 문서에서 정의하는 '상태(State)'
컴퓨터 공학에서 '상태'는 문맥에 따라 다르게 쓰인다. 딥러닝 프레임워크와 병렬 연산의 맥락에서 **상태(State)**란, **"함수나 연산이 외부의 명시적인 인자 전달 없이도 스스로 기억하고, 시간이 지남에 따라 암묵적으로 변경하는 내부 메모리 값"**을 뜻한다.

우리가 흔히 아는 NumPy의 난수 생성 방식을 보자.

```python
import numpy as np
np.random.seed(42)          
a = np.random.normal()      
b = np.random.normal()      
```

위 코드에서 `np.random.normal()`은 겉보기엔 입력 인자가 없는 빈 괄호 `()`이다. 하지만 내부적으로는 이전에 설정된 `42`라는 상태(State)를 읽어와 난수 `a`를 도출하고, 그 즉시 내부 메모리의 상태 값을 다른 값(예: `43`)으로 몰래 갱신한다. 이 때문에 다음 호출 시 난수 `b`는 `a`와 다른 값이 나올 수 있다. 이렇게 함수 외부에 숨겨져 알아서 갱신되는 값을 **전역 상태(Global State)**라고 한다.

이러한 은닉된 전역 상태는 매번 시드(Seed)를 넘겨주지 않아도 되므로 **직관적이고 압도적인 개발 편의성**을 제공한다. 이것이 NumPy가 이 방식을 채택한 이유다.

---

## 2. 멘탈 모델과 비교 철학 (Mental Model & Comparative Philosophy)

### 순수 함수 (Pure Function)의 강제
JAX는 앞서 설명한 "내부 상태가 몰래 갱신되는 마법"을 시스템 차원에서 전면 부정한다. 대신 **모든 함수가 순수 함수(Pure Function)여야 함**을 강제한다.

!!! abstract "프레임워크 철학: 순수 함수란?"
    1. **동일 입력, 동일 출력:** 어떤 함수에 동일한 인자 $X$를 넣으면 언제 실행하든 무조건 동일한 결과 $Y$가 나와야 한다.
    2. **부작용(Side Effect) 없음:** 함수가 실행되는 동안 시스템의 다른 전역 변수나 상태를 절대로 읽거나 쓰지(변경하지) 않아야 한다.

JAX에서 난수 생성은 암묵적 상태 갱신이 아니라, 철저히 **수학적 매핑**인 $f(\text{Key}) \rightarrow \text{Value}$ 의 형태를 띤다.
이 수식은 **"독립 변수 $\small \text{Key}$를 함수 $f$에 명시적으로 투입했을 때, 오직 그 $\small \text{Key}$에 의해서만 유일하게 결정되는 종속 변수 $\small \text{Value}$가 도출된다"**는 것을 뜻한다. 즉, 입력 $\small \text{Key}$가 같으면 반환되는 난수도 영원히 같다.

### 왜 굳이 순수 함수인가? (XLA 컴파일러의 한계와 병렬화)
왜 JAX는 개발자를 귀찮게 만들면서까지 순수 함수를 고집할까? 그 이유는 JAX의 심장인 XLA 컴파일러 때문이다.

XLA는 코드를 실시간으로 실행하지 않고, 전체 연산 그래프(HLO)를 먼저 그린 뒤 **극단적인 연산 재배치 및 병렬 최적화**를 수행하여 GPU 기계어로 굽는다(JIT 컴파일).
만약 난수 생성기가 '전역 상태'를 가진다면, XLA 컴파일러는 심각한 병목(Bottleneck)에 직면한다. 

* **Read-After-Write (RAW) 해저드:** `난수 A`를 생성하는 연산이 내부 상태를 갱신(Write)한 뒤에야, 비로소 `난수 B`를 생성하는 연산이 그 상태를 읽을(Read) 수 있다.
* **병렬화 불가능:** 상태 의존성 때문에 XLA는 `난수 A`와 `난수 B`를 서로 다른 GPU 코어에 동시에(병렬로) 할당할 수 없다. 울며 겨자 먹기로 코드를 순차적(Sequential)으로 실행해야 하므로 GPU의 수천 개 코어가 놀게 된다.

JAX가 순수 함수를 강제하는 것은, XLA 컴파일러에게 **"이 연산들은 서로 상태를 공유하지 않는 완전히 독립된 수학 식이니, 안심하고 수백 개의 GPU 코어에 동시에 흩뿌려라!"**라는 강력한 보증 수표를 제공하기 위함이다.

!!! question "그럼 PyTorch는 어떻게 상태를 가지면서도 병렬 처리를 할까?"
    PyTorch는 JAX와 전혀 다른 철학(트레이드오프)을 선택했다.
    **PyTorch의 해결책:** PyTorch는 전역 상태를 유지하면서 다중 GPU/스레드 환경을 지원하기 위해, 내부적으로 각 디바이스나 스레드마다 독립적인 `torch.Generator` 인스턴스(상태 객체)를 꼼꼼하게 메모리에 할당한다. 또한 동시 접근 시 충돌을 막기 위해 복잡한 락(Lock) 메커니즘이나 스레드 로컬 스토리지(TLS)를 사용한다.
    **트레이드오프의 결과:** PyTorch는 이 무거운 메모리/동기화 오버헤드를 프레임워크가 짊어짐으로써 **개발자의 편의성**을 극대화했다. 반면, JAX는 이러한 프레임워크 단의 오버헤드와 잠재적인 컴파일러 병목마저 완전히 제거하기 위해 편의성을 포기하고 **궁극의 실행 속도와 병렬 효율**을 취한 것이다.

---

## 3. 구체적 해결책: 명시적 Key와 Split (The Specific Solution & Rationale)

JAX는 이 문제를 **PRNGKey**와 **Split**이라는 독특한 기법으로 해결한다.

```python
from jax import random

key = random.PRNGKey(0)         # (1)!
val1 = random.normal(key)       # (2)!
val2 = random.normal(key)       # (3)!

key, subkey = random.split(key) # (4)!
val3 = random.normal(subkey)    # (5)!
```

1. 여기서 `0`은 특별한 매직 넘버가 아니다. 단순히 시스템을 초기화하기 위해 프로그래머가 임의로 지정한 정수 리터럴이다 (예: 42나 1234).
2. `val1`을 생성할 때 `key`를 명시적으로 전달한다.
3. **완벽하게 동일한 난수 발생:** 동일한 `key`를 넣었으므로, 순수 함수의 원칙에 따라 `val1`과 완전히 동일한 값이 반환된다.
4. 새로운 난수가 필요하므로, 기존의 `key`를 두 갈래의 독립된 키(`key`, `subkey`)로 **쪼갠다(split)**.
5. 파생된 새로운 독립 변수 `subkey`를 전달하여 비로소 새로운 난수 `val3`을 도출한다.

!!! info "제약(Constraint) vs 관례(Convention)"
    이 `split` 과정은 프로그래머의 스타일이나 JAX 커뮤니티의 관례가 아니다. JAX 하에서 독립된 난수를 얻기 위한 **수학적/시스템적 절대 제약(Fundamental Constraint)**이다. `split`을 누락하면 모델의 모든 파라미터가 영원히 동일한 난수로 초기화된다.

### 왜 '생성(Generate)'이 아니라 '쪼갬(Split)'인가?
JAX의 PRNG는 내부적으로 기존과 다른 해시(Hash) 기반 알고리즘(Threefry 등)을 사용한다. 
'생성'은 무언가 새로운 임의의 상태가 만들어짐을 암시하지만, `split`은 **하나의 부모 키로부터 암호학적으로 안전하게, 절대 겹치지 않는(Non-overlapping) 두 개의 의사 난수 스트림(Stream)을 결정론적으로 파생시킨다**는 기하학적 의미를 명확히 하기 위해 선택된 명명법이다.

---

## 4. `mnist.py` 코드 심층 분석 (Deep Dive Mechanics)

이 철학이 `src/mnist.py`의 파라미터 초기화에 어떻게 적용되었는지 살펴보자.

```python
def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2): # (1)!
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)                           # (2)!
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    keys = random.split(key, len(sizes))                           # (3)!
    
    return [
        random_layer_params(m, n, k, scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]
```

1. **`scale=1e-2` (관례):** 여기서 0.01을 곱해주는 것은 네트워크 학습 시 활성화 함수가 포화되는 것을 막기 위한 **경험적 관례(Pragmatic Choice)**다. 논리적 필연성이 아니며 He나 Xavier 초기화로 대체될 수 있다.
2. **레이어 내부 분할:** 하나의 신경망 은닉층은 항상 가중치 행렬($W$)과 편향 벡터($b$)를 동시에 가진다. 이 둘 모두 독립적인 난수여야 하므로, 하나의 은닉층에 도달한 단일 `key`를 즉시 `w_key`와 `b_key` 두 갈래로 분할한다.
3. **루프 외부의 병렬 분할 패턴:** 여러 개의 층(Layer)을 순회하며 그때그때 루프 내부에서 키를 갱신하는 것은 함수형 프로그래밍스럽지 않다. 대신 필요한 전체 층의 개수(`len(sizes)`)만큼 **독립적인 키의 배열을 한 번에 쫙 분할해 놓고**, 리스트 컴프리헨션을 통해 각 층에 하나씩 매핑(`zip`)한다. 부작용(Side effect)을 최소화하는 매우 JAX다운 디자인 패턴이다.

---

*[PRNG]: Pseudo-Random Number Generator. 결정론적인 알고리즘을 통해 난수처럼 보이는 수열을 생성하는 프로그램.
*[XLA]: Accelerated Linear Algebra. 연산 그래프를 분석해 커널 퓨전(Kernel Fusion) 및 병렬화를 수행하는 구글의 머신러닝 컴파일러.
*[HLO]: High Level Optimizer. XLA 컴파일러가 소스 코드를 최적화하기 위해 구축하는 추상적인 연산 의존성 그래프.
*[JIT]: Just-In-Time Compilation. 파이썬 코드를 실행하는 시점에 기계어로 즉시 컴파일하여 속도를 비약적으로 높이는 기술.
