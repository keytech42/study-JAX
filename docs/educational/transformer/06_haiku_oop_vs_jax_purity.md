# JAX Deep Dive: 순수 함수형 엔진 위의 객체 지향 (Haiku OOP vs JAX Purity)

이 문서는 JAX 생태계에 입문한 개발자가 가장 혼란을 겪는 **"철학의 모순"**을 다룬다. 앞선 [PRNG Deep Dive](../fundamentals/05_jax_prng_and_state_deep_dive.md) 문서에서 우리는 "JAX는 내부 상태(State)를 극도로 혐오하며, 모든 것을 순수 함수로 작성해야 한다"고 배웠다. 그런데 `src/transformer.py`를 보면 우리는 버젓이 `class DenseBlock(hk.Module):` 이라며 상태를 품는 객체 지향 프로그래밍(OOP)을 하고 있다.

이 거대한 모순이 어떻게 성립하는지, 그리고 왜 PyTorch처럼 처음부터 OOP를 쓰지 않고 굳이 `Haiku`라는 중간 다리를 거쳐야만 하는지 심층적으로 해체한다.

---

## 1. 핵심 문제와 용어 정의 (The Core Problem & Contextual Definitions)

### 딥러닝에서 '파라미터(Parameter)'와 '상태(State)'의 의미
일반적인 소프트웨어 공학에서 객체(Object)는 자신만의 데이터(상태)를 가진다. 딥러닝 신경망에서 이 데이터란 곧 **가중치(Weights)와 편향(Bias)**, 즉 파라미터를 의미한다.

문제는 **인간의 사고방식**과 **JAX의 연산 방식**이 정면으로 충돌한다는 점이다.

* **인간의 사고방식 (OOP):** "Dense 레이어라는 상자(객체)를 만들고, 그 안에 가중치(상태)를 숨겨둔 다음, 데이터 $x$를 통과시키자." (`layer(x)`)
* **JAX의 연산 방식 (Pure Function):** "전역 상태나 숨겨진 값은 존재해서는 안 된다. 가중치 $W$와 데이터 $x$를 모두 명시적인 인자로 받아 곱해버리는 거대한 수학 공식을 만들어라." (`f(W, x)`)

이 충돌을 해결하지 않으면, 수십억 개의 파라미터를 가진 트랜스포머 모델을 JAX의 순수 함수 스타일로 짤 때, 개발자는 매번 함수를 호출할 때마다 수만 개의 행렬을 손수 매개변수로 던져주어야 하는 "매개변수 지옥(Parameter Hell)"에 빠지게 된다.

---

## 2. 멘탈 모델과 비교 철학 (Mental Model & Comparative Philosophy)

### 프레임워크들의 갈림길: 상태를 누가 관리할 것인가?

이 문제를 해결하기 위해 딥러닝 프레임워크들은 각자의 철학적 선택을 내렸다.

!!! question "PyTorch는 이 모순을 어떻게 해결했는가?"
    PyTorch는 **"인간의 직관(OOP)이 최우선"**이라는 철학을 가진다. `nn.Module`을 상속받은 클래스를 만들면, PyTorch 프레임워크 자체가 그 객체 내부의 가중치(상태)를 추적하고 업데이트한다. 즉, **"상태를 허용"**했다.
    **트레이드오프:** 이 방식은 코드를 읽고 디버깅하기에는 최고다. 하지만 XLA 같은 극단적인 AOT(Ahead-of-Time) 컴파일러의 입장에서 볼 때, 이런 동적인 객체와 은닉된 상태들은 연산 그래프를 미리 최적화하고 수천 개의 TPU에 안전하게 흩뿌리는 데 엄청난 방해물이 된다.

!!! abstract "JAX/Haiku의 철학: 환상(Illusion)으로서의 객체 지향"
    JAX 코어 엔진 자체는 절대로 양보하지 않는다. **무조건 상태 없는 순수 함수여야만 한다.**
    그래서 구글 딥마인드(DeepMind)는 JAX 위에 **Haiku**라는 신경망 라이브러리를 올렸다. Haiku의 역할은 단순명료하다. 개발자가 익숙한 PyTorch 스타일(OOP)로 코드를 짤 수 있게 **'환상'**을 제공한 뒤, 실행하기 직전에 **내부 상태를 모조리 적출하여 순수한 $f(W, x)$ 형태의 함수로 강제 변환(Transformation)**해 버리는 것이다.

---

## 3. 구체적 해결책: Haiku Transform (The Specific Solution)

어떻게 객체 지향 코드가 순수 함수로 둔갑할 수 있을까? 그 핵심 기전이 바로 `hk.transform`이다. 코드가 어떻게 변환되는지 개념적으로 살펴보자.

**1. 인간이 작성하는 코드 (OOP 스타일)**
```python
import haiku as hk

def my_network(x):
    # 내부에 상태(w, b)를 품는 객체를 생성하고 호출한다.
    layer = hk.Linear(128) 
    return layer(x)
```
이 상태로는 JAX가 컴파일(`jax.jit`)할 수 없다. `hk.Linear` 내부에 숨겨진 가중치 상태 때문이다.

**2. Haiku가 변환한 결과 (순수 함수 스타일)**
```python
# hk.transform(my_network)를 거치면, 
# JAX가 좋아하는 상태 없는(Stateless) 두 개의 순수 함수가 튀어나온다.

pure_init, pure_apply = hk.transform(my_network)

# 1. 상태 초기화 함수 (순수함수: 난수 Key를 넣으면 초기 가중치 딕셔너리를 뱉음)
params = pure_init(jax.random.PRNGKey(42), x)

# 2. 연산 적용 함수 (순수함수: 숨겨진 상태 없이 파라미터를 명시적으로 받음)
output = pure_apply(params, x) 
```

!!! info "제약(Constraint) vs 관례(Convention)"
    Haiku 코드 블록을 작성할 때 파이썬의 클래스를 사용하는 것은, JAX가 이를 요구해서가 아니라 오직 **인간 개발자의 인지적 부하(Cognitive Load)를 줄이기 위한 임의적 관례(Convention)**일 뿐이다. 시스템적 관점에서는 결국 모든 클래스가 벗겨지고 순수한 튜플(파라미터 묶음)과 행렬 곱 연산식으로 분해(Flatten)된다.

---

## 4. `transformer.py` 코드 심층 분석 (Deep Dive Mechanics)

이제 실제 우리 프로젝트의 코드인 `DenseBlock`을 살펴보자.

```python
class DenseBlock(hk.Module):                 # (1)!
    def __init__(self, init_scale: float, widening_factor: int = 4, name: str | None = None):
        super().__init__(name=name)
        self._init_scale = init_scale        # (2)!
        self._widening_factor = widening_factor

    def __call__(self, x: jax.Array) -> jax.Array: # (3)!
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)
```

1. **`hk.Module` 상속:** PyTorch의 `nn.Module`과 완벽히 동일한 외형을 취한다. 이는 딥러닝 연구자들이 새로운 프레임워크를 배울 때 느끼는 진입 장벽을 낮추기 위한 설계다.
2. **하이퍼파라미터 저장:** `__init__`에서는 가중치 행렬 같은 진짜 '상태(State)'를 생성하지 않는다. 단지 몇 배로 차원을 늘릴 것인지 등 구조에 대한 **설정값(Configuration)**만 파이썬 인스턴스의 변수로 저장해 둔다.
3. **`__call__` 매직 메서드:** 파이썬에서 객체 인스턴스를 함수처럼 호출(`block(x)`)할 수 있게 해준다. Haiku의 `transform` 엔진은 런타임에 이 `__call__` 메서드를 추적(Trace)하여 연산 그래프를 그린다. 이때 `hk.Linear`를 만나면 "아, 여기에 가중치 행렬 $W$와 $b$가 필요하군"이라고 기록만 해둔 뒤, 나중에 `pure_init` 함수를 통해 한꺼번에 생성해낸다.

### 왜 굳이 `__call__`을 쓰는가?
`forward()`나 `apply()` 같은 일반 메서드 이름을 써도 무방하다. 하지만 딥러닝 커뮤니티에서는 수학적인 표기법인 $y = f(x)$를 코드에 가장 자연스럽게 녹여내기 위해, 객체 자체를 함수처럼 취급할 수 있는 파이썬의 `__call__`을 활용하는 것이 강력한 **관례(Convention)**로 자리 잡았다.

---

*[AOT]: Ahead-Of-Time. 파이썬 스크립트가 실행되는 도중(Just-In-Time)이 아니라, 실제 연산이 GPU에 올라가기 전에 전체 그래프를 미리 파악하고 기계어로 번역/최적화하는 방식.
*[XLA]: Accelerated Linear Algebra. 연산 그래프를 분석해 커널 퓨전(Kernel Fusion) 및 병렬화를 수행하는 구글의 머신러닝 컴파일러. JAX의 속도를 책임지는 핵심 엔진.
*[HLO]: High Level Optimizer. XLA 컴파일러가 소스 코드를 최적화하기 위해 구축하는 추상적인 연산 의존성 그래프.
