# JAX Deep Dive: 인터프리터 설계와 추적기(Tracer)의 마법

이 문서는 이전 문서에서 타협(생략)했던 심연의 영역, **"Haiku는 대체 어떻게 파이썬의 OOP 코드 속에서 가중치를 적출해 내는가?"**에 대한 완벽한 기술적 해답을 제시한다.

이를 이해하기 위해서는 JAX가 파이썬 코드를 기계어(XLA)로 번역하기 위해 사용하는 핵심 기전인 **추적(Tracing)**과 **추적기(Tracer)** 객체의 동작 원리, 그리고 컴파일러 설계의 근본적인 한계를 해부해야 한다.

---

## 1. 핵심 문제: 동적 언어와 정적 컴파일러의 간극

파이썬(Python)은 한 줄씩 코드를 읽고 실행하는 인터프리터 언어(동적 언어)다. `if`문이나 `for`문이 실행 시간에 결정되며, 변수의 타입도 그때그때 바뀐다.
반면 XLA 컴파일러는 C++처럼 미리 전체 구조가 확정된 정적 연산 그래프(Static Computation Graph)를 요구한다. 모든 행렬의 크기(Shape)와 데이터 타입(Dtype)이 사전에 완벽히 계산되어 있어야 GPU 메모리를 미리 할당하고 연산을 재배치할 수 있기 때문이다.

**문제의 본질:** 동적이고 유연한 파이썬 함수 `def f(x): ...` 를 어떻게 뼈대만 앙상하게 남은 정적 수학 그래프(XLA HLO)로 변환할 것인가?

---

## 2. 멘탈 모델과 비교 철학: 그래프를 그리는 세 가지 방법

이 간극을 메우기 위해 딥러닝 프레임워크들은 각기 다른 철학을 채택했다.

!!! question "PyTorch는 어떻게 코드를 그래프로 만드는가?"
    *   **과거의 PyTorch (Eager Execution):** 그래프를 "미리" 그리지 않는다. 파이썬이 `x + y`를 실행할 때 백그라운드(C++)에서 즉시 연산을 수행하고, 그 결과를 바탕으로 뒤늦게 미분(Autograd)을 위한 동적 그래프를 쌓아 나간다. (매우 직관적이지만 AOT 컴파일과 최적화에는 불리하다.)
    *   **최신 PyTorch (`torch.compile` & Dynamo):** 파이썬의 **바이트코드(Bytecode)** 자체를 후킹(Hooking)하여 분석한다. 파이썬이 실행할 명령어의 흐름을 가로채서 그래프로 만든다. 매우 강력한 엔지니어링이지만, 내부 구조가 극도로 복잡하다.

!!! question "TensorFlow는 어떻게 코드를 그래프로 만드는가?"
    *   **AutoGraph (`tf.function`):** 파이썬 소스 코드의 **추상 구문 트리(AST, Abstract Syntax Tree)**를 직접 파싱하여 분석한다. 파이썬의 `for`문이나 `if`문을 텍스트 레벨에서 분석하여 텐서플로우 전용 그래프 노드(`tf.while_loop`, `tf.cond`)로 억지로 번역한다. 버그가 났을 때 추적하기가 악명 높게 어렵다.

!!! abstract "JAX의 철학: 가짜 실행을 통한 추적 (Tracing as Mock Execution)"
    JAX는 소스 코드를 뜯어보거나(AST) 바이트코드를 해킹하지 않는다. 대신 **JAX는 파이썬 인터프리터 위에서 코드를 한 번 "가짜로 실행(Mock Execution)" 시켜본다.**
    이때 파이썬 함수에 진짜 데이터(예: `np.array`)를 넣는 것이 아니라, 오직 모양(Shape)과 타입(Dtype) 정보만 가진 텅 빈 스파이 객체인 **추적기(Tracer)**를 집어넣는다.

---

## 3. 심층 기전: `jax.core.Tracer`의 동작 원리

JAX의 핵심은 파이썬의 매직 메서드(Magic Methods) 오버라이딩에 있다.

### Tracer 객체의 실체
JAX의 `jit` 컴파일러는 함수를 처음 호출할 때 입력값 `x`를 `jax.core.Tracer` (구체적으로는 `ShapedArray`)라는 특수한 객체로 감싼다.

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    print("현재 x의 정체:", x) # (1)!
    return x * 2

# 함수 호출
f(jnp.array([1.0, 2.0]))
```

1. 함수가 JIT 컴파일될 때 이 `print`문이 실행되는데, 출력 결과는 실제 배열 `[1.0, 2.0]`이 아니라 `Traced<ShapedArray(float32[2])>`이다!

파이썬 인터프리터가 `x * 2`라는 코드를 실행하려고 시도할 때, `x`는 실제 숫자가 아니라 `Tracer` 객체이므로 파이썬의 `__mul__` 매직 메서드가 호출된다.

```python
class Tracer:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __mul__(self, other):
        # 1. 실제 곱셈 연산을 하지 않는다!
        # 2. 대신 중앙 기록 장치(Jaxpr Builder)에 "곱셈 연산 노드를 추가해 줘"라고 기록한다.
        jaxpr_builder.add_node("multiply", inputs=[self, other])
        
        # 3. 새로운 결과물의 모양을 계산해서 새로운 Tracer를 반환한다.
        return Tracer(self.shape, self.dtype)
```

이처럼 `Tracer` 객체가 파이썬 함수의 흐름을 따라 이리저리 굴러다니면서 만나는 모든 연산(`+`, `-`, `jnp.sum` 등)을 중간 표현식인 **Jaxpr (JAX Expression)**라는 리스트에 차곡차곡 기록해 나가는 과정이 바로 **추적(Tracing)**이다.
추적이 끝나면 JAX는 완성된 Jaxpr 그래프를 XLA 컴파일러에 넘겨 초고속 기계어로 번역한다.

---

## 4. Haiku의 기적: Tracer를 가로채어 상태(State) 적출하기

이제 다시 우리의 메인 질문으로 돌아오자. `hk.transform`은 어떻게 파이썬 OOP 코드(`DenseBlock`)에서 가중치(파라미터)를 모아낼 수 있을까?

Haiku는 JAX의 이 "추적(Tracing)" 시스템에 무임승차(Piggyback)하는 천재적인 라이브러리다.

```python
import haiku as hk

def my_network(x):
    layer = hk.Linear(128) # (1)!
    return layer(x)

pure_init, pure_apply = hk.transform(my_network)
```

### `pure_init`이 호출될 때 일어나는 일

1. 개발자가 `pure_init(key, x)`를 호출하면, Haiku는 내부적으로 빈 딕셔너리(`params = {}`)를 하나 만든다.
2. 그리고 JAX에게 지시하여 입력 데이터 `x`를 `Tracer` 객체로 바꾼 뒤 `my_network` 함수를 가짜로 실행(Trace)시킨다.
3. `layer(x)`가 호출될 때 (즉 `hk.Linear.__call__` 내부가 실행될 때), `hk.Linear`는 파라미터가 필요한 위치(`w`, `b`)에서 일반적인 난수 생성을 하는 대신, **현재 진행 중인 추적 과정의 흐름을 가로챈다(Intercept).**
4. `hk.Linear`는 난수 `key`를 사용해 실제 가중치 행렬 $W$를 생성한 다음, 이를 Haiku가 미리 만들어둔 전역 딕셔너리 `params`에 `{"Linear": {"w": W, "b": b}}` 형태로 **몰래 기록(Register)**해 둔다.
5. 동시에 함수 내부의 연산은 JAX의 `Tracer`에 의해 정상적으로 Jaxpr 그래프로 기록된다.
6. 가짜 실행이 끝나면, Haiku는 기록이 완료된 꽉 찬 `params` 딕셔너리를 사용자에게 반환한다!

!!! info "제약(Constraint) vs 관례(Convention): 부작용(Side Effect)의 위험성"
    이 놀라운 추적 시스템에는 치명적인 **제약(Constraint)**이 있다. Tracer는 파이썬 제어 흐름만 따라가기 때문에, 파이썬의 외부 상태를 변경하는 코드(예: 전역 리스트에 `append` 하기, 파일에 쓰기)는 **컴파일(Trace)될 때 단 한 번만 실행되고, 이후 XLA로 최적화된 코드가 실행될 때는 완전히 증발해 버린다.** 이것이 바로 앞선 [PRNG Deep Dive](./05_jax_prng_and_state_deep_dive.md) 문서에서 "JAX는 무조건 순수 함수여야 한다"고 강력하게 주장했던 기술적 밑바탕이다.

---

## 5. 결론: 환상을 걷어내다

결론적으로, 우리가 `src/transformer.py`에 작성한 `class DenseBlock(hk.Module)`는 진짜로 객체 인스턴스가 되어 메모리에 상주하며 텐서 연산을 수행하는 녀석이 아니다.

Haiku의 OOP 클래스들은 파이썬 코드가 한 번 가짜로 실행(Trace)되는 동안, **자기 자신에게 필요한 파라미터가 무엇인지 딕셔너리에 명시적으로 등록해 주고 장렬히 산화하는 일회성 "템플릿(Template)" 또는 "공장(Factory)" 역할**을 할 뿐이다.
모든 추적이 끝난 후 남는 것은 철저히 분해되어 상태(가중치 딕셔너리)와 분리된, 극도로 순수한 $f(W, x)$ 형태의 수학적 그래프(Jaxpr/HLO)뿐이다. 이것이 바로 JAX 생태계가 OOP의 탈을 쓰고 극단적인 순수 함수형 철학을 유지하는 메커니즘이다.

---

*[Tracer]: JAX가 파이썬 함수의 연산 흐름을 추적하기 위해 삽입하는, Shape과 Dtype 정보만을 가진 스파이(가짜) 객체.
*[Jaxpr]: JAX Expression. 파이썬 코드가 Tracer에 의해 추적된 후 생성되는, 부작용이 없고 상태가 분리된 JAX의 자체적인 중간 그래프 표현식.
*[AST]: Abstract Syntax Tree. 소스 코드를 문자열 그대로 해석하여 문법적 구조를 트리 형태로 나타낸 것. TensorFlow의 AutoGraph가 이를 파싱해 사용한다.
*[Bytecode]: 파이썬 인터프리터가 실행하기 위해 소스 코드를 컴파일한 중간 단계의 저수준 코드. PyTorch Dynamo가 이 바이트코드를 분석해 그래프를 생성한다.
