# JAX & Haiku Deep Dive: DenseBlock & `__call__`

이 문서는 트랜스포머의 `DenseBlock`(또는 MLP, Feed-Forward Network) 모듈과 파이썬/Haiku 환경에서 `__call__` 메서드가 왜 그토록 중요한지를 분석한다.

## 1. 파이썬과 Haiku에서 `__call__`의 의미

객체 지향 프로그래밍(OOP)에서 클래스는 보통 상태(변수)와 행동(메서드)을 묶어둔다. 그렇다면 왜 `forward()`나 `run()` 같은 이름 대신 굳이 `__call__`이라는 "매직 메서드(Magic Method)"를 사용할까?

### Pythonic Convention (콜러블 객체)

파이썬에서 클래스 내부에 `__call__`을 정의하면, 그 클래스로 만든 **인스턴스를 마치 함수처럼 호출(Callable)**할 수 있게 된다.
```python
# 일반 메서드 사용 시
block = DenseBlock()
output = block.forward(x)

# __call__ 사용 시
block = DenseBlock()
output = block(x)  # 함수처럼 매우 깔끔해짐
```
수학적인 함수 $y = f(x)$ 의 형태를 코드에 가장 자연스럽게 녹여내는 파이썬의 관습(Convention)이다.

### Haiku의 Functional Transformation

JAX는 모든 것이 '순수 함수'여야 한다고 앞서 설명했다. Haiku는 우리가 작성한 객체 지향 코드(클래스)를 순수 함수로 바꾸기 위해 `hk.transform`이라는 마법을 부린다.
이때 Haiku 엔진은 내부적으로 **우리가 모듈을 어떻게 "호출"하는지**를 추적(Trace)하여 연산 그래프를 그린다. 즉, 인스턴스가 함수처럼 호출될 때(`__cavll__`이 실행될 때) 비로소 가중치가 필요한지, 입력의 모양이 무엇인지 판단하고 GPU 메모리에 파라미터를 초기화한다. `__call__`은 딥러닝에서 '순전파(Forward pass)'를 정의하는 프레임워크 간의 암묵적인 표준이다.

---

## 2. DenseBlock과 잔차 연결 (Residual Connection)

트랜스포머 아키텍처에서 `DenseBlock`은 Attention 레이어 직후에 위치하며, 각 토큰(단어)의 표현을 개별적으로 더 풍부하게 만들어주는 역할을 한다.

```python
    def __call__(self, x: jax.Array) -> jax.Array:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        
        # 1. 확장 (Expand)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        
        # 2. 비선형성 부여
        x = jax.nn.gelu(x)
        
        # 3. 축소 (Project back)
        return hk.Linear(hiddens, w_init=initializer)(x)
```

### 왜 다시 원래 차원(`hiddens`)으로 되돌려놓아야 하는가?

질문하신 내용의 핵심이다. 트랜스포머의 모든 서브 레이어(Attention, DenseBlock 등)를 통과한 후에는 항상 **잔차 연결(Residual Connection)**이 기다리고 있다.

$$\text{Output} = x_{original} + SubLayer(x_{original})$$

이 공식에서 덧셈(`+`)이 성립하려면 텐서의 형태(Shape)가 완벽하게 동일해야 한다.
1. `hk.Linear`를 통해 차원을 크게 넓혀서(보통 4배) 데이터가 가진 복잡한 특징을 비선형 함수(GELU)로 충분히 섞어준다.
2. 하지만 블록을 나갈 때는 **반드시** 두 번째 `hk.Linear`를 거쳐 다시 원래의 차원(`hiddens`)으로 되돌려 놓아야만, 밖에서 기다리고 있는 원본 입력값 $x$와 무사히 더해질 수 있다.
