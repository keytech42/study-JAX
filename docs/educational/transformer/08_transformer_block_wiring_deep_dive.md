# JAX Deep Dive: 트랜스포머 블록 조립과 Pre-LN 철학

이 문서는 이전에 개별적으로 살펴보았던 [Self-Attention](./01_self_attention_deep_dive.md), [DenseBlock](./02_dense_block_deep_dive.md), 그리고 [Layer Normalization](./03_layer_norm_deep_dive.md)이 어떻게 하나의 완전한 **트랜스포머 블록(Transformer Block)**으로 조립되는지 다룬다.

단순히 `src/transformer.py`의 코드를 나열하는 것을 넘어, **"왜 정규화(Normalization)를 어텐션 연산 이전에 하는가(Pre-LN)?"**라는 아키텍처 설계의 본질적인 질문과 역사적 맥락을 파헤친다.

---

## 1. 핵심 문제: 층이 깊어질수록 학습이 붕괴되는 현상

트랜스포머 모델의 성능을 끌어올리는 가장 확실한 방법은 층(Layer)을 수십, 수백 개로 깊게 쌓는 것이다. 하지만 층이 깊어질수록 심각한 문제에 직면한다.

입력 데이터 $x$가 거대한 트랜스포머 블록을 통과하면서 값의 분산(Variance)이 통제 불능 상태로 커지거나 작아진다. 이렇게 되면 오차 역전파(Backpropagation) 과정에서 기울기 폭발(Gradient Exploding)이나 소실(Vanishing)이 발생하여 모델 학습이 완전히 멈춰버린다. 이를 해결하기 위해 레이어 정규화(Layer Normalization)를 어디에 배치할 것인가가 딥러닝 아키텍처의 최대 화두 중 하나였다.

---

## 2. 멘탈 모델과 비교 철학: Post-LN vs Pre-LN

이 블록 배치의 딜레마를 해결하기 위해 역사적으로 두 가지 철학이 팽팽하게 대립했다.

!!! question "초기 트랜스포머(Post-LN)의 방식과 한계"
    2017년 구글의 오리지널 논문("Attention Is All You Need")은 **Post-LN (사후 정규화)** 방식을 택했다.
    수식: $x = \text{LayerNorm}(x + \text{Sublayer}(x))$
    이 방식은 "먼저 연산을 하고, 원본과 더한 뒤에, 튀어버린 값을 정규화로 예쁘게 누르자"는 인간의 직관에 완벽히 부합한다.
    **트레이드오프:** 이 구조는 수렴만 시킬 수 있다면 모델의 최종 성능이 매우 높다. 그러나, 층이 깊어질수록 출력 레이어 근처에서 발생한 거대한 기울기(Gradient)가 아래로 내려가면서 증폭되어 초기 층의 학습을 망쳐버린다. 이를 막기 위해 초기 학습률을 극도로 낮췄다가 서서히 올리는 **Learning Rate Warm-up**이라는 까다로운 휴리스틱(꼼수)을 반드시 동원해야만 했다.

!!! abstract "최신 트랜스포머(Pre-LN)의 철학"
    우리 코드(`src/transformer.py`)와 최신 거대 언어 모델(GPT-3, LLaMA 등)이 채택한 방식은 **Pre-LN (사전 정규화)**이다.
    수식: $x = x + \text{Sublayer}(\text{LayerNorm}(x))$
    이는 "원본 데이터 $x$ 자체는 그대로 뻗어나가는 고속도로(메인 스트림)로 보존하고, 옆길(Sublayer)로 빠져서 연산할 녀석들만 정규화를 시켜서 보내자"는 철학이다. 

### 왜 대세는 Pre-LN이 되었는가?
Pre-LN 구조에서는 잔차 연결(Residual Connection)로 이어지는 **메인 스트림 $x$가 단 한 번도 정규화에 의해 방해받지 않고 맨 마지막 층까지 그대로 다이렉트로 연결**된다. 
따라서 맨 마지막 층에서 발생한 오차 기울기가 어떠한 변형이나 축소 없이 모델의 가장 첫 번째 층까지 고속도로를 타고 즉각적으로 도달할 수 있다. 이로 인해 학습이 압도적으로 안정화되어, 골치 아픈 Warm-up 기법 없이도 수십, 수백 층의 모델을 쉽게 훈련시킬 수 있게 되었다.

---

## 3. 구체적 해결책: TransformerBlock 조립

이제 이론을 바탕으로 우리의 `TransformerBlock`이 어떻게 조립되었는지 살펴보자. 

트랜스포머 블록은 항상 두 개의 서브 레이어(Sublayer)로 구성된다.
1. **Self-Attention Sublayer:** 토큰 간의 맥락(Context)을 파악하는 곳.
2. **DenseBlock (MLP) Sublayer:** 파악된 맥락을 바탕으로 각 토큰의 의미를 개별적으로 심화시키는 곳.

우리 코드는 이 두 서브 레이어에 완벽하게 **Pre-LN** 구조를 적용했다.

---

## 4. `transformer.py` 코드 심층 분석

`src/transformer.py`에 구현된 `TransformerBlock.__call__` 메서드를 분해해보자.

```python
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        # 1. Self-Attention Sublayer (with Pre-LN)
        attn_out = SelfAttention(                                      # (1)!
            num_heads=self._num_heads,
            key_size=self._key_size,
            w_init=hk.initializers.VarianceScaling(self._w_init_scale),
            name="attention",
        )(query=layer_norm(x, name="attn_ln"), mask=mask)              # (2)!
        x = x + attn_out                                               # (3)!

        # 2. DenseBlock Sublayer (with Pre-LN)
        dense_out = DenseBlock(
            init_scale=self._w_init_scale,
            widening_factor=self._widening_factor,
            name="mlp",
        )(layer_norm(x, name="mlp_ln"))                                # (4)!
        x = x + dense_out                                              # (5)!

        return x
```

1. Haiku의 특징이다. `__call__` 내부에서 객체를 생성하고 즉시 괄호 `()`를 붙여 함수처럼 호출한다. ([Haiku OOP vs JAX Purity](./06_haiku_oop_vs_jax_purity.md) 참고)
2. **Pre-LN의 핵심 1:** `query` 인자로 $x$를 넘기기 전에 `layer_norm(x)`를 씌워서 넘긴다. 옆길로 빠지는 데이터만 정규화하는 것이다.
3. **잔차 연결(Residual Connection):** 서브 레이어의 결과물인 `attn_out`을 아무런 조작을 가하지 않은 **순수한 원본 $x$**와 더한다. 이것이 고속도로다.
4. **Pre-LN의 핵심 2:** 마찬가지로 `DenseBlock`에 넘기기 직전에 다시 한번 정규화를 수행한다.
5. **두 번째 잔차 연결:** DenseBlock을 통과한 값을 또다시 메인 스트림 $x$에 더해준다. 이로써 하나의 완전한 트랜스포머 블록이 완성된다.

!!! info "제약(Constraint) vs 관례(Convention): 파라미터 공유 없음"
    위 코드에서 `layer_norm(x, name="attn_ln")`과 `layer_norm(x, name="mlp_ln")`처럼 굳이 서로 다른 `name`을 부여한 것은 디버깅을 위한 시각적 관례가 아니다. Haiku는 `name`을 기준으로 가중치 딕셔너리를 관리한다. 만약 이름을 다르게 주지 않으면, 두 정규화 레이어가 학습 가능한 $\gamma$(Scale)와 $\beta$(Offset) 파라미터를 **공유(Share)**해버리는 치명적인 논리적 오류가 발생한다. 어텐션과 MLP가 요구하는 정규화의 뉘앙스가 완전히 다르기 때문에 파라미터를 철저히 분리해야 하는 **시스템적 제약**이다.

---

*[Pre-LN]: Pre-Layer Normalization. 연산(Attention 또는 MLP)을 수행하기 직전에 데이터를 정규화하는 아키텍처 방식. 학습 안정성이 뛰어나 최신 모델의 표준이다.
*[Post-LN]: Post-Layer Normalization. 연산을 수행하고 원본과 더한 직후에 데이터를 정규화하는 초기 트랜스포머의 방식.
*[Warm-up]: 모델 학습 초기 단계에서 Learning Rate(학습률)를 0에 가깝게 설정했다가 서서히 목표치까지 끌어올리는 스케줄링 꼼수(Heuristic). Post-LN의 초기 불안정성을 억제하기 위해 쓰였다.
