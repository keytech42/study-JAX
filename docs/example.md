# Markdown Execution with Matplotlib & Caching

이 문서는 `markdown-exec`를 활용하여 **Matplotlib 시각화**, **코드/결과 스위칭 UI(Tabs)**, **캐싱(Caching)**, 그리고 **파일별 독립 실행 환경(Session)**을 구성한 최종 PoC입니다.

## 1. 런타임 UI 스위칭 (Tabs 기능)
`markdown-exec`는 기본적으로 코드 블록과 실행 결과를 상하로 배치합니다. 하지만 `tabs="true"` 속성을 사용하면 코드와 결과를 탭(Tab) 형태로 분리하여 독자가 런타임에 클릭으로 스위칭하며 볼 수 있습니다.

```python exec="on" tabs="true"
import sys
print("이 출력은 'Result' 탭에 나타나며, 소스 코드는 'Code' 탭에서 확인할 수 있습니다.")
print(f"현재 실행 중인 Python 환경: {sys.prefix}")
```

## 2. Matplotlib 시각화 연동
시각화 결과물(그래프)을 마크다운 문서에 렌더링하려면 결과를 이미지(SVG 또는 Base64 PNG)로 변환한 후 HTML로 직접 출력해야 합니다.
블록에 `html="true"` 속성을 주어 렌더링을 허용합니다.

```python exec="on" tabs="true" html="true" session="example_md_session" cache="true"
import io
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 그래프 그리기
plt.figure(figsize=(6, 4))
plt.plot(x, y, label="sin(x)", color="blue")
plt.title("Matplotlib in MkDocs")
plt.legend()
plt.tight_layout()

# 결과를 SVG로 변환하여 HTML로 출력
buf = io.StringIO()
plt.savefig(buf, format='svg')
plt.close() # 메모리 누수 방지

# HTML 태그로 감싸서 출력 (결과 탭에 렌더링됨)
print(f"<div style='text-align: center;'>{buf.getvalue()}</div>")
```

## 3. 세션(Session)을 이용한 파일 단위 환경 독립
`session="<세션이름>"` 속성을 지정하면 동일한 세션 이름을 가진 블록끼리는 변수와 임포트 상태를 공유합니다.
파일마다 고유한 세션 이름(예: `session="example_md_session"`)을 부여하면, 파일별로 완전히 독립된 실행 환경을 보장할 수 있습니다.

```python exec="on" session="example_md_session"
# 위 블록에서 선언한 x와 y 변수를 그대로 재사용할 수 있습니다.
print(f"이전 블록에서 생성한 x 배열의 크기: {x.shape}")
print("동일한 session 이름을 사용했기 때문에 상태가 유지됩니다.")
```

## 4. 빌드 성능 및 캐싱 가이드
데이터 전처리나 모델 학습 등 실행 시간이 오래 걸리는 코드는 `cache="true"` 속성을 사용하여 결과를 디스크에 캐싱할 수 있습니다.
코드가 변경되지 않으면 다음 빌드 시에는 캐시된 결과를 즉시 가져옵니다.

### 💡 전체 강제 재빌드 (Cache Invalidation) 방법
캐시를 완전히 무시하고 처음부터 모든 파이썬 코드를 다시 실행하여 빌드하고 싶다면, 로컬 환경(UV 기반)에서 다음 명령어들을 활용할 수 있습니다.

1. **캐시 파일 삭제 후 빌드:** `markdown-exec`는 프로젝트 루트의 특정 숨김 디렉토리(보통 `.cache/` 또는 운영체제 임시 폴더)에 데이터를 캐싱합니다. 명시적으로 환경을 초기화하려면 터미널에서 캐시 폴더를 비우는 스크립트를 실행하거나, `mkdocs build --clean` 옵션을 활용합니다.
2. **코드 블록 ID 변경:** 특정 블록만 강제로 다시 실행하고 싶다면 해당 코드 블록에 `id="new-id"` 속성을 변경하거나 추가하면 해당 블록의 캐시가 무효화됩니다.
