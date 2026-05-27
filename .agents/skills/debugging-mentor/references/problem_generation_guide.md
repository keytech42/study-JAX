이 문서는 멘토가 문제를 실제 파일로 생성할 때 반드시 참조해야 하는 아키텍처 및 렌더링 가이드라인이다.

## 1. 디렉토리 구조 및 인덱싱 (Indexing)

- **기본 저장 위치:** 모든 문제는 `docs/educational/debugging/` 하위에 저장된다.
- **폴더 명명 규칙:** `{순번: 3자리}-{주제}` 패턴을 따른다. (예: `001-jax-shape-mismatch/`, `002-silent-state-mutation/`)
- **인덱싱(Index) 파일 업데이트:** 새로운 문제를 생성한 직후, 반드시 `docs/educational/debugging/README.md` (Index 파일)의 마크다운 테이블에 해당 문제의 메타데이터와 링크를 추가(Append)하여 목차를 유지한다.

## 2. 코드 파일 (`problem.py` 등)

- 철저히 실무 코드(Production)처럼 작성한다.
- **절대 멘토의 주석이나 힌트를 이 파일에 넣지 않는다.** 
- 멘티와 합의하여 지정된 노이즈 레벨(Low/Medium/High)에 따라 더미 코드나 복잡한 파이프라인을 섞어 실전성을 높인다.

## 3. 설명서 파일 (`README.md` 및 Frontmatter)

- 파일 최상단에 **반드시 YAML Frontmatter를 작성**하여 다음의 스키마에 맞게 핵심 메타데이터를 기입한다.
  - `id`: (String) `{순번: 3자리}` 형식의 고유 식별자 (예: "001")
  - `title`: (String) 문제의 주제
  - `difficulty`: (String) "Beginner", "Intermediate", "Advanced", "Expert" 중 택 1
  - `noise_level`: (String) "Low", "Medium", "High" 중 택 1
  - `tags`: (List[String]) 핵심 기술 태그 배열
  - `created_at`: (String) 문제 생성일 ("YYYY-MM-DD" 형식)
  - `approved`: (Boolean) 해당 문제가 실제로 테스트되어 목표(학습 의도)에 부합하게 디버깅이 완수되는지 검증되었는지의 여부를 나타낸다. 최초 생성 시 미검증 상태라면 `false`로 설정하고, 멘토가 정상 작동 및 부작용 없음을 완전히 확인(혹은 자동 검증 과정을 거친) 이후에만 `true`로 변경한다.
- 문제의 배경(Context)과 달성해야 할 목표(Goal)를 명시한다.
- 멘토의 Socratic 힌트와 최종 해설은 마크다운의 `<details>` 태그 안에 숨겨두어, 학습자가 코드를 탐색하다가 원할 때만 펼쳐볼 수 있게 한다.

**README.md 템플릿 예시:**
```markdown
---
id: 001
title: "JAX 차원 불일치 (Shape Mismatch)"
difficulty: "Intermediate"
noise_level: "Medium"
tags: ["JAX", "shape", "broadcasting", "zip"]
created_at: "YYYY-MM-DD"
approved: true
---

# Debugging Problem: [문제 제목]

## Context
[문제 상황에 대한 실무적 배경 설명]

## Goal
[해결해야 할 구체적인 목표와 발생하고 있는 에러 현상]

## Mentor's Guidance (Hint)
<details>
<summary>💡 Hint 1: [첫 번째 관문 힌트 제목]</summary>

여기에 Socratic 멘토링 힌트를 작성합니다. (예: "로그의 마지막 줄을 볼까요? 변수 x의 차원이 어떻게 되나요?")

</details>

<details>
<summary>💡 Hint 2: [두 번째 관문 힌트 제목]</summary>

추가적인 힌트 내용.

</details>

<details>
<summary>✅ Solution (Spoiler)</summary>

여기에 버그의 근본 원인과 해결 코드를 설명합니다.

</details>
```
