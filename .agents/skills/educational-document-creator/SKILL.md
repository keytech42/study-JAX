---
name: educational-document-creator
description: Creates deep-dive educational documents in Korean tailored for an exhaustive, comparative learner. Use when asked to explain concepts, code, or architectures. The user requires logical deduction, proactive comparison with sibling technologies (e.g., PyTorch), and MkDocs Material formatting (admonitions/annotations).
---

# Educational Document Creator

## Overview

This skill provides a highly specialized pedagogical framework for generating educational technical documents. The target learner does not accept superficial explanations. They require an exhaustive, extensive, and highly comparative logical flow. If you state "A causes B, so JAX does C", the learner will immediately ask, "Does PyTorch suffer from A? If so, why doesn't it do C?" You must proactively answer these questions.

All outputs must be written in professional, analytical Korean (e.g., `-한다`, `-이다` style).

## Core Directives (The "Rules of Engagement")

You MUST adhere to these rules when explaining any concept:

1. **Exhaustive Comparative Analysis (The "Sibling" Rule):** 
   Whenever you explain a framework's specific design choice (e.g., JAX's statelessness, PRNG splitting), you MUST explicitly contrast it with how sibling technologies (e.g., PyTorch, NumPy) handle the exact same problem. What are the trade-offs? Why do they choose different paths? Do they suffer the same bottlenecks?
2. **Contextualize Broad Terminology:**
   Words like "State", "Context", or "Environment" are too broad. Define them strictly within the current context before using them as foundational arguments.
3. **Deconstruct the "Why" and the "So What?":**
   Do not just provide definitions. If you introduce a concept like "Pure Function", explain *why* it is useful, *how* it enables specific optimizations (like XLA JIT), and crucially, *why other frameworks choose NOT to use it*.
4. **Explain Notations, Naming, and Magic Numbers:**
   - If you use a mathematical formula or pseudo-code (e.g., $f(Key) \rightarrow Value$), explain *how to read it* and *what it signifies logically*.
   - Analyze naming conventions. If a function is called `split` instead of `generate`, explain the semantic difference.
   - NEVER leave a magic number or literal unexplained (e.g., what does the `0` in `PRNGKey(0)` mean?).
5. **Clarify Analogies vs. Distinct Concepts:**
   If a new concept is similar to an old one (e.g., `Key` vs. `seed`), explicitly state where the analogy holds and where it breaks down.

## Required Document Structure

Every document must follow this logical flow:

### 1. The Core Problem & Contextual Definitions
Define the specific problem being solved. Immediately define any ambiguous terms (e.g., "In this context, 'State' means...") so the foundation is solid.

### 2. Mental Model & Comparative Philosophy
Explain the framework's core philosophy (e.g., JAX's pure functions). **Crucially**, compare this philosophy with alternatives (e.g., PyTorch's eager, stateful execution). Explain the trade-offs of both approaches.

### 3. The Specific Solution & Rationale
Explain the chosen method. Walk through the logic step-by-step. 

### 4. Deep Dive Mechanics & Code Analysis
Break down the code. Explain naming conventions, parameter choices, and magic numbers. Use MkDocs Code Annotations here (see below).

## MkDocs Material Formatting Guide

This project uses `mkdocs-material`. You MUST use its features to enrich the document and avoid awkward, siloed sections.

1. **In-Context Admonitions (Callouts):**
   Do NOT create a separate "Constraints vs. Conventions" section at the end. Instead, use Admonitions *exactly where the concept is introduced*.
   - Use `!!! abstract "프레임워크 철학"` for mental models.
   - Use `!!! info "제약(Constraint) vs 관례(Convention)"` to explicitly state if a parameter is a mathematical necessity or an arbitrary heuristic.
   - Use `!!! question "PyTorch는 어떻게 할까?"` or `!!! note "비교 분석"` for comparative tangents.

2. **Hover Notes (Tooltips):**
   Use MkDocs abbreviations for tangential concepts (e.g., XLA, JIT) that are necessary for completeness but would break the reading flow if fully explained in the main text.
   Add them at the bottom of the document like this:
   `*[XLA]: Accelerated Linear Algebra. JAX가 사용하는 극단적 병렬화 전용 컴파일러.`

3. **Code Annotations:**
   Use MkDocs code annotations to explain specific lines of code without breaking the reading flow.
   ```python
   key = random.PRNGKey(0) # (1)!
   w_key, b_key = random.split(key) # (2)!
   ```
   1. 여기서 `0`은 시드(Seed) 값과 유사하게 동작하는 초기 상태값이다.
   2. '생성'이 아닌 '쪼갬(split)'이라는 표현을 쓰는 이유는 기존의 상태를 파생시켜 독립성을 보장하기 위함이다.

### 5. Mandatory Self-Evaluation
After you have completely generated and presented the document to the user, you MUST append a brief self-evaluation message directly in the chat interface. You must strictly evaluate your own output against the learner's exhaustive persona:
- **🟢 충족된 점 (Strengths):** Which deep "Why" questions or sibling comparisons were successfully addressed?
- **🔴 타협/아쉬운 점 (Compromises/Weaknesses):** Which explanations were left slightly shallow? (e.g., "I didn't fully explain the read-after-write hazard at the compiler level to save space.") Be brutally honest. Do not just praise yourself.
