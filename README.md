# JAX Deep Learning & AI-Driven Debugging Platform

This project is an active educational workspace designed to explore the core philosophies of JAX and practice real-world debugging, moving beyond simple code replication.

## Project Philosophy

This repository is built with the intent of deeply understanding the underlying mechanics of numerical computing rather than merely achieving functional code.

- **Deep Dive Learning:** We explore JAX's stateless nature, broadcasting behaviors, and XLA compiler mechanics. This is achieved through interactive Python REPL scripts integrated with structured MkDocs documentation.
- **Socratic Debugging Training:** Critical, real-world bugs are not simply fixed and forgotten. Instead, they are preserved as persistent "Debugging Problems" with the assistance of a custom AI mentor, serving as continuous training assets.

## Architecture & Features

### 1. MkDocs Material & Jupyter Notebook Rendering
The documentation environment allows execution results to be viewed directly in the browser. It features interactive hover annotations to provide immediate, contextual explanations of complex code blocks without breaking the reading flow.

### 2. Autonomous Agent Skills
This workspace integrates two custom AI agent skills powered by the Google Antigravity SDK:
- **Educational Document Creator:** Automates the conversion of fragmented code experiments and discussions into structured, comprehensive MkDocs materials.
- **Debugging Mentor:** A strict, senior-level mentor persona. It utilizes the Socratic method and a strict 5-fail limit to build debugging resilience, ensuring learners trace the root cause before any direct answers are provided.

### 3. Debugging Problem Repository
Significant bugs encountered during development are systematically archived in `docs/educational/debugging/`. These permanent training problems incorporate deliberate code noise, YAML frontmatter for metadata tracking, and hidden mentor hints via HTML `<details>` tags to simulate realistic debugging scenarios.

## Quick Start

```bash
# Synchronize project dependencies
uv sync

# Run the local documentation server
mkdocs serve
```

## Directory Guide

- `src/`: Pure JAX practice implementations (e.g., MNIST, Transformers).
- `docs/educational/`: Deep-dive educational documentation rendered via MkDocs.
- `docs/educational/debugging/`: The repository of practical debugging problems curated by the AI mentor.
- `.agents/skills/`: Custom AI agent skill configurations and architectural guidelines.

## References

This project utilizes various educational resources and literature to guide the learning process.

- ***Deep Learning with Jax*** by Grigory Sapunov (primary reference)
- *(Additional sources and research papers will be added as the curriculum expands)*
