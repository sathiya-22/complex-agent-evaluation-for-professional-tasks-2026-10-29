The Real Problem: Complex Agent Evaluation for Professional Tasks

Evaluating AI agents on complex, professional assignments, such as tax preparation or legal document analysis, presents a significant challenge. Current methods often lack the robustness, standardization, and nuance required to accurately assess agent performance in real-world scenarios. This deficiency directly impacts the reliability and trustworthiness of agentic systems, making it difficult for developers and businesses to confidently deploy them in critical applications. Without clear, objective benchmarks, improving these agents becomes a trial-and-error process, hindering innovation and adoption.

Why This Project Shape/Stack Was Chosen: Python CLI with Structured Data and Fixtures

This prototype is a Python CLI tool because:
1.  **Ease of Use for Developers:** Python is a de-facto standard in the AI/ML community, making it accessible for developers to integrate into their existing workflows. A CLI provides a straightforward interface for running evaluations.
2.  **Structured Data for Complex Scenarios:** Professional tasks involve highly structured inputs and expected outputs. Python's data structures and libraries (like `pydantic` for validation) are excellent for defining and manipulating these complex data models.
3.  **Focus on Evaluation Logic, Not UI/Service Overhead:** The core problem is the *evaluation logic* itself. A CLI allows us to focus solely on this, without the overhead of building a web service, GUI, or distributed system, which are premature for a proof-of-concept for evaluation.
4.  **Provider-Agnostic Design:** The evaluation core does not call an LLM directly. Instead, it processes *agent transcripts* and *human-provided ground truth*. This makes it entirely provider-agnostic, working with any agent system that can output its steps in a defined format.
5.  **Fixture-First Development:** The design emphasizes using fixture data (sample agent transcripts and ground truths) to ensure the tool is immediately usable and verifiable without any API keys or external dependencies. This allows for clear demonstration of its core value.

Setup and Usage Instructions (Zero API Keys Required):

1.  **Clone the repository:**
    ```bash
    git clone <this-repo-url>
    cd <this-repo-directory>
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Run the evaluation with the provided fixture data:**
    ```bash
    python main.py evaluate --task-definition fixtures/tax_task_definition.json --agent-transcript fixtures/agent_transcript_good.json --ground-truth fixtures/ground_truth_good.json
    python main.py evaluate --task-definition fixtures/tax_task_definition.json --agent-transcript fixtures/agent_transcript_bad.json --ground-truth fixtures/ground_truth_good.json
    ```
    This will output evaluation results to the console based on the defined metrics.

Optional Real-LLM Adapter:

This prototype does *not* include an optional real-LLM adapter. The core problem is the *evaluation* of agent behavior, not the generation of that behavior or its comparison to LLM outputs directly. The system evaluates *against ground truth* and structured expectations, which are inherently human-defined for professional tasks. LLMs could potentially be used to *generate* ground truth or *explain* agent failures, but that is a separate problem space from the core evaluation mechanism demonstrated here. This tool is designed to work with *any* agent's output, regardless of how that output was generated.
