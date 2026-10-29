import typer
import json
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

from src.models import AgentTranscript, GroundTruth, TaskDefinition, OverallEvaluation
from src.evaluator import AgentEvaluator

app = typer.Typer(help="Agentic AI Evaluation CLI")
console = Console()

@app.command()
def evaluate(
    task_definition: Path = typer.Option(..., "--task-definition", "-t", help="Path to the task definition JSON file."),
    agent_transcript: Path = typer.Option(..., "--agent-transcript", "-a", help="Path to the agent transcript JSON file."),
    ground_truth: Path = typer.Option(..., "--ground-truth", "-g", help="Path to the ground truth JSON file.")
):
    """
    Evaluates an AI agent's performance against a task definition and ground truth.
    """
    console.print(Panel("[bold blue]Starting Agent Evaluation[/bold blue]", expand=False))

    try:
        task_def_data = json.loads(task_definition.read_text())
        agent_transcript_data = json.loads(agent_transcript.read_text())
        ground_truth_data = json.loads(ground_truth.read_text())
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: File not found - {e.filename}[/bold red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error: Invalid JSON in file - {e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        task_def = TaskDefinition.model_validate(task_def_data)
        agent_trans = AgentTranscript.model_validate(agent_transcript_data)
        gt = GroundTruth.model_validate(ground_truth_data)
    except Exception as e:
        console.print(f"[bold red]Error validating input data models: {e}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Task: [bold green]{task_def.task_name}[/bold green] (ID: {agent_trans.task_id})")
    console.print(f"Agent Transcript: [cyan]{agent_transcript.name}[/cyan]")
    console.print(f"Ground Truth: [cyan]{ground_truth.name}[/cyan]")
    console.print("")

    evaluator = AgentEvaluator(task_definition=task_def, ground_truth=gt)
    overall_evaluation = evaluator.evaluate(agent_transcript=agent_trans)

    console.print(Panel(f"[bold yellow]Overall Score: {overall_evaluation.overall_score:.2f}%[/bold yellow]", expand=False))

    results_table = Table(title="Detailed Evaluation Results")
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Score (%)", style="magenta")
    results_table.add_column("Details", style="green")

    for result in overall_evaluation.results:
        results_table.add_row(
            result.metric,
            f"{result.score:.2f}",
            json.dumps(result.details, indent=2)
        )
    console.print(results_table)

    console.print("\n[bold blue]Evaluation Complete.[/bold blue]")

if __name__ == "__main__":
    app()
