import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


DEFAULT_SELECTOR_PROMPT = """You are selecting the next speaker in a multi-agent discussion.

Available roles:
{roles}

Conversation so far:
{history}

Choose exactly one agent from {participants} to speak next.
Prefer forward progress and avoid repetition.
"""


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_task(config: Mapping[str, Any], cli_task: str | None, cli_task_file: str | None) -> str:
    if cli_task:
        return cli_task
    if cli_task_file:
        return Path(cli_task_file).read_text(encoding="utf-8")
    task = config.get("task")
    if not task:
        raise ValueError("No task provided. Add 'task' to the JSON or pass --task / --task-file.")
    return str(task)


def build_model_client(config: Mapping[str, Any]) -> OpenAIChatCompletionClient:
    model_cfg = config.get("model_client", {})
    provider = model_cfg.get("provider", "openai")
    kwargs = dict(model_cfg.get("kwargs", {}))
    if provider != "openai":
        raise ValueError(
            f"Unsupported model_client.provider={provider!r}. This script currently supports only 'openai'."
        )
    if "model" not in kwargs:
        raise ValueError("model_client.kwargs.model is required.")
    return OpenAIChatCompletionClient(**kwargs)


def build_agents(config: Mapping[str, Any], model_client: OpenAIChatCompletionClient) -> list[AssistantAgent]:
    agents_cfg = config.get("agents", [])
    if not agents_cfg:
        raise ValueError("The config must contain a non-empty 'agents' list.")

    agents: list[AssistantAgent] = []
    for agent_cfg in agents_cfg:
        if "name" not in agent_cfg:
            raise ValueError(f"Every agent needs a 'name'. Bad entry: {agent_cfg!r}")
        agent = AssistantAgent(
            name=agent_cfg["name"],
            model_client=model_client,
            description=agent_cfg.get("description"),
            system_message=agent_cfg.get("system_message", "You are a helpful AI assistant."),
        )
        agents.append(agent)
    return agents


def build_termination(config: Mapping[str, Any]):
    term_cfg = config.get("termination", {})
    termination = None

    text_mention = term_cfg.get("text_mention")
    max_messages = term_cfg.get("max_messages")

    if text_mention:
        termination = TextMentionTermination(str(text_mention))
    if max_messages is not None:
        max_term = MaxMessageTermination(int(max_messages))
        termination = max_term if termination is None else (termination | max_term)

    return termination


def build_team(config: Mapping[str, Any], agents: list[AssistantAgent], model_client: OpenAIChatCompletionClient):
    team_cfg = config.get("team", {})
    team_type = team_cfg.get("type", "selector")
    termination = build_termination(config)
    name = team_cfg.get("name")
    description = team_cfg.get("description")
    max_turns = team_cfg.get("max_turns")

    if team_type == "selector":
        return SelectorGroupChat(
            agents,
            model_client=model_client,
            name=name,
            description=description,
            termination_condition=termination,
            max_turns=max_turns,
            selector_prompt=team_cfg.get("selector_prompt", DEFAULT_SELECTOR_PROMPT),
            allow_repeated_speaker=bool(team_cfg.get("allow_repeated_speaker", False)),
            max_selector_attempts=int(team_cfg.get("max_selector_attempts", 3)),
        )

    if team_type == "round_robin":
        return RoundRobinGroupChat(
            agents,
            name=name,
            description=description,
            termination_condition=termination,
            max_turns=max_turns,
        )

    raise ValueError("team.type must be either 'selector' or 'round_robin'.")


def message_to_record(message: Any) -> dict[str, Any]:
    if hasattr(message, "dump"):
        return dict(message.dump())
    return {
        "type": type(message).__name__,
        "source": getattr(message, "source", None),
        "content": getattr(message, "content", str(message)),
    }


def is_chat_message_record(record: Mapping[str, Any]) -> bool:
    msg_type = str(record.get("type", ""))
    return msg_type.endswith("Message")


def message_text(message: Any) -> str:
    if hasattr(message, "to_text"):
        try:
            return str(message.to_text())
        except Exception:
            pass
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return str(message)
    try:
        return json.dumps(content, ensure_ascii=False, indent=2)
    except TypeError:
        return str(content)


def extract_final_answer(task_result: TaskResult, final_cfg: Mapping[str, Any]) -> str:
    preferred_source = final_cfg.get("source")
    strip_tokens = [str(x) for x in final_cfg.get("strip_mentions", [])]

    chosen = None
    if preferred_source:
        for message in reversed(task_result.messages):
            if getattr(message, "source", None) == preferred_source and hasattr(message, "to_text"):
                chosen = message
                break

    if chosen is None:
        for message in reversed(task_result.messages):
            if hasattr(message, "to_text") and getattr(message, "source", None) != "user":
                chosen = message
                break

    if chosen is None:
        return ""

    text = message_text(chosen).strip()
    for token in strip_tokens:
        text = text.replace(token, "").strip()
    return text + "\n"


async def run_from_config(
    config_path: str,
    task_override: str | None = None,
    task_file: str | None = None,
    answer_out: str | None = None,
    run_dir_override: str | None = None,
) -> None:
    config = load_json(config_path)
    task = resolve_task(config, task_override, task_file)

    outputs_cfg = config.get("outputs", {})
    run_dir = Path(run_dir_override or outputs_cfg.get("run_dir", "autogen_run"))
    run_dir.mkdir(parents=True, exist_ok=True)

    answer_path = Path(answer_out or outputs_cfg.get("answer_path", run_dir / "answer.txt"))
    transcript_path = run_dir / outputs_cfg.get("transcript_file", "transcript.jsonl")
    result_path = run_dir / outputs_cfg.get("result_file", "task_result.json")
    state_path = run_dir / outputs_cfg.get("state_file", "team_state.json")

    model_client = build_model_client(config)
    try:
        agents = build_agents(config, model_client)
        team = build_team(config, agents, model_client)

        final_result: TaskResult | None = None

        with open(transcript_path, "w", encoding="utf-8") as transcript_file:
            async for item in team.run_stream(task=task):
                if isinstance(item, TaskResult):
                    final_result = item
                    continue

                record = message_to_record(item)
                transcript_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if is_chat_message_record(record):
                    source = record.get("source", "unknown")
                    content = record.get("content", "")
                    print(f"\n[{source}]\n{content}\n")

        if final_result is None:
            raise RuntimeError("The team finished without returning a TaskResult.")

        result_payload = {
            "stop_reason": final_result.stop_reason,
            "messages": [message_to_record(m) for m in final_result.messages],
        }
        write_json(result_path, result_payload)

        final_answer = extract_final_answer(final_result, config.get("final_answer", {}))
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(final_answer, encoding="utf-8")

        state = await team.save_state()
        write_json(state_path, state)

        print("Saved files:")
        print(f"- Transcript: {transcript_path}")
        print(f"- Task result: {result_path}")
        print(f"- Team state:  {state_path}")
        print(f"- Answer:      {answer_path}")
    finally:
        await model_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AutoGen multi-agent team from a JSON config.")
    parser.add_argument("config", help="Path to the JSON configuration file.")
    parser.add_argument("--task", help="Override the task from the config with a literal string.")
    parser.add_argument("--task-file", help="Read the task from a text/markdown file.")
    parser.add_argument(
        "--answer-out",
        help="Path where the final textual answer will be saved. Overrides outputs.answer_path.",
    )
    parser.add_argument(
        "--run-dir",
        help="Directory where transcript.jsonl, task_result.json, and team_state.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_from_config(
            config_path=args.config,
            task_override=args.task,
            task_file=args.task_file,
            answer_out=args.answer_out,
            run_dir_override=args.run_dir,
        )
    )


if __name__ == "__main__":
    main()
