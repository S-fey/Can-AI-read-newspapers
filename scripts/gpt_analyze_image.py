#!/usr/bin/env python3
"""
Ask GPT-5.5 to answer newspaper front-page questions from an image.

The template groups related questions into "tasks", separated by blank rows
(rows whose "QUESTION" field is empty) - e.g. all the D1.x masthead questions
form one task, all the D2.x layout questions form the next, and so on. Every
question belonging to a task is sent to GPT-5.5 in a SINGLE API call, and the
model returns one answer per question, keyed by question ID (e.g. "D1.1"),
which is written back into the matching row.

Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python analyze_newspaper.py newspaper.jpg
    python analyze_newspaper.py newspaper.jpg --prompt prompt.txt --template template.json --output answer.json
"""

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path

from openai import OpenAI

MODEL = "gpt-5.5"
QUESTION_ID_RE = re.compile(r"[A-Za-z]+\d+\.\d+")


def image_to_data_url(image_path: Path) -> str:
    """Read an image file and return it as a base64 data URL."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def question_id(question_text: str) -> str:
    """Pull the leading ID (e.g. 'D1.7') off a question string."""
    match = QUESTION_ID_RE.match(question_text.strip())
    if not match:
        raise ValueError(f"Could not find a question ID in: {question_text!r}")
    return match.group(0)


def group_into_tasks(questions: list) -> list:
    """Split the template into tasks: consecutive non-empty QUESTION rows,
    separated by one or more blank rows."""
    tasks, current = [], []
    for item in questions:
        if item.get("QUESTION", "").strip():
            current.append(item)
        elif current:
            tasks.append(current)
            current = []
    if current:
        tasks.append(current)
    return tasks


def parse_json_object(text: str) -> dict:
    """Pull the JSON object out of the model's reply.

    The prompt tells the model to return raw JSON only, but this guards
    against the odd stray ```fence or sentence around it.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(text[start:end + 1])


def build_batch_instructions(base_instructions: str, ids: list) -> str:
    """Extend the single-question system prompt to cover a batch of questions."""
    return (
        f"{base_instructions}\n\n"
        "---\n"
        "BATCH MODE: you will receive MULTIPLE questions in one message, each "
        "starting with its own ID (e.g. \"D1.1\"). Apply every rule above "
        "independently and separately to each question - do not let one "
        "question's answer influence another's.\n\n"
        "Return EXCLUSIVELY a single JSON object whose keys are exactly these "
        f"question IDs: {', '.join(ids)}. Each value must itself be an object "
        "with the fields AI_Answer, AI_Evidence, AI_Explanation, formatted per "
        "the rules above. Answer every question - do not merge, skip, or "
        "reorder them. No text outside the JSON object."
    )


def ask_task(client: OpenAI, base_instructions: str, image_data_url: str, task_items: list) -> dict:
    """Send every question in a task to GPT-5.5 in a single call."""
    ids = [question_id(item["QUESTION"]) for item in task_items]
    instructions = build_batch_instructions(base_instructions, ids)
    questions_text = "\n\n".join(item["QUESTION"] for item in task_items)

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        # generous headroom: each question needs its own Answer/Evidence/Explanation
        max_output_tokens=1000 + 500 * len(task_items),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": questions_text},
                # "high" detail so small print/captions stay legible
                {"type": "input_image", "image_url": image_data_url, "detail": "original"},
            ],
        }],
    )
    return parse_json_object(response.output_text)


def main():
    parser = argparse.ArgumentParser(description="Answer newspaper front-page questions with GPT-5.5.")
    parser.add_argument("image", help="Path to the newspaper front-page image")
    parser.add_argument("--prompt", default="prompt.txt", help="Path to the system prompt file")
    parser.add_argument("--template", default="template.json", help="Path to the question template JSON")
    parser.add_argument("--output", default="answer.json", help="Where to write the filled-in answers")
    args = parser.parse_args()

    base_instructions = Path(args.prompt).read_text(encoding="utf-8").strip()
    questions = json.loads(Path(args.template).read_text(encoding="utf-8"))
    image_data_url = image_to_data_url(Path(args.image))

    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    tasks = group_into_tasks(questions)
    total_questions = sum(len(t) for t in tasks)
    print(f"{total_questions} questions grouped into {len(tasks)} tasks (1 API call each)")

    for i, task_items in enumerate(tasks, start=1):
        ids = [question_id(item["QUESTION"]) for item in task_items]
        print(f"[{i}/{len(tasks)}] {', '.join(ids)}")
        try:
            answers = ask_task(client, base_instructions, image_data_url, task_items)
            for item in task_items:
                qid = question_id(item["QUESTION"])
                answer = answers.get(qid)
                if answer is None:
                    print(f"  warning: no answer returned for {qid}")
                    answer = {}
                item["AI_Answer"] = answer.get("AI_Answer", "")
                item["AI_Evidence"] = answer.get("AI_Evidence", "")
                item["AI_Explanation"] = answer.get("AI_Explanation", "")
        except Exception as e:
            print(f"  failed: {e}")
            for item in task_items:
                item["AI_Answer"] = f"ERROR: {e}"

        # Write after every task so a crash midway doesn't lose progress
        Path(args.output).write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done -> {args.output} ({len(tasks)} API calls total, was {total_questions})")


if __name__ == "__main__":
    main()