# turns recipes.json into training_data.jsonl for fine-tuning (instruction -> recipe output)

import json
from pathlib import Path
from typing import Any


def _get_recipes_path() -> Path:
    return Path(__file__).resolve().parent / "recipes.json"


def _get_output_path() -> Path:
    return Path(__file__).resolve().parent / "training_data.jsonl"


def _format_recipe_response(recipe: dict[str, Any]) -> str:
    name = recipe.get("name", "Unknown")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", "")
    return (
        f"Recipe: {name}\n"
        f"Ingredients: {', '.join(ingredients)}\n"
        f"Instructions: {instructions}"
    )


def _instruction_for_ingredients(ingredients: list[str]) -> str:
    if not ingredients:
        return "I have no specific ingredients. Suggest a recipe from your list."
    return "I have " + ", ".join(ingredients) + ". Suggest a recipe I can make."


def build_training_examples(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for recipe in recipes:
        ingredients = recipe.get("ingredients", [])
        output = _format_recipe_response(recipe)
        instruction = _instruction_for_ingredients(ingredients)
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })
        if len(ingredients) >= 2:
            subset = ingredients[:2]
            instruction_sub = _instruction_for_ingredients(subset)
            examples.append({
                "instruction": instruction_sub,
                "input": "",
                "output": output,
            })
    return examples


def build_chat_format(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # user/assistant messages for HF style training
    out = []
    for ex in examples:
        user = ex["instruction"]
        if ex.get("input"):
            user = user + "\n" + ex["input"]
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["output"]},
            ]
        })
    return out


def main() -> Path:
    with open(_get_recipes_path(), encoding="utf-8") as f:
        recipes = json.load(f)
    examples = build_training_examples(recipes)
    chat_examples = build_chat_format(examples)
    output_path = _get_output_path()
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chat_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    path = main()
    print("Training data written to:", path)
