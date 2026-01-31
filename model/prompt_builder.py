# build the prompt we send to the model (recipe list + user message)

from typing import Any


SYSTEM_PROMPT = """You are a helpful recipe assistant. You suggest recipes ONLY from the provided recipe list. Do not invent or hallucinate recipes. If the user's ingredients match one or more recipes below, recommend the best match(es) and briefly explain why. If no recipe matches well, say so politely and suggest they try different ingredients from the list. Keep responses concise and structured."""


def format_recipe_for_prompt(recipe: dict[str, Any]) -> str:
    name = recipe.get("name", "Unknown")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", "")
    return f"- **{name}**\n  Ingredients: {', '.join(ingredients)}\n  Instructions: {instructions}"


def format_recipe_for_response(recipe: dict[str, Any]) -> str:
    """Single recipe as plain text for fallback response."""
    name = recipe.get("name", "Unknown")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", "")
    return (
        f"Recipe: {name}\n"
        f"Ingredients: {', '.join(ingredients)}\n"
        f"Instructions: {instructions}"
    )


def build_recipe_prompt(
    user_message: str,
    matching_recipes: list[dict[str, Any]],
    include_all_recipes: bool = False,
    all_recipes: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system = SYSTEM_PROMPT

    if matching_recipes:
        recipe_text = "\n\n".join(format_recipe_for_prompt(r) for r in matching_recipes)
        context = (
            "Here are the recipes that match the user's ingredients (use ONLY these):\n\n"
            + recipe_text
        )
    else:
        if include_all_recipes and all_recipes:
            recipe_text = "\n\n".join(
                format_recipe_for_prompt(r) for r in all_recipes[:15]
            )
            context = (
                "No recipes matched the user's ingredients exactly. "
                "Here is a subset of available recipes for reference:\n\n"
                + recipe_text
            )
        else:
            context = (
                "No recipes in the dataset match the user's ingredients. "
                "Politely tell the user and suggest they try different ingredients."
            )

    user_prompt = f"{context}\n\n---\n\nUser message: {user_message}\n\nRespond with a helpful recipe suggestion based only on the recipes above."

    return system, user_prompt
