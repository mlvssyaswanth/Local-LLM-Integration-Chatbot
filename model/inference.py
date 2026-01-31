# calls ollama with recipe context so we don't get random recipes

import re
from typing import Any

from dataset.loader import RecipeLoader
from model.prompt_builder import build_recipe_prompt


def _parse_ingredients_from_message(message: str) -> list[str]:
    # split on comma, "and", etc
    message = message.strip().lower()
    for sep in [",", "/", " and ", " & "]:
        message = message.replace(sep, " ")
    tokens = [t.strip() for t in re.split(r"\s+", message) if t.strip()]
    return tokens if tokens else [message] if message else []


class RecipeInferenceEngine:
    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        recipe_loader: RecipeLoader | None = None,
        max_recipe_context: int = 5,
    ) -> None:
        self.model_name = model_name
        self.loader = recipe_loader or RecipeLoader()
        self.max_recipe_context = max_recipe_context

    def _call_ollama(self, system: str, user: str) -> str:
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError(
                "Ollama Python client not installed. Run: pip install ollama"
            ) from e

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            msg = str(e).lower()
            if "connection" in msg or "refused" in msg or "not found" in msg:
                raise RuntimeError(
                    "Ollama is not running or model not found. "
                    "Start Ollama and run: ollama run " + self.model_name
                ) from e
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def suggest_recipe(self, user_message: str) -> str:
        ingredients = _parse_ingredients_from_message(user_message)
        matching = self.loader.find_by_ingredients(
            ingredients,
            max_results=self.max_recipe_context,
        )
        all_recipes = self.loader.get_all() if not matching else None
        system, user_prompt = build_recipe_prompt(
            user_message,
            matching_recipes=matching,
            include_all_recipes=not matching,
            all_recipes=all_recipes,
        )
        return self._call_ollama(system, user_prompt)
