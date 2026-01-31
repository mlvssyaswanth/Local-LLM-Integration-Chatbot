# load recipes and search by ingredients

import json
from pathlib import Path
from typing import Any


def _get_recipes_path() -> Path:
    return Path(__file__).resolve().parent / "recipes.json"


def load_recipes() -> list[dict[str, Any]]:
    path = _get_recipes_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize(s: str) -> str:
    return s.strip().lower()


def _expand_for_match(ingredient: str) -> set[str]:
    # handle egg/eggs etc so we match either way
    n = _normalize(ingredient)
    out = {n}
    if n.endswith("s") and len(n) > 1:
        out.add(n[:-1])
    if not n.endswith("s") and len(n) > 1:
        out.add(n + "s")
    return out


def _ingredients_match(recipe_ingredients: list[str], user_ingredients: list[str]) -> tuple[bool, int]:
    recipe_set = {_normalize(i) for i in recipe_ingredients}
    user_expanded = set()
    for i in user_ingredients:
        user_expanded |= _expand_for_match(i)
    overlap = recipe_set & user_expanded
    return len(overlap) > 0, len(overlap)


class RecipeLoader:
    def __init__(self) -> None:
        self._recipes: list[dict[str, Any]] | None = None

    @property
    def recipes(self) -> list[dict[str, Any]]:
        if self._recipes is None:
            self._recipes = load_recipes()
        return self._recipes

    def find_by_ingredients(
        self,
        ingredients: list[str],
        max_results: int = 5,
        min_matches: int = 1,
    ) -> list[dict[str, Any]]:
        # returns recipes that have at least one of the ingredients, sorted by match count
        if not ingredients:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for recipe in self.recipes:
            has_match, count = _ingredients_match(
                recipe.get("ingredients", []),
                ingredients,
            )
            if has_match and count >= min_matches:
                scored.append((count, recipe))

        scored.sort(key=lambda x: (-x[0], x[1]["name"]))
        return [r for _, r in scored[:max_results]]

    def get_all(self) -> list[dict[str, Any]]:
        return list(self.recipes)
