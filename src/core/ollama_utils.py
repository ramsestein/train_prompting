"""Utilidades para descubrir y seleccionar modelos Ollama."""
import sys

import requests

from . import config


def list_ollama_models():
    """Devuelve la lista de modelos instalados en Ollama, o [] si no disponible."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def pick_ollama_model():
    """Lista los modelos Ollama disponibles e invita al usuario a elegir uno."""
    models = list_ollama_models()
    if not models:
        sys.exit(
            "Error: Ollama no está disponible o no tiene modelos instalados.\n"
            "  Asegúrate de que Ollama esté corriendo: ollama serve"
        )
    print("\nModelos Ollama disponibles:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input(f"Elige un modelo [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print(f"  Opción inválida, introduce un número entre 1 y {len(models)}.")
