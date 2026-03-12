"""Globals compartidos y carga de configuración desde .env y archivos Markdown."""
from pathlib import Path

# Poblado en main() desde .env
API_KEYS: dict = {}

# URL base de Ollama (API local)
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Cuando True, deshabilita el modo 'think' en modelos Ollama que lo soporten (ej. qwen3.5:35b)
NO_THINK: bool = False

# Tamaño del contexto enviado a Ollama (num_ctx). El default de Ollama es 2048, insuficiente
# para prompts largos + thinking. Se puede sobreescribir con --ollama-num-ctx.
OLLAMA_NUM_CTX: int = 16384


def load_env(env_path=".env"):
    """Carga variables de un archivo .env sin dependencias externas."""
    env = {}
    path = Path(env_path)
    if not path.is_file():
        # Fallback: buscar junto a la raíz del proyecto (un nivel arriba del script)
        path = Path(__file__).resolve().parent.parent / ".env"
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def load_strategies(strategies_path=None):
    """Carga estrategias de optimización desde un archivo Markdown con secciones ## Nombre."""
    candidates = []
    if strategies_path:
        candidates.append(Path(strategies_path))
    candidates += [
        Path(__file__).resolve().parent.parent / "optimization_strategies.md",
        Path("optimization_strategies.md"),
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if not path:
        return []
    strategies = []
    current_name = None
    current_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current_name and current_lines:
                strategies.append({"name": current_name, "text": "\n".join(current_lines).strip()})
            current_name = line[3:].strip()
            current_lines = []
        elif current_name and line.strip() and not line.startswith("#"):
            current_lines.append(line)
    if current_name and current_lines:
        strategies.append({"name": current_name, "text": "\n".join(current_lines).strip()})
    return strategies
