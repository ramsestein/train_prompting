"""Lectura/escritura de ficheros y parsing de anotaciones BRAT."""
import glob
import os
import random
import re
from pathlib import Path


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_ann(ann_text):
    """Devuelve lista de tuplas (tipo, texto) desde formato BRAT .ann."""
    entities = []
    for line in ann_text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].startswith("T"):
            type_info = parts[1]
            first_token = type_info.split(" ")[0]
            entity_type = first_token
            entity_text = parts[2].strip()
            entities.append((entity_type, entity_text))
    return entities


def extract_entity_types(brat_dir):
    """Recorre todos los .ann de un directorio y extrae el set de tipos de entidad."""
    types = set()
    for ann_path in glob.glob(str(Path(brat_dir) / "*.ann")):
        for etype, _ in parse_ann(load_text(ann_path)):
            types.add(etype)
    return types


def parse_model_output(output, valid_types):
    """
    Parsea el texto anotado inline con el formato [**TIPO: texto**].
    Extrae lista de tuplas (tipo, texto).
    """
    entities = []
    for match in re.finditer(r"\[\*\*([A-ZÁÉÍÓÚa-záéíóú_]+)\s*:\s*([^\]\*]+?)\*\*\]", output):
        etype = match.group(1).strip().upper()
        etext = match.group(2).strip()
        if etype in valid_types and etext:
            entities.append((etype, etext))
    return entities


def get_training_samples(brat_dir, n=10):
    txt_files = sorted(glob.glob(str(Path(brat_dir) / "*.txt")))
    selected = random.sample(txt_files, min(n, len(txt_files)))
    samples = []
    for txt_path in selected:
        ann_path = txt_path[:-4] + ".ann"
        if os.path.exists(ann_path):
            ann_raw = load_text(ann_path)
            samples.append({
                "name": os.path.basename(txt_path),
                "text": load_text(txt_path),
                "ann_raw": ann_raw,
                "entities": parse_ann(ann_raw),
            })
    return samples
