# train_prompt

Optimización iterativa de prompts de sistema mediante evaluación automatizada contra anotaciones ground truth en formato BRAT.

Diseñado originalmente para tareas de anonimización de texto clínico (corpus MEDDOCAN), pero funciona con cualquier tarea de etiquetado de entidades siempre que se tenga un corpus en formato BRAT (`.txt` + `.ann`).

---

## Arquitectura del sistema

El sistema combina ideas de **algoritmos evolutivos** (mutación, selección, presión selectiva) con **optimización guiada por LLM** (el optimizador "lee" los errores y genera mutaciones inteligentes en lugar de aleatorias). El resultado es un loop cerrado donde el prompt evoluciona hacia el óptimo de la métrica elegida, evaluando baseline y candidato sobre el **mismo batch** antes de aceptar cambios.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     LOOP PRINCIPAL                          │
                    │                                                             │
            best_pipeline ─► [Worker LLM] ─► baseline_metrics                              │
                    │                                                             │
                    ▼                                                             │
                [Advisors + Optimizador LLM] ─► candidate_pipeline                      │
                    │                                                             │
                    ▼                                                             │
               [Worker LLM en el mismo batch] ─► candidate_metrics                     │
                    │                                                             │
                    ▼                                                             │
                  ┌──────────────────────────────┐                                    │
                  │ Aceptación por umbral (ε)    │                                    │
                  │ Δ = score(cand) - score(base)│                                    │
                  │ ¿Δ > ε?                       │                                    │
                  └──────────────┬───────────────┘                                    │
                       sí / no                                                   │
                      ▼      ▼                                                   │
                  promover cand  conservar best                                      │
                    └─────────────────────────────────────────────────────────────┘
```

---

## Lógica de entrenamiento en profundidad

### 1. El prompt como individuo evolutivo

En lugar de optimizar pesos numéricos (como en ML clásico), aquí **el individuo que evoluciona es el texto del prompt**. En cada iteración se congela el mejor pipeline actual (`best`) como baseline, se evalúa en un batch, y luego se generan candidatos mutados desde ese baseline. Cada candidato se evalúa en el **mismo batch** y se acepta solo si mejora por encima de un umbral `ε`. Esto se comporta como un esquema **(1+λ) con aceptación por umbral**.

```
best_pipeline = {main, review_1, review_2, ...}   ← parent/baseline
      │
    [evaluar baseline en batch B]
      │
      ▼
score_base = metric(best_pipeline, B)
      │
    [mutar desde best, hasta λ intentos]
      │
      ▼
candidate_pipeline = {main', review_1', ...}
      │
    [evaluar candidato en el mismo batch B]
      │
      ▼
Δ = score_cand - score_base
      │
  Δ > ε ?
    sí → aceptar candidato como nuevo best
    no → rechazar candidato y conservar best
```

La diferencia clave con un algoritmo evolutivo clásico es que **las mutaciones no son aleatorias**: el optimizador LLM recibe los errores concretos (FP, FN, métricas por tipo de entidad) y genera cambios dirigidos. Además, cuando hay rechazos en una misma iteración, el optimizador recibe feedback de esos intentos fallidos para evitar repetirlos.

### 2. El pipeline como cromosoma compuesto

El individuo no es un solo prompt, sino un **pipeline de prompts encadenados**:

```
pipeline = {
    "main":     "prompt principal que anota el texto",
    "review_1": "prompt que corrige la salida del main",
    "review_2": "prompt que corrige la salida de review_1",
    "review_3": "prompt que corrige la salida de review_2"
}
```

Esto es análogo a un cromosoma con **genes variables**: el pipeline empieza solo con `main` y puede crecer hasta 4 genes (main + 3 reviews). El optimizador puede mutar cualquier gen independientemente o de forma coordinada.

**Ejecución del pipeline** en cada muestra:

```
texto ──► [main] ──► salida_1 ──► [review_1] ──► salida_2 ──► [review_2] ──► salida_final
```

Cada review recibe como input la salida del paso anterior y puede corregirla: añadir entidades omitidas, eliminar falsos positivos, o ambas cosas. Las reviews actúan como **capas de refinamiento** sucesivas, similar a las capas de una red neuronal pero operando en el espacio de texto natural.

### 3. Presión selectiva adaptativa

El sistema aplica distintos mecanismos de presión según el estado del entrenamiento:

#### 3.1 Estrategias de optimización rotatorias

Cuando el sistema lleva ciclos sin mejorar o con mejoras demasiado pequeñas, inyecta una **estrategia de enfoque** en el prompt del optimizador. Hay 12 estrategias predefinidas (Chain of Thought, Few-shot, Anclas negativas, etc.) que se ciclan secuencialmente:

```
ciclo sin mejora #2  → Estrategia[0]: "Cadena de pensamiento"
ciclo sin mejora #4  → Estrategia[1]: "Divide y vencerás por tipo"
ciclo sin mejora #6  → Estrategia[2]: "Anclas negativas"
...
```

Estas estrategias fuerzan al optimizador a explorar direcciones de mutación diferentes, evitando que quede atrapado en un óptimo local. Es el equivalente a aumentar la tasa de mutación cuando la fitness se estanca.

#### 3.2 Crecimiento del pipeline (añadir reviews)

Cuando el estancamiento supera `--review-step` ciclos (default: 5), el sistema permite al optimizador **añadir una nueva review** al pipeline. Esto amplía el espacio de búsqueda al dar al modelo otra oportunidad de corregir errores. Es análogo a **añadir un nuevo gen al cromosoma** cuando la estructura actual ha alcanzado su límite expresivo.

```
ciclos sin mejora < 5   → solo mutar prompts existentes
ciclos sin mejora ≥ 5   → permitir añadir review_1
ciclos sin mejora ≥ 10  → permitir añadir review_2 (si review_1 ya existe)
ciclos sin mejora ≥ 15  → permitir añadir review_3 (máximo)
```

Las reviews candidatas se prueban inmediatamente: si la iteración con la nueva review no mejora el best, se descarta. Si mejora, se incorpora al best y queda permanente.

#### 3.3 Alternativas para reviews estancadas

Si una review lleva ciclos sin contribuir a mejoras, el sistema sugiere al optimizador que la **rediseñe completamente** con un enfoque alternativo de un banco de 18 alternativas (segunda pasada de cobertura, filtro de FP, especialización por tipo, etc.). Estas se rotan para maximizar la diversidad:

```
review_1 estancada ─► ciclo 1: "Segunda pasada de cobertura"
                   ─► ciclo 2: "Filtro de falsos positivos"
                   ─► ciclo 3: "Especialización en entidades de tiempo"
                   ...
```

Esto equivale a **resetear un gen** que ha convergido prematuramente, reintroduciendo diversidad genética.

#### 3.4 Refinado por párrafo

Cada 3 ciclos consecutivos sin mejora, el sistema selecciona un **párrafo aleatorio** del prompt principal y lo envía al optimizador para que lo reescriba de forma más clara y precisa. Este micro-refinado actúa como una **mutación puntual** que no altera la estructura global pero puede desbloquear mejoras locales.

### 4. Subagentes Advisors: análisis especializado de errores

En cada iteración, antes de que el optimizador genere la mutación, se ejecutan **subagentes especializados** que analizan los errores desde una perspectiva enfocada:

#### Advisor de Recall (analista de FN)
- Recibe: la cadena de prompts + solo las muestras con entidades no detectadas
- Analiza: patrones en los FN (tipos más fallados, contextos donde no detecta, formatos problemáticos)
- Genera: un informe de ~400 palabras con recomendaciones concretas para detectar más entidades

#### Advisor de Precision (analista de FP)
- Recibe: la cadena de prompts + solo las muestras con falsos positivos
- Analiza: patrones en los FP (tipos sobredetectados, contextos confusos, entidades clínicas marcadas como personales)
- Genera: un informe de ~400 palabras con reglas de exclusión y criterios más estrictos

#### Lógica de activación según la métrica

| `--metric` | Advisor activado | Criterio |
|---|---|---|
| `recall` | Solo recall advisor | Siempre |
| `precision` | Solo precision advisor | Siempre |
| `f1` | **El de la métrica más baja** | Si R < P → recall advisor; si P ≤ R → precision advisor |

Cuando se optimiza F1, el sistema identifica cuál de las dos métricas componentes está más baja y activa solo ese advisor. Esto concentra la señal de mejora en el flanco más débil, evitando enviar consejos contradictorios al optimizador.

Los informes se inyectan en el system message del optimizador central como bloques prioritarios, de modo que las mutaciones generadas estén informadas por el análisis especializado.

### 5. El optimizador central: mutación dirigida

El optimizador recibe un paquete completo de información:

| Componente | Contenido |
|---|---|
| **Cadena de prompts actual** | `main` + cada `review_N` |
| **Resultados del batch** | Para cada muestra: texto (500 chars), ground truth, predicciones, métricas individuales |
| **Métrica promedio** | F1/P/R actual |
| **Informe del advisor** | Análisis especializado de FN o FP (según la métrica) |
| **Estrategia sugerida** | Una de las 12 del banco (si hay estancamiento) |
| **Alternativas de review** | Sugerencia de rediseño para reviews estancadas |
| **Documentación de contexto** | Archivo externo opcional (CSV, guías, etc.) |

Con todo esto, el optimizador genera una nueva versión completa del pipeline. La respuesta se parsea usando separadores exactos (`---PROMPT:main---`, `---PROMPT:review_1---`, etc.) para extraer cada prompt individualmente.

### 6. Evaluación: fitness function

La función de fitness por pipeline es la **media aritmética** de la métrica elegida sobre las N muestras del batch:

$$\text{fitness} = \frac{1}{N} \sum_{i=1}^{N} m_i$$

donde $m_i$ es F1, precision o recall de la muestra $i$. Las métricas individuales se calculan por **exact match** entre tuplas (tipo, texto):

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

Una predicción es TP solo si tanto el tipo de entidad como el texto extraído coinciden exactamente con el ground truth. Esto penaliza tanto los errores de clasificación (tipo incorrecto) como los de extracción (límites incorrectos).

Para decidir aceptación dentro de la iteración, se compara baseline vs candidato en el mismo batch:

$$
\Delta = \text{fitness}(candidate, B) - \text{fitness}(baseline, B)
$$

Se acepta el candidato solo si:

$$
\Delta > \epsilon
$$

donde $\epsilon$ se controla con `--accept-epsilon`.

### 7. Convergencia y terminación

El entrenamiento termina cuando se cumple alguna de estas condiciones:

1. **Se agotan las iteraciones** (`--iterations`, default 100)
2. **Early stopping**: `--patience` ciclos consecutivos sin aceptar ningún candidato (default 30)

Al terminar, el sistema restaura y guarda el **mejor pipeline visto** durante todo el entrenamiento, no el último. Esto garantiza que incluso si las últimas iteraciones degeneraron, el resultado final es el óptimo observado.

### Resumen: analogía con algoritmos evolutivos

| Concepto evolutivo | Implementación en train_prompt |
|---|---|
| Individuo / cromosoma | Pipeline de prompts (main + reviews) |
| Gen | Cada prompt individual del pipeline |
| Fitness function | Media de la métrica sobre N muestras |
| Operador de mutación | LLM optimizador (mutación dirigida, no aleatoria) |
| Selección | Aceptación en mismo batch (1+λ): baseline vs candidato con umbral ε |
| Tasa de mutación variable | Estrategias inyectadas en estancamiento |
| Crossover | No se usa (población de 1) |
| Ampliación del genoma | Añadir reviews al pipeline |
| Reset de gen | Alternativas para reviews estancadas |
| Mutación puntual | Refinado de párrafo aleatorio |
| Presión selectiva externa | Advisors especializados (recall / precision) |

---

## Estructura del proyecto

```
train_prompt/
├── src/
│   ├── train.py                  # Entry point: main() con el loop evolutivo
│   ├── core/
│   │   ├── config.py             # Globals compartidos y carga de .env
│   │   ├── api.py                # Llamadas a LLM (OpenAI, DeepSeek, Ollama nativo)
│   │   ├── brat.py               # I/O de ficheros y parsing BRAT
│   │   ├── metrics.py            # Cálculo de precision/recall/F1
│   │   ├── worker.py             # Ejecución del pipeline worker + reviews
│   │   ├── optimizer.py          # Generación y optimización de prompts
│   │   ├── advisors.py           # Subagentes especializados (recall/precision)
│   │   ├── ollama_utils.py       # Descubrimiento de modelos Ollama
│   │   └── logger.py             # Log de iteraciones a disco
│   └── tests/
│       └── test_no_think.py      # 18 tests (unitarios + mock + integración)
├── SPACCC_MEDDOCAN/
│   └── corpus/
│       ├── train/brat/           # Corpus de entrenamiento (.txt + .ann)
│       ├── dev/brat/             # Corpus de desarrollo
│       └── test/brat/            # Corpus de test
├── prompt.txt                    # Prompt inicial (se sobreescribe cada iteración)
├── optimization_strategies.md    # 12 estrategias de mutación (Chain of Thought, Few-shot, etc.)
├── review_alternatives.md        # 18 alternativas para rediseñar reviews estancadas
├── etiquetas_*.csv               # Mapeo de etiquetas (referencia)
└── .env                          # Claves de API (no subir al repositorio)
```

---

## Requisitos

- Python 3.8+
- Librería `requests` (`pip install requests`)
- Al menos una fuente de modelo:
  - **OpenAI**: clave en `.env` → `OPENAI_API_KEY`
  - **DeepSeek**: clave en `.env` → `DEEPSEEK_API_KEY`
  - **Ollama** (local): Ollama corriendo en `localhost:11434`

### Archivo `.env`

```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

---

## Ejecución

### Uso básico

```bash
python src/train.py --brat-dir SPACCC_MEDDOCAN/corpus/train/brat --prompt prompt.txt
```

Usa `gpt-3.5-turbo` como worker y `deepseek-chat` como optimizador por defecto.

### Con modelos Ollama

**Selector interactivo** — muestra los modelos que tengas instalados:
```bash
python src/train.py --brat-dir SPACCC_MEDDOCAN/corpus/train/brat --prompt prompt.txt --worker-model ollama
```

**Nombre directo** — si ya sabes el nombre del modelo:
```bash
python src/train.py --brat-dir SPACCC_MEDDOCAN/corpus/train/brat --prompt prompt.txt --worker-model qwen3.5:35b --no-think
```

### Con modelos cloud

```bash
# Worker GPT-4o + optimizador DeepSeek (por defecto)
python src/train.py --brat-dir SPACCC_MEDDOCAN/corpus/train/brat --prompt prompt.txt --worker-model gpt-4o

# Worker Ollama local + optimizador GPT-4o
python src/train.py --brat-dir SPACCC_MEDDOCAN/corpus/train/brat --prompt prompt.txt \
  --worker-model llama3:8b --optimizer-model gpt-4o
```

---

## Parámetros

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `--brat-dir` / `-d` | (obligatorio) | Carpeta con los pares `.txt` / `.ann` BRAT |
| `--prompt` / `-p` | (obligatorio) | Archivo con el prompt inicial (se sobreescribe) |
| `--worker-model` | `gpt-3.5-turbo` | Modelo que procesa los textos. `ollama` → selector interactivo |
| `--optimizer-model` | `deepseek-chat` | Modelo que optimiza el prompt. `ollama` → selector interactivo |
| `--no-think` | `False` | Deshabilita thinking en modelos Ollama (envía `think: false`) |
| `--ollama-num-ctx` | `16384` | Tamaño del contexto para Ollama (`num_ctx`) |
| `--iterations` / `-n` | `100` | Número máximo de iteraciones |
| `--samples` / `-s` | `10` | Textos muestreados por iteración |
| `--metric` / `-m` | `f1` | Métrica a optimizar: `f1`, `precision` o `recall` |
| `--parallel` | `1` | Llamadas worker en paralelo (útil con modelos cloud rápidos) |
| `--patience` | `30` | Early stopping: ciclos sin mejora antes de parar |
| `--review-step` | `5` | Añade una nueva review cada N ciclos sin mejora |
| `--max-reviews` | `3` | Número máximo de reviews encadenadas |
| `--accept-epsilon` | `0.002` | Umbral mínimo para aceptar candidato en el mismo batch (`Δmétrica > ε`) |
| `--candidate-attempts` | `1` | Intentos máximos de mutación+evaluación por iteración en el mismo batch |
| `--context` / `-c` | — | Archivo de contexto adicional (CSV, TXT…) para el optimizador |
| `--log` | `training_log.txt` | Archivo de log junto al prompt |
| `--strategies` | `optimization_strategies.md` | Estrategias de optimización del prompt principal |
| `--review-alternatives` | `review_alternatives.md` | Alternativas para rediseñar reviews estancadas |
| `--seed` | — | Semilla para reproducibilidad del muestreo |

### Aceptación de candidato en el mismo batch

En cada iteración el flujo es:

1. Evaluar el pipeline activo (baseline/parent) en el batch actual.
2. Generar un candidato desde ese baseline.
3. Evaluar el candidato en el **mismo batch**.
4. Aceptar solo si `Δmétrica > --accept-epsilon`.
5. Si no mejora, se rechaza y se conserva el baseline para la siguiente iteración.

Si `--candidate-attempts > 1`, se pueden probar varios candidatos dentro de la misma iteración; los intentos rechazados se reportan al optimizador para evitar repetir cambios que empeoraron.

---

## Selección de modelos

El modelo se determina por el prefijo del nombre:

| Prefijo / valor | Proveedor | API |
|---|---|---|
| `gpt-`, `o1-`, `o3-`, `o4-` | OpenAI | `/v1/chat/completions` (SSE) |
| `deepseek-` | DeepSeek | `/v1/chat/completions` (SSE) |
| `ollama` | Ollama (interactivo) | `/api/chat` (nativo) |
| cualquier otro nombre | Ollama (directo) | `/api/chat` (nativo) |

> **Nota sobre Ollama**: Se usa el API nativo (`/api/chat`) en lugar del compatible OpenAI (`/v1/chat/completions`) porque solo el nativo respeta correctamente `think: false` en modelos con reasoning (ej. qwen3.5:35b).

---

## Formato de los datos BRAT

Cada muestra consiste en un par de archivos:

- `nombre.txt` — Texto clínico en bruto
- `nombre.ann` — Anotaciones en formato BRAT:
  ```
  T1	NOMBRE_SUJETO_ASISTENCIA 10 22	Juan García
  T2	FECHAS 45 55	10/05/2023
  ```

El modelo worker debe devolver el mismo texto con entidades marcadas inline:
```
... [**NOMBRE_SUJETO_ASISTENCIA: Juan García**] ingresó el [**FECHAS: 10/05/2023**] ...
```

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `prompt.txt` | Prompt actual (se actualiza cada iteración) |
| `prompt_best.txt` | Mejor prompt visto durante el entrenamiento |
| `prompt_review1.txt` | Prompt de la primera review (si existe) |
| `prompt_review2.txt` | Prompt de la segunda review (si existe) |
| `prompt_review3.txt` | Prompt de la tercera review (si existe) |
| `training_log.txt` | Log detallado de cada iteración con métricas y cambios |

---

## Ejemplo de sesión completa

```bash
# 1. Asegurarse de tener Ollama corriendo (si se usa localmente)
ollama serve

# 2. Lanzar el entrenamiento con modelo local y 20 iteraciones
python src/train.py \
  --brat-dir SPACCC_MEDDOCAN/corpus/train/brat \
  --prompt prompt.txt \
  --worker-model qwen3.5:35b \
  --no-think \
  --optimizer-model deepseek-chat \
  --iterations 20 \
  --samples 5 \
  --metric f1 \
  --patience 10
```

Al terminar, `prompt_best.txt` contiene el prompt con mejor F1 conseguido.
