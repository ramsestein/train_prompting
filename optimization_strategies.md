# Estrategias de optimización de prompts
#
# Este archivo define los "empujes" de enfoque que se inyectan en el optimizador
# cuando detecta que el rendimiento lleva varios ciclos sin mejorar o mejora poco.
#
# FORMATO: cada estrategia empieza con una línea "## Nombre" y su descripción
# a continuación (párrafo de texto libre). Puedes añadir, eliminar o modificar
# estrategias. El sistema las cicla en orden, una cada 2 ciclos de bajo rendimiento.
#
# ─────────────────────────────────────────────────────────────────────────────

## Cadena de pensamiento (Chain of Thought)
Instruye al modelo para que razone en voz alta paso a paso antes de dar la respuesta final. Añade indicaciones como: "Primero lee el texto completo, luego identifica cada posible entidad, evalúa si cumple los criterios, y finalmente formatea la salida". El razonamiento explícito reduce los errores de categoría y mejora la cobertura.

## Divide y vencerás por tipo de entidad
En lugar de pedir al modelo que detecte todos los tipos a la vez, instrúyele para que haga un barrido secuencial: "Primero localiza solo NOMBRES DE PERSONA, luego FECHAS, luego LUGARES, etc.". Reduce la omisión de entidades poco frecuentes y facilita la especialización por categoría.

## Anclas negativas (qué NO hacer)
Especifica explícitamente casos que NO deben anotarse. Por ejemplo: "NO anotes términos médicos generales ni nombres de enfermedades, solo datos que identifiquen a una persona concreta". Las prohibiciones explícitas reducen los falsos positivos en categorías ambiguas.

## Pocos ejemplos en el prompt (Few-shot)
Añade 2-3 ejemplos completos de texto → anotación esperada directamente en el prompt. Escoge ejemplos que cubran los tipos de entidades con más errores. Los ejemplos concretos son especialmente efectivos para casos límite y entidades con ortografía variable.

## Verificación interna (Self-check)
Después de anotar, pide al modelo que verifique su propio trabajo antes de responder: "Antes de devolver tu respuesta, comprueba: ¿Has anotado todas las menciones? ¿Alguna anotación es un término genérico que no identifica a nadie? Corrige si es necesario." Reduce tanto FN como FP.

## Árbol de pensamiento (Tree of Thought)
Pide al modelo que considere varias hipótesis antes de decidir por cada entidad: "Para cada posible entidad, evalúa si cumple los criterios de identificación personal. Ante la duda, decide explícitamente si incluirla o no y por qué." Útil cuando hay confusión sistemática entre categorías similares.

## Sensibilidad al contexto local
Indica al modelo que use el contexto inmediato (5-10 palabras alrededor) para confirmar cada entidad: "No etiquetes una palabra de forma aislada; verifica que el contexto confirma que es un dato personal real y no un uso genérico del mismo término."

## Umbral conservador (maximizar precision)
Ajusta la sensibilidad hacia la precisión: "Solo anota una entidad si estás completamente seguro de que es un dato de identificación personal. Ante cualquier duda, NO la anotes." Útil cuando hay muchos falsos positivos en los resultados actuales.

## Umbral agresivo (maximizar recall)
Ajusta la sensibilidad hacia la cobertura: "Ante la duda, incluye la entidad. Es preferible marcar algún término de más que dejar sin anonimizar datos personales reales." Útil cuando hay muchos falsos negativos y el recall es bajo.

## Descomposición en subtareas con roles
Divide el problema asignando roles explícitos al modelo: "Actúa como un anotador médico experto en protección de datos. Tu única responsabilidad es detectar y marcar información que permitiría identificar a pacientes o personal sanitario. Ignora todo lo demás."

## Consistencia y normalización
Añade instrucciones sobre homogeneidad en la salida: "Si una misma persona u organización aparece varias veces en el texto, anonimiza TODAS las menciones, incluso pronombres o abreviaturas que se refieran a ella. Mantén la capitalización original exacta dentro de las marcas."

## Meta-prompt reflexivo
Pide al modelo que reflexione sobre el propio prompt antes de ejecutar: "Lee primero este prompt completo y confirma que entiendes la tarea. Si algo es ambiguo, asume la interpretación más conservadora en cuanto a privacidad. Luego procesa el texto."
