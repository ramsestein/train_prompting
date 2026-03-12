# Usos alternativos para los slots de review
#
# Cuando una review lleva ≥1 ciclo sin aportar mejora, el optimizador recibe
# una de estas sugerencias de forma rotatoria para que rediseñe esa review
# con un enfoque completamente diferente.
#
# FORMATO: sección "## Nombre" seguida de descripción libre.
# Puedes añadir, eliminar o reordenar entradas.
# ─────────────────────────────────────────────────────────────────────────────

## Segunda pasada de cobertura
Rediseña este prompt de review para que su única misión sea encontrar entidades que el paso anterior haya omitido. El modelo debe releer el texto completo buscando activamente menciones adicionales de cada tipo de entidad, asumiendo que la anotación anterior está incompleta. Al final devuelve el texto con TODAS las anotaciones (las anteriores más las nuevas que encuentre).

## Filtro de falsos positivos
Rediseña este prompt de review para que actúe como auditor estricto: revisa cada anotación existente y elimina las que no sean genuinamente datos de identificación personal. El modelo debe argumentar internamente si cada anotación cumple el criterio de privacidad; solo mantiene las que superan el escrutinio. No puede añadir nuevas, solo depurar.

## Especialización en entidades de tiempo
Rediseña este prompt de review para que se focalice exclusivamente en expresiones temporales: fechas, horas, duraciones, edades, años de nacimiento, referencias a eventos temporales específicos. Debe verificar todas las del texto y completar las que falten, ignorando las demás categorías.

## Especialización en entidades de lugar
Rediseña este prompt de review para que se focalice exclusivamente en datos geográficos e institucionales: calles, ciudades, países, centros médicos, hospitales, consultas, plantas, habitaciones. Revisa el texto buscando solo estos datos y completa cualquier omisión.

## Especialización en personas y profesionales
Rediseña este prompt de review para que se centre exclusivamente en nombres propios de personas: pacientes, médicos, enfermeras, familiares, con sus títulos y cargos asociados. Debe confirmar que cada mención (incluyendo alias, apodos, iniciales) esté anotada.

## Verificador de coherencia entre menciones
Rediseña este prompt de review para que compruebe la coherencia a lo largo de todo el texto: si una persona, lugar u organización aparece varias veces (incluso con forma abreviada, pronombre o referencia elíptica), todas las menciones deben estar anotadas. El modelo debe rastrear cada entidad desde su primera aparición.

## Cazador de entidades compuestas y límites incorrectos
Rediseña este prompt de review para detectar: (1) entidades que deberían anotarse como una sola unidad pero aparecen parcialmente anotadas o divididas, y (2) anotaciones cuyos límites son incorrectos (cortadas demasiado pronto o incluyendo palabras extra). Corrige ambos tipos de error.

## Buscador por patrones lingüísticos
Rediseña este prompt de review para que busque entidades mediante patrones sintácticos habituales: nombres tras títulos ("Dr.", "Dra.", "Sr.", "Sra.", "Prof."), fechas en formatos DD/MM/AAAA o AAAA-MM-DD, edades tras "de" o "con", números de historia clínica, códigos postales. Detecta todo lo que encaje en estos patrones.

## Detector de referencias implícitas
Rediseña este prompt de review para encontrar referencias indirectas a personas o lugares ya mencionados: pronombres personales en contexto identificable, elipsis, epítetos ("el médico", "la paciente", "el familiar"), siempre que inequívocamente se refieran a una entidad concreta del texto. Anótalas con su tipo correspondiente.

## Crítico inverso (abogado del diablo)
Rediseña este prompt de review para que adopte el rol contrario: cuestiona cada anotación existente con el argumento de por qué NO debería estar anonimizada. Tras el debate interno, mantiene solo las que claramente identifiquen a una persona real. El objetivo es reducir el ruido y los falsos positivos de forma agresiva.

## Re-anotador por párrafos independientes
Rediseña este prompt de review para que procese el texto párrafo a párrafo de forma independiente, anotando cada uno como si fuese un documento nuevo, y luego fusione los resultados. Este enfoque reduce los errores causados por el contexto acumulado y mejora la cobertura en textos largos.

## Completador de series y listas
Rediseña este prompt de review para detectar series incompletas: si el texto enumera varios datos del mismo tipo (ej: una lista de diagnósticos, varios médicos, múltiples fechas) y algunos están anotados pero otros no, completa la serie. Presta especial atención a enumeraciones y listas con comas o puntos y coma.

## Verificador semántico por tipo
Rediseña este prompt de review para ir tipo por tipo de entidad (uno en cada pasada mental) y verificar que todas las instancias de ese tipo en el texto están anotadas. Para cada tipo: busca, confirma o añade. Al final combina todo en la salida con el formato inline.

## Normalizador y estandarizador
Rediseña este prompt de review para que normalice la salida: asegura que el texto anotado es exactamente el texto original con únicamente marcas inline añadidas, que las etiquetas usan la capitalización correcta, que no hay marcas duplicadas, y que el texto fuera de las marcas no está alterado.

## Especialista en datos de contacto
Rediseña este prompt de review para focalizarse en datos de contacto y acceso: teléfonos, correos electrónicos, números de historia clínica, DNI/NIF, números de seguridad social, matrículas de vehículo, y cualquier código numérico que identifique a una persona en sistemas de información.

## Segunda opinión desde cero
Rediseña este prompt de review para que el modelo ignore completamente la anotación anterior y realice una anotación independiente desde el texto en blanco como si nunca hubiese visto el resultado previo. Luego fusiona ambas anotaciones (la recibida más la nueva) preservando la unión de ambas (maximiza recall).

## Analizador de contexto clínico vs personal
Rediseña este prompt de review para distinguir con precisión entre terminología médica (diagnósticos, fármacos, procedimientos) que NO debe anotarse y datos personales que SÍ deben anotarse. El modelo debe reclasificar las anotaciones dudosas usando esta distinción como criterio principal.

## Auditor de privacidad estricto
Rediseña este prompt de review para que adopte la perspectiva de un responsable de protección de datos: ¿qué información, si apareciera en un documento filtrado, permitiría identificar al paciente o al personal? Anota todo lo que pase este test, aunque no encaje exactamente en una categoría estándar.
