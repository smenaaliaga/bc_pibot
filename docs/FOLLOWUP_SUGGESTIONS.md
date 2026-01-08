# Preguntas Sugeridas (Follow-up Suggestions)

## Descripción

Esta funcionalidad genera automáticamente preguntas sugeridas contextuales basadas en las entidades normalizadas almacenadas en memoria después de cada respuesta del asistente.

## Arquitectura

### 1. Generación de Sugerencias (`agent_graph.py`)

La función `_generate_suggested_questions(state: AgentState)` genera hasta 3 preguntas contextuales basándose en:

- **Indicador**: El indicador económico mencionado (ej: IMACEC, PIB)
- **Componente**: Sectores o componentes específicos (ej: minero, no minero)
- **Estacionalidad**: Si los datos son desestacionalizados o no
- **Periodo**: Rango temporal de los datos
- **Intención**: El tipo de consulta del usuario (metodología, datos, etc.)

#### Ejemplos de Sugerencias Generadas:

**Para consultas sobre IMACEC:**
- "¿Cómo estuvo el IMACEC minero?"
- "¿Cuál es el IMACEC desestacionalizado?"
- "¿Qué mide el IMACEC?"

**Para consultas metodológicas:**
- "¿Cuál es el último valor del [indicador]?"

**Para consultas sobre PIB:**
- "¿Cuál es la variación del PIB por sectores?"

### 2. Emisión de Marcadores (`memory_node`)

En el nodo `memory_node`, después de persistir la respuesta en memoria:

1. Se invocan `_generate_suggested_questions(state)`
2. Las preguntas se formatean dentro de marcadores especiales:
   ```
   ##FOLLOWUP_START
   suggestion_1=¿Pregunta sugerida 1?
   suggestion_2=¿Pregunta sugerida 2?
   suggestion_3=¿Pregunta sugerida 3?
   ##FOLLOWUP_END
   ```
3. Los marcadores se agregan al `output` final

### 3. Parseo y Renderizado (`app.py`)

En la interfaz de Streamlit:

1. **Parseo del Stream**: El `handle_chunk` detecta los marcadores `##FOLLOWUP_START/END` y extrae las sugerencias en `markers_followup`

2. **Renderizado de Botones**: 
   - Las sugerencias se muestran en columnas (máx. 3)
   - Cada sugerencia es un botón clickeable con estilo "secondary"
   - Los botones usan `use_container_width=True` para mejor presentación

3. **Interacción del Usuario**:
   - Al hacer click, la pregunta se almacena en `st.session_state.pending_question`
   - Se ejecuta un `st.rerun()` 
   - En el próximo ciclo, la pregunta se procesa como si el usuario la hubiera escrito

## Flujo de Datos

```
Usuario hace consulta
    ↓
classify_node extrae entidades normalizadas
    ↓
Entidades se almacenan en state (indicator_context, component_context, etc.)
    ↓
Respuesta se genera en data_node/rag_node/fallback_node
    ↓
memory_node:
  - Persiste respuesta en memoria
  - Persiste entidades en facts
  - Genera preguntas sugeridas basadas en entidades
  - Agrega marcadores ##FOLLOWUP al output
    ↓
app.py parsea marcadores y renderiza botones
    ↓
Usuario hace click → se procesa como nueva consulta
```

## Beneficios

1. **Contextual**: Las sugerencias están basadas en el contexto específico de la conversación
2. **Descubrimiento**: Ayuda a los usuarios a explorar datos relacionados sin pensar en las preguntas
3. **Flujo Natural**: Las preguntas sugeridas continúan la conversación de forma coherente
4. **Memoria Persistente**: Usa las entidades almacenadas en facts para mayor precisión

## Configuración

No requiere configuración adicional. La funcionalidad está activa por defecto cuando:
- El sistema de memoria está habilitado (`_MEMORY`)
- Se detectan entidades normalizadas en la clasificación
- La respuesta se genera exitosamente

## Testing

Para probar la funcionalidad:

1. Haz una consulta sobre un indicador específico:
   ```
   ¿Cuál fue el IMACEC de diciembre 2024?
   ```

2. Observa las preguntas sugeridas que aparecen al final de la respuesta

3. Haz click en una sugerencia y verifica que se procese correctamente

## Personalización

Para modificar las preguntas sugeridas, edita la función `_generate_suggested_questions` en:
- **Archivo**: `orchestrator/graph/agent_graph.py`
- **Línea**: ~227

Puedes agregar:
- Nuevas condiciones basadas en otros indicadores
- Más lógica contextual
- Diferentes tipos de sugerencias según la intención
