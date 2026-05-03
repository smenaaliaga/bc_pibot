import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator2 import Orchestrator, InMemoryMemoryAdapter, adapt_memory_handler

mem = adapt_memory_handler() or InMemoryMemoryAdapter()
orch = Orchestrator(memory=mem)

text = (
    "Esta es una respuesta de prueba. Contiene varias oraciones para comprobar el chunking. "
    "La idea es que el orquestador devuelva trozos legibles en lugar de un bloque enorme. "
    "Aquí va otra frase más para simular salida más larga. Y otra más. Y otra más para asegurar agrupamiento."
)

print('--- stream_chunks=True ---')
for i, c in enumerate(orch.stream('Pregunta de prueba para chunking', stream_chunks=True, chunk_size=80)):
    print(f'CHUNK {i+1}:', repr(c))

print('\n--- stream_chunks=False ---')
for i, c in enumerate(orch.stream('Pregunta de prueba para chunking', stream_chunks=False)):
    print(f'REPLY {i+1}:', repr(c))
