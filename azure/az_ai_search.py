# https://srch-dckjldgkw5zcs.search.windows.net
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, CorsOptions,
)
import numpy as np
import re, unicodedata
from uuid import uuid4

service_endpoint = " https://srch-dckjldgkw5zcs.search.windows.net"
index_name = "qa-pib-test"
VECTOR_DIMS = 1536

cred = DefaultAzureCredential()
admin = SearchIndexClient(endpoint=service_endpoint, credential=cred)
search = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=cred)

def ensure_index():
    try:
        admin.get_index(index_name)
        print(f"Índice '{index_name}' ya existe.")
        return
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMS,
            vector_search_profile_name="vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw")],
    )
    admin.create_index(SearchIndex(
        name=index_name, fields=fields, vector_search=vector_search,
        cors_options=CorsOptions(allowed_origins=["*"], max_age_in_seconds=300),
    ))
    print(f"Índice '{index_name}' creado.")

def slugify(text: str) -> str:
    import unicodedata, re
    norm = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_\-=]+","-", norm).strip("-").lower()
    from uuid import uuid4
    return slug or f"doc-{uuid4().hex}"

def main():
    ensure_index()

    title = "¿Qué es el Producto Interno Bruto (PIB)?"
    content = (
        "El PIB es el valor total de los bienes y servicios finales producidos en una economía en un período "
        "determinado. En Chile, se calcula desde tres enfoques: producción, gasto e ingreso. El enfoque de producción "
        "mide el valor agregado por actividad, el de gasto incluye consumo, inversión y exportaciones netas, y el de "
        "ingreso analiza remuneraciones, excedentes de explotación e impuestos."
    )

    # ➜ Genera el embedding a partir de title + content (aquí simulado con random)
    text_for_embedding = f"{title}\n{content}"
    vector = np.random.rand(VECTOR_DIMS).astype("float32").tolist()

    doc = {
        "id": slugify(title),
        "title": title,
        "content": content,
        "contentVector": vector,
    }

    res = search.upload_documents(documents=[doc])
    print([{"key": r.key, "ok": r.succeeded} for r in res])

if __name__ == "__main__":
    main()




'''
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
import json
import numpy as np

service_endpoint = "https://srch-dckjldgkw5zcs.search.windows.net"
credential = DefaultAzureCredential()
client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

indexes = client.list_indexes()

for index in indexes:
   index_dict = index.as_dict()
   print(json.dumps(index_dict, indent=2))


index_name = "pib"
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

document = {
    "id": "¿Qué es el Producto Interno Bruto (PIB)?",
    "content": "El PIB es el valor total de los bienes y servicios finales producidos en una economía en un período determinado. En Chile, se calcula desde tres enfoques: producción, gasto e ingreso. El enfoque de producción mide el valor agregado generado en cada actividad económica, el de gasto incluye consumo, inversión y exportaciones netas, y el de ingreso analiza las remuneraciones, excedentes de explotación e impuestos. Es un indicador central para evaluar el crecimiento económico y la salud general de la economía.",
    "contentVector": np.random.rand(1536).tolist() 
}


result = search_client.upload_documents(documents=[document])
print("Resultado de la inserción:", result)
'''