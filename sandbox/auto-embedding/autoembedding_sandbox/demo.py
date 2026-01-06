"""Development script for Auto-Embedding indexes on MongoDB Community with Search Edition

The service setup follows that described in https://github.com/mongodb-js/mongodb-community-search-setup.

A) Add indexes
0. Create two empty collections.
1. create_search_index with the existing vectorSearch index with type==vector, where one must compute vectors.
2. create_search_index with the NEW vectorSearch index with type==autoEmbed, where one must specify
        - modality=text, model=voyage-4-lite, and path points to the text strings, not vectors.

B) Perform searches
3. Add documents
        - For type==vector, we will need to call out to the voyage API
        - We may wish to add the same docs to two different tables to compare results
4. Vector Search: type==vector
5. Vector Search: type==autoEmbed
"""

import os
import time

from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo_search_utils import bulk_embed_and_insert_texts
from pymongo_search_utils.operation import execute_search_query

MONGODB_URI = "mongodb://localhost:27017/?directConnection=true"
VOYAGE_API_KEY = os.environ["VOYAGEAI_API_KEY"]

embedding_model = VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model="voyage-3.5")
embedding_func = embedding_model.embed_documents
text_path = "content"
byov_path = "content_embeddings"


client = MongoClient(MONGODB_URI)
db = client["auto_embedding_demo"]
print(f"{client.admin.command({'ping': 1}) =}")
server_info = client.server_info()
print(f"{server_info['version'] =}")

# Create two empty collections
coll_name_byov = "BYOVector"
coll_name_auto = "AutoEmbed"
for coll_name in [coll_name_byov, coll_name_auto]:
    db.drop_collection(coll_name)
    coll = db.create_collection(coll_name)
    print(f" {coll_name} Collection created")
coll_byov = db[coll_name_byov]
coll_auto = db[coll_name_auto]

# 1. Create a vectorSearch index where one provides embedding vectors ("type": "vector").
idx_byovector = SearchIndexModel(
    name="BYOVectorIndex",
    type="vectorSearch",
    definition={
        "fields": [
            {
                "type": "vector",
                "path": byov_path,  # "content_embeddings"
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
)
idx_name = coll_byov.create_search_index(idx_byovector)
time.sleep(2)
print(f"vector search index '{idx_name}' created.")

# 2. Create a vectorSearch index where one provides the model ("type": "autoEmbed").
idx_auto = SearchIndexModel(
    name="AutoEmbedIndex",
    type="vectorSearch",
    definition={
        "fields": [
            {
                "type": "text",  #  todo: soon "autoEmbed",
                "path": text_path,  # "content",
                "model": "voyage-3.5",
                # "modality": "text",  todo soon
            }
        ]
    },
)
idx_name = coll_auto.create_search_index(idx_auto)
time.sleep(2)
print(f"vector search index '{idx_name}' created.")


# 3 Add documents
# a. To existing bring your own vector collection
# For existing vector style, we have a utility to embed and insert.
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


documents = [
    Document(page_content="In 2023, I visited Paris", metadata={"keywords": "MongoDB"}),
    Document(
        page_content="In 2022, I visited New York",
        metadata={"keywords": "Atlas"},
    ),
    Document(
        page_content="In 2021, I visited New Orleans",
        metadata={"keywords": "Search"},
    ),
    Document(
        page_content="Sandwiches are beautiful. Sandwiches are fine.",
        metadata={"keywords": "is awesome"},
    ),
]

n_docs = len(documents)

texts, metadatas = zip(
    *[(doc.page_content, doc.metadata) for doc in documents], strict=False
)
bulk_embed_and_insert_texts(
    texts=texts,
    metadatas=metadatas,
    embedding_func=embedding_func,
    collection=coll_byov,
    text_key=text_path,
    embedding_key=byov_path,
)
time.sleep(2)
print(f"{coll_byov.find_one({}) = }")

# b. To our collection with an auto-embedding vector search index.
docs = [{text_path: t, **m} for t, m in zip(texts, metadatas, strict=False)]
result = coll_auto.insert_many(docs)
time.sleep(2)
print(f"{coll_auto.find_one({}) = }")


# 4. Vector Search: type==vector

query = "Where did I visit most recently?"
query_vector = embedding_model.embed_query(query)

response_byov = execute_search_query(
    query_vector=query_vector,
    collection=coll_byov,
    embedding_key=byov_path,
    text_key=text_path,
    index_name=idx_byovector.document["name"],
)

print(f"{response_byov = }")


# 5. Vector Search: type==autoEmbed

pipeline = [
    {
        "$vectorSearch": {
            "index": idx_auto.document["name"],
            "path": text_path,
            "query": query,
            "limit": n_docs,
            "numCandidates": n_docs * 10,
        }
    },
    {"$set": {"score": {"$meta": "vectorSearchScore"}}},
]
response_auto = list(coll_auto.aggregate(pipeline))
print(f"{response_auto = }")
