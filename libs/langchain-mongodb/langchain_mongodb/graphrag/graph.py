import json
import logging
from typing import Any, Dict, List, Optional, TypeAlias, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.results import BulkWriteResult

from langchain_mongodb.graphrag import prompts, schema

from .prompts import rag_prompt
from .schema import entity_schema

logger = logging.getLogger(__name__)

# Represents an entity in the knowledge graph with ID, type, properties, and relationships fields
# See .schema for full schema
Entity: TypeAlias = Dict[str, Any]


class MongoDBGraphStore:
    """GraphRAG DataStore

    GraphRAG is a ChatModel that provides responses to semantic queries.
    As in Vector RAG, we augment the Chat Model's training data
    with relevant information that we collect from  documents.

    In Vector RAG, one uses an "Embedding" model that converts both
    the query, and the potentially relevant documents, into vectors,
    which can then be compared, and the most similar supplied to the
    Chat Model as context to the query.

    In Graph RAG, one uses an Entity-Extraction model that converts both
    the query, and the potentially relevant documents, into graphs. These are
    composed of nodes that are entities (nouns) and edges that are relationships.
    The idea is that the graph can find connections between entities and
    hence answer questions that require more than one connection.

    It is also about finding common entities in documents,
    combining the properties found and hence providing richer context than Vector RAG,
    especially in certain cases.

    When a document is extracted, each entity is represented by a single
    MongoDB Document, and relationships are defined in a nested field named
    'relationships'. The schema, and an example, are described in the
    :data:`~langchain_mongodb.graphrag.prompts.entity_context` prompts module.

    When a query is made, the model extracts the entities and relationships from it,
    then traverses the graph starting from each of the entities found.
    The connected entities and relationships form the context
    that is included with the query to the Chat Model.

    Requirements:
        Documents
        Entity Extraction Model
        Prompt Template
        Query
        Chat Model

    Example Query: "Does Casey Clements work at MongoDB?"
        ==> Entities: [(name=Casey Clements, type=person), (name=MongoDB, type=Corporation)]
        ==> Relationships: [(name=EMPLOYEE), ]
    Example document ingested into Collection:
        "MongoDB employees everyone."

    From this information, one would be able to deduce that Casey Clements DOES work at MongoDB.
    The Graph is there to provide context/information, not answer the question.
    The LLM will know that everyone contains every person.
    But how will the LLM know to categorize Casey Clements as a person? It could be an organization, for example.
        - One would add hints or examples in their prompt to the Entity Extraction Model.
        - This is the right track. Even if we're not focussing on Entity Extraction, we care about providing an API to allow it
         ==> e.g. Could give specific types to include, and that one must pick one, or if ambiguous, throw away information.

    Here are a few examples of so-called multi-hop questions where GraphRAG excels:
        - What is the connection between ACME Corporation and GreenTech Ltd.?
        - Who is leading the SolarGrid Initiative, and what is their role?
        - Which organizations are participating in the SolarGrid Initiative?
        - What is John Doe’s role in ACME’s renewable energy projects?
        - Which company is headquartered in San Francisco and involved in the SolarGrid Initiative?
    """

    def __init__(
        self,
        collection: Collection,
        entity_extraction_model: BaseChatModel,
        entity_prompt: ChatPromptTemplate = prompts.entity_prompt,
        query_prompt: ChatPromptTemplate = prompts.query_prompt,
        max_depth: int = 2,
        validate: bool = False,
        entity_examples: str = None,
        entity_name_examples: str = None,
    ):
        """
        Args:
            collection: Collection representing an Entity Graph
            entity_extraction_model: LLM for converting documents into Graph of Entities and Relationships
            entity_prompt: Prompt to fill graph store with entities following schema
            query_prompt: Prompt extracts entities and relationships as search starting points.
            max_depth: Maximum recursion depth in graph traversal.
            validate: If True, entity schema will be validated on every insert or update.
            entity_examples: A string containing any number of additional examples to provide as context for entity extraction.
            entity_name_examples: A string appended to schema.name_extraction_instructions containing examples.
        """
        self.entity_extraction_model = entity_extraction_model
        self.entity_prompt = entity_prompt
        self.query_prompt = query_prompt
        self.max_depth = max_depth
        if validate:
            collection.database.command(
                "collMod",
                collection.name,
                validator={"$jsonSchema": self.entity_schema},
            )
        self.collection = collection

        if entity_examples:
            self.entity_prompt.messages.insert(
                1,
                SystemMessagePromptTemplate.from_template(
                    f"## Additional Examples \n {entity_examples}"
                ),
            )

        if entity_name_examples:
            self.query_prompt.messages.insert(
                1,
                SystemMessagePromptTemplate.from_template(
                    f"## Additional Examples \n {entity_name_examples}"
                ),
            )

    @property
    def entity_schema(self):
        """JSON Schema Object of Entities. Will be applied if validate is True.

        See Also:
            `$jsonSchema <https://www.mongodb.com/docs/manual/reference/operator/query/jsonSchema/>`_
        """
        return schema.entity_schema

    def add_documents(
        self, documents: Union[Document, List[Document]]
    ) -> List[BulkWriteResult]:
        """Extract entities and upsert into the collection.

        Each entity is represented by a single MongoDB Document.
        Existing entities identified in documents will be updated.

        Args:
            documents: list of textual documents and associated metadata.
        Returns:
            List containing metadata on entities inserted and updated, one value for each input document
        """
        documents = [documents] if isinstance(documents, Document) else documents
        results = []
        for doc in documents:
            # Call LLM to find all Entities in doc
            entities = self.extract_entities(doc.page_content)
            logger.debug(f"Entities found: {[e["ID"] for e in entities]}")
            # Insert new, or combine with existing, entity
            operations = []
            for entity in entities:
                operations.append(
                    UpdateOne(
                        filter={"ID": entity["ID"]},  # Match on ID
                        update={
                            "$setOnInsert": {  # Set if upsert
                                "ID": entity["ID"],
                                "type": entity["type"],
                            },
                            "$addToSet": {  # Update without overwriting
                                **{
                                    f"properties.{k}": v
                                    for k, v in entity.get("properties", {}).items()
                                },
                                **{
                                    f"relationships.{k}": {"$each": v}
                                    for k, v in entity.get("relationships", {}).items()
                                },
                            },
                        },
                        upsert=True,
                    )
                )

            # Execute bulk write for the entities
            if operations:
                results.append(self.collection.bulk_write(operations))
        return results

    def extract_entities(self, raw_document: str, **kwargs: Any) -> List[Entity]:
        """Extract entities and their relations using chosen prompt and LLM.

        Args:
            raw_document: A single text document as a string. Typically prose.
        Returns:
            List of Entity dictionaries.
        """
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.entity_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(
            dict(input_document=raw_document, entity_schema=entity_schema)
        )
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and trailing ```
        json_string = (
            response.content.removeprefix("```json").removesuffix("```").strip()
        )
        extracted = json.loads(json_string)
        return extracted["entities"]

    def extract_entity_names(self, raw_document: str, **kwargs: Any) -> List[str]:
        """Extract entity names from a document for similarity_search.

        The second entity extraction has a different form and purpose than
        the first as we are looking for starting points of our search and
        paths to follow. We aim to find source nodes,  but no target nodes or edges.

        Args:
            raw_document: A single text document as a string. Typically prose.
        Returns:
            List of entity names / IDs.
        """
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = self.query_prompt | self.entity_extraction_model
        # Invoke on a document to extract entities and relationships
        response: AIMessage = chain.invoke(dict(input_document=raw_document))
        # Post-Process output string into list of entity json documents
        # Strip the ```json prefix and suffix
        json_string = (
            response.content.removeprefix("```json").removesuffix("```").strip()
        )
        return json.loads(json_string)

    def find_entity_by_name(self, name: str) -> Optional[List[Entity]]:
        """Utility to get Entity dict from Knowledge Graph / Collection.
        Args:
            name: ID string to look for
        Returns:
            List of Entity dicts if any match name
        """
        return list(self.collection.find({"ID": name}))

    def related_entities(
        self, starting_entities: List[str], max_depth=3
    ) -> List[Entity]:
        """Traverse Graph along relationship edges to find connected entities.

        Args:
            starting_entities: Traversal begins with documents whose IDs match these strings.
            max_depth: Recursion continues until no more matching documents are found,
            or until the operation reaches a recursion depth specified by the maxDepth parameter

        Returns:
            List of connected entities.
        """
        pipeline = [
            # Begin by finding starting points.
            {"$match": {"ID": {"$in": starting_entities}}},
            {
                "$graphLookup": {
                    "from": self.collection.name,
                    "startWith": {
                        # Extract all relationship targets dynamically
                        "$reduce": {
                            "input": {"$objectToArray": "$relationships"},
                            "initialValue": [],
                            "in": {"$concatArrays": ["$$value", "$$this.v.target"]},
                        }
                    },
                    "connectFromField": "ID",  # Match on entity ID
                    "connectToField": "ID",  # Use the target entity's ID
                    "as": "connections",  # Output field for related entities
                    "maxDepth": max_depth or self.max_depth,  # Cap traversal depth
                }
            },
            # Unwind the connections array to process each connection as its own document
            {"$unwind": "$connections"},
            # Replace the root with the connection document
            {"$replaceRoot": {"newRoot": "$connections"}},
            # Remove duplicates
            {
                "$group": {
                    "_id": "$ID",  # Group by `ID`
                    "document": {"$first": "$$ROOT"},  # Keep the first
                }
            },
            # Restore the original document structure
            {"$replaceRoot": {"newRoot": "$document"}},
            # Exclude MongoDB's internal `_id` field
            {"$project": {"_id": 0}},
        ]
        return list(self.collection.aggregate(pipeline))

    def similarity_search(self, input_document: str) -> List[Entity]:
        """Retrieve list of connected Entities found via traversal of KnowledgeGraph.

        1. Use LLM & Prompt to find entities within the input_document itself.
        2. Find Entity Nodes that match those found in the input_document.
        3. Traverse the graph using these as starting points.

        Args:
            input_document: String to find relevant documents for.
        Returns:
            List of connected Entity dictionaries.
        """
        starting_ids: List[str] = self.extract_entity_names(input_document)
        return self.related_entities(starting_ids)

    def chat_response(
        self,
        query: str,
        chat_model: BaseChatModel = None,
        prompt: ChatPromptTemplate = None,
    ) -> AIMessage:
        """Responds to a query given information found in Knowledge Graph.

        Args:
            query: Query to send the chat_model
            chat_model: ChatBot. Defaults to entity_extraction_model.
            prompt: Alternative Prompt Template. Defaults to prompts.rag_prompt
        Returns:
            Response Message. response.content contains text.
        """
        if chat_model is None:
            chat_model = self.entity_extraction_model
        if prompt is None:
            prompt = rag_prompt

        # Perform Retrieval on knowledge graph
        related_entities = self.similarity_search(query)
        # Combine the llm with the prompt template to form a chain
        chain: RunnableSequence = prompt | chat_model
        # Invoke with query
        return chain.invoke(
            dict(
                query=query,
                related_entities=related_entities,
                entity_schema=entity_schema,
            )
        )


# TODO
#   - Rename "properties" in schema to "attributes"
#   - Update Design Doc
#   - Prompts:
#       - Can we pass variables in the dict that aren't used? (overspecified)
#       - Add: Constraints in Prompts:
#           - relationship names/types. Need to describe this well. keys in the relationships object.
#           - entity types
