from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

output_format = """
A valid json document with a single top-level key 'entities'.
Its value should be an array of the entities inferred.

Each Entity will be represented by a single JSON Document. It will have the following fields.
* ID: A unique identifier for the entity (e.g., UUID, name).
* type: A string specifying the type of the entity (e.g., “Person”, “Organization”).
* relationships: Stored as embedded key-value pairs. Keys are relationship types, values are lists of target entity IDs, along with additional metadata describing the relationship to that entity.
* properties: A dictionary containing key-value pairs of attributes describing the entity. Properties should not include things that could be entities. When in doubt, make something an entity.

## Example Entity structure
{{
  "ID": "Alice Palace",
  "type": "Person",
  "properties": {{
    "position": "CEO",
    "startDate": "2018-01-01"
  }},
  "relationships": {{
    "employer": [
      {{
        "target": "MongoDB"
      }}
    ],
    "friend": [
      {{
        "target": "Jarnail Singh",
        "properties": {{
          "since": "2019-05-01"
        }}
      }},
      {{
        "target": "Jasbinder Kaur",
        "properties": {{
          "since": "2015-05-01"
        }}
      }}
    ]
  }}
}}
"""


query_context_entities_only = """
You are a meticulous analyst tasked with extracting information from documents to form
knowledge graphs of entities (nodes) and their relationships (edges).

You will be provided a short document (query) from which you infer the entity names.
You need not think about relationships between the entities. You only need names.

Provide your response as valid json list of entity IDs,
names or human-readable identifiers, found in the text.

 ## Examples:
 1. input:  "Jack works at ACME in New York"
    output: ["Jack", "ACME", "New York"]

 In this example, you would identify 3 entities:
 Jack of type person; ACME of type organization; New York of type place.

 2. input: "In what continent is Brazil?
    output: ["Brazil"]

This example is in the form of a question. There is one entity,

3. input: "For legal and operational purposes, many governments and organizations adopt specific definitions."
   output: []

In the final example, there are no entities.
Though there are concepts and nouns that might be types or properties of entities,
there is nothing here that could be seen as being a unique identifier or name.
"""

extraction_template = """
## Overview
You are a meticulous analyst tasked with extracting information from unstructured text
to build a knowledge graph in a structured json format of entities (nodes) and their relationships (edges).
The graph will be stored in a MongoDB Collection and traversed using $graphLookup
from starting points of entity IDs and relationship types.

Use the following as guidelines.

- Simplicity: The graph should have as few entities and relationship types as needed to convey the information in the input.
- Consistency: Connections can only be made if entities and relationships use consistent naming
- Generality: The graph should be useful for describing the concepts in not just this document but other similar documents.
- Accuracy: Do not add any information that is not explicitly mentioned in the text.

INPUT: You will be provided a text document.
OUTPUT: You will produce valid json according the "Output Schema" section below.

## Entities

An entity in a knowledge graph is a uniquely identifiable object or concept,
such as a person, organization, location, object, or event,
represented as a node with attributes (properties) and relationships to other entities,
enabling structured and meaningful connections within the graph.

Extract all entities mentioned, using their full names as unique IDs when available.

Maintain Entity Consistency when extracting entities. If an entity, such as "John Doe",
is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph.
In this example, use "John Doe" as the entity ID.
Define required fields (e.g., ID, name,, type) and allow optional properties.

Do not nest organizations as properties of another entity. they should be separate entities with their own unique IDs.

## Relationships

Relationships represent edges in the knowledge graph. Relationships describe a specific edge type.
Ensure consistency and generality in relationship names when constructing knowledge schemas.
Instead of using specific and momentary types such as 'worked_at', use more general and timeless relationship types
like 'employee'. Add details as properties. Make sure to use general and timeless relationship types!

If synonyms are found in the document, choose the most general and use consistently.

If a relationship is bidirectional, each entity should contain the relationship with the other entity as target.
For example, if Casey works at MongoDB, MongoDB is an employer of Casey, and Casey is an employee of MongoDB.

## Output Schema
{output_schema}
"""

rag_template = """
## Context
You are a meticulous analyst tasked with extracting information in the form of knowledge graphs
comprised of entities (nodes) and their relationships (edges).

Based on the user input (query) that will be provided, you have already retrieved information
from the knowledge graph in the form of a list of entities known to be related to those in the Query.

From the context retrieved alone, please respond to the Query.
Your response should be a string of concise prose.

## Entity Schema
The entities have the following schema:

{entity_schema}

## Entities Found to be Related to Query
{related_entities}
"""

entity_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            extraction_template.format(output_schema=output_format)
        ),
        HumanMessagePromptTemplate.from_template("INPUT: {input_document}"),
    ]
)

query_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(query_context_entities_only),
        HumanMessagePromptTemplate.from_template("{input_document}"),
    ]
)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(rag_template),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)


# TODO -
#  Parameterize this to constrain
#  - entity types
#  - relationships types,
#  Add spaceholder for examples
