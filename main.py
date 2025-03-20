import json
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from langgraph.graph import StateGraph

llm = ChatGroq(model="qwen-2.5-coder-32b")
graph_db = Neo4jGraph(enhanced_schema=True)


class AgentState(TypedDict):
    question: str
    cypher_statement: str
    query_result: str


text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

User input: {question}
Cypher query:"""
            ),
        ),
    ]
)

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()


def generate_cypher(state: AgentState) -> AgentState:
    print("Generating cypher statement:")
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "schema": graph_db.schema,
        }
    )
    print(generated_cypher, "\n")
    return {"cypher_statement": generated_cypher}


def execute_cypher(state: AgentState) -> AgentState:
    print("Executing cypher statement:")
    try:
        records = json.dumps(graph_db.query(state.get("cypher_statement")))
    except Exception as e:
        records = str(e)
    print(records, "\n")
    return {"query_result": records}


generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a succinct, definitive answer to the user's question.

Respond as if you are answering the question directly.

Results: {results}
Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()


def generate_final_answer(state: AgentState) -> AgentState:
    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"), "results": state.get("query_result")}
    )
    print(final_answer)
    return {"answer": final_answer}


agent_graph_builder = StateGraph(AgentState)
agent_graph_builder.add_node(generate_cypher)
agent_graph_builder.add_node(execute_cypher)
agent_graph_builder.add_node(generate_final_answer)
agent_graph_builder.set_entry_point("generate_cypher")
agent_graph_builder.add_edge("generate_cypher", "execute_cypher")
agent_graph_builder.add_edge("execute_cypher", "generate_final_answer")
agent_graph_builder.set_finish_point("generate_final_answer")
agent_graph = agent_graph_builder.compile()

agent_graph.invoke({"question": input("Ask a question: ")})
