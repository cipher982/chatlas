import json
import operator
from typing import Annotated
from typing import Sequence
from typing import TypedDict

from langchain.chat_models.base import BaseChatModel
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage
from langchain_core.messages import FunctionMessage
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_graph_agent(llm: BaseChatModel, db_uri: str):
    # Set up the tools
    db_engine = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db_engine, llm=llm)
    tools = toolkit.get_tools()
    tool_executor = ToolExecutor(tools)
    # Set up the model
    model = ChatOpenAI(temperature=0, streaming=True)
    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)

    def call_model(state, model=model):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    def call_tool(state, tool_executor=tool_executor):
        messages = state["messages"]
        last_message = messages[-1]
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
        )
        response = tool_executor.invoke(action)
        function_message = FunctionMessage(content=str(response), name=action.tool)
        return {"messages": [function_message]}

    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"

    # Define the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)  # Now uses closure
    workflow.add_node("action", call_tool)  # Now uses closure
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    workflow.add_edge("action", "agent")

    # Compile the graph
    app = workflow.compile()

    # Use it
    inputs = {"messages": [HumanMessage(content="Your initial message")]}
    app.invoke(inputs)
