from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import tools_condition
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="agentic-langgraph"

llm=init_chat_model("groq:llama-3.3-70b-versatile")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
def make_tool_graph():
    @tool
    def multiply(a:int, b:int)->int:
        """Multiply a and b

        Args:
            a (int): first int
            b (int): second int

        Returns:
            int: output int
        """
        return a*b

    tool_node = ToolNode([multiply])

    llm_with_tools = llm.bind_tools([multiply])

    def tool_calling_llm(state:State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )
    builder.add_edge("tools", END)

    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()