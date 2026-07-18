from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class BloggerState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] 
    user_input: str 
    human_feedback: str

    editorial_plan: List[Dict[str, Any]]
    full_calendar: List[Dict[str, Any]]
    current_draft: str
   
    kg_summary: str                      
    action_results: Dict[str, List[str]]   
    reasoning_trace: List[Dict[str, str]]
    research_summary: str
    fact_check_report: str
    tool_call_count: int
    