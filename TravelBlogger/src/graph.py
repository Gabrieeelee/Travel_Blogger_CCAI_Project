from langgraph.graph import StateGraph, START, END
from typing import Literal
from src.state import BloggerState
from src.agents.nodes import ( 
    planner_node, 
    research_subgraph, 
    recap,
    drafter_node, 
    human_in_the_loop_node, 
    fact_checking_node,
    kg_updater_node
)


builder = StateGraph(BloggerState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", research_subgraph) 
builder.add_node("recap", recap)
builder.add_node("fact_checker", fact_checking_node)
builder.add_node("drafter", drafter_node)
builder.add_node("human_review", human_in_the_loop_node)
builder.add_node("kg_updater", kg_updater_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher") 
builder.add_edge("researcher", "recap") 
builder.add_edge("recap", "fact_checker") 
builder.add_edge("fact_checker", "drafter")
builder.add_edge("drafter", "human_review")
#builder.add_edge("human_review", "kg_updater")
builder.add_edge("kg_updater", END)

app = builder.compile()

