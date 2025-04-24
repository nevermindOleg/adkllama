from google.adk.agents import Agent

root_agent = Agent(
    name="retrieval_agent",
    model="gemini-2.0-flash",
    description="Agent answers questions using retrieved information.",
    instruction=(
        "Answer user questions by retrieving relevant information. "
        "Use the `query_knowledge_base` tool first. "
        "Prioritize retrieved information. Cite the source."
    ),
)