# rag_agent/agent.py

from google.adk.agents import Agent

# Define the root agent for the application.
# This agent is designed to answer user questions by leveraging a knowledge base
# through a retrieval tool.
root_agent = Agent(
    name="retrieval_agent",
    model="gemini-2.0-flash", # Or another suitable model for text generation
    description="An agent that answers questions using retrieved information from a knowledge base.",
    instruction=(
        "You are an agent designed to answer user questions by retrieving relevant information. "
        "When a user asks a question, first use the `query_knowledge_base` tool to find relevant documents or information. "
        "Synthesize the information retrieved from the tool to formulate your answer. "
        "If the retrieval tool does not provide sufficient information, you can try to answer based on your internal knowledge, but prioritize retrieved information when available. "
        "Always cite the source if the retrieval tool provides source information."
    ),
    # The 'tools' list should contain instances of tools available to the agent.
    # The 'query_knowledge_base' tool instance would be defined elsewhere (e.g., in a tools module)
    # and imported here.
    # tools=[
    #     query_knowledge_base # Example: Assuming 'query_knowledge_base' is an imported tool instance
    # ]
)