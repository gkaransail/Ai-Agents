from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent,MCPClient 
import os 
import asyncio


"""Simple chat example with built-in conversation memory.
This example demonstrates how to use the MCPClient to interact with an MCP server.
It uses the built-in conversation memory to store the conversation history.
"""


async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory"""
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    #Config file path
    config_file = "browser_mcp.json"

    print("Initializing MCP Agent")

    #Create MCP Client and agent with memory enabled 
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="llama3-70b-8192")  # or any other valid Groq model


    #Create agent with memory_enabled = True 

    agent = MCPAgent(
        llm = llm, 
        client=client,
        max_steps=15,
        memory_enabled=True,
    )


    print("Agent initialized")
    print("type 'exit' or 'quit' to exit")
    print("type 'clear' to clear the memory")
    print("====================================================")

    try:
        #Main chat loop
        while True :
            #Get user input
            user_input = input("You: ")

            #Exit condition
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            #Clear memory   
            if user_input.lower() in ["clear", "reset"]:
                agent.clear_conversation_history()
                print("Memory cleared")
                continue 

            #Get response from agent
            print("\n Assistant:", end = "",flush=True)
             
            try:
                #Run the agent with the user input (memory handled internally)
                response = await agent.run(user_input)
                print(response)
            
            except Exception as e:
                print(f"Error: {e}")
    finally:
        #Clean up 
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())


    