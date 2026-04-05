
from langchain.chat_models import init_chat_model
from constants import *

from langchain.tools import tool



model = init_chat_model(
    MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    use_responses_api=True,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers.
    Args:
        a: The first number.
        b: The second number.
    
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers.
    Args:
        a: The first number.
        b: The second number.
    
    """
    return a + b


@tool
def devide(a: int, b: int) -> float:
    """Devide two numbers.
    Args:
        a: The first number.
        b: The second number.
    
    """
    return a / b


tools = [multiply, add, devide]

tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)



from langchain.messages import SystemMessage, HumanMessage, ToolCall

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


from langgraph.func import entrypoint, task


system_message = SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )

@task
def call_llm(messages: list[BaseMessage]):
    return model_with_tools.invoke(
        [system_message] + messages
    )

@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()
    while True:
        if not model_response.tool_calls:
            break

        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]

        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()
    messages = add_messages(messages, model_response)

    return messages

from rich.console import Console
from rich.pretty import Pretty
from rich.rule import Rule

console = Console()


def main():
    messages = [HumanMessage(content="Add 3 and 4,  using the tool.")]
    for i, chunk in enumerate(agent.stream(messages, stream_mode="updates"), 1):
        console.print(Rule(f"chunk {i}"))
        console.print(Pretty(chunk, expand_all=True))
        console.print()

if __name__ == "__main__":
    main()