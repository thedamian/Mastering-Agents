import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(city: str) -> str:
	"""Return the current weather for the requested city."""
	return f"The temperature in {city} is 72°F and raining!"


def run_tool_call_demo(question: str) -> List[str]:
	"""Run a simple LangChain tool calling demo and return console lines."""

	load_dotenv()

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"Missing OPENAI_API_KEY. Please set it in your environment or .env file."
		)

	llm = ChatOpenAI(
		model="gpt-5-nano",
		temperature=0,
		api_key=api_key,
	)

	llm_with_tools = llm.bind_tools([get_weather])

	transcript: List[str] = []
	transcript.append(f"User: {question}")

	messages: List[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
		SystemMessage(
			content=(
				"You help travelers prepare for their trips. Whenever someone asks about"
				" what to wear in a city, ALWAYS call the `get_weather` tool for that"
				" city before answering. Provide a friendly, concise recommendation"
				" afterward."
			)
		),
		HumanMessage(content=question),
	]

	first_response = llm_with_tools.invoke(messages)
	planning_text = first_response.content or "[planning via tool call]"
	transcript.append(f"Assistant (planning): {planning_text}")

	if not getattr(first_response, "tool_calls", None):
		transcript.append("Assistant: (No tool call was made.)")
		return transcript

	messages.append(first_response)

	for tool_call in first_response.tool_calls:
		tool_args = tool_call["args"]
		tool_name = tool_call["name"]
		result = get_weather.invoke(tool_args)
		transcript.append(
			f"Tool `{tool_name}` called with args {tool_args} -> {result}"
		)
		messages.append(
			ToolMessage(content=result, tool_call_id=tool_call["id"])
		)

	final_response = llm_with_tools.invoke(messages)
	transcript.append(f"Assistant: {final_response.content}")

	return transcript


def main() -> None:
	question = "I'm going to paris tomorrow. Do I need a raincoat or a winter coat?"

	transcript = run_tool_call_demo(question)

	print("=== LangChain Tool Calling Demo ===")
	for line in transcript:
		print(line)


if __name__ == "__main__":
	main()
