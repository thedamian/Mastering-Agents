from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

############################
#   LLM  
############################
llm = ChatOpenAI(model="gpt-5-mini")

############################
#   BLOBBY WRITER AGENT  
############################
blob_writer_prompt = SystemMessagePromptTemplate.from_template("""
You are BLOBWRITER-9000.
Your job is to write long, freeform raw content ("blob content").
Do NOT format. Do NOT censor.
""")

blob_writer_chain = LLMChain(llm=llm, prompt=blob_writer_prompt)

def blob_writer_tool(query: str):
    return blob_writer_chain.run(query)

blob_writer = Tool(
    name="blob_writer",
    func=blob_writer_tool,
    description="Writes large raw text blobs from a prompt."
)

############################
#   CENSORING AGENT  
############################
censor_prompt = SystemMessagePromptTemplate.from_template("""
You are a STRICT CONTENT CENSORBOT.
You remove profanity, hate speech, and unsafe text.
""")

censor_chain = LLMChain(llm=llm, prompt=censor_prompt)

def censor_tool(text: str):
    return censor_chain.run(text)

censor_agent = Tool(
    name="censor_agent",
    func=censor_tool,
    description="Cleans and censors unsafe or inappropriate text."
)

############################
#   SEO AGENT  
############################
seo_prompt = SystemMessagePromptTemplate.from_template("""
You are an SEO EXPERT AGENT.
You analyze content for readability, keywords, structure, and ranking.
""")

seo_chain = LLMChain(llm=llm, prompt=seo_prompt)

def seo_tool(text: str):
    return seo_chain.run(text)

seo_agent = Tool(
    name="seo_agent",
    func=seo_tool,
    description="Analyzes text for SEO score and improvements."
)

############################
#   MANAGER / SUPERVISOR AGENT  
############################
supervisor_system_prompt = """
You are the MANAGER AGENT.

Your job:
1. Understand the user request.
2. Break the task into steps.
3. Decide which agent (tool) should execute each step:
   - blob_writer
   - censor_agent
   - seo_agent
4. Combine the results.
5. Return the final answer.

Only call tools when needed.
"""

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, [blob_writer, censor_agent, seo_agent])

supervisor = PlanAndExecute(planner=planner, executor=executor)

############################
#   RUN THE WHOLE SYSTEM  
############################
result = supervisor.run("Write me 3 paragraphs about AI in healthcare, clean it, and optimize for SEO.")

print(result)
        