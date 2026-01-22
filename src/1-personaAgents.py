import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------
# Load API KEY from .env
# -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# You can choose: "gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5-mini" ...
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -----------------------------------------------------------
# 1. blog Writing Agent
# -----------------------------------------------------------
blog_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a blog Writing Agent. You write long-form, casual, unstructured content about a topic."),
    ("user", "Write a blog about: {topic}")
])

blog_agent = blog_prompt | llm

# -----------------------------------------------------------
# 2. SEO Checking Agent
# -----------------------------------------------------------
seo_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SEO Checking Agent. Analyze content and provide optimization suggestions."),
    ("user", "Content:\n{content}\n\nGive SEO analysis and improvements.")
])

seo_agent = seo_prompt | llm

# -----------------------------------------------------------
# 3. Fact Checking Agent
# -----------------------------------------------------------
fact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Fact Checking Agent. Identify factual claims and verify them."),
    ("user", "Fact-check the following content:\n\n{content}")
])

fact_agent = fact_prompt | llm

# -----------------------------------------------------------
# 4. Manager Agent (Coordinator)
# -----------------------------------------------------------
def manager_chain(topic):
    # Step 1: blog writing
    blog = blog_agent.invoke({"topic": topic}).content

    # Step 2: SEO check
    seo = seo_agent.invoke({"content": blog}).content

    # Step 3: fact check
    facts = fact_agent.invoke({"content": blog}).content

    # Step 4: Manager summary
    summary = llm.invoke([
        ("system", "Combine all workflow outputs into a final structured report."),
        ("user", f"""
        Topic: {topic}

        === blog Content ===
        {blog}

        === SEO Analysis ===
        {seo}

        === Fact Check ===
        {facts}

        Create a final summarized Manager Report.
        """)
    ]).content

    return blog, seo, facts, summary


# -----------------------------------------------------------
# Use the Manager_Chain to call each of the other "persona" agents
# -----------------------------------------------------------
topic = "Is AI going to be replacing junior programmers any time soon?"


print("Get the Manager_Chain to call each of the other 'persona' agents.... ")
print("Topic we will be making a blog for is: ", topic)
print("Please wait... This will take a minute or two...")
print("-----------------------------------------------------------")
blog, seo, facts, report = manager_chain(topic)

print("📝 blog Content")
print(blog)

print("🔎 SEO Analysis")
print(seo)

print("✔️ Fact Checking")
print(facts)

print("📘 Final Manager Report")
print(report)
