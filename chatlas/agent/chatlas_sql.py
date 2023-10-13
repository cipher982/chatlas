"""Chatlas Agent for workin with SQL."""

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, SystemMessage
from langchain.utilities import SQLDatabase

from chatlas.prompts.prompts_sql import PREFIX, SUFFIX, FUNCS_SUFFIX

TOP_K = 5
INPUT_VARIABLES = None
CALLBACK_MANAGER = None
VERBOSE = True
MAX_ITERATIONS = 15
MAX_EXECUTION_TIME = None
EARLY_STOPPING_METHOD = "force"


def create_chatlas(llm: BaseChatModel, db: str, functions: bool = False) -> AgentExecutor:
    # prefix = PREFIX
    # suffix = SUFFIX_WITH_DF
    # number_of_head_rows = 5
    # callback_manager = None

    # Setup memory for contextual conversation
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Set DB engine
    db_engine = SQLDatabase.from_uri(db)

    # Create toolkit
    toolkit = SQLDatabaseToolkit(db=db_engine, llm=llm)
    tools = toolkit.get_tools()

    # Set prompts
    prefix = PREFIX.format(dialect=toolkit.dialect, top_k=TOP_K)

    if not functions:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=INPUT_VARIABLES,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=CALLBACK_MANAGER,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    else:
        messages = [
            SystemMessage(content=prefix),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=FUNCS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        input_variables = ["input", "agent_scratchpad"]
        _prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

        # llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=CALLBACK_MANAGER,
        )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=CALLBACK_MANAGER,
        verbose=VERBOSE,
        max_iterations=MAX_ITERATIONS,
        max_execution_time=MAX_EXECUTION_TIME,
        early_stopping_method=EARLY_STOPPING_METHOD,
    )
