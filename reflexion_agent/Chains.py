from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime

from reflexion_agent.schema import GenEssayFormat, ReviseEssayFormat

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4-turbo")
json_parser = JsonOutputToolsParser(return_id=True)
pydantic_tool_parser = PydanticToolsParser(tools=[GenEssayFormat])

gen_ans_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

gen_instruction = "Generate a ~250 words detailed answer."

generate_ans_chain = gen_ans_prompt_template.partial(
    first_instruction=gen_instruction
) | llm.bind_tools(tools=[GenEssayFormat], tool_choice="GenEssayFormat")
validator = PydanticToolsParser(tools=[GenEssayFormat])

revise_instruction = """
Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.

"""

revise_chain = gen_ans_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseEssayFormat], tool_choice="ReviseEssayFormat")
#
# if __name__ == "__main__":
#     print("Invoking the chain")
#
#     human_question = HumanMessage(content="""Write about Retrieval Augmentation Generation in LLMs. And also list some related sources related to that.""")
#
#     final_chain = gen_ans_prompt_template.partial(first_instruction=gen_instruction) | llm.bind_tools(tools=[GenEssayFormat], tool_choice="GenEssayFormat")  | pydantic_tool_parser
#
#     result = final_chain.invoke(input={"messages": [human_question]})
#
#     print(result)

#
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
#
# load_dotenv()
# import datetime
#
# from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
#
# from schema import AnswerQuestion, ReviseAnswer
#
# llm = ChatOpenAI(model="gpt-3.5-turbo")
# parser = JsonOutputToolsParser(return_id=True)
#
# actor_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are expert researcher.
# Current time: {time}
#
# 1. {first_instruction}
# 2. Reflect and critique your answer. Be severe to maximize improvement.
# 3. Recommend search queries to research information and improve your answer.""",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#         ("system", "Answer the user's question above using the required format."),
#     ]
# ).partial(
#     time=lambda: datetime.datetime.now().isoformat(),
# )
#
#
# first_responder = actor_prompt_template.partial(
#     first_instruction="Provide a detailed ~250 word answer."
# ) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
# validator = PydanticToolsParser(tools=[AnswerQuestion])
#
#
# revise_instructions = """Revise your previous answer using the new information.
#     - You should use the previous critique to add important information to your answer.
#         - You MUST include numerical citations in your revised answer to ensure it can be verified.
#         - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
#             - [1] https://example.com
#             - [2] https://example.com
#     - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
# """
#
#
# revisor = actor_prompt_template.partial(
#     first_instruction=revise_instructions
# ) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
#
# if __name__ == "__main__":
#     print("Invoking the chain")
#
#     human_question = HumanMessage(content="""Write about Retrieval Augmentation Generation in LLMs. And also list some related sources related to that.""")
#
#     final_chain = actor_prompt_template.partial(
#     first_instruction="Provide a detailed ~250 word answer."
# ) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")  | PydanticToolsParser(tools=[AnswerQuestion])
#
#     result = final_chain.invoke(input={"messages": [human_question]})
#
#     print(result)