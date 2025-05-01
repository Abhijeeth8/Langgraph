from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

critique_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a viral twitter influencer that grades tweets. So generate critique the tweet provided to you and also generate recommendations for the user
And always provide detailed recommendations including the requests for length, virality, style, etc."""
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

tweet_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             """You are a Viral twitter tech influencer expert in writing tweets related to latest trends in technology.
Generate best twitter tweets possible for the user.
And if the user provides any critiques to your tweets please adapt them and revise your previous tweet accordingly and respond with updated tweets."""),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(temperature=0.5)

critique_chain = critique_prompt | llm
tweet_gen_chain = tweet_gen_prompt | llm