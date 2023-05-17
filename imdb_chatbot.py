import gradio as gr
import random
import time

import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import os
import pinecone

index_name = "imdb-langchain-self-retriever-demo"


from langchain.base_language import BaseLanguageModel
from langchain.vectorstores import Pinecone, VectorStore
from langchain.chains.query_constructor.ir import StructuredQuery, Visitor
from typing import Any, Dict, List, Optional, Type, cast
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from query_constructor.base import load_query_constructor_chain


def _get_builtin_translator(vectorstore_cls: Type[VectorStore]) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {
        Pinecone: PineconeTranslator
    }
    if vectorstore_cls not in BUILTIN_TRANSLATORS:
        raise ValueError(
            f"Self query retriever with Vector Store type {vectorstore_cls}"
            f" not supported."
        )
    return BUILTIN_TRANSLATORS[vectorstore_cls]()


class J2InstructSelfQueryRetriever(SelfQueryRetriever):
    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            vectorstore: VectorStore,
            document_contents: str,
            metadata_field_info: List[AttributeInfo],
            structured_query_translator: Optional[Visitor] = None,
            chain_kwargs: Optional[Dict] = None,
            **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore.__class__)
        chain_kwargs = chain_kwargs or {}
        if "allowed_comparators" not in chain_kwargs:
            chain_kwargs[
                "allowed_comparators"
            ] = structured_query_translator.allowed_comparators
        if "allowed_operators" not in chain_kwargs:
            chain_kwargs[
                "allowed_operators"
            ] = structured_query_translator.allowed_operators
        llm_chain = load_query_constructor_chain(
            llm, document_contents, metadata_field_info, **chain_kwargs
        )
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json['completions'][0]['data']["text"]


content_handler = ContentHandler()
llm = SagemakerEndpoint(
    endpoint_name="j2-jumbo-instruct",
    region_name="us-east-1",
    model_kwargs={"temperature": 0, "maxTokens": 60, "stopSequences": ["###"]},
    content_handler=content_handler
)

metadata_field_info = [
    AttributeInfo(
        name="originalTitle",
        description="The title of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="countries",
        description="The countries from where the movie is originated",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="genres",
        description="The genres of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="url",
        description="The image URL for the movie",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The overall user rating of the movie",
        type="float",
    ),
    AttributeInfo(
        name="numberOfVotes",
        description="The number of votes for the movie rating",
        type="int",
    ),
    AttributeInfo(
        name="isAdult",
        description="Indicator for the movie whether it is an adult film",
        type="boolean",
    ),
    AttributeInfo(
        name="runtimeMinutes",
        description="Total runtime in minutes of the movie",
        type="int",
    ),
    AttributeInfo(
        name="year",
        description="The year when the movie was released",
        type="int",
    ),
    AttributeInfo(
        name="casts",
        description="The cast members for the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="directors",
        description="The directors for the movie",
        type="string or list[string]",
    )
]

embeddings = HuggingFaceEmbeddings()
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

document_content_description = "Brief summary of a movie"
retriever = J2InstructSelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info,
                                                  verbose=True)

from langchain.chains.question_answering import load_qa_chain

prompt_template = """
The following content are the answer to the question: {question}.

{context}

Instruction: Based on the above documents, reevaluate the question and provide the most accurate answer.
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

chain = load_qa_chain(llm=llm, chain_type="stuff", **chain_type_kwargs)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(question, chat_history):
        results = retriever.get_relevant_documents(question)
        print(results)

        for result in results:
            metadata = result.metadata
            movie_name = metadata['originalTitle']
            plot = result.page_content
            new_page_content = f"Movie name: {movie_name}. \nMovie plot: {plot}"
            result.page_content = new_page_content

        reply = chain.run(input_documents=results, question=question)
        bot_message = reply
        chat_history.append((question, bot_message))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=False, server_name="0.0.0.0")