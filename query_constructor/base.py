"""LLM Chain for turning a user text query into a structured query."""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence

from langchain import BasePromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chains.query_constructor.ir import (
    Comparator,
    Operator,
    StructuredQuery,
)
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
    DEFAULT_EXAMPLES,
    DEFAULT_PREFIX,
    DEFAULT_SCHEMA,
    DEFAULT_SUFFIX,
    EXAMPLE_PROMPT,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException
from langchain import PromptTemplate

IMDB_EXAMPLE_PROMPT_TEMPLATE = """\
###

Data Source:
{data_source}

User Query:
{user_query}

Structured Request:
{structured_request}
"""

IMDB_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["data_source", "user_query", "structured_request"],
    template=IMDB_EXAMPLE_PROMPT_TEMPLATE,
)


IMDB_DEFAULT_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

```json
{{{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}}}
```
The response must not include any '<< Example' snippets in the response. Include only JSON markdown code snippet.
The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical \
operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation \
statements): one or more statements to appy the operation to

Make sure that you only use the comparators and logical operators listed above and \
no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes and only make \
comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return "NO_FILTER" for the filter value.\
"""

IMDB_DEFAULT_SUFFIX = """\
###

Data Source:
```json
{{{{
    content: {content},
    attributes: {attributes}
}}}}
```

User Query:
{{query}}

Structured Request:
"""

IMDB_DATA_SOURCE = """\
```json
{
    "content": "Brief summary of a movie",
    "attributes": {
        "originalTitle": {
            "description": "The title of the movie",
            "type": "string or list[string]" 
        },
        "countries" : {
            "description" : "The countries from where the movie is originated",
            "type" : "string or list[string]"
        },
        "genres" : {
            "description" : "The genres of the movie",
            "type" :  "string or list[string]"
        },
        "url" : {
            "description" : "The image URL for the movie",
            "type" : "string"
        },
        "rating" : {
             "description" : "The overall user rating of the movie",
             "type" : "float"
        },
        "numberOfVotes" : {
            "description" : "The number of votes for the movie rating",
            "type" : "int"
        },
        "isAdult" : { 
            "description" : "Indicator for the movie whether it is an adult film",
            "type" : "boolean"
        },
        "runtimeMinutes" : { 
            "description" : "Total runtime in minutes of the movie",
            "type" :  "int"
        },
        "year" : {
            "description" : "The year when the movie was released",
            "type" : "int"
        },
        "casts" : {
            description="The cast members for the movie",
            type="string or list[string]"
        },
        "directors" : {
            description="The directors for the movie",
            type="string or list[string]"
        }
    }
}
```\
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

IMDB_FULL_ANSWER = """\
```json
{{
    "query": "female reporter",
    "filter" : "and(gt(\\"rating\\", 6.0), eq(\\"year\\", 1966), eq(\\"cast\\", \\"Mary Smith\\")"
}}"""

IMDB_QUERY_ATTRIBUTE_ANSWER = """\
```json
{{
    "query": "police and bank robbers",
    "filter" : "eq(\\"genre\\", \\"Horror\\")"
}}"""

IMDB_NO_FILTER_ANSWER = """\
```json
{{
    "query": "",
    "filter": "NO_FILTER"
}}
```\
"""

IMDB_QUERY_ATTRIBUTE_ONLY_ANSWER = """\
```json
{{
    "query": "",
    "filter" : "eq(\\"genre\\", \\"Funny\\")"
}}"""

IMDB_EXAMPLES = [
    {
        "data_source": IMDB_DATA_SOURCE  ,
        "user_query": "What are the movies about female reporter with ratings greater than 6.0 and released in 1966 and Mary Smith as a cast member",
        "structured_request": IMDB_FULL_ANSWER,
    },
    {
        "data_source": IMDB_DATA_SOURCE,
        "user_query": "Tell me about a movie that has female reporter",
        "structured_request": IMDB_NO_FILTER_ANSWER,
    },
    {
        "data_source": IMDB_DATA_SOURCE,
        "user_query": "What are some horror movies that also contains police and bank robbers?",
        "structured_request": IMDB_QUERY_ATTRIBUTE_ANSWER,
    },
    {
        "data_source": IMDB_DATA_SOURCE,
        "user_query": "What are some funny movies?",
        "structured_request": IMDB_QUERY_ATTRIBUTE_ONLY_ANSWER,
    }
]
class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    ast_parse: Callable
    """Callable that parses dict into internal representation of query language."""

    def parse(self, text: str) -> StructuredQuery:
        try:
            expected_keys = ["query", "filter"]
            parsed = parse_json_markdown(text, expected_keys)
            if len(parsed["query"]) == 0:
                parsed["query"] = " "
            if parsed["filter"] == "NO_FILTER" or not parsed["filter"]:
                parsed["filter"] = None
            else:
                parsed["filter"] = self.ast_parse(parsed["filter"])
            return StructuredQuery(query=parsed["query"], filter=parsed["filter"])
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )

    @classmethod
    def from_components(
        cls,
        allowed_comparators: Optional[Sequence[Comparator]] = None,
        allowed_operators: Optional[Sequence[Operator]] = None,
    ) -> StructuredQueryOutputParser:
        ast_parser = get_parser(
            allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
        )
        return cls(ast_parse=ast_parser.parse)


def _format_attribute_info(info: Sequence[AttributeInfo]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=2).replace("{", "{{").replace("}", "}}")


def _get_prompt(
    document_contents: str,
    attribute_info: Sequence[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
) -> BasePromptTemplate:
    attribute_str = _format_attribute_info(attribute_info)
    examples = examples or DEFAULT_EXAMPLES
    allowed_comparators = allowed_comparators or list(Comparator)
    allowed_operators = allowed_operators or list(Operator)
    schema = IMDB_DEFAULT_SCHEMA.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join(allowed_operators),
    )
    prefix = DEFAULT_PREFIX.format(schema=schema)
    suffix = IMDB_DEFAULT_SUFFIX.format(
        i=len(examples) + 1, content=document_contents, attributes=attribute_str
    )
    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return FewShotPromptTemplate(
        examples=IMDB_EXAMPLES,
        example_prompt=IMDB_EXAMPLE_PROMPT,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
        output_parser=output_parser,
    )

def load_query_constructor_chain(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: List[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
    **kwargs: Any,
) -> LLMChain:
    prompt = _get_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
    )
    return LLMChain(llm=llm, prompt=prompt, **kwargs)
