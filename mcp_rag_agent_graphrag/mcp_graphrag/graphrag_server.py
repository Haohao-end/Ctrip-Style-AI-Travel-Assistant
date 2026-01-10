import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_communities,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_relationships,
    # read_indexer_covariates,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch, GlobalSearchResult
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

ENTITY_NODES_TABLE = 'create_final_nodes'
ENTITY_EMBEDDING_TABLE = 'create_final_entities'
COMMUNITIES_TABLE = 'create_final_communities'
COMMUNITY_REPORT_TABLE = 'create_final_community_reports'
TEXT_UNIT_TABLE = 'create_final_text_units'
RELATIONSHIP_TABLE = 'create_final_relationships'
# COVARIATE_TABLE = 'create_final_covariates'

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2

api_type = OpenaiApiType.OpenAI



# 阿里通义
api_key  = os.getenv('API_KEY')
api_base = os.getenv('BASE_URL')
llm_model = os.getenv("MODEL")

print("api_key----------:",api_key)
print("api_base---------:",api_base)
print("llm_model:-------:",llm_model)
embedding_model = 'text-embedding-v2'
llm_temperature = 0.0
json_mode = False


# Ollama
# api_key = ''
# api_base = 'http://localhost:11434/v1'
# llm_model = 'ollama.mistral:latest'  # 建议使用上下文窗口不小于32K的模型
# embedding_model = 'ollama.quentinz/bge-large-zh-v1.5:latest'
# llm_temperature = 0.0
# json_mode = True


llm = ChatOpenAI(
    api_key=api_key,
    api_type=api_type,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    api_base=api_base,
    api_version='2024-02-15-preview',  # just for AzureOpenAI
    model=llm_model,
    max_retries=20
)

text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_type=api_type,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    api_base=api_base,  # http://localhost:11434/api for Ollama
    api_version='2024-02-15-preview',  # just for AzureOpenAI
    model=embedding_model,
    deployment_name=embedding_model,  # just for AzureOpenAI
    max_retries=20
)

token_encoder = tiktoken.get_encoding('cl100k_base')

local_context_params = {
    'text_unit_prop': 0.5,
    'community_prop': 0.1,
    'conversation_history_max_turns': 5,
    'conversation_history_user_turns_only': True,
    'top_k_mapped_entities': 10,
    'top_k_relationships': 10,
    'include_entity_rank': True,
    'include_relationship_weight': True,
    'include_community_rank': False,
    'return_candidate_context': False,

    # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    'embedding_vectorstore_key': EntityVectorStoreKey.ID,

    # change this based on the token limit you have on your model
    # (if you are using a model with 8k limit, a good setting could be 5000)
    'max_tokens': 12_000
}

global_context_params = {
    # False means using full community reports. True means using community short summaries.
    'use_community_summary': False,
    'shuffle_data': True,
    'include_community_rank': True,
    'min_community_rank': 0,
    'community_rank_name': 'rank',
    'include_community_weight': True,
    'community_weight_name': 'occurrence weight',
    'normalize_community_weight': True,

    # change this based on the token limit you have on your model
    # (if you are using a model with 8k limit, a good setting could be 5000)
    'max_tokens': 12_000,

    'context_name': 'Reports'
}

llm_params = {
    # change this based on the token limit you have on your model
    # (if you are using a model with 8k limit, a good setting could be 1000=1500)
    'max_tokens': 2_000,

    'temperature': llm_temperature
}

map_llm_params = {
    'max_tokens': 1000,
    'temperature': llm_temperature,
    'response_format': {'type': 'json_object'}
}

reduce_llm_params = {
    # change this based on the token limit you have on your model
    # (if you are using a model with 8k limit, a good setting could be 1000-1500)
    'max_tokens': 2000,

    'temperature': llm_temperature
}


def build_local_context_builder() -> LocalSearchMixedContext:
    entity_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_NODES_TABLE}.parquet')
    entity_embedding_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet')

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name='default-entity-description',
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    print(f'Entity count: {len(entity_df)}')
    entity_df.head()

    relationship_df = pd.read_parquet(f'{DATA_DIR}/{RELATIONSHIP_TABLE}.parquet')
    relationships = read_indexer_relationships(relationship_df)
    print(f'Relationship count: {len(relationship_df)}')
    relationship_df.head()

    # NOTE: covariates are turned off by default, because they generally need prompt tuning to be valuable
    # Please see the GRAPHRAG_CLAIM_* settings
    # covariate_df = pd.read_parquet(f'{DATA_DIR}/{COVARIATE_TABLE}.parquet')
    # claims = read_indexer_covariates(covariate_df)
    # logger.info(f'Claim records: {len(claims)}')
    # covariates = {'claims': claims}

    report_df = pd.read_parquet(f'{DATA_DIR}/{COMMUNITY_REPORT_TABLE}.parquet')
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    print(f'Report records: {len(report_df)}')
    report_df.head()

    text_unit_df = pd.read_parquet(f'{DATA_DIR}/{TEXT_UNIT_TABLE}.parquet')
    text_units = read_indexer_text_units(text_unit_df)
    print(f'Text unit records: {len(text_unit_df)}')
    text_unit_df.head()

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,

        # if you did not run covariates during indexing, set this to None
        # covariates=covariates,

        entity_text_embeddings=description_embedding_store,

        # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        embedding_vectorstore_key=EntityVectorStoreKey.ID,

        text_embedder=text_embedder,
        token_encoder=token_encoder
    )

    return context_builder


def build_local_search_engine() -> LocalSearch:
    return LocalSearch(
        llm=llm,
        context_builder=build_local_context_builder(),
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,

        # free form text describing the response type and format, can be anything,
        # e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        response_type='multiple paragraphs'
    )


def build_local_question_gen() -> LocalQuestionGen:
    return LocalQuestionGen(
        llm=llm,
        context_builder=build_local_context_builder(),
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params
    )


def build_global_search_engine() -> GlobalSearch:
    community_df = pd.read_parquet(f'{DATA_DIR}/{COMMUNITIES_TABLE}.parquet')
    entity_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_NODES_TABLE}.parquet')
    report_df = pd.read_parquet(f'{DATA_DIR}/{COMMUNITY_REPORT_TABLE}.parquet')
    entity_embedding_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet')

    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f'Total report count: {len(report_df)}')
    print(f'Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}')
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,

        # default to None if you don't want to use community weights for ranking
        entities=entities,

        token_encoder=token_encoder
    )

    return GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,

        # change this based on the token limit you have on your model
        # (if you are using a model with 8k limit, a good setting could be 5000)
        max_data_tokens=12_000,

        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,

        # set this to True will add instruction to encourage the LLM to incorporate
        # general knowledge in the response, which may increase hallucinations,
        # but could be useful in some use cases.
        allow_general_knowledge=False,

        # set this to False if your LLM model does not support JSON mode.
        json_mode=json_mode,

        context_builder_params=global_context_params,
        concurrent_coroutines=32,

        # free form text describing the response type and format, can be anything,
        # e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        response_type='multiple paragraphs'
    )


def embed_community_reports(
        input_dir: str,
        embedder: OpenAIEmbedding,
        community_report_table: str = COMMUNITY_REPORT_TABLE
):
    '''Embeds the full content of the community reports and saves the DataFrame with embeddings to the output path.'''
    input_path = Path(input_dir) / f'{community_report_table}.parquet'
    output_path = Path(input_dir) / f'{community_report_table}_with_embeddings.parquet'

    if not Path(output_path).exists():
        print('Embedding file not found. Computing community report embeddings...')

        report_df = pd.read_parquet(input_path)

        if 'full_content' not in report_df.columns:
            error_msg = f"'full_content' column not found in {input_path}"
            raise ValueError(error_msg)

        report_df['full_content_embeddings'] = report_df.loc[:, 'full_content'].apply(
            lambda x: embedder.embed(x)
        )

        # Save the DataFrame with embeddings to the output path
        report_df.to_parquet(output_path)
        print(f'Embeddings saved to {output_path}')
        return report_df
    print(f'Embeddings file already exists at {output_path}')
    return pd.read_parquet(output_path)


def build_drift_search_engine() -> DRIFTSearch:
    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_NODES_TABLE}.parquet')
    entity_embedding_df = pd.read_parquet(f'{DATA_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet')

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name='default-entity-description',
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    print(f'Entity count: {len(entity_df)}')
    entity_df.head()

    relationship_df = pd.read_parquet(f'{DATA_DIR}/{RELATIONSHIP_TABLE}.parquet')
    relationships = read_indexer_relationships(relationship_df)

    print(f'Relationship count: {len(relationship_df)}')
    relationship_df.head()

    text_unit_df = pd.read_parquet(f'{DATA_DIR}/{TEXT_UNIT_TABLE}.parquet')
    text_units = read_indexer_text_units(text_unit_df)

    print(f'Text unit records: {len(text_unit_df)}')
    text_unit_df.head()

    report_df = embed_community_reports(DATA_DIR, text_embedder)
    reports = read_indexer_reports(
        report_df,
        entity_df,
        COMMUNITY_LEVEL,
        content_embedding_col='full_content_embeddings'
    )

    context_builder = DRIFTSearchContextBuilder(
        chat_llm=llm,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=reports,
        entity_text_embeddings=description_embedding_store,
        text_units=text_units
    )

    return DRIFTSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder
    )


def local_search(query) -> SearchResult:
    search_engine = build_local_search_engine()
    return search_engine.search(query)

import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx
import json

#创建一个对象
mcp = FastMCP("graphrag")

@mcp.tool()
async def local_asearch(query) -> str:
    """xxx"""
    search_engine = await build_local_search_engine()
    result = await search_engine.asearch(query)
    print("search_result:", type(result.response),result.response)
    return result.response


async def local_astream_search(query) -> AsyncGenerator:
    search_engine = build_local_search_engine()
    async for chunk in search_engine.astream_search(query):
        yield chunk


def global_search(query) -> GlobalSearchResult:
    search_engine = build_global_search_engine()
    return search_engine.search(query)


async def global_asearch(query) -> GlobalSearchResult:
    search_engine = build_global_search_engine()
    return await search_engine.asearch(query)


async def global_astream_search(query) -> AsyncGenerator:
    search_engine = build_global_search_engine()
    async for chunk in search_engine.astream_search(query):
        yield chunk


async def drift_asearch(query) -> SearchResult:
    search_engine = build_drift_search_engine()
    return await search_engine.asearch(query)


def local_search_demo():
    query = 'Who is Scrooge, and what are his main relationships?'
    result = local_search(query)
    print(result.context_data)
    print(result.response)


async def local_asearch_demo():
    query = 'Who is Scrooge, and what are his main relationships?'
    result = await local_asearch(query)
    print(result.context_data)
    print(result.response)


async def local_astream_search_demo():
    query = 'Who is Scrooge, and what are his main relationships?'
    response, context_data = '', {}
    async for chunk in local_astream_search(query):
        if isinstance(chunk, str):
            response += chunk
        else:
            context_data = chunk
    print(context_data)
    print(response)


async def question_generation_demo():
    question_history = [
        'Tell me about Agent Mercer',
        'What happens in Dulce military base?'
    ]
    question_generator = build_local_question_gen()
    candidate_questions = await question_generator.agenerate(
        question_history=question_history, context_data=None, question_count=5
    )
    print(candidate_questions.response)


def global_search_demo():
    query = 'What are the top themes in this story?'
    result = global_search(query)
    print(result.context_data)
    print(result.response)


async def global_asearch_demo():
    query = 'What are the top themes in this story?'
    result = await global_asearch(query)
    print(result.context_data)
    print(result.response)


async def global_astream_search_demo():
    query = 'What are the top themes in this story?'
    response, context_data = '', {}
    async for chunk in global_astream_search(query):
        if isinstance(chunk, str):
            response += chunk
        else:
            context_data = chunk
    print(context_data)
    print(response)


async def drift_asearch_demo():
    query = 'Who is agent Mercer?'
    result = await local_asearch(query)
    print(result.context_data)
    print(result.response)


if __name__ == '__main__':
    DATA_DIR = './doupocangqiong/output'
    LANCEDB_URI = f'{DATA_DIR}/lancedb'
    '''测试需要打开这一行代码
    asyncio.run(local_asearch("xxx"))
    #mcp.run(transport="stdio")
    '''

    '''联调需要与graphrag_client.py一起测试'''
    mcp.run(transport="stdio")
