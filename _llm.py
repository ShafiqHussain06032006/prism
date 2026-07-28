import numpy as np

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from ollama import AsyncClient
from dataclasses import asdict, dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None
global_gemini_openai_client = None  # Gemini via OpenAI-compatible endpoint

# ---------------------------------------------------------------------------
# Gemini client helpers
# ---------------------------------------------------------------------------
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

def get_gemini_openai_client_instance():
    """Return a reusable AsyncOpenAI client pointed at Gemini's OpenAI-compat endpoint."""
    global global_gemini_openai_client
    if global_gemini_openai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        base_url = os.environ.get("GEMINI_BASE_URL", GEMINI_OPENAI_BASE_URL)
        global_gemini_openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return global_gemini_openai_client

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        # set OLLAMA_HOST or pass in host="http://127.0.0.1:11434"
        global_ollama_client = AsyncClient()  # Adjust base URL if necessary        
    return global_ollama_client

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int    
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func_raw: callable
    best_model_name: str    
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None    
    best_model_func: callable = None    
    cheap_model_func: callable = None
    

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func_raw = azure_openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,    
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw = azure_gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw  = azure_gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


######  Ollama configuration

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Send the request to Ollama
    response = await ollama_client.chat(
        model=model,
        messages=messages
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']


async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,
    embedding_model_name = "nomic-embed-text",
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete ,
    best_model_name = "gemma2:latest", # need to be a solid instruct model
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = "olmo2",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)
###### DeepSeek Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 使用DeepSeek API
    import httpx
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DeepSeek API调用
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096)
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return content

async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # 使用硅基流动的BAAI/bge-m3嵌入模型
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

# DeepSeek + BAAI/bge-m3 配置
deepseek_bge_config = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,
    
    best_model_func_raw = deepseek_complete,
    best_model_name = "deepseek-chat",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
    
    cheap_model_func_raw = deepseek_complete,
    cheap_model_name = "deepseek-chat",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


# ===========================================================================
# Google Gemini Configuration
# ===========================================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def gemini_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """Embed texts using Google's native Generative AI embedding API.

    Uses ``google-generativeai`` SDK directly because the OpenAI-compat
    endpoint does not expose an embeddings route for Gemini models.

    Supported model: ``models/text-embedding-004`` (768 dims).
    Falls back to OpenAI-compat endpoint when GEMINI_USE_OPENAI_EMBED=1.
    """
    use_compat = os.environ.get("GEMINI_USE_OPENAI_EMBED", "0") == "1"
    if use_compat:
        # Fallback: use OpenAI-compatible endpoint (works if Gemini exposes it)
        client = get_gemini_openai_client_instance()
        response = await client.embeddings.create(
            model=model_name, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)

        # Process in batches of 100 (API limit per request)
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document",
            )
            all_embeddings.extend(result["embedding"])
        return np.array(all_embeddings)
    except ImportError:
        # google-generativeai not installed — fall back to OpenAI-compat
        client = get_gemini_openai_client_instance()
        response = await client.embeddings.create(
            model=model_name, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Chat completion via Gemini's OpenAI-compatible endpoint with KV caching."""
    client = get_gemini_openai_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Remove unsupported kwargs that OpenAI might accept but Gemini does not
    kwargs.pop("response_format", None)

    response = await client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    content = response.choices[0].message.content
    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return content


async def gemini_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Best / analysis model using Gemini."""
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_caption_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Caption / vision model using Gemini (gemini-2.0-flash or gemini-1.5-pro).

    Gemini vision models accept image/video data as base64 via the
    OpenAI-compatible messages format.  Any ``image_url`` content parts
    already present in *history_messages* are forwarded unchanged.
    """
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Gemini preset — uses native embedding API + OpenAI-compat chat
gemini_config = LLMConfig(
    # Embeddings: google text-embedding-004, 768-dim
    embedding_func_raw=gemini_embedding,
    embedding_model_name="models/text-embedding-004",
    embedding_dim=768,
    embedding_max_token_size=2048,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,

    # Analysis model (high quality reasoning)
    best_model_func_raw=gemini_complete,
    best_model_name="gemini-2.5-pro",
    best_model_max_token_size=32768,
    best_model_max_async=8,

    # Processing model (fast, cheap preprocessing)
    cheap_model_func_raw=gemini_complete,
    cheap_model_name="gemini-2.0-flash",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)
