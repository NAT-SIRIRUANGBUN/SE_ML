import openai as _openai
from dotenv import load_dotenv
import os
import json
from typing import List
import asyncio

load_dotenv()


def query_function(func):
    def wrapper(*args, **kwargs):
        max_tries = 3
        now_try = 0
        while now_try < max_tries:
            try:
                res = func(*args, **kwargs)
                break
            except Exception as e:
                now_try += 1
                if now_try == max_tries:
                    raise e
        return res
    return wrapper


class OpenAIClient:
    """Synchronous OpenAI client wrapper."""

    def __init__(self) -> None:
        self.client = _openai.OpenAI()

    @query_function
    def ask_json(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1, tools=None):
        completion_response = self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
            tools=tools
        )
        if with_token:
            return {
                'result': json.loads(completion_response.choices[0].message.content)['result'],
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return json.loads(completion_response.choices[0].message.content)

    @query_function
    def ask_plain_text(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1, tools=None):
        completion_response = self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            tools=tools
        )
        if with_token:
            return {
                'result': completion_response.choices[0].message.content,
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return completion_response.choices[0].message.content

    def text_embedding(self, input: list, model: str = 'text-embedding-3-large', with_token: bool = False):
        embedding_list = []
        sum_token = 0
        index = 0
        step = 500
        while index < len(input):
            embedding_response = self.client.embeddings.create(
                input=input[index:index + step],
                model=model
            )
            index += step
            embedding_list += embedding_response.data
            sum_token = embedding_response.usage.total_tokens
        if with_token:
            return {'embedding': embedding_list, 'token': sum_token}
        return embedding_list

    @query_function
    def query(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1, tools=None):
        completion_response = self.client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            tools=tools,
            tool_choice="auto",
            input=prompt,
            temperature=temperature
        )
        if with_token:
            return {
                'result': completion_response.output[1].content[0].text,
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return completion_response.output[1].content[0].text


class AsyncOpenAIClient:
    """Asynchronous OpenAI client wrapper."""

    def __init__(self) -> None:
        self.client = _openai.AsyncOpenAI()

    async def ask_one_json_async(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1):
        completion_response = await self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        if with_token:
            return {
                'result': json.loads(completion_response.choices[0].message.content)['result'],
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return json.loads(completion_response.choices[0].message.content)

    async def _ask_many_async(self, prompt_list: List[str], model: str = 'gpt-4o', with_token: bool = False, temperature=1):
        tasks = [
            self.client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            for prompt in prompt_list
        ]
        results = await asyncio.gather(*tasks)
        response_list = []
        for completion_response in results:
            if with_token:
                response_list.append({
                    'result': json.loads(completion_response.choices[0].message.content)['result'],
                    'input_token': completion_response.usage.prompt_tokens,
                    'completion_token': completion_response.usage.completion_tokens,
                    'total_token': completion_response.usage.total_tokens
                })
            else:
                response_list.append(json.loads(completion_response.choices[0].message.content))
        return response_list

    def ask_many_json(self, prompt_list: List[str], model: str = 'gpt-4o', with_token: bool = False, temperature=1):
        return asyncio.run(self._ask_many_async(prompt_list, model, with_token, temperature))

    @query_function
    def ask_json(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1):
        completion_response = self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        if with_token:
            return {
                'result': json.loads(completion_response.choices[0].message.content)['result'],
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return json.loads(completion_response.choices[0].message.content)

    @query_function
    def ask_plain_text(self, prompt: str, model: str = 'gpt-4o', with_token: bool = False, temperature=1):
        completion_response = self.client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature
        )
        if with_token:
            return {
                'result': completion_response.choices[0].message.content,
                'input_token': completion_response.usage.prompt_tokens,
                'completion_token': completion_response.usage.completion_tokens,
                'total_token': completion_response.usage.total_tokens
            }
        return completion_response.choices[0].message.content

OpenAI = OpenAIClient
AsyncOpenAI = AsyncOpenAIClient