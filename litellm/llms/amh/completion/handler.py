import json
from copy import deepcopy
from typing import Any, Callable, List, Optional, Union, cast


import httpx
import requests

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    Usage,
    get_secret,
)

from ..common_utils import AWSEventStreamDecoder, AMHError
from .transformation import AMHConfig

amh_config = AMHConfig()


class APIGEERequest:
    def __init__(self, url, data):
        self.url = url
        self.body = data


class AMHLLM(BaseAWSLLM):
    def _prepare_request(
        self,
        model: str,
        data: dict,
        messages: List[AllMessageValues],
        litellm_params: dict,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
    ):
        domain = "https://dev.api.ally.com/amp/inference/utils"
        base_url = f"{domain}/amh-llm-amh-deploy-200285-dev-f-cia-180800-utils-amh-llm-api-proxy/{model}/v1-0/infer"
        prepped_request = APIGEERequest(base_url, data)

        return prepped_request

    def _get_credentials(
            self, params: dict
    ) -> tuple[str | None, str | None]:
        return params.get('client_id', None), params.get('client_secret', None)

    def get_access_token_from_apigee(
            self,
            litellm_params: dict,
            header: Optional[dict] = None
    ):

        if 'token' in litellm_params:
            return litellm_params['token']

        if header is not None:
            if 'token' in litellm_params:
                return header['token']

        url = litellm_params.get(
            "auth_url",
            "https://secure-dev.ally.com/acs/v1/access/token"
        )

        client_id = litellm_params.get(
            "client_id", "c2MY2Y3NwyuYI6dF5sSx7AVTfpu8rSsa"
        )

        client_secret = litellm_params.get(
            "client_secret",
            "PvFESDqZFsAyfJYT125GyiuN7zIjoXlb"
        )

        r = requests.post(
            url,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
        )
        self.token = r.json()["access_token"]
        return self.token

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_prompt_dict={},
        hf_model_name=None,
        logger_fn=None,
        acompletion: bool = False,
        headers: dict = {},
    ):
        apigee_token = self.get_access_token_from_apigee(litellm_params)

        # Non-Streaming completion CALL
        _data = amh_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        prepared_request_args = {
            "model": model,
            "data": _data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "messages": messages,
        }
        prepared_request = self._prepare_request(**prepared_request_args)

        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {apigee_token}",
        }

        try:
            sync_handler = _get_httpx_client()

            # LOGGING
            logging_obj.pre_call(
                input=[],
                api_key="",
                additional_args={
                    "complete_input_dict": _data,
                    "api_base": prepared_request.url,
                    "headers": header,
                },
            )

            # make sync httpx post request here
            try:
                sync_response = sync_handler.post(
                    url=prepared_request.url,
                    headers=header,  # type: ignore
                    data=json.dumps(prepared_request.body),
                    timeout=timeout,
                )

                if sync_response.status_code != 200:
                    raise AMHError(
                        status_code=sync_response.status_code,
                        message=sync_response.text,
                    )
            except Exception as e:
                # LOGGING
                logging_obj.post_call(
                    input=[],
                    api_key="",
                    original_response=str(e),
                    additional_args={"complete_input_dict": _data},
                )
                raise e
        except Exception as e:
            verbose_logger.error("Sagemaker error %s", str(e))
            status_code = (
                getattr(e, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode", 500)
            )
            error_message = (
                getattr(
                    e, "response", {}
                ).get("Error", {}).get("Message", str(e))
            )
            if "Inference Component Name header is required" in error_message:
                error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
            raise AMHError(
                status_code=status_code, message=error_message
            )

        return amh_config.transform_response(
            model=model,
            raw_response=sync_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=_data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
            litellm_params=litellm_params,
        )

    async def make_async_call(
        self,
        api_base: str,
        headers: dict,
        data: str,
        logging_obj,
        client=None,
    ):
        try:
            if client is None:
                client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.SAGEMAKER
                )  # Create a new client if none provided
            response = await client.post(
                api_base,
                headers=headers,
                data=data,
                stream=True,
            )

            if response.status_code != 200:
                raise AMHError(
                    status_code=response.status_code, message=response.text
                )

            decoder = AWSEventStreamDecoder(model="")
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=1024)
            )

            return completion_stream

            # LOGGING
            logging_obj.post_call(
                input=[],
                api_key="",
                original_response="first stream response received",
                additional_args={"complete_input_dict": data},
            )

        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise AMHError(
                status_code=error_code, message=err.response.text
            )
        except httpx.TimeoutException:
            raise AMHError(
                status_code=408, message="Timeout error occurred."
            )
        except Exception as e:
            raise AMHError(status_code=500, message=str(e))

    async def async_streaming(
        self,
        messages: List[AllMessageValues],
        model: str,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
        credentials,
        aws_region_name: str,
        optional_params,
        encoding,
        model_response: ModelResponse,
        model_id: Optional[str],
        logging_obj: Any,
        litellm_params: dict,
        headers: dict,
    ):
        data = await amh_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params={**optional_params, "stream": True},
            litellm_params=litellm_params,
            headers=headers,
        )
        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
            "messages": messages,
        }
        prepared_request = await asyncified_prepare_request(
            **prepared_request_args
        )
        # Fixes https://github.com/BerriAI/litellm/issues/8889
        if model_id is not None:
            prepared_request.headers.update(
                {"X-Amzn-SageMaker-Inference-Component": model_id}
            )

        if not prepared_request.body:
            raise ValueError("Prepared request body is empty")

        completion_stream = await self.make_async_call(
            api_base=prepared_request.url,
            headers=prepared_request.headers,  # type: ignore
            data=cast(str, prepared_request.body),
            logging_obj=logging_obj,
        )
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="sagemaker",
            logging_obj=logging_obj,
        )

        # LOGGING
        logging_obj.post_call(
            input=[],
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response

    async def async_completion(
        self,
        messages: List[AllMessageValues],
        model: str,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
        credentials,
        aws_region_name: str,
        encoding,
        model_response: ModelResponse,
        optional_params: dict,
        logging_obj: Any,
        model_id: Optional[str],
        headers: dict,
        litellm_params: dict,
    ):
        timeout = 300.0
        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.SAGEMAKER
        )

        data = await amh_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
            "messages": messages,
        }

        prepared_request = await asyncified_prepare_request(
            **prepared_request_args
        )
        # LOGGING
        logging_obj.pre_call(
            input=[],
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": prepared_request.url,
                "headers": prepared_request.headers,
            },
        )
        try:
            if model_id is not None:
                # Add model_id as InferenceComponentName header
                # boto3 doc: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                prepared_request.headers.update(
                    {"X-Amzn-SageMaker-Inference-Component": model_id}
                )
            # make async httpx post request here
            try:
                if prepared_request.headers is not None:
                    with open('output.txt', 'w') as f:
                        # f.write(str(messages))
                        f.write(str(prepared_request.headers))

                response = await async_handler.post(
                    url=prepared_request.url,
                    headers=prepared_request.headers,  # type: ignore
                    data=prepared_request.body,
                    timeout=timeout,
                )

                if response.status_code != 200:
                    raise AMHError(
                        status_code=response.status_code, message=response.text
                    )
            except Exception as e:
                # LOGGING
                logging_obj.post_call(
                    input=data["messages"],  # TODO changed 'input' to 'messages'
                    api_key="",
                    original_response=str(e),
                    additional_args={"complete_input_dict": data},
                )
                raise e
        except Exception as e:
            error_message = f"{str(e)}"
            if "Inference Component Name header is required" in error_message:
                error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
            raise AMHError(status_code=500, message=error_message)
        return amh_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
            litellm_params=litellm_params,
        )

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        custom_prompt_dict={},
        litellm_params=None,
        logger_fn=None,
    ):
        """
        Supports Huggingface Jumpstart embeddings like GPT-6B
        """
        # BOTO3 INIT
        import boto3

        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop(
            "aws_secret_access_key", None
        )
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_region_name = optional_params.pop("aws_region_name", None)

        if aws_access_key_id is not None:
            # uses auth params passed to completion
            # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
            client = boto3.client(
                service_name="sagemaker-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )
        else:
            # aws_access_key_id is None, assume user is trying to auth using env variables
            # boto3 automaticaly reads env variables

            # we need to read region name from env
            # I assume majority of users use .env for auth
            region_name = (
                get_secret("AWS_REGION_NAME")
                or aws_region_name  # get region from config file if specified
                or "us-west-2"  # default to us-west-2 if region not specified
            )
            client = boto3.client(
                service_name="sagemaker-runtime",
                region_name=region_name,
            )

        # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
        inference_params = deepcopy(optional_params)
        inference_params.pop("stream", None)

        # Load Config
        config = litellm.SagemakerConfig.get_config()
        for k, v in config.items():
            if (
                k not in inference_params
            ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        # HF EMBEDDING LOGIC
        data = json.dumps({"inputs": input}).encode("utf-8")

        # LOGGING
        request_str = f"""
        response = client.invoke_endpoint(
            EndpointName={model},
            ContentType="application/json",
            Body=f"{data!r}",  # Use !r for safe representation
            CustomAttributes="accept_eula=true",
        )"""  # type: ignore
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data, "request_str": request_str
            },
        )
        # EMBEDDING CALL
        try:
            response = client.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
        except Exception as e:
            status_code = (
                getattr(e, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode", 500)
            )
            error_message = (
                getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            )
            raise AMHError(
                status_code=status_code, message=error_message
            )

        response = json.loads(response["Body"].read().decode("utf8"))
        # LOGGING
        logging_obj.post_call(
            input=input,
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": data},
        )

        print_verbose(f"raw model_response: {response}")
        if "embedding" not in response:
            raise AMHError(
                status_code=500, message="embedding not found in response"
            )
        embeddings = response["embedding"]

        if not isinstance(embeddings, list):
            raise AMHError(
                status_code=422,
                message=f"Response not in expected format - {embeddings}",
            )

        output_data = []
        for idx, embedding in enumerate(embeddings):
            output_data.append(
                {"object": "embedding", "index": idx, "embedding": embedding}
            )

        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

        input_tokens = 0
        for text in input:
            input_tokens += len(encoding.encode(text))

        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
            ),
        )

        return model_response
