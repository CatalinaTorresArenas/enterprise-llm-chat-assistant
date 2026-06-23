import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from core.config import load_project_env


load_project_env()


class LLMClientError(RuntimeError):
    pass


class BaseLLMClient:
    api_style = "disabled"
    provider_name = "disabled"

    def is_enabled(self) -> bool:
        return False

    def get_unavailable_reason(self) -> Optional[str]:
        return None

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise LLMClientError("LLM client is disabled.")


class DisabledLLMClient(BaseLLMClient):
    def __init__(self, reason: Optional[str] = None, provider_name: str = "disabled"):
        self.reason = reason
        self.provider_name = provider_name

    def get_unavailable_reason(self) -> Optional[str]:
        return self.reason


class OpenAIResponsesLLMClient(BaseLLMClient):
    api_style = "responses"
    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def get_unavailable_reason(self) -> Optional[str]:
        if self.is_enabled():
            return None
        return "Missing OpenAI API key or model configuration."

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "input": messages,
            "tools": tools,
            "temperature": 0.2,
        }

        body = _post_json(
            url=f"{self.base_url}/responses",
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )

        output = body.get("output") or []
        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for item in output:
            item_type = item.get("type")

            if item_type == "function_call":
                raw_arguments = item.get("arguments") or "{}"
                parsed_arguments = _safe_json_loads(raw_arguments, default={})
                tool_calls.append({
                    "id": item.get("call_id") or item.get("id"),
                    "name": item.get("name"),
                    "arguments": parsed_arguments if isinstance(parsed_arguments, dict) else {},
                })
                continue

            if item_type == "message":
                for content in item.get("content") or []:
                    content_type = content.get("type")
                    if content_type in {"output_text", "text"}:
                        text_value = content.get("text")
                        if isinstance(text_value, dict):
                            text_value = text_value.get("value", "")
                        if text_value:
                            text_chunks.append(str(text_value))

        return {
            "content": "\n".join(chunk.strip() for chunk in text_chunks if str(chunk).strip()).strip(),
            "tool_calls": tool_calls,
            "finish_reason": body.get("status"),
            "raw_response": body,
        }


class OpenAICompatibleChatLLMClient(BaseLLMClient):
    api_style = "chat_completions"
    provider_name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        provider_name: str = "openai_compatible",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.provider_name = provider_name

    def is_enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def get_unavailable_reason(self) -> Optional[str]:
        if self.is_enabled():
            return None
        return f"Missing configuration for provider '{self.provider_name}'."

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [_responses_tool_to_chat_tool(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0.2,
        }

        body = _post_json(
            url=f"{self.base_url}/chat/completions",
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )

        choices = body.get("choices") or []
        if not choices:
            raise LLMClientError("LLM returned no choices.")

        message = choices[0].get("message") or {}
        tool_calls: List[Dict[str, Any]] = []

        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            parsed_arguments = _safe_json_loads(raw_arguments, default={})
            tool_calls.append({
                "id": item.get("id"),
                "name": function.get("name"),
                "arguments": parsed_arguments if isinstance(parsed_arguments, dict) else {},
            })

        return {
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
            "finish_reason": choices[0].get("finish_reason"),
            "raw_response": body,
        }


class BedrockConverseLLMClient(BaseLLMClient):
    api_style = "bedrock_converse"
    provider_name = "bedrock"

    def __init__(
        self,
        model: str,
        region_name: str,
        timeout_seconds: int = 30,
        aws_profile: Optional[str] = None,
    ):
        self.model = model
        self.region_name = region_name
        self.timeout_seconds = timeout_seconds
        self.aws_profile = aws_profile
        self._client = None
        self._boot_error: Optional[str] = None

        try:
            import boto3
            from botocore.config import Config

            session_kwargs = {}
            if aws_profile:
                session_kwargs["profile_name"] = aws_profile

            session = boto3.Session(**session_kwargs)
            config = Config(read_timeout=timeout_seconds, connect_timeout=timeout_seconds)
            self._client = session.client(
                "bedrock-runtime",
                region_name=region_name,
                config=config,
            )
        except Exception as exc:
            self._boot_error = str(exc)
            self._client = None

    def is_enabled(self) -> bool:
        return self._client is not None and bool(self.model and self.region_name)

    def get_unavailable_reason(self) -> Optional[str]:
        if self.is_enabled():
            return None
        if self._boot_error:
            return self._boot_error
        if not self.model:
            return "Missing Bedrock model configuration."
        if not self.region_name:
            return "Missing Bedrock region configuration."
        return "Bedrock client is unavailable."

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self._client:
            raise LLMClientError(
                f"Bedrock client is not available: {self._boot_error or 'unknown initialization error'}"
            )

        system_blocks, bedrock_messages = _chat_messages_to_bedrock_messages(messages)
        payload: Dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
        }

        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            payload["toolConfig"] = {
                "tools": [_responses_tool_to_bedrock_tool(tool) for tool in tools],
                "toolChoice": {"auto": {}},
            }

        try:
            body = self._client.converse(**payload)
        except Exception as exc:
            raise LLMClientError(f"Bedrock converse error: {exc}") from exc

        output = body.get("output") or {}
        message = output.get("message") or {}
        content_blocks = message.get("content") or []

        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content_blocks:
            if "text" in block and block["text"]:
                text_chunks.append(str(block["text"]))
                continue

            tool_use = block.get("toolUse")
            if tool_use:
                tool_calls.append({
                    "id": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "arguments": tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {},
                })

        return {
            "content": "\n".join(chunk.strip() for chunk in text_chunks if str(chunk).strip()).strip(),
            "tool_calls": tool_calls,
            "finish_reason": body.get("stopReason"),
            "raw_response": body,
        }


def _safe_json_loads(raw: Any, default: Any):
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return default


def _responses_tool_to_chat_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    function_block = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
    }

    if "strict" in tool:
        function_block["strict"] = tool.get("strict", True)

    return {
        "type": "function",
        "function": function_block,
    }


def _responses_tool_to_bedrock_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "json": tool["parameters"],
            },
        }
    }


def _chat_messages_to_bedrock_messages(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    system_blocks: List[Dict[str, str]] = []
    bedrock_messages: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_blocks.append({"text": content})
            continue

        if role == "user":
            if isinstance(content, str):
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": content}],
                })
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                blocks = []
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    arguments = _safe_json_loads(function.get("arguments"), default={})
                    if not isinstance(arguments, dict):
                        arguments = {}
                    blocks.append({
                        "toolUse": {
                            "toolUseId": tool_call.get("id"),
                            "name": function.get("name"),
                            "input": arguments,
                        }
                    })
                if blocks:
                    bedrock_messages.append({
                        "role": "assistant",
                        "content": blocks,
                    })
                continue

            if isinstance(content, str) and content.strip():
                bedrock_messages.append({
                    "role": "assistant",
                    "content": [{"text": content}],
                })
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            output_payload = _safe_json_loads(content, default=content)
            blocks = [{
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": [{"json": output_payload}],
                }
            }]
            bedrock_messages.append({
                "role": "user",
                "content": blocks,
            })

    return system_blocks, bedrock_messages


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_headers(api_key),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                raise LLMClientError("LLM returned an empty response body.")
            return json.loads(raw)

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise LLMClientError(f"LLM HTTP error {exc.code}: {detail}") from exc

    except urllib.error.URLError as exc:
        raise LLMClientError(f"LLM connection error: {exc}") from exc

    except json.JSONDecodeError as exc:
        raise LLMClientError(f"LLM returned invalid JSON: {exc}") from exc


def build_llm_client() -> BaseLLMClient:
    provider = os.getenv("AGENT_LLM_PROVIDER", "openai").strip().lower()

    if provider in {"", "disabled", "none"}:
        return DisabledLLMClient(
            reason="LLM provider is disabled by configuration.",
            provider_name="disabled",
        )

    timeout_seconds = int(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "30").strip() or "30")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("AGENT_LLM_API_KEY", "").strip()
        model = os.getenv("AGENT_LLM_MODEL", "").strip() or "gpt-4.1-mini"
        base_url = os.getenv("AGENT_LLM_BASE_URL", "").strip() or "https://api.openai.com/v1"
        if not api_key:
            return DisabledLLMClient(
                reason="Missing OPENAI_API_KEY or AGENT_LLM_API_KEY for provider 'openai'.",
                provider_name="openai",
            )
        return OpenAIResponsesLLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    if provider in {"openai_compatible", "ollama"}:
        base_url = os.getenv("AGENT_LLM_BASE_URL", "").strip()
        api_key = os.getenv("AGENT_LLM_API_KEY", "").strip()
        model = os.getenv("AGENT_LLM_MODEL", "").strip()

        if provider == "ollama" and not base_url:
            base_url = "http://localhost:11434/v1"
        if provider == "ollama" and not api_key:
            api_key = "ollama"
        if provider == "ollama" and not model:
            model = "llama3.1"

        if not base_url or not api_key or not model:
            return DisabledLLMClient(
                reason=f"Missing base URL, API key or model for provider '{provider}'.",
                provider_name=provider,
            )

        return OpenAICompatibleChatLLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            provider_name=provider,
        )

    if provider == "bedrock":
        model = os.getenv("AGENT_LLM_MODEL", "").strip()
        region_name = (
            os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
            or os.getenv("BEDROCK_REGION", "").strip()
            or "us-east-1"
        )
        aws_profile = os.getenv("AWS_PROFILE", "").strip() or None

        if not model:
            return DisabledLLMClient(
                reason="Missing AGENT_LLM_MODEL for provider 'bedrock'.",
                provider_name="bedrock",
            )

        client = BedrockConverseLLMClient(
            model=model,
            region_name=region_name,
            timeout_seconds=timeout_seconds,
            aws_profile=aws_profile,
        )
        if not client.is_enabled():
            return DisabledLLMClient(
                reason=client.get_unavailable_reason() or "Bedrock client is unavailable.",
                provider_name="bedrock",
            )
        return client

    return DisabledLLMClient(
        reason=f"Unsupported provider '{provider}'.",
        provider_name=provider or "unknown",
    )


def get_llm_runtime_metadata() -> Dict[str, Optional[str]]:
    provider = os.getenv("AGENT_LLM_PROVIDER", "openai").strip().lower() or "openai"
    model = os.getenv("AGENT_LLM_MODEL", "").strip()
    base_url = os.getenv("AGENT_LLM_BASE_URL", "").strip()
    region_name = (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("BEDROCK_REGION", "").strip()
    )

    if provider == "openai":
        if not model:
            model = "gpt-4.1-mini"
        if not base_url:
            base_url = "https://api.openai.com/v1"

    if provider == "ollama":
        if not base_url:
            base_url = "http://localhost:11434/v1"
        if not model:
            model = "llama3.1"

    if provider == "bedrock":
        base_url = None

    return {
        "provider": provider,
        "model": model or None,
        "base_url": base_url or None,
        "region_name": region_name or None,
    }
