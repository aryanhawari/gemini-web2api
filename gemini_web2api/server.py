"""HTTP server exposing OpenAI + Google native endpoints over the Gemini protocol.

Endpoints:
  GET  /                                                  health JSON
  GET  /v1/models                                         OpenAI model list
  POST /v1/chat/completions                               OpenAI chat (stream + tools + images)
  POST /v1/responses                                      OpenAI Responses API (Codex CLI)
  GET  /v1beta/models                                     Google native model list
  POST /v1beta/models/{model}:generateContent             Google native, non-streaming
  POST /v1beta/models/{model}:streamGenerateContent       Google native, streaming

Built on stdlib ThreadingHTTPServer — no web framework. Chunked request bodies
are decoded manually. Auth (optional): Authorization: Bearer, x-api-key,
x-goog-api-key or ?key= against config.api_keys (empty list = open).
"""

import json
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .gemini import GeminiError, RateLimitedError
from .models import (THINK_SUFFIX_RE, is_known_model, list_model_infos,
                     resolve_model)
from .multimodal import UploadError, detect_image_mime, make_file_name
from .tools import (PromptError, google_contents_to_messages, messages_to_prompt,
                    normalize_tools, parse_tool_calls, resolve_image,
                    responses_input_to_messages)

_CREATED_TS = int(time.time())


class App:
    """Wires config + gemini client + uploader together; injectable for tests."""

    def __init__(self, config, gemini=None, uploader=None, image_engine=True):
        self.config = config
        if gemini is None:
            from .gemini import GeminiClient
            gemini = GeminiClient(config)
        if uploader is None:
            from .multimodal import ImageUploader
            uploader = ImageUploader(gemini)
        self.gemini = gemini
        self.uploader = uploader
        self.semaphore = threading.BoundedSemaphore(
            max(1, int(config.max_concurrent_requests)))
        self.image_engine = None
        if image_engine is True:
            try:
                from .library_engine import LibraryEngine
                if LibraryEngine.available():
                    self.image_engine = LibraryEngine(config, gemini.cookies)
            except Exception:
                self.image_engine = None

    def warmup(self):
        """Prefetch the build label in the background; never fatal."""
        try:
            self.gemini.ensure_bl()
        except GeminiError as exc:
            self._warn(f"build-label prefetch failed: {exc}")

    @staticmethod
    def _warn(message):
        sys.stderr.write(f"[gemini-web2api] WARN {message}\n")

    def upload_images(self, images):
        """image specs -> [ref, filename] pairs for the payload. Raises UploadError (-> 502)."""
        refs = []
        for spec in images or []:
            data, mime = resolve_image(spec, proxy=self.config.proxy)
            mime = mime or detect_image_mime(data)
            if not mime:
                raise UploadError("unsupported image format (magic bytes not recognised)")
            try:
                refs.append((self.uploader.upload(data, mime), make_file_name(mime)))
            except UploadError:
                raise
            except GeminiError:
                raise
            except Exception as exc:
                raise UploadError(str(exc))
        return refs

    def prepare_image_data(self, images):
        """image specs -> [(bytes, mime)] for the library engine."""
        pairs = []
        for spec in images or []:
            data, mime = resolve_image(spec, proxy=self.config.proxy)
            mime = mime or detect_image_mime(data)
            if not mime:
                raise UploadError("unsupported image format (magic bytes not recognised)")
            pairs.append((data, mime))
        return pairs

    def complete_with_images(self, prompt, images, model_info):
        """Full-text generation for image prompts via the best available engine."""
        if self.image_engine is not None:
            return self.image_engine.generate(prompt, self.prepare_image_data(images))
        return self.gemini.generate(prompt, self.upload_images(images), model_info)

    def images_via_engine(self, images):
        """True when image requests should take the library-engine path."""
        return bool(images) and self.image_engine is not None


def _usage(prompt, text):
    pt = max(1, len(prompt) // 4)
    ct = max(1, len(text) // 4)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def _chat_chunk(cid, created, model, delta, finish=None):
    return {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _tc_wire(call, index):
    return {"id": call["id"], "type": "function", "index": index,
            "function": {"name": call["function"]["name"],
                         "arguments": call["function"]["arguments"]}}


def _resp_object(rid, created, model, output, prompt, text):
    usage = _usage(prompt, text)
    return {
        "id": rid, "object": "response", "created_at": created, "status": "completed",
        "model": model, "output": output,
        "usage": {"input_tokens": usage["prompt_tokens"],
                  "output_tokens": usage["completion_tokens"],
                  "total_tokens": usage["total_tokens"]},
        "metadata": {},
    }


def _resp_message_item(text, status="completed"):
    return {"type": "message", "id": "msg_" + uuid.uuid4().hex[:24], "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


class BodyTooLargeError(Exception):
    pass


def make_handler(app):
    """Returns a request handler class bound to the given App."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"gemini-web2api/{__version__}"
        _response_started = False

        # ------------------------------------------------ plumbing

        def log_message(self, fmt, *args):
            if app.config.log_requests:
                sys.stderr.write("[{}] {} {}\n".format(
                    self.log_date_time_string(), self.address_string(), fmt % args))

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status, message, err_type="invalid_request_error", code=None):
            self._send_json({"error": {"message": message, "type": err_type, "code": code}},
                            status)

        def _start_stream(self, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self._response_started = True

        def _sse(self, payload, event=None):
            data = payload if isinstance(payload, str) else json.dumps(payload)
            chunk = (f"event: {event}\n" if event else "") + f"data: {data}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        def _read_body(self):
            limit = int(app.config.max_body_mb * 1_000_000)
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                data = b""
                while True:
                    size_line = self.rfile.readline(65536).strip()
                    if b";" in size_line:
                        size_line = size_line.split(b";", 1)[0]
                    try:
                        size = int(size_line or b"0", 16)
                    except ValueError:
                        return data
                    if size == 0:
                        self.rfile.readline(65536)  # trailing CRLF
                        return data
                    remaining = size
                    while remaining > 0:
                        part = self.rfile.read(min(remaining, 65536))
                        if not part:
                            return data
                        data += part
                        remaining -= len(part)
                        if len(data) > limit:
                            raise BodyTooLargeError()
                    self.rfile.readline(65536)  # CRLF after each chunk
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                raise BodyTooLargeError()
            data = b""
            while len(data) < length:
                part = self.rfile.read(length - len(data))
                if not part:
                    break
                data += part
            return data

        # ------------------------------------------------ auth

        def _api_key(self):
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            for name in ("x-api-key", "x-goog-api-key"):
                value = self.headers.get(name)
                if value:
                    return value.strip()
            query = parse_qs(urlparse(self.path).query)
            if query.get("key"):
                return query["key"][0]
            return None

        def _authorized(self):
            keys = app.config.api_keys
            if not keys:
                return True
            return self._api_key() in keys

        # ------------------------------------------------ verbs

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type, x-api-key, x-goog-api-key")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            try:
                self._route_get()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                self._route_post()
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ------------------------------------------------ routing

        def _route_get(self):
            path = urlparse(self.path).path
            if path in ("/", "/health"):
                cookie_state = "not set"
                cookies = getattr(app.gemini, "cookies", None)
                if cookies is not None:
                    cookie_state = ("loaded" if any(cookies.get())
                                    else "file empty")
                self._send_json({
                    "status": "ok", "version": __version__,
                    "default_model": app.config.default_model,
                    "models": len(list_model_infos()),
                    "cookie": cookie_state,
                    "images": (app.image_engine.status()
                               if app.image_engine is not None
                               else "native engine (pip install gemini-webapi for best results)"),
                })
                return
            if not self._authorized():
                self._send_error_json(401, "invalid API key", "authentication_error")
                return
            if path == "/v1/models":
                data = [{"id": m.name, "object": "model", "created": _CREATED_TS,
                         "owned_by": "gemini-web2api"} for m in list_model_infos()]
                self._send_json({"object": "list", "data": data})
            elif path == "/v1beta/models":
                models = [{"name": f"models/{m.name}", "displayName": m.name,
                           "description": m.description or m.name,
                           "supportedGenerationMethods": ["generateContent",
                                                          "streamGenerateContent"]}
                          for m in list_model_infos()]
                self._send_json({"models": models})
            else:
                self._send_error_json(404, f"not found: {path}")

        def _route_post(self):
            path = unquote(urlparse(self.path).path)
            if not self._authorized():
                self._send_error_json(401, "invalid API key", "authentication_error")
                return
            try:
                raw = self._read_body()
            except BodyTooLargeError:
                self._send_error_json(
                    413, f"request body exceeds {app.config.max_body_mb:g} MB limit",
                    "request_too_large")
                return
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._send_error_json(400, "request body is not valid JSON")
                return
            try:
                with app.semaphore:
                    if path == "/v1/chat/completions":
                        self._chat_completions(data)
                    elif path == "/v1/responses":
                        self._responses(data)
                    elif path.startswith("/v1beta/models/") and ":" in path:
                        model_path, action = path.rsplit(":", 1)
                        model = model_path.split("/v1beta/models/", 1)[1]
                        if action == "generateContent":
                            self._google_generate(data, model, stream=False)
                        elif action == "streamGenerateContent":
                            self._google_generate(data, model, stream=True)
                        else:
                            self._send_error_json(404, f"unknown action :{action}")
                    else:
                        self._send_error_json(404, f"not found: {path}")
            except PromptError as exc:
                self._fail(400, str(exc))
            except UploadError as exc:
                self._fail(502, f"image upload failed: {exc}", "upstream_error")
            except RateLimitedError as exc:
                self._fail(429, str(exc), "upstream_error")
            except GeminiError as exc:
                self._fail(502, str(exc), "upstream_error")
            except Exception as exc:  # noqa: BLE001 - last-resort guard
                self._fail(500, f"internal error: {exc}", "internal_error")

        def _resolve_model_or_error(self, requested):
            """Strict model validation: unknown explicit names -> 400 (config-gated)."""
            name = (requested or "").strip()
            if name:
                clean = name
                m = THINK_SUFFIX_RE.search(clean)
                if m:
                    clean = clean[: m.start()]
                if not is_known_model(clean):
                    if app.config.strict_models:
                        supported = ", ".join(sorted(m.name for m in list_model_infos()))
                        self._send_error_json(
                            400, f"model '{name}' is not supported. Supported models: {supported}",
                            "invalid_request_error", "model_not_found")
                        return None
                    return resolve_model(name)  # legacy silent fallback
            return resolve_model(name or app.config.default_model)

        def _fail(self, status, message, err_type="invalid_request_error"):
            if self._response_started:
                # headers already flushed (SSE) — nothing to do but hang up
                self.close_connection = True
                return
            self._send_error_json(status, message, err_type)

        # ------------------------------------------------ OpenAI chat

        def _chat_completions(self, data):
            model_info = self._resolve_model_or_error(data.get("model"))
            if model_info is None:
                return
            messages = data.get("messages") or []
            if not messages:
                self._send_error_json(400, "messages[] is required")
                return
            tools = normalize_tools(data.get("tools"))
            prompt, images = messages_to_prompt(messages, tools=tools or None,
                                                tool_choice=data.get("tool_choice", "auto"))
            engine_images = app.images_via_engine(images)
            file_refs = [] if engine_images else (app.upload_images(images) if images else [])
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            wants_stream = bool(data.get("stream"))

            if tools or engine_images:
                # full response is needed to parse tool_call blocks; image
                # requests via the library engine are non-streamed as well
                if engine_images:
                    text = app.complete_with_images(prompt, images, model_info)
                    clean, calls = parse_tool_calls(text) if tools else (text, [])
                else:
                    text = app.gemini.generate(prompt, file_refs, model_info)
                    clean, calls = parse_tool_calls(text)
                if wants_stream:
                    self._start_stream("text/event-stream; charset=utf-8")
                    self._sse(_chat_chunk(cid, created, model_info.name,
                                          {"role": "assistant", "content": ""}))
                    if calls:
                        self._sse(_chat_chunk(
                            cid, created, model_info.name,
                            {"tool_calls": [_tc_wire(c, i) for i, c in enumerate(calls)]}))
                        finish = "tool_calls"
                    else:
                        self._sse(_chat_chunk(cid, created, model_info.name,
                                              {"content": clean}))
                        finish = "stop"
                    self._sse(_chat_chunk(cid, created, model_info.name, {}, finish))
                    self._sse("[DONE]")
                    return
                message = {"role": "assistant", "content": clean or None}
                if calls:
                    message["tool_calls"] = [_tc_wire(c, i) for i, c in enumerate(calls)]
                self._send_json({
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": model_info.name,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": _usage(prompt, clean or json.dumps(calls)),
                })
                return

            if wants_stream:
                gen = app.gemini.stream_events(prompt, file_refs, model_info)
                try:
                    first = next(gen)
                except StopIteration:
                    first = None
                self._start_stream("text/event-stream; charset=utf-8")
                self._sse(_chat_chunk(cid, created, model_info.name,
                                      {"role": "assistant", "content": ""}))
                pieces = []

                def _emit(event):
                    kind, delta = event
                    if kind == "thought":
                        self._sse(_chat_chunk(cid, created, model_info.name,
                                              {"reasoning_content": delta}))
                    else:
                        pieces.append(delta)
                        self._sse(_chat_chunk(cid, created, model_info.name,
                                              {"content": delta}))

                if first is not None:
                    _emit(first)
                for event in gen:
                    _emit(event)
                text = "".join(pieces)
                final = _chat_chunk(cid, created, model_info.name, {}, "stop")
                final["usage"] = _usage(prompt, text)
                self._sse(final)
                self._sse("[DONE]")
            else:
                reasoning = []
                pieces = []
                for kind, delta in app.gemini.stream_events(prompt, file_refs, model_info):
                    (reasoning if kind == "thought" else pieces).append(delta)
                text = "".join(pieces)
                message = {"role": "assistant", "content": text}
                if reasoning:
                    message["reasoning_content"] = "".join(reasoning)
                self._send_json({
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": model_info.name,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "stop"}],
                    "usage": _usage(prompt, text),
                })

        # ------------------------------------------------ OpenAI Responses

        def _responses(self, data):
            model_info = self._resolve_model_or_error(data.get("model"))
            if model_info is None:
                return
            messages = responses_input_to_messages(data.get("input"),
                                                   data.get("instructions"))
            tools = normalize_tools(data.get("tools"))
            prompt, images = messages_to_prompt(messages, tools=tools or None,
                                                tool_choice=data.get("tool_choice", "auto"))
            engine_images = app.images_via_engine(images)
            file_refs = [] if engine_images else (app.upload_images(images) if images else [])
            rid = "resp_" + uuid.uuid4().hex[:24]
            created = int(time.time())

            if tools or engine_images:
                if engine_images:
                    text = app.complete_with_images(prompt, images, model_info)
                    clean, calls = parse_tool_calls(text) if tools else (text, [])
                else:
                    text = app.gemini.generate(prompt, file_refs, model_info)
                    clean, calls = parse_tool_calls(text)
                output = []
                if clean:
                    output.append(_resp_message_item(clean))
                for call in calls:
                    output.append({"type": "function_call",
                                   "id": "fc_" + uuid.uuid4().hex[:24],
                                   "call_id": call["id"],
                                   "name": call["function"]["name"],
                                   "arguments": call["function"]["arguments"],
                                   "status": "completed"})
                resp_obj = _resp_object(rid, created, model_info.name, output, prompt,
                                        clean or json.dumps(calls))
                if data.get("stream"):
                    self._stream_responses_tools(resp_obj)
                else:
                    self._send_json(resp_obj)
                return

            gen = app.gemini.stream_generate(prompt, file_refs, model_info)
            try:
                first = next(gen)
            except StopIteration:
                first = None
            if data.get("stream"):
                self._stream_responses_text(rid, created, model_info.name, prompt,
                                            gen, first)
            else:
                pieces = []
                if first is not None:
                    pieces.append(first)
                pieces.extend(gen)
                text = "".join(pieces)
                self._send_json(_resp_object(
                    rid, created, model_info.name,
                    [_resp_message_item(text)] if text else [], prompt, text))

        def _stream_responses_text(self, rid, created, model, prompt, gen, first):
            msg_id = "msg_" + uuid.uuid4().hex[:24]
            seq = [0]

            def ev(etype, extra):
                payload = {"type": etype, "sequence_number": seq[0]}
                seq[0] += 1
                payload.update(extra)
                self._sse(payload, event=etype)

            self._start_stream("text/event-stream; charset=utf-8")
            ev("response.created", {"response": {"id": rid, "object": "response",
                                                 "created_at": created, "status": "in_progress",
                                                 "model": model, "output": []}})
            ev("response.in_progress", {"response": {"id": rid, "status": "in_progress"}})
            item = {"type": "message", "id": msg_id, "status": "in_progress",
                    "role": "assistant", "content": []}
            ev("response.output_item.added", {"output_index": 0, "item": item})
            part = {"type": "output_text", "text": "", "annotations": []}
            ev("response.content_part.added",
               {"item_id": msg_id, "output_index": 0, "content_index": 0, "part": part})

            pieces = []
            if first is not None:
                pieces.append(first)
                ev("response.output_text.delta",
                   {"item_id": msg_id, "output_index": 0, "content_index": 0,
                    "delta": first, "logprobs": []})
            for delta in gen:
                pieces.append(delta)
                ev("response.output_text.delta",
                   {"item_id": msg_id, "output_index": 0, "content_index": 0,
                    "delta": delta, "logprobs": []})
            text = "".join(pieces)

            ev("response.output_text.done",
               {"item_id": msg_id, "output_index": 0, "content_index": 0, "text": text})
            ev("response.content_part.done",
               {"item_id": msg_id, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}})
            done_item = {"type": "message", "id": msg_id, "status": "completed",
                         "role": "assistant",
                         "content": [{"type": "output_text", "text": text,
                                      "annotations": []}]}
            ev("response.output_item.done", {"output_index": 0, "item": done_item})
            resp_obj = _resp_object(rid, created, model, [done_item] if text else [],
                                    prompt, text)
            ev("response.completed", {"response": resp_obj})

        def _stream_responses_tools(self, resp_obj):
            seq = [0]

            def ev(etype, extra):
                payload = {"type": etype, "sequence_number": seq[0]}
                seq[0] += 1
                payload.update(extra)
                self._sse(payload, event=etype)

            self._start_stream("text/event-stream; charset=utf-8")
            ev("response.created", {"response": {"id": resp_obj["id"],
                                                 "object": "response",
                                                 "created_at": resp_obj["created_at"],
                                                 "status": "in_progress",
                                                 "model": resp_obj["model"],
                                                 "output": []}})
            ev("response.in_progress", {"response": {"id": resp_obj["id"],
                                                     "status": "in_progress"}})
            for index, item in enumerate(resp_obj["output"]):
                ev("response.output_item.added", {"output_index": index, "item": item})
                if item["type"] == "function_call":
                    ev("response.function_call_arguments.done",
                       {"item_id": item["id"], "output_index": index,
                        "arguments": item["arguments"]})
                ev("response.output_item.done", {"output_index": index, "item": item})
            ev("response.completed", {"response": resp_obj})

        # ------------------------------------------------ Google native

        def _google_generate(self, data, model, stream):
            model = (model or "").rsplit("/", 1)[-1]
            model_info = self._resolve_model_or_error(model)
            if model_info is None:
                return
            contents = data.get("contents") or []
            si = data.get("systemInstruction") or data.get("system_instruction")
            tools_raw = data.get("tools")
            tool_config = data.get("toolConfig") or data.get("tool_config")
            messages, images, tool_choice = google_contents_to_messages(
                contents, si, tool_config)
            prompt, images2 = messages_to_prompt(
                messages, tools=normalize_tools(tools_raw) or None,
                tool_choice=tool_choice, google_format=True)
            all_images = images + images2
            engine_images = app.images_via_engine(all_images)
            file_refs = ([] if engine_images else
                         (app.upload_images(all_images) if all_images else []))

            alt_sse = parse_qs(urlparse(self.path).query).get("alt", [""])[0] == "sse"

            def gusage(text):
                usage = _usage(prompt, text)
                return {"promptTokenCount": usage["prompt_tokens"],
                        "candidatesTokenCount": usage["completion_tokens"],
                        "totalTokenCount": usage["total_tokens"]}

            def emit(obj):
                if stream:
                    self._sse(obj) if alt_sse else self.wfile.write(
                        (json.dumps(obj) + "\n").encode("utf-8"))
                else:
                    self._send_json(obj)

            def full_text():
                if engine_images:
                    return app.complete_with_images(prompt, all_images, model_info)
                return app.gemini.generate(prompt, file_refs, model_info)

            if tools_raw or engine_images:
                if stream:
                    self._start_stream("text/event-stream" if alt_sse
                                       else "application/x-ndjson")
                text = full_text()
                calls = (parse_tool_calls(text, google_format=True)[1]
                         if tools_raw else [])
                if calls:
                    parts = [{"functionCall": {"name": c["name"], "args": c["args"]}}
                             for c in calls]
                else:
                    parts = [{"text": text}]
                emit({"candidates": [{"content": {"parts": parts, "role": "model"},
                                      "finishReason": "STOP", "index": 0,
                                      "safetyRatings": []}],
                      "usageMetadata": gusage(text),
                      "modelVersion": model_info.name})
                return

            if stream:
                content_type = ("text/event-stream" if alt_sse
                                else "application/x-ndjson")
                self._start_stream(content_type)
                gen = app.gemini.stream_generate(prompt, file_refs, model_info)
                try:
                    first = next(gen)
                except StopIteration:
                    first = None
                pieces = []
                if first is not None:
                    pieces.append(first)
                    emit({"candidates": [{"content": {"parts": [{"text": first}],
                                                      "role": "model"}, "index": 0}],
                          "modelVersion": model_info.name})
                for delta in gen:
                    pieces.append(delta)
                    emit({"candidates": [{"content": {"parts": [{"text": delta}],
                                                      "role": "model"}, "index": 0}],
                          "modelVersion": model_info.name})
                text = "".join(pieces)
                emit({"candidates": [{"content": {"parts": [], "role": "model"},
                                      "finishReason": "STOP", "index": 0}],
                      "usageMetadata": gusage(text),
                      "modelVersion": model_info.name})
            else:
                text = app.gemini.generate(prompt, file_refs, model_info)
                emit({"candidates": [{"content": {"parts": [{"text": text}],
                                                  "role": "model"},
                                      "finishReason": "STOP", "index": 0,
                                      "safetyRatings": []}],
                      "usageMetadata": gusage(text),
                      "modelVersion": model_info.name})

    return Handler


def serve(config):
    """Blocking entrypoint: builds the app, starts the HTTP server, serves forever."""
    app = App(config)
    handler = make_handler(app)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    server.daemon_threads = True
    threading.Thread(target=app.warmup, daemon=True).start()

    auth = "API key" if config.api_keys else "open (no auth)"
    print(f"gemini-web2api v{__version__} listening on http://{config.host}:{config.port}")
    print(f"  endpoints : /v1/chat/completions, /v1/responses, /v1beta/models/*, /v1/models")
    print(f"  default   : {config.default_model} | auth: {auth}"
          + (" | cookie: yes" if config.cookie_file else " | cookie: no (anonymous)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
