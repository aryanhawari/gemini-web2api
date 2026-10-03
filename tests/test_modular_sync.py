"""Unit + live-server tests for gemini-web2api.

Everything upstream is mocked — no network access is needed. The server tests
spin a real ThreadingHTTPServer on an ephemeral port and talk HTTP to it.
"""

import base64
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_web2api.config import Config, CookieStore, load_config
from gemini_web2api.gemini import (GeminiError, RateLimitedError, RetryableError,
                                   build_inner, clean_text, extract_build_label,
                                   extract_line_text, iter_deltas, parse_response)
from gemini_web2api.models import DEFAULT_MODEL, ModelInfo, resolve_model
from gemini_web2api.multimodal import detect_image_mime
from gemini_web2api.server import App, make_handler
from gemini_web2api.tools import (google_contents_to_messages, messages_to_prompt,
                                  normalize_tools, parse_tool_calls, resolve_image,
                                  responses_input_to_messages)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeGemini:
    def __init__(self, text="Hello from Gemini!"):
        self.text = text
        self.calls = []

    def generate(self, prompt, file_refs=None, model_info=None):
        self.calls.append({"prompt": prompt, "file_refs": list(file_refs or []),
                           "model": model_info.name if model_info else None,
                           "think": model_info.think if model_info else None})
        return self.text

    def stream_generate(self, prompt, file_refs=None, model_info=None):
        self.calls.append({"prompt": prompt, "file_refs": list(file_refs or []),
                           "model": model_info.name if model_info else None,
                           "think": model_info.think if model_info else None})
        for i in range(0, len(self.text), 5):
            yield self.text[i:i + 5]


TOOL_TEXT = 'Let me check.\n```tool_call\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'


class FakeUploader:
    def __init__(self, fail=False):
        self.fail = fail
        self.uploaded = []

    def upload(self, data, mime):
        if self.fail:
            raise RuntimeError("upstream upload exploded")
        self.uploaded.append((len(data), mime))
        return "/uploaded/fake-ref-%d" % len(self.uploaded)


def make_app(fake=None, uploader=None, **cfg_overrides):
    config = Config(**cfg_overrides)
    # image_engine=False keeps image tests on the (mocked) native path
    return App(config, gemini=fake or FakeGemini(), uploader=uploader or FakeUploader(),
               image_engine=False)


def make_server(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def request(port, method, path, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=20)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, payload, headers or {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def parse_sse(raw):
    """Returns [(event_name_or_None, data_str)] for an SSE byte stream."""
    events = []
    name = None
    for line in raw.decode("utf-8", "replace").split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event: "):
            name = line[7:].strip()
        elif line.startswith("data: "):
            events.append((name, line[6:]))
            name = None
    return events


def png_bytes():
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class ModelResolutionTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(resolve_model("gemini-3.7-flash").mode, 1)
        self.assertEqual(resolve_model("gemini-3.5-flash-thinking").mode, 2)
        self.assertEqual(resolve_model("gemini-3.1-pro").mode, 3)
        self.assertEqual(resolve_model("gemini-auto").mode, 4)
        self.assertEqual(resolve_model("gemini-3.5-flash-thinking-lite").mode, 5)
        self.assertEqual(resolve_model("gemini-flash-lite").mode, 6)

    def test_think_suffix(self):
        info = resolve_model("gemini-3.5-flash-thinking@think=2")
        self.assertEqual(info.name, "gemini-3.5-flash-thinking")
        self.assertEqual(info.think, 2)
        self.assertEqual(resolve_model("gemini-auto@think=0").think, 0)

    def test_unknown_falls_back(self):
        self.assertEqual(resolve_model("gpt-9000").name, DEFAULT_MODEL)
        self.assertEqual(resolve_model(None).name, DEFAULT_MODEL)

    def test_extra_flags(self):
        info = resolve_model("gemini-3.1-pro-enhanced")
        self.assertEqual(info.extra, {31: 2, 80: 3})


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

class PayloadBuildTest(unittest.TestCase):
    def test_persistent_flags(self):
        inner = build_inner("hi", None, resolve_model("gemini-3.6-flash"))
        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_temporary_flags(self):
        inner = build_inner("hi", None, resolve_model("gemini-3.6-flash"),
                            temporary_chats=True)
        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_model_fields(self):
        info = resolve_model("gemini-3.1-pro-enhanced")
        inner = build_inner("hi", None, info)
        self.assertEqual(inner[79], 3)
        self.assertEqual(inner[17], [[4]])
        self.assertEqual(inner[31], 2)
        self.assertEqual(inner[80], 3)

    def test_file_refs(self):
        inner = build_inner("look", [["ref1", "img1.png"], ["ref2", "img2.jpg"]], None)
        self.assertEqual(inner[0][0], "look")
        self.assertEqual(inner[0][3], [[["ref1"], "img1.png"], [["ref2"], "img2.jpg"]])

    def test_file_refs_plain_strings(self):
        inner = build_inner("look", ["ref1"], None)
        entry = inner[0][3][0]
        self.assertEqual(entry[0], ["ref1"])
        self.assertTrue(entry[1].startswith("input_"))

    def test_no_refs_is_null(self):
        inner = build_inner("hi", [], None)
        self.assertIsNone(inner[0][3])


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def frame_line(text, pad=30, shape="current"):
    inner = [None] * 20
    if shape == "current":      # candidate[1] = ["text"] (2026 format)
        inner[4] = [["rc", [text]]]
    else:                        # candidate[1] = [[None, ["text"]]] (legacy)
        inner[4] = [["rc", [[None, [text]]]]]
    frame = ["wrb.fr", None, json.dumps(inner), None] + [None] * pad
    return json.dumps([frame])


class ResponseParseTest(unittest.TestCase):
    def test_parse_response(self):
        body = ")]}'\n" + frame_line("Hello world") + "\n" + json.dumps([["di", 42]]) + "\n"
        self.assertEqual(parse_response(body), "Hello world")

    def test_progressive_text_picks_longest(self):
        body = ")]}'\n" + frame_line("Hello") + "\n" + frame_line("Hello world") + "\n"
        self.assertEqual(parse_response(body), "Hello world")

    def test_parse_response_legacy_shape(self):
        body = ")]}'\n" + frame_line("Legacy shape", shape="legacy") + "\n"
        self.assertEqual(parse_response(body), "Legacy shape")

    def test_bard_error(self):
        body = ")]}'\n" + frame_line("x") + "\nBardErrorInfo somewhere\n"
        with self.assertRaises(GeminiError):
            parse_response(body)

    def test_error_frame(self):
        line = json.dumps([["er", 5, "rejected"]])
        with self.assertRaises(GeminiError):
            extract_line_text(line)

    def test_stream_deltas(self):
        lines = [frame_line("Hello", pad=5), frame_line("Hello world", pad=5)]
        self.assertEqual(list(iter_deltas(lines)), ["Hello", " world"])

    def test_stream_rewrite_raises(self):
        lines = [frame_line("Hello", pad=5), frame_line("Goodbye", pad=5)]
        with self.assertRaises(RetryableError):
            list(iter_deltas(lines))

    def test_clean_text(self):
        dirty = "```python?code_reference&code_event_index=7\nprint(1)\n``` see http://googleusercontent.com/card_content/2"
        cleaned = clean_text(dirty)
        self.assertNotIn("?code_reference", cleaned)
        self.assertNotIn("card_content", cleaned)
        self.assertIn("print(1)", cleaned)

    def test_clean_text_code_stdout(self):
        dirty = "```text?code_stdout&code_event_index=1\n42\n```"
        cleaned = clean_text(dirty)
        self.assertNotIn("?code_stdout", cleaned)
        self.assertIn("42", cleaned)

    def test_build_label_cfb2h(self):
        html = '"x":"y","cfb2h":"boq_gemini-web-uiserver_20261002.02_p0","z"'
        self.assertEqual(extract_build_label(html),
                         "boq_gemini-web-uiserver_20261002.02_p0")

    def test_build_label_legacy(self):
        html = 'src="/js/boq_assistant-bard-web-server_20251201.04_p0/x.js"'
        self.assertEqual(extract_build_label(html),
                         "boq_assistant-bard-web-server_20251201.04_p0")
        self.assertIsNone(extract_build_label("<html>nothing</html>"))


# ---------------------------------------------------------------------------
# Prompt building & tools
# ---------------------------------------------------------------------------

class PromptAndToolsTest(unittest.TestCase):
    def test_multiturn_prompt(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
        prompt, images = messages_to_prompt(messages)
        self.assertEqual(images, [])
        self.assertIn("[System instruction]: Be brief.", prompt)
        self.assertIn("Hi\n\n[Assistant]: Hello!\n\nBye", prompt)

    def test_tool_prompt_injection(self):
        tools = normalize_tools([{"type": "function", "function": {
            "name": "get_weather", "description": "w",
            "parameters": {"type": "object", "properties": {}}}}])
        prompt, _ = messages_to_prompt([{"role": "user", "content": "hi"}],
                                       tools=tools, tool_choice="auto")
        self.assertIn("# Tool Use", prompt)
        self.assertIn("```tool_call", prompt)
        self.assertIn("get_weather", prompt)
        none_p, _ = messages_to_prompt([{"role": "user", "content": "hi"}],
                                       tools=tools, tool_choice="none")
        self.assertIn("Do NOT call", none_p)
        req_p, _ = messages_to_prompt([{"role": "user", "content": "hi"}],
                                      tools=tools, tool_choice="required")
        self.assertIn("MUST call at least one", req_p)

    def test_tool_result_line(self):
        prompt, _ = messages_to_prompt([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                                                  "function": {"name": "get_weather",
                                                               "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "name": "get_weather", "content": "20C"},
        ], tools=normalize_tools([]) or None)
        self.assertIn("```tool_call", prompt)
        self.assertIn("[Tool result for get_weather]: 20C", prompt)

    def test_parse_tool_calls_openai(self):
        clean, calls = parse_tool_calls(TOOL_TEXT)
        self.assertEqual(clean, "Let me check.")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"})

    def test_parse_tool_calls_google(self):
        text = '{"name": "flip", "args": {"n": 1}}'
        clean, calls = parse_tool_calls(text, google_format=True)
        self.assertEqual(calls, [{"name": "flip", "args": {"n": 1}}])
        self.assertEqual(clean, "")

    def test_google_contents_conversion(self):
        messages, images, choice = google_contents_to_messages(
            [{"role": "user", "parts": [{"text": "hello"}]},
             {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {"a": 1}}}]}],
            system_instruction={"parts": [{"text": "sys"}]},
            tool_config={"functionCallingConfig": {"mode": "NONE"}})
        self.assertEqual(messages[0]["content"], "sys")
        self.assertIn("```function_call", messages[2]["content"])
        self.assertEqual(choice, "none")

    def test_google_function_response(self):
        messages, _, _ = google_contents_to_messages(
            [{"role": "user", "parts": [
                {"functionResponse": {"name": "f", "response": {"ok": True}}}]}])
        self.assertIn("[Tool result for f]: {\"ok\": true}", messages[0]["content"])

    def test_responses_input_conversion(self):
        messages = responses_input_to_messages([
            {"type": "message", "role": "user", "content": "weather?"},
            {"type": "function_call", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}",
             "call_id": "call_9"},
            {"type": "function_call_output", "call_id": "call_9", "output": "20C"},
        ], instructions="Be brief.")
        self.assertEqual(messages[0]["role"], "system")
        joined = messages_to_prompt(messages)[0]
        self.assertIn("[Tool result for get_weather]: 20C", joined)

    def test_normalize_tools_google_dict(self):
        tools = normalize_tools({"functionDeclarations": [{"name": "f"}]})
        self.assertEqual(tools[0]["name"], "f")

    def test_resolve_image_data_url(self):
        data, mime = resolve_image(
            {"source": "data:image/png;base64," + base64.b64encode(png_bytes()).decode()})
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual(mime, "image/png")

    def test_detect_mime(self):
        self.assertEqual(detect_image_mime(b"\xff\xd8\xff" + b"\x00" * 8), "image/jpeg")
        self.assertEqual(detect_image_mime(b"RIFF\x00\x00\x00\x00WEBP"), "image/webp")
        self.assertIsNone(detect_image_mime(b"not an image at all...."))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigTest(unittest.TestCase):
    def test_env_overrides(self):
        from gemini_web2api.config import _apply_env_overrides
        cfg = Config()
        _apply_env_overrides(cfg, {
            "PROXY_API_KEY": "sk-test-1",
            "PROXY_API_KEYS": "sk-test-2, sk-test-3",
            "RETRY_ATTEMPTS": "5",
            "MAX_CONCURRENT_REQUESTS": "3",
        })
        self.assertEqual(cfg.api_keys, ["sk-test-1", "sk-test-2", "sk-test-3"])
        self.assertEqual(cfg.retry_attempts, 5)
        self.assertEqual(cfg.max_concurrent_requests, 3)

    def test_env_file_parsing(self):
        from gemini_web2api.config import load_env_file
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('# comment\nPROXY_API_KEY="sk-abc"\nPORT=9999\nBAD LINE\n\n')
            env = load_env_file(path)
        self.assertEqual(env["PROXY_API_KEY"], "sk-abc")
        self.assertEqual(env["PORT"], "9999")
        self.assertNotIn("BAD LINE", env)

    def test_env_config_load(self):
        old = os.environ.get("GEMINI_WEB2API_CONFIG")
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"port": 9999, "api_keys": ["k1"], "temporary_chats": True}, fh)
            os.environ["GEMINI_WEB2API_CONFIG"] = path
            os.chdir(tmp)  # isolate from any real .env in the repo cwd
            try:
                cfg = load_config()
            finally:
                os.chdir(old_cwd)
                if old is None:
                    os.environ.pop("GEMINI_WEB2API_CONFIG", None)
                else:
                    os.environ["GEMINI_WEB2API_CONFIG"] = old
        self.assertEqual(cfg.port, 9999)
        self.assertEqual(cfg.api_keys, ["k1"])
        self.assertTrue(cfg.temporary_chats)

    def test_cookie_store_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cookie.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"cookie": "SID=abc; SAPISID=fromcookie"}, fh)
            cookie, sapisid = CookieStore(path).get()
            self.assertEqual(cookie, "SID=abc; SAPISID=fromcookie")
            self.assertEqual(sapisid, "fromcookie")

    def test_cookie_store_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cookie.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("SID=1; HSID=2; SAPISID=sa1")
            cookie, sapisid = CookieStore(path).get()
            self.assertIn("HSID=2", cookie)
            self.assertEqual(sapisid, "sa1")


# ---------------------------------------------------------------------------
# Live server: plain endpoints
# ---------------------------------------------------------------------------

class ServerEndpointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeGemini("The Eiffel Tower is in Paris.")
        cls.server = make_server(make_app(cls.fake))
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_health(self):
        status, body = request(self.port, "GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_health_alias(self):
        status, body = request(self.port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_v1_models(self):
        status, body = request(self.port, "GET", "/v1/models")
        self.assertEqual(status, 200)
        data = json.loads(body)
        ids = [m["id"] for m in data["data"]]
        self.assertIn("gemini-3.6-flash", ids)
        self.assertIn("gemini-3.1-pro-enhanced", ids)

    def test_v1beta_models(self):
        status, body = request(self.port, "GET", "/v1beta/models")
        self.assertEqual(status, 200)
        models = json.loads(body)["models"]
        self.assertTrue(any(m["name"] == "models/gemini-auto" for m in models))

    def test_chat_non_stream(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "model": "gemini-3.5-flash-thinking@think=2",
            "messages": [{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": "Where is the Eiffel Tower?"}],
        })
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["object"], "chat.completion")
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], self.fake.text)
        self.assertEqual(data["model"], "gemini-3.5-flash-thinking")
        self.assertGreater(data["usage"]["total_tokens"], 0)
        call = self.fake.calls[-1]
        self.assertIn("[System instruction]: Be brief.", call["prompt"])
        self.assertEqual(call["think"], 2)

    def test_unknown_model_400(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "model": "gpt-9000",
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        err = json.loads(body)["error"]
        self.assertIn("not supported", err["message"])
        self.assertIn("gemini-3.6-flash", err["message"])

    def test_empty_model_defaults(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model"], DEFAULT_MODEL)

    def test_chat_stream_chunk_order(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "model": "gemini-3.6-flash", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        events = parse_sse(body)
        chunks = [json.loads(d) for _, d in events if d != "[DONE]"]
        self.assertEqual(events[-1][1], "[DONE]")
        self.assertEqual(chunks[0]["choices"][0]["delta"].get("role"), "assistant")
        text = "".join(c["choices"][0]["delta"].get("content", "")
                       for c in chunks[1:] if c.get("choices"))
        self.assertEqual(text, self.fake.text)
        finishes = [c["choices"][0]["finish_reason"] for c in chunks[1:] if c.get("choices")]
        self.assertEqual(finishes[-1], "stop")
        self.assertIn("usage", chunks[-1])  # final chunk carries usage

    def test_responses_non_stream(self):
        status, body = request(self.port, "POST", "/v1/responses", {
            "model": "gemini-3.6-flash",
            "instructions": "Be brief.",
            "input": "hello",
        })
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["object"], "response")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["output"][0]["content"][0]["text"], self.fake.text)

    def test_responses_stream_event_sequence(self):
        status, body = request(self.port, "POST", "/v1/responses", {
            "model": "gemini-3.6-flash", "stream": True, "input": "hello"})
        self.assertEqual(status, 200)
        events = [(name, json.loads(d) if d != "[DONE]" else d) for name, d in parse_sse(body)]
        names = [n for n, _ in events]
        self.assertEqual(names[0], "response.created")
        self.assertEqual(names[1], "response.in_progress")
        self.assertIn("response.output_text.delta", names)
        self.assertEqual(names[-1], "response.completed")
        seqs = [payload["sequence_number"] for _, payload in events if isinstance(payload, dict)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))
        completed = dict(events)["response.completed"]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertGreater(completed["usage"]["total_tokens"], 0)

    def test_google_generate_content(self):
        status, body = request(
            self.port, "POST",
            "/v1beta/models/models/gemini-3.6-flash:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}],
             "systemInstruction": {"parts": [{"text": "sys"}]}})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["candidates"][0]["content"]["parts"][0]["text"], self.fake.text)
        self.assertEqual(data["candidates"][0]["finishReason"], "STOP")
        self.assertIn("usageMetadata", data)
        call = self.fake.calls[-1]
        self.assertIn("[System instruction]: sys", call["prompt"])

    def test_google_stream_sse(self):
        status, body = request(
            self.port, "POST",
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]})
        self.assertEqual(status, 200)
        events = parse_sse(body)
        chunks = [json.loads(d) for _, d in events]
        text = "".join(p.get("text", "")
                       for c in chunks if c.get("candidates")
                       for p in c["candidates"][0]["content"]["parts"])
        self.assertEqual(text, self.fake.text)
        self.assertEqual(chunks[-1]["candidates"][0]["finishReason"], "STOP")

    def test_chunked_request_body(self):
        payload = json.dumps({"model": "gemini-3.6-flash",
                              "messages": [{"role": "user", "content": "chunked"}]}).encode()
        raw = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               b"Content-Type: application/json\r\n"
               b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
        raw += format(len(payload), "x").encode() + b"\r\n" + payload + b"\r\n0\r\n\r\n"
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        try:
            sock.sendall(raw)
            data = b""
            while True:
                part = sock.recv(65536)
                if not part:
                    break
                data += part
        finally:
            sock.close()
        self.assertIn(b'"chat.completion"', data)


# ---------------------------------------------------------------------------
# Live server: tool calling
# ---------------------------------------------------------------------------

class ServerToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeGemini(TOOL_TEXT)
        cls.server = make_server(make_app(cls.fake))
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_chat_tool_calls_non_stream(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "weather lookup",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}}}}}]})
        self.assertEqual(status, 200)
        data = json.loads(body)
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"city": "Paris"})

    def test_chat_tool_calls_stream_single_chunk(self):
        status, body = request(self.port, "POST", "/v1/chat/completions", {
            "model": "gemini-3.6-flash", "stream": True,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]})
        self.assertEqual(status, 200)
        events = parse_sse(body)
        chunks = [json.loads(d) for _, d in events[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        self.assertTrue(any("tool_calls" in d for d in deltas))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_function_call_output(self):
        status, body = request(self.port, "POST", "/v1/responses", {
            "model": "gemini-3.6-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Weather?"},
                {"type": "function_call", "name": "get_weather",
                 "arguments": "{\"city\": \"Paris\"}", "call_id": "call_1"},
                {"type": "function_call_output", "call_id": "call_1", "output": "20C"},
            ]})
        self.assertEqual(status, 200)
        self.assertIn("[Tool result for get_weather]: 20C", self.fake.calls[-1]["prompt"])


# ---------------------------------------------------------------------------
# Live server: images
# ---------------------------------------------------------------------------

class ServerImageTest(unittest.TestCase):
    DATA_URL = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()

    def test_upload_success_passes_ref(self):
        fake, uploader = FakeGemini("saw it"), FakeUploader()
        server = make_server(make_app(fake, uploader))
        try:
            status, body = request(server.server_address[1], "POST",
                                   "/v1/chat/completions", {
                                       "messages": [{"role": "user", "content": [
                                           {"type": "text", "text": "what is this?"},
                                           {"type": "image_url",
                                            "image_url": {"url": self.DATA_URL}}]}]})
            self.assertEqual(status, 200)
            pair = fake.calls[-1]["file_refs"][0]
            self.assertEqual(pair[0], "/uploaded/fake-ref-1")
            self.assertTrue(pair[1].startswith("input_"))
            self.assertEqual(uploader.uploaded[0][1], "image/png")
        finally:
            server.shutdown()

    def test_upload_failure_502(self):
        fake, uploader = FakeGemini("x"), FakeUploader(fail=True)
        server = make_server(make_app(fake, uploader))
        try:
            status, body = request(server.server_address[1], "POST",
                                   "/v1/chat/completions", {
                                       "messages": [{"role": "user", "content": [
                                           {"type": "image_url",
                                            "image_url": {"url": self.DATA_URL}}]}]})
            self.assertEqual(status, 502)
            self.assertIn("image upload failed", json.loads(body)["error"]["message"])
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# Live server: auth
# ---------------------------------------------------------------------------

class ServerAuthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(make_app(FakeGemini("ok"), api_keys=["sk-secret"]))
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_health_open(self):
        status, _ = request(self.port, "GET", "/")
        self.assertEqual(status, 200)

    def test_missing_key_401(self):
        status, _ = request(self.port, "GET", "/v1/models")
        self.assertEqual(status, 401)

    def test_bearer_ok(self):
        status, _ = request(self.port, "GET", "/v1/models",
                            headers={"Authorization": "Bearer sk-secret"})
        self.assertEqual(status, 200)

    def test_query_key_ok(self):
        status, _ = request(self.port, "GET", "/v1/models?key=sk-secret")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
