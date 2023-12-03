import argparse
import json
import os
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

import torch

from base import BaseRequest, BatchRequest, ModelWrapper, ServerConfig, SingleRequest  # pylint: disable=import-error


model: ModelWrapper = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        constructor: BaseRequest
        if self.path == SingleRequest.path:
            constructor = SingleRequest
        elif self.path == BatchRequest.path:
            constructor = BatchRequest
        else:
            self._send_error(400, "Unsupported URL")
            return

        request = None
        try:
            request = json.loads(post_data.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON data")
            return

        if request is None:
            self._send_error(400, "Invalid JSON data")
            return

        try:
            request = self._parse_request(request, constructor)
        except ValueError:
            self._send_error(400, "Invalid JSON data")
            return

        output = model(request.get_model_input())

        response_data = output.to_json()

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(response_data.encode("utf-8"))

    def do_GET(self):
        if self.path == "/info":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = json.dumps(model.info())
            self.wfile.write(response_data.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _send_error(self, error_code: int, reason: str):
        self.send_response(error_code)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(reason)

    def _parse_request(self, request: Dict[str, Any], constructor: BaseRequest) -> BaseRequest:
        return constructor.new(request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="google/flan-t5-xl")
    default_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    parser.add_argument("--config-file", type=str, default=default_config_file)
    return parser.parse_args()


def main(args: argparse.Namespace):
    global model  # pylint: disable=global-statement

    server_config = ServerConfig.new(args.config_file)
    server_address = (server_config.address, server_config.port)
    if server_config.verbose:
        log = server_config.log or os.path.join(os.path.dirname(os.path.realpath(__file__)), "log", "server.log")
        log_file = open(log, "w")
    else:
        log_file = open(os.path.devnull, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    print("Initializing the model...", flush=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ModelWrapper(args.model_name, device)
    print("Done.", flush=True)

    with ThreadingHTTPServer(server_address, Handler) as httpd:
        print(f"Server started serving on port {server_config.port}.", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    args = parse_args()
    main(args)
