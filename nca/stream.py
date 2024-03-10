import multiprocessing as mp
import socket
import threading
from contextlib import closing

import cv2
import numpy as np
from flask import Flask, Response, request
from werkzeug.serving import make_server


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class VideoStream(threading.Thread):
    def __init__(self) -> None:
        threading.Thread.__init__(self)
        self.app = Flask(__name__)
        self.port = _find_free_port()
        self.server = make_server("0.0.0.0", self.port, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        self.shutdown = False

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.shutdown = True
        self.server.shutdown()

    @property
    def url(self):
        return f"http://localhost:{self.port}/video_feed"

    def _repr_html_(self):
        return f"<img src='{self.url}' />"

    def _gen_frames(self):
        while not self.shutdown:
            frame = np.random.randint(0, 255, (500, 500, 3))
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + frame + b"\r\n")
