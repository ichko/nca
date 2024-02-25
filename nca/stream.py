from flask import Flask, render_template, Response
import cv2
import numpy as np
import multiprocessing as mp
import threading
import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class VideoStream:
    def __init__(self, frames_generator) -> None:
        self.frames_generator = frames_generator
        self.app = Flask(__name__)

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        self.server = threading.Thread(target=self._run)

    def start(self):
        # mp.set_start_method("fork")
        # self.server = mp.Process(target=self._run)
        self.server.start()

    def stop(self):
        self.server.terminate()

    def _gen_frames(self):
        while True:
            frame = np.random.randint(0, 255, (500, 500, 3))
            ret, buffer = cv2.imencode(".png", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + frame + b"\r\n")

    def _run(self):
        self.port = find_free_port()
        self.app.run(debug=True, port=self.port)
