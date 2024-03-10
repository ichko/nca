import threading
from IPython.display import display
from ipywidgets import Output


def display_generator(displayable_iterator):
    display_handler = display(None, display_id=True)

    def run():
        for displayable in displayable_iterator:
            display_handler.update(displayable)
            if not threading.current_thread().should_run:
                break

    thread = threading.Thread(target=run)
    thread.should_run = True

    def stop_thread():
        thread.should_run = False

    thread.stop = stop_thread
    thread.start()

    return thread


def stop_all_display_threads():
    for t in threading.enumerate():
        if hasattr(t, "stop"):
            t.stop()
