import threading
from IPython.display import display
from ipywidgets import Output, HBox, VBox
import time


def display_generator(displayable_iterator, sleep=0):
    display_handler = display(None, display_id=True)

    for displayable in displayable_iterator:
        if type(displayable) is list:
            rows = []
            for displayable_row in displayable:
                if type(displayable_row) is list:
                    displayable_row = HBox([as_ipw(i) for i in displayable_row])
                else:
                    displayable_row = as_ipw(displayable_row)
                rows.append(displayable_row)
            display_handler.update(VBox(rows))
        else:
            # displayable = as_ipw(displayable)
            display_handler.update(displayable)

        current_thread = threading.current_thread()
        if hasattr(current_thread, "should_run") and not current_thread.should_run:
            break

        if sleep != 0:
            time.sleep(sleep)


def display_generator_async(displayable_iterator, sleep=0):
    thread = threading.Thread(
        target=lambda: display_generator(displayable_iterator, sleep),
        daemon=True,
    )
    thread.should_run = True

    def stop_thread():
        thread.should_run = False

    thread.stop = stop_thread
    thread.start()

    return thread


def display_generator_decorator(sleep=0, block=False):
    def decorator(generator_ctor):
        generator = generator_ctor()
        if block:
            display_generator(generator, sleep)
        else:
            display_generator_async(generator, sleep)

    return decorator


def as_ipw(obj):
    out = Output()
    with out:
        display(obj)
    return out


def stop_all_display_threads():
    for t in threading.enumerate():
        if hasattr(t, "stop"):
            t.stop()
