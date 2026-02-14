import threading
import queue
from config import settings
import atexit


class AsyncFileLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename: str):
        if hasattr(self, '_initialized'):
            return

        self.filename = filename
        self.log_queue = queue.Queue()

        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

        atexit.register(self.shutdown)

        self._initialized = True

    def _write_loop(self):
        while True:
            item = self.log_queue.get()

            if item is None:
                break

            try:
                with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                    f.write(str(item) + '\n')
            except Exception as e:
                print(f"错误：[AsyncFileLogger] 写入文件时失败: {e}")
            finally:
                self.log_queue.task_done()

    def record(self, data):
        self.log_queue.put(data)

    def shutdown(self):
        self.log_queue.join()
        self.log_queue.put(None)
        self.writer_thread.join(timeout=2)


TOKEN_LOGGER_FILE = settings.file_load_path.token_file
token_logger = AsyncFileLogger(filename=TOKEN_LOGGER_FILE)
