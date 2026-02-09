import threading
import queue
from config import settings
import atexit


class AsyncFileLogger:
    """
    一个异步的、线程安全的日志记录器。
    它使用一个专用的后台线程来处理文件写入，避免阻塞主线程。
    """
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
        # 创建一个无上限的任务队列
        self.log_queue = queue.Queue()

        # 创建并启动消费者（写入）线程
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

        # 注册一个退出处理器，确保程序关闭前队列中的所有内容都被写入
        atexit.register(self.shutdown)

        self._initialized = True

    def _write_loop(self):
        """消费者线程的循环体，不断从队列中获取并写入数据。"""
        while True:
            # .get() 方法会阻塞，直到队列中有可用的项
            item = self.log_queue.get()

            # 收到“毒丸”信号，结束循环
            if item is None:
                break

            try:
                # 'a' (append) 模式, newline=''
                with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                    f.write(str(item) + '\n')
            except Exception as e:
                print(f"错误：[AsyncFileLogger] 写入文件时失败: {e}")
            finally:
                # 标记任务完成
                self.log_queue.task_done()

    def record(self, data):
        """
        【非阻塞】的记录方法。主线程调用此方法。
        它只负责将数据放入队列，然后立即返回。
        """
        self.log_queue.put(data)

    def shutdown(self):
        """在程序退出时被调用，以确保所有待处理的日志都被写入。"""
        # 等待队列中的所有项都被处理
        self.log_queue.join()
        # 发送“毒丸”信号来优雅地停止写入线程
        self.log_queue.put(None)
        # 等待写入线程真正结束
        self.writer_thread.join(timeout=2)


TOKEN_LOGGER_FILE = settings.file_load_path.token_file
token_logger = AsyncFileLogger(filename=TOKEN_LOGGER_FILE)
