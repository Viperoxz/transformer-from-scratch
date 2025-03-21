import torch

# Biến toàn cục để kiểm soát việc chạy ví dụ
RUN_EXAMPLES = True  # Có thể thay đổi thành False để tắt ví dụ

def is_interactive_notebook():
    """Kiểm tra xem code có chạy trong môi trường notebook hay không."""
    return __name__ == "__main__"


def show_example(fn, args=[]):
    """Hiển thị kết quả của một hàm ví dụ nếu đang chạy trực tiếp và RUN_EXAMPLES bật."""
    # if __name__ == "__main__" and RUN_EXAMPLES:
    if RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    """Thực thi một hàm ví dụ nếu đang chạy trực tiếp và RUN_EXAMPLES bật."""
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    """Optimizer giả lập cho mục đích kiểm tra/debug."""
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        super().__init__([], {})  # Khởi tạo lớp cha với tham số rỗng

    def step(self):
        """Bỏ qua bước tối ưu hóa."""
        pass  # Thay None bằng pass cho Pythonic hơn

    def zero_grad(self, set_to_none=False):
        """Bỏ qua việc đặt gradient về 0."""
        pass


class DummyScheduler:
    """Scheduler giả lập cho mục đích kiểm tra/debug."""
    def step(self):
        """Bỏ qua bước cập nhật learning rate."""
        pass