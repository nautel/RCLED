from collections import defaultdict, deque
import torch.distributed as dist



class SmoothedValue(object):
    """
    English:
    Track a series of values and provide access to smoothed values over a
    window or the global series average.

    Tiếng Việt:
    Theo dõi các giá trị và cung cấp quyền truy cập đến các giá trị được làm mịn thông qua
    một cửa sổ hoặc giá trị trung bình toàn cục của chuỗi
    """
    def __init__(self, window_size=20, fmt=None):
        # fmt là format của chuỗi đầu vào
        if fmt is None:
            fmt = "{median:.6f ({global_avg:.6f})}"
        # deque: chỉ giữ lại giá trị gần đây nhất với kích thước chỉ định
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # cập nhập chuỗi deque với n giá trị mới
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        # đồng bộ hóa các quá trình khác nhau
        if not is_dist_avail_and_initialized():
            return





class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True