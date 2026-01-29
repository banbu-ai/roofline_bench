import datetime
import time

class DateTimeUtils:
    """
    一个用于获取当前日期、时间及时间戳的工具类。
    """

    @staticmethod
    def get_current_date() -> str:
        """
        获取当前日期，格式为 'YYYY-MM-DD'。
        例如: '2024-04-23'
        """
        return datetime.datetime.now().strftime('%Y-%m-%d')

    @staticmethod
    def get_current_time() -> str:
        """
        获取当前时间，格式为 'HH:MM:SS'。
        例如: '14:30:55'
        """
        return datetime.datetime.now().strftime('%H:%M:%S')

    @staticmethod
    def get_current_datetime() -> str:
        """
        获取当前的完整日期和时间，格式为 'YYYY-MM-DD HH:MM:SS'。
        例如: '2024-04-23 14:30:55'
        """
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def get_current_timestamp() -> str:
        """
        获取当前时间戳（秒）。
        例如: 1713863455
        """
        return str(int(time.time()))

# --- 使用示例 ---
if __name__ == '__main__':
    print(f"当前日期: {DateTimeUtils.get_current_date()}")
    print(f"当前时间: {DateTimeUtils.get_current_time()}")
    print(f"当前完整日期时间: {DateTimeUtils.get_current_datetime()}")
    print(f"当前时间戳 (秒): {DateTimeUtils.get_current_timestamp()}")