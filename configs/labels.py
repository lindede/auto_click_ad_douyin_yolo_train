"""
labels.py —— 整个系统的唯一标签数据源
所有工具、训练脚本、推理模块均从此文件读取标签定义。
新增/修改标签只需改动此文件。
"""

from enum import Enum


class PageClass(Enum):
    """页面分类标签（用于 yolo11n-cls 分类模型）"""
    DouyinEarningTasksPageInfo      = 0   # 抖音APP - 赚钱任务页面
    DouyinOpenChestRewardsPageInfo  = 1   # 抖音APP - 获得开宝箱奖励页面
    DouyinTotalRewardsPageInfo      = 2   # 抖音APP - 累计获得奖励页面
    DouyinNextAdPageInfo            = 3   # 抖音APP - 再看一个视频额外获得页面
    DouyinAdShowPageInfo            = 4   # 抖音APP - 广告展示页面
    DouyinHomePageInfo              = 5   # 抖音APP - 首页页面
    AndroidDesktopPageInfo          = 6   # Android系统 - 桌面页面
    OtherPageInfo                   = 7   # 其他页面

    @classmethod
    def names(cls) -> list[str]:
        return [c.name for c in cls]

    @classmethod
    def from_index(cls, idx: int) -> "PageClass":
        return list(cls)[idx]

    @classmethod
    def from_name(cls, name: str) -> "PageClass":
        return cls[name]


class DetClass(Enum):
    """目标检测标签（用于 yolo11n 检测模型）"""
    system_time                       = 0   # 系统时间
    earning_task_ready_button         = 1   # 赚钱任务-可点击按钮
    earning_task_waiting_button       = 2   # 赚钱任务-等待中按钮
    coins                             = 3   # 金币图标/数量
    watch_ad_to_earn_button           = 4   # 看广告赚金币按钮
    happy_to_receive_button           = 5   # 开心收下按钮
    rate_and_get_coins_button         = 6   # 评分获得金币按钮
    ad_title                          = 7   # 广告标题
    get_reward_ready_button           = 8   # 领取奖励-可点击按钮
    get_reward_waiting_button         = 9   # 领取奖励-等待中按钮
    receive_reward_to_next_adA_button = 10  # 领取奖励并看下一个广告按钮
    insist_exit_button                = 11  # 坚持退出按钮
    douyin_home_button                = 12  # 抖音首页按钮
    douyin_me_button                  = 13  # 抖音我的按钮
    douyin_lite_lanch_icon            = 14  # 抖音启动图标
    kuaishou_lite_lanch_icon          = 15  # 快手启动图标
    open_box_for_coins_button         = 16  # 开宝箱得金币按钮
    open_box_for_coins_waiting_button = 17  # 开宝箱得金币等待按钮


    @classmethod
    def names(cls) -> list[str]:
        return [c.name for c in cls]

    @classmethod
    def from_index(cls, idx: int) -> "DetClass":
        return list(cls)[idx]

    @classmethod
    def from_name(cls, name: str) -> "DetClass":
        return cls[name]

