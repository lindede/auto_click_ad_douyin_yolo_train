"""
labels.py —— 整个系统的唯一标签数据源
所有工具、训练脚本、推理模块均从此文件读取标签定义。
新增/修改标签只需改动此文件。
"""

from enum import Enum


class PageClass(Enum):
    """页面分类标签（用于 yolo11n-cls 分类模型）"""
    HomePage      = 0   # 主页面
    VideoPage      = 1   # 视频页面
    TaskPage       = 2   # 任务页面
    AdPage         = 3   # 广告页面
    PopupPage      = 4   # 弹窗页面
    AndroidDesktopPage = 5   # Android 桌面页面
    OtherPage      = 6   # 其他页面

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
    # 主页面元素
    ho_earning_tasks_button           = 0   # 赚钱任务按钮
    ho_earning_tasks_assistive_button = 1   # 赚钱任务悬浮球按钮

    # 任务页面元素
    ta_daily_coins_box                = 2   # 今日可得金币显示框
    ta_total_coins_box                = 3   # 累计可得金币显示框
    ta_watch_video_button             = 4   # 去看剧赚金币按钮（需配置冷却时间）
    ta_watch_ad_button                = 5   # 看下一个广告按钮（需配置冷却时间）
    ta_turn_card_button               = 6   # 翻卡赚钱按钮（需配置冷却时间）
    ta_open_the_chest_button          = 7   # 开宝箱/领金币按钮（需配置冷却时间）
    ta_sroll_bottom_box               = 8   # 底部边界框（已滚动到底部）
    ta_sroll_top_box                  = 9   # 顶部边界框（已滚动到顶部）

    # 广告页面元素
    ad_remaining_time_button                  = 10   # 广告剩余时间按钮
    ad_remaining_time_and_xxx_coins_button    = 11   # 广告剩余时间+可得金币按钮
    ad_remaining_time_and_xxx_coins_box       = 12   # 广告剩余时间+可得金币显示框
    ad_close_button                           = 13   # 广告关闭按钮
    ad_open_app_and_experience_button         = 14   # 打开并体验按钮（含体验时间和金币数）
    ad_open_app_button                        = 15   # 打开App按钮

    # 弹窗页面元素
    po_back_button                    = 16   # 返回按钮
    po_ad_coins_box                   = 17   # 下一个广告可得金币数显示框
    po_watch_ad_get_xxx_coins_button  = 18   # 看下一个广告再赚xxxx金币按钮
    po_watch_ad_button                = 19   # 看下一个广告按钮
    po_exit_button                    = 20   # 退出按钮
    po_happy_to_receive_button        = 21   # 开心收下按钮
    po_close_button                   = 22   # 关闭按钮    

    # 安卓桌面元素
    de_douyin_lite_app_icon         = 23   # 抖音极速版图标
    de_kuaishou_lite_app_icon       = 24   # 快手极速版图标
    de_hongguo_app_icon             = 25   # 红果免费短剧图标
    de_jinritoutiao_lite_app_icon   = 26   # 今日头条极速版图标
    de_wukongliulanqi_app_icon      = 27   # 悟空浏览器图标
    de_baidu_lite_app_icon          = 28   # 百度极速版图标
    de_close_app_button             = 29   # 关闭App按钮

    @classmethod
    def names(cls) -> list[str]:
        return [c.name for c in cls]

    @classmethod
    def from_index(cls, idx: int) -> "DetClass":
        return list(cls)[idx]

    @classmethod
    def from_name(cls, name: str) -> "DetClass":
        return cls[name]

