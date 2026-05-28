"""
SubtitleTranslator — 入库后自动提取字幕并使用 LLM 翻译为双语 ASS 字幕。

触发：TransferComplete 事件（仅处理监听目录中的文件）
字幕源：外挂 .srt/.ass → 内嵌字幕轨 (ffmpeg 提取)
翻译引擎：Agent Loop (翻译→验证→反馈→重试, 最多 3 轮)
输出：双语 ASS（译文大字在上, 原文小字在下）
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.core.event import eventmanager, Event
from app.log import logger
from app.plugins import _PluginBase
from app.schemas.types import EventType

# 懒加载——避免 openai/pysrt 未安装时整个插件崩溃
_translator = None
_subtitle_utils = None
_prompt = None


def _lazy_import():
    """延迟导入依赖模块 (需要 openai + pysrt)"""
    global _translator, _subtitle_utils, _prompt
    if _translator is None:
        from . import translator as _translator
        from . import subtitle_utils as _subtitle_utils
        from . import prompt as _prompt
    return _translator, _subtitle_utils, _prompt


class SubtitleTranslator(_PluginBase):
    """字幕翻译插件：文件入库后自动翻译字幕"""

    # 插件元信息（官方规范）
    plugin_name: str = "字幕翻译"
    plugin_desc: str = "文件入库后自动提取字幕并使用 LLM 翻译为双语 ASS 字幕。"
    plugin_icon: str = "autosubtitles.jpeg"
    plugin_version: str = "1.3"
    plugin_author: str = "wushuyi2018"
    author_url: str = "https://github.com/wushuyi2018"
    plugin_config_prefix: str = "subtitletranslator_"
    plugin_order: int = 20
    auth_level: int = 1

    # ---------- 插件属性 ----------
    _enabled: bool = False
    _api_base: str = ""
    _api_key: str = ""
    _model: str = ""
    _watch_dir: str = ""
    _batch_size: int = 10
    _context_window: int = 3
    _reflect_mode: bool = False
    _temperature: float = 0.3
    _translated_count: int = 0  # 累计翻译字幕数
    _notify: bool = True

    _translator: Optional[Any] = None

    # ---------- 生命周期 ----------

    def init_plugin(self, config: dict = None):
        """
        生效插件配置

        :param config: 配置信息字典
        """
        if config:
            self._enabled = config.get("enabled", False)
            self._api_base = config.get("api_base", "")
            self._api_key = config.get("api_key", "")
            self._model = config.get("model", "deepseek-v4-flash")
            self._watch_dir = config.get("watch_dir", "")
            self._batch_size = int(config.get("batch_size", 10))
            self._context_window = int(config.get("context_window", 3))
            self._reflect_mode = config.get("reflect_mode", False)
            self._temperature = float(config.get("temperature", 0.3))
            self._notify = config.get("notify", True)

        # 初始化翻译引擎 (懒加载依赖)
        if self._enabled and self._api_key:
            try:
                tr, _, _ = _lazy_import()
                self._translator = tr.TranslatorEngine(
                    api_base=self._api_base,
                    api_key=self._api_key,
                    model=self._model,
                    batch_size=self._batch_size,
                    context_window=self._context_window,
                    reflect_mode=self._reflect_mode,
                    temperature=self._temperature,
                )
                logger.info(
                    f"[SubtitleTranslator] 翻译引擎已启动: "
                    f"model={self._model}, batch={self._batch_size}, "
                    f"context={self._context_window}, reflect={self._reflect_mode}"
                )
            except Exception as e:
                logger.error(f"[SubtitleTranslator] 初始化失败: {e}")
                self._translator = None

    def get_state(self) -> bool:
        """获取插件运行状态"""
        return self._enabled and bool(self._api_key)

    def stop_service(self):
        """停止插件服务"""
        self._enabled = False
        self._translator = None
        logger.info("[SubtitleTranslator] 插件已停止")

    # ---------- 配置表单 ----------

    @staticmethod
    def get_form() -> Tuple[Optional[List[dict]], Dict[str, Any]]:
        return (
            [
                {
                    "component": "VForm",
                    "content": [
                        {
                            "component": "VRow",
                            "content": [
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 4},
                                    "content": [
                                        {
                                            "component": "VSwitch",
                                            "props": {
                                                "model": "enabled",
                                                "label": "启用插件",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 4},
                                    "content": [
                                        {
                                            "component": "VSwitch",
                                            "props": {
                                                "model": "notify",
                                                "label": "完成后通知",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 4},
                                    "content": [
                                        {
                                            "component": "VSwitch",
                                            "props": {
                                                "model": "reflect_mode",
                                                "label": "反思翻译 (多1次API调用)",
                                            },
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "component": "VRow",
                            "content": [
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 6},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "api_base",
                                                "label": "API 地址",
                                                "placeholder": "https://api.deepseek.com/v1",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 6},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "api_key",
                                                "label": "API Key",
                                                "type": "password",
                                            },
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "component": "VRow",
                            "content": [
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 3},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "model",
                                                "label": "模型名称",
                                                "placeholder": "deepseek-v4-flash",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 3},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "temperature",
                                                "label": "温度参数",
                                                "type": "number",
                                                "placeholder": "0.3",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 3},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "watch_dir",
                                                "label": "监听目录名",
                                                "placeholder": "待翻译",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "component": "VCol",
                                    "props": {"cols": 12, "md": 3},
                                    "content": [
                                        {
                                            "component": "VTextField",
                                            "props": {
                                                "model": "batch_size",
                                                "label": "每批句数",
                                                "type": "number",
                                            },
                                        }
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ],
            {
                "enabled": False,
                "api_base": "https://api.deepseek.com/v1",
                "api_key": "",
                "model": "deepseek-v4-flash",
                "temperature": 0.3,
                "watch_dir": "",
                "batch_size": 10,
                "context_window": 3,
                "reflect_mode": False,
                "notify": True,
            },
        )

    # ---------- 事件处理 ----------

    @eventmanager.register(EventType.TransferComplete)
    def on_transfer_complete(self, event: Event):
        """
        文件整理完成事件处理

        :param event: 事件对象 (event.data.transferinfo, event.data.mediainfo)
        """
        if not self._enabled or not self._translator:
            return

        transfer_info = event.data.transferinfo
        if not transfer_info:
            return

        # 检查监听目录：target_diritem 的路径中应包含监听目录名
        target_path = ""
        if transfer_info.target_diritem:
            target_path = transfer_info.target_diritem.path or ""
        if self._watch_dir and self._watch_dir not in target_path:
            return

        # 处理每一个入库的文件
        file_list = transfer_info.file_list_new or []
        for file_path in file_list:
            self._process_file(file_path)

    def _process_file(self, file_path: str):
        """
        处理单个视频文件

        :param file_path: 视频文件绝对路径
        """
        file_name = os.path.basename(file_path)
        logger.info(f"[SubtitleTranslator] 开始处理: {file_name}")

        # 1. 查找字幕源
        subtitle_source = self._find_subtitle_source(file_path)
        if not subtitle_source:
            logger.info(f"[SubtitleTranslator] 跳过: {file_name} (无可用字幕)")
            return

        # 2. 检查是否已有翻译
        output_path = self._get_output_path(file_path)
        if os.path.exists(output_path):
            logger.info(f"[SubtitleTranslator] 跳过: {file_name} (已有翻译)")
            return

        # 3. 解析字幕
        try:
            _, su, _ = _lazy_import()
            subs = su.parse_srt(subtitle_source)
        except Exception as e:
            logger.error(f"[SubtitleTranslator] 解析字幕失败: {file_name} - {e}")
            return
        if not subs:
            logger.info(f"[SubtitleTranslator] 跳过: {file_name} (字幕为空)")
            return

        # 4. 翻译
        try:
            translated = self._translator.translate(subs)
        except Exception as e:
            logger.error(f"[SubtitleTranslator] 翻译失败: {file_name} - {e}")
            if self._notify:
                self.post_message(
                    title="❌ 字幕翻译失败",
                    text=f"文件：{file_name}\n错误：{str(e)[:200]}",
                )
            return

        # 5. 检测 HDR + 写入双语 ASS
        try:
            _, su, _ = _lazy_import()
            is_hdr = su.detect_hdr(file_path)
            su.write_ass_bilingual(
                output_path, subs, translated,
                is_hdr=is_hdr,
                model=self._model,
            )
            self._translated_count += len(subs)
        except Exception as e:
            logger.error(f"[SubtitleTranslator] 写入文件失败: {file_name} - {e}")
            return

        # 6. 通知
        token_info = self._translator.get_token_stats()
        cost_info = self._translator.get_cost()

        logger.info(
            f"[SubtitleTranslator] 翻译完成: {file_name} "
            f"({len(subs)} 句, "
            f"prompt={token_info['prompt_tokens']}, "
            f"completion={token_info['completion_tokens']}, "
            f"cache={token_info['cache_hit_tokens']}, "
            f"费用=¥{cost_info['total_cost']:.4f})"
        )

        if self._notify:
            self.post_message(
                title="✅ 字幕翻译完成",
                text=(
                    f"文件：{file_name}\n"
                    f"句数：{len(subs)}\n"
                    f"Prompt Token：{token_info['prompt_tokens']}\n"
                    f"Completion Token：{token_info['completion_tokens']}\n"
                    f"缓存命中：{token_info['cache_hit_tokens']}\n"
                    f"费用：¥{cost_info['total_cost']:.4f}"
                ),
            )

    # ---------- 辅助方法 ----------

    def _find_subtitle_source(self, video_path: str) -> Optional[str]:
        """
        查找字幕源：外挂 > 内嵌提取

        :param video_path: 视频文件路径
        :return: 字幕文件路径 或 None
        """
        # 优先使用同目录外挂字幕
        _, su, _ = _lazy_import()
        external = su.find_external_subtitles(video_path)
        if external:
            logger.info(f"[SubtitleTranslator] 使用外挂字幕: {os.path.basename(external)}")
            return external

        # 尝试从视频中提取内嵌字幕
        logger.info(f"[SubtitleTranslator] 未找到外挂字幕，尝试提取内嵌字幕")
        extracted = su.extract_embedded_subtitle(video_path)
        if extracted:
            logger.info(f"[SubtitleTranslator] 已提取内嵌字幕: {os.path.basename(extracted)}")
            return extracted

        return None

    def _get_output_path(self, video_path: str) -> str:
        """
        获取输出文件路径

        :param video_path: 视频文件路径
        :return: {原文件名}.chs.ass
        """
        base = os.path.splitext(video_path)[0]
        return f"{base}.chs.ass"

    def get_page(self) -> Optional[List[dict]]:
        """插件详情页"""
        if not self._translator:
            return [
                {
                    "component": "VAlert",
                    "props": {
                        "type": "info",
                        "text": "插件未启用或未配置 API Key。",
                    },
                }
            ]

        token_info = self._translator.get_token_stats()
        cost_info = self._translator.get_cost()

        return [
            {
                "component": "VRow",
                "content": [
                    {
                        "component": "VCol",
                        "props": {"cols": 12, "md": 4},
                        "content": [
                            {
                                "component": "VCard",
                                "props": {"title": "翻译统计"},
                                "content": [
                                    {
                                        "component": "VCardText",
                                        "props": {
                                            "text": (
                                                f"已翻译字幕数：{self._translated_count:,}"
                                            ),
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VCol",
                        "props": {"cols": 12, "md": 4},
                        "content": [
                            {
                                "component": "VCard",
                                "props": {"title": "Token 统计"},
                                "content": [
                                    {
                                        "component": "VCardText",
                                        "props": {
                                            "text": (
                                                f"Prompt Token: {token_info['prompt_tokens']:,}\n"
                                                f"Completion Token: {token_info['completion_tokens']:,}\n"
                                                f"缓存命中: {token_info['cache_hit_tokens']:,}\n"
                                                f"缓存未命中: {token_info['cache_miss_tokens']:,}"
                                            ),
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VCol",
                        "props": {"cols": 12, "md": 4},
                        "content": [
                            {
                                "component": "VCard",
                                "props": {"title": "费用统计"},
                                "content": [
                                    {
                                        "component": "VCardText",
                                        "props": {
                                            "text": (
                                                f"缓存命中: ¥{cost_info['cache_hit_cost']:.4f}\n"
                                                f"缓存未命中: ¥{cost_info['cache_miss_cost']:.4f}\n"
                                                f"Completion: ¥{cost_info['completion_cost']:.4f}\n"
                                                f"**合计: ¥{cost_info['total_cost']:.4f}**"
                                            ),
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
