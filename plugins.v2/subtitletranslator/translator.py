"""
翻译引擎：Agent Loop + 上下文注入 + Token 精确统计。

借鉴：
  - VideoCaptioner: agent_loop (翻译→验证→反馈→重试, MAX_STEPS=3)
  - llm-subtrans: 上下文注入 (前后 N 句原文)
  - AutoSubv2Custom v2.5.4: 精确计费 (cache_hit + cache_miss + completion)
"""
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt import STANDARD_PROMPT, REFLECT_PROMPT

# 懒加载 openai (MP 环境可能未安装)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# DeepSeek 定价 (¥/1K tokens)
# https://api-docs.deepseek.com/zh-cn/quick_start/pricing
DEEPSEEK_PRICE_CACHE_HIT = 0.0001      # 缓存命中
DEEPSEEK_PRICE_CACHE_MISS = 0.001      # 缓存未命中 (= prompt)
DEEPSEEK_PRICE_COMPLETION = 0.002      # 输出


class TranslatorEngine:
    """
    LLM 字幕翻译引擎

    核心流程：
      1. 分批 (batch_size 句/批)
      2. 上下文注入 (前后 context_window 句原文)
      3. Agent Loop 翻译 (翻译→验证 JSON 键完整性→反馈→重试)
      4. Reflect 反思模式 (可选, 初始翻译→反思→原生重写)
    """

    MAX_STEPS = 3          # Agent loop 最大重试次数
    DEFAULT_TEMPERATURE = 0.3  # 翻译默认温度
    REQUEST_INTERVAL = 0.5 # 请求间隔 (秒)
    MAX_RETRIES = 3        # HTTP 重试次数
    RETRY_DELAY = 3.0      # HTTP 重试间隔 (秒)

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "deepseek-v4-flash",
        batch_size: int = 10,
        context_window: int = 3,
        reflect_mode: bool = False,
        temperature: float = 0.3,
    ):
        """
        初始化翻译引擎

        :param api_base:  API 地址 (OpenAI 兼容)
        :param api_key:   API Key
        :param model:     模型名称
        :param batch_size: 每批翻译句数
        :param context_window: 上下文窗口大小 (前后各 N 句)
        :param reflect_mode: 是否启用反思翻译
        :param temperature: LLM 温度参数
        """
        self.client = None
        if OPENAI_AVAILABLE:
            self.client = OpenAI(base_url=api_base, api_key=api_key)
        else:
            logger.warning("[Translator] openai 未安装，翻译功能不可用")
        self.model = model
        self.batch_size = batch_size
        self.context_window = context_window
        self.reflect_mode = reflect_mode
        self._temperature = temperature

        # Token 统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cache_hit_tokens = 0
        self.total_cache_miss_tokens = 0
        self.total_batches = 0
        self.total_retries = 0

    # ---------- 公共接口 ----------

    def translate(self, subtitles: List[Dict[str, str]]) -> Dict[int, str]:
        """
        翻译全部字幕

        :param subtitles: 字幕列表 [{"text": "原文", "start": ..., "end": ...}, ...]
        :return: 翻译结果 {index: "译文", ...}
        """
        result: Dict[int, str] = {}

        total_batches = (len(subtitles) + self.batch_size - 1) // self.batch_size
        logger.info(f"[Translator] 开始翻译: {len(subtitles)} 句, {total_batches} 批")

        for batch_num, start_idx in enumerate(range(0, len(subtitles), self.batch_size)):
            batch = subtitles[start_idx : start_idx + self.batch_size]

            # 上下文注入
            context = self._build_context(subtitles, start_idx)

            # 翻译
            translated = self._translate_batch(batch, context, batch_num, total_batches)
            for i, text in translated.items():
                result[start_idx + i] = text

            self.total_batches += 1

            # 请求间隔 (避免触发 API 速率限制)
            time.sleep(self.REQUEST_INTERVAL)

        logger.info(
            f"[Translator] 翻译完成: {len(subtitles)} 句, "
            f"retries={self.total_retries}"
        )
        return result

    def get_token_stats(self) -> Dict[str, int]:
        """获取 Token 统计"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cache_hit_tokens": self.total_cache_hit_tokens,
            "cache_miss_tokens": self.total_cache_miss_tokens,
        }

    def get_cost(self) -> Dict[str, float]:
        """获取费用统计 (DeepSeek 定价)"""
        cache_hit_cost = self.total_cache_hit_tokens / 1000 * DEEPSEEK_PRICE_CACHE_HIT
        cache_miss_cost = self.total_cache_miss_tokens / 1000 * DEEPSEEK_PRICE_CACHE_MISS
        completion_cost = self.total_completion_tokens / 1000 * DEEPSEEK_PRICE_COMPLETION

        return {
            "cache_hit_cost": cache_hit_cost,
            "cache_miss_cost": cache_miss_cost,
            "completion_cost": completion_cost,
            "total_cost": cache_hit_cost + cache_miss_cost + completion_cost,
        }

    # ---------- 批次翻译 ----------

    def _translate_batch(
        self,
        batch: List[Dict[str, str]],
        context: Dict[str, Any],
        batch_num: int,
        total_batches: int,
    ) -> Dict[int, str]:
        """
        翻译一批字幕 (Agent Loop)

        :param batch: 当前批次的字幕列表
        :param context: 上下文 (previous_lines, next_lines)
        :param batch_num: 批次序号 (0-based)
        :param total_batches: 总批次数
        :return: {index_in_batch: "译文", ...}
        """
        # 构造输入字典: {"0": "原文1", "1": "原文2", ...}
        subtitle_dict = {str(i): item["text"] for i, item in enumerate(batch)}

        # 选择 prompt
        prompt_name = "reflect" if self.reflect_mode else "standard"
        prompt_template = REFLECT_PROMPT if self.reflect_mode else STANDARD_PROMPT

        # 填充 prompt
        system_prompt = prompt_template.format(
            context_previous=context.get("previous", ""),
            context_next=context.get("next", ""),
        )

        logger.debug(
            f"[Translator] 批次 {batch_num + 1}/{total_batches}: "
            f"{len(batch)} 句, reflect={self.reflect_mode}"
        )

        # Agent Loop 翻译
        result = self._agent_loop(system_prompt, subtitle_dict)

        # 反思模式: 提取 native_translation
        if self.reflect_mode and isinstance(result, dict):
            processed = {}
            for k, v in result.items():
                if isinstance(v, dict) and "native_translation" in v:
                    processed[int(k)] = v["native_translation"]
                else:
                    processed[int(k)] = str(v)
            return processed

        return {int(k): str(v) for k, v in result.items()}

    # ---------- Agent Loop ----------

    def _agent_loop(
        self, system_prompt: str, subtitle_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Agent Loop: 翻译 → 验证 → 反馈 → 重试

        :param system_prompt: 系统提示词
        :param subtitle_dict: 待翻译字幕 {"0": "原文", ...}
        :return: 翻译结果 {"0": "译文", ...}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(subtitle_dict, ensure_ascii=False)},
        ]

        last_result = {}

        for step in range(self.MAX_STEPS):
            raw_response = self._call_llm(messages)
            if not raw_response:
                logger.warning(f"[Translator] Agent step {step + 1}: LLM 返回空")
                continue

            # 尝试解析 JSON
            try:
                result = self._parse_json(raw_response)
            except json.JSONDecodeError as e:
                logger.warning(f"[Translator] Agent step {step + 1}: JSON 解析失败 - {e}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON format. Output ONLY a valid JSON object. "
                            "Error: " + str(e)
                        ),
                    }
                )
                self.total_retries += 1
                continue

            last_result = result

            # 验证键完整性
            is_valid, error_msg = self._validate_response(result, subtitle_dict)
            if is_valid:
                logger.debug(f"[Translator] Agent step {step + 1}: ✅ 验证通过")
                return result

            # 反馈修正
            logger.debug(f"[Translator] Agent step {step + 1}: ❌ {error_msg}")
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {error_msg}\n\n"
                        f"Fix ALL errors above and output a COMPLETE JSON dictionary "
                        f"with ALL {len(subtitle_dict)} keys."
                    ),
                }
            )
            self.total_retries += 1

        logger.warning(
            f"[Translator] Agent loop 耗尽 {self.MAX_STEPS} 步，"
            f"返回最后结果 (可能不完整)"
        )
        return last_result

    # ---------- 验证 ----------

    @staticmethod
    def _validate_response(
        response: Dict[str, Any], expected: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        验证翻译结果键完整性

        :param response: LLM 返回的翻译结果
        :param expected: 期望的键集合
        :return: (是否通过验证, 错误信息)
        """
        if not isinstance(response, dict):
            return False, f"Expected a JSON object, got {type(response).__name__}"

        expected_keys = set(expected.keys())
        actual_keys = set(response.keys())

        if expected_keys == actual_keys:
            return True, ""

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        error_parts = []
        if missing:
            error_parts.append(
                f"Missing keys: {sorted(missing, key=int)} — you MUST translate ALL items"
            )
        if extra:
            error_parts.append(
                f"Extra keys: {sorted(extra, key=int)} — these do not exist in the input, REMOVE them"
            )

        return False, "; ".join(error_parts)

    # ---------- 上下文 ----------

    def _build_context(
        self, all_subs: List[Dict[str, str]], current_start: int
    ) -> Dict[str, str]:
        """
        构造上下文窗口

        :param all_subs: 全部字幕
        :param current_start: 当前批次的起始索引
        :return: {"previous": "前文原文", "next": "后文原文"}
        """
        context = {"previous": "", "next": ""}

        if self.context_window <= 0:
            return context

        # 前文
        prev_start = max(0, current_start - self.context_window)
        if prev_start < current_start:
            prev_lines = []
            for i in range(prev_start, current_start):
                prev_lines.append(f'"{all_subs[i]["text"]}"')
            context["previous"] = "\n".join(prev_lines)

        # 后文
        next_end = min(len(all_subs), current_start + self.batch_size + self.context_window)
        next_start = current_start + self.batch_size
        if next_start < next_end:
            next_lines = []
            for i in range(next_start, next_end):
                next_lines.append(f'"{all_subs[i]["text"]}"')
            context["next"] = "\n".join(next_lines)

        return context

    # ---------- LLM 调用 ----------

    def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        调用 LLM API (带重试)

        :param messages: 消息列表
        :return: 响应文本 或 None
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self._temperature,
                )
                self._record_usage(response)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Translator] API 调用失败 (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"[Translator] API 调用全部失败: {last_error}")
        return None

    def _record_usage(self, response: Any) -> None:
        """
        记录 Token 用量 (支持 DeepSeek 缓存统计)

        :param response: OpenAI API 响应对象
        """
        usage = response.usage
        if not usage:
            return

        self.total_prompt_tokens += usage.prompt_tokens or 0
        self.total_completion_tokens += usage.completion_tokens or 0

        # DeepSeek 缓存统计
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            details = usage.prompt_tokens_details
            self.total_cache_hit_tokens += getattr(details, "cached_tokens", 0) or 0
            self.total_cache_miss_tokens += (
                (usage.prompt_tokens or 0) - (getattr(details, "cached_tokens", 0) or 0)
            )
        else:
            # 无缓存细节时，全部计入 miss
            self.total_cache_miss_tokens += usage.prompt_tokens or 0

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON (带容错)

        :param text: LLM 原始响应
        :return: 解析后的字典
        """
        text = text.strip()

        # 去掉可能的 markdown 代码块包裹
        if text.startswith("```"):
            # 去掉第一行 ```json 或 ```
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else ""
            # 去掉最后的 ```
            if text.rstrip().endswith("```"):
                text = text.rstrip()[: text.rstrip().rfind("```")]

        return json.loads(text)
