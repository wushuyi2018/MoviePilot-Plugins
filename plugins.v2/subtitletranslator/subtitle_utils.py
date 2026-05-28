"""
字幕工具：SRT 解析、ASS 双语输出、内嵌字幕提取、文本清洗、智能断行。

借鉴：
  - ai-trans.py: 双语 ASS 格式、样式标签剥离
  - 修复换行123.py: SRT→ASS 转换、编码检测、三套 HDR/SDR 预设
"""
import json as json_lib
import logging
import os
import re
import subprocess
from typing import Dict, List, Optional

# 懒加载 pysrt (MP 环境可能未安装)
try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------- ASS 样式预设 ----------

ASS_STYLES = {
    "sdr": (
        "Style: Default,文泉驿微米黑,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,93,100,0,0,1,3,0.6,2,10,10,10,1"
    ),
    "hdr": (
        "Style: Default,文泉驿微米黑,20,&H009E9E9E,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,95,100,0,0,1,0.6,0.2,2,10,10,10,1"
    ),
}


def detect_hdr(video_path: str) -> bool:
    """
    检测视频是否为 HDR (通过色彩传递函数)

    :param video_path: 视频文件路径
    :return: True = HDR, False = SDR (或无法检测时默认 SDR)
    """
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False

        streams = json_lib.loads(result.stdout).get("streams", [])
        if not streams:
            return False

        video_stream = streams[0]
        # HDR 特征: 色彩传递函数为 smpte2084 (PQ) 或 arib-std-b67 (HLG)
        transfer = video_stream.get("color_transfer", "")
        color_primaries = video_stream.get("color_primaries", "")
        bits = video_stream.get("bits_per_raw_sample", 8)

        is_hdr = (
            transfer in ("smpte2084", "arib-std-b67")
            or int(bits) >= 10
            or color_primaries in ("bt2020",)
        )
        logger.debug(
            f"[SubtitleUtils] 视频检测: transfer={transfer}, "
            f"primaries={color_primaries}, bits={bits}, hdr={is_hdr}"
        )
        return is_hdr
    except Exception as e:
        logger.warning(f"[SubtitleUtils] HDR 检测失败, 默认 SDR: {e}")
        return False


def find_external_subtitles(video_path: str) -> Optional[str]:
    """
    查找同目录下的外挂字幕文件

    :param video_path: 视频文件路径
    :return: 字幕文件路径 (优先级: .srt > .ass) 或 None
    """
    directory = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]

    # 优先级: 精准匹配 > 模糊匹配
    extensions = [".srt", ".ass", ".ssa"]
    for ext in extensions:
        candidate = os.path.join(directory, f"{base}{ext}")
        if os.path.exists(candidate):
            # 排除已有的翻译输出
            if not candidate.endswith(".chs.ass") and not candidate.endswith(".chs.srt"):
                return candidate

    # 模糊匹配 (文件名包含基础名的字幕)
    for f in sorted(os.listdir(directory)):
        f_lower = f.lower()
        if not any(f_lower.endswith(ext) for ext in extensions):
            continue
        if f_lower.endswith(".chs.ass") or f_lower.endswith(".chs.srt"):
            continue
        if base.lower() in f_lower:
            return os.path.join(directory, f)

    return None


def extract_embedded_subtitle(video_path: str) -> Optional[str]:
    """
    用 ffmpeg 提取视频内嵌字幕轨

    :param video_path: 视频文件路径
    :return: 提取后的 SRT 文件路径 或 None
    """
    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]

    # 1. 检查是否有字幕轨
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "s", video_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        streams = json_lib.loads(result.stdout).get("streams", [])
        if not streams:
            return None
    except Exception as e:
        logger.warning(f"[SubtitleUtils] ffprobe 失败: {e}")
        return None

    # 2. 提取第一个字幕轨
    track_index = streams[0]["index"]
    output_path = os.path.join(video_dir, f"{video_base}.extracted.srt")

    try:
        extract_cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", video_path,
            "-map", f"0:s:{track_index}",
            output_path,
        ]
        subprocess.run(extract_cmd, check=True, timeout=60)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        logger.warning(f"[SubtitleUtils] ffmpeg 提取失败: {e}")

    return None


def parse_srt(srt_path: str) -> List[Dict]:
    """
    解析 SRT 文件为结构化数据

    :param srt_path: SRT 文件路径
    :return: [{"start": datetime, "end": datetime, "text": "清洗后的文本"}, ...]
    """
    subs = _open_srt(srt_path)
    result = []
    for sub in subs:
        text = clean_text(sub.text)
        if not text:
            continue
        result.append({
            "start": sub.start,
            "end": sub.end,
            "text": text,
        })
    return result


def write_ass_bilingual(
    output_path: str,
    originals: List[Dict],
    translated: Dict[int, str],
    is_hdr: bool = False,
    model: str = "LLM",
):
    r"""
    写入双语 ASS 字幕文件

    ASS 格式：
      译文行: 大字 + 粗体 + 描边 (上方)
      原文行: 小字 (下方, "\N" 换行分隔)

    :param output_path: 输出文件路径
    :param originals: 原始字幕 [{"start", "end", "text"}, ...]
    :param translated: 翻译结果 {index: "译文", ...}
    :param is_hdr: 是否 HDR 视频 (自动选择样式)
    :param model: 翻译模型名称 (用于标注行)
    """
    video_style = "hdr" if is_hdr else "sdr"

    # 译文样式 (内联覆盖)
    target_style = (
        "{\\fn文泉驿微米黑\\fs20\\b1\\bord0.8}"
    )
    source_style = (
        "{\\fn微软雅黑\\fs15\\b0}"
    )

    lines = []

    # ASS 头部
    style_line = ASS_STYLES.get(video_style, ASS_STYLES["sdr"])
    lines.extend([
        "[Script Info]",
        "; Generated by SubtitleTranslator (MoviePilot Plugin)",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        style_line,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])

    # 标注行 (\\N 是 ASS 换行符)
    lines.append(
        "Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,"
        f"本字幕由 {model} 翻译" + "\\N" + f"视频样式: {video_style}"
    )

    # 内容行
    for idx, sub in enumerate(originals):
        trans_text = translated.get(idx, "") or ""
        orig_text = sub["text"] or ""

        # 译文智能断行
        trans_text = smart_line_break(trans_text, max_chars=20)

        start = _convert_time(sub["start"])
        end = _convert_time(sub["end"])

        # 一行两条: 译文 \N 原文
        full_text = f"{target_style}{trans_text}\\N{source_style}{orig_text}"
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{full_text}"
        )

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------- 文本处理 ----------

# ASS 标签 + SRT 格式标签
TAG_PATTERN = re.compile(r"<[^>]+>|\{[^}]*\}")


def clean_text(text: str) -> str:
    """
    清洗字幕文本：剥离格式标签 + 规范化换行

    :param text: 原始字幕文本
    :return: 纯文本
    """
    # 剥离 ASS 标签 ({...}) 和 SRT HTML 标签 (<...>)
    text = TAG_PATTERN.sub("", text)
    # 规范化换行符
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\r\n", " ").replace("\n", " ")
    # 压缩连续空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def smart_line_break(text: str, max_chars: int = 20) -> str:
    """
    中文智能断行：优先在标点处切断，≤2 行

    :param text: 待断行的文本
    :param max_chars: 每行最大字符数 (中文约 20, 英文约 42)
    :return: 断行后的文本 ("\n" 分隔)
    """
    if len(text) <= max_chars:
        return text

    # 中文标点和空格优先切断
    break_points = r"[。，！？；：、\s]"
    mid = len(text) // 2

    # 找最接近中间位置的断点
    matches = list(re.finditer(break_points, text))
    best = None
    for m in matches:
        pos = m.end() if m.group().strip() else m.start()
        if max_chars * 0.4 < pos < max_chars * 1.3:
            if best is None or abs(pos - mid) < abs(best.end() - mid):
                best = m

    if best:
        break_pos = best.end() if best.group().strip() else best.start()
        first = text[:break_pos].rstrip()
        second = text[break_pos:].lstrip()
        # 截断第二行
        if len(second) > max_chars:
            second = second[:max_chars]
        return f"{first}\n{second}"

    # 硬切
    first = text[:max_chars].rstrip()
    second = text[max_chars : max_chars * 2].lstrip()
    return f"{first}\n{second}" if second else first


# ---------- 辅助 ----------

def _open_srt(srt_path: str) -> pysrt.SubRipFile:
    """
    打开 SRT 文件 (编码检测)

    :param srt_path: SRT 文件路径
    :return: SubRipFile 对象
    """
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return pysrt.open(srt_path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"无法识别字幕编码: {srt_path}")


def _convert_time(time_obj) -> str:
    """
    SRT 时间对象 → ASS 时间字符串

    :param time_obj: pysrt.SubRipTime 对象
    :return: "H:MM:SS.cc" 格式 (百分之一秒)
    """
    cs = round(time_obj.milliseconds / 10)
    total_sec = time_obj.seconds + time_obj.minutes * 60 + time_obj.hours * 3600
    total_sec += cs // 100
    cs %= 100
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours}:{minutes:02}:{seconds:02}.{cs:02}"
