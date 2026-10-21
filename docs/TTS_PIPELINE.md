# 基础 TTS 管线

这份文档描述 csw 负责的前半部分：把 `PerformanceCoach` 的输出转换成可播放、可下载的 16 kHz WAV。精确停顿、情绪对照和人工评价属于后续实验层。

## 数据流

```text
PerformanceCoach response
        ↓
extract_spoken_script()
        ↓
prepare_tts_text()
        ↓
SiliconFlowTTS.synthesize()
        ↓
WAV header normalization + format validation
        ↓
TTSResult → Streamlit playback / download
```

## 输入协议

`PerformanceCoach` 可以返回带边界的脚本：

```text
<SCRIPT>
[SETUP]大家好，今天聊聊我的网购经历。
[PAUSE=0.8]
[PUNCHLINE]省了二十块，却多花了二百八。
</SCRIPT>
```

`extract_spoken_script` 会移除边界外的解释性文字。`prepare_tts_text` 会移除内部结构标签，并保留一条面向 CosyVoice 的中文表演指令。现阶段停顿标记仍被转换为文本省略号；精确静音拼接由后续表演控制实验负责。

## 输出保证

当 `response_format` 为 `wav` 时，适配器会：

1. 修正流式响应中可能存在的 RIFF/data 长度占位值；
2. 解析 WAV 头，确认响应是有效音频；
3. 检查实际采样率是否等于配置的 `TTS_SAMPLE_RATE`；
4. 返回声道数、音频时长和生成耗时。

调用示例：

```python
from openmic.tts import SiliconFlowTTS, TTSConfig

config = TTSConfig.from_env()
result = SiliconFlowTTS(config).synthesize(script, speed=1.0)

print(result.sample_rate, result.duration_seconds, result.generation_seconds)
with open("openmic.wav", "wb") as audio_file:
    audio_file.write(result.audio)
```

## 本地验证

```bash
PYTHONPATH=src python -m unittest tests.test_tts -v
```

真实 API 验证需要在本地 `.env` 中提供 `TTS_API_KEY` 和 `TTS_BASE_URL`。密钥不应写入仓库或提交到 Git。
