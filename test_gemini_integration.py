#!/usr/bin/env python3
"""
快速测试脚本：验证 Gemini API 翻译集成
===========================================
检查列表：
  1. 环境变量加载（GEMINI_API_KEY）
  2. 静态翻译词典覆盖率
  3. 双语标签格式化
  4. 缓存机制
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

print("=" * 60)
print("RQ1 Gemini API 集成验证测试")
print("=" * 60)

# Test 1: 环境变量
print("\n[Test 1] 环境变量检查")
gemini_key = os.environ.get("GEMINI_API_KEY", "")
if gemini_key:
    print(f"  ✓ GEMINI_API_KEY 已设置 (前8字符: {gemini_key[:8]}...)")
else:
    print(f"  ⚠ GEMINI_API_KEY 未设置，使用本地缓存翻译")

# Test 2: 导入并检查静态词典
print("\n[Test 2] 静态翻译词典")
from unsupervised_classification.RQ1.viz_utils import (
    STATIC_TRANSLATIONS,
    translate_entities,
    bilingual_label,
    _load_cache,
    _save_cache
)

sample_keys = ["安倍", "統一教会", "bishop", "神棍", "基督狗"]
print(f"  总词条数: {len(STATIC_TRANSLATIONS)}")
print(f"  覆盖示例:")
for key in sample_keys:
    trans = STATIC_TRANSLATIONS.get(key, STATIC_TRANSLATIONS.get(key.lower(), "❌ 未覆盖"))
    status = "✓" if trans != "❌ 未覆盖" else "✗"
    print(f"    {status} {key:<12} → {trans}")

# Test 3: 缓存机制
print("\n[Test 3] 缓存机制")
cache = _load_cache()
print(f"  缓存文件: {Path(__file__).parent / 'unsupervised_classification/RQ1/data/translation_cache.json'}")
print(f"  缓存条目数: {len(cache)}")
if cache:
    sample_cached = list(cache.items())[:3]
    print(f"  样本缓存:")
    for orig, trans in sample_cached:
        print(f"    {orig} → {trans}")

# Test 4: 翻译函数（仅静态词典，不调 API）
print("\n[Test 4] translate_entities 函数 (仅静态词典)")
test_entities = ["安倍", "統一教会", "カルト", "Pope", "unknown_entity_xyz"]
trans_result = translate_entities(
    test_entities,
    api_key=None,  # 不调 API，只用静态词典
    use_api=False
)
print(f"  输入: {test_entities}")
print(f"  输出:")
for ent, trans in trans_result.items():
    symbol = "✓" if trans != ent else "⚠"
    print(f"    {symbol} {ent:<20} → {trans}")

# Test 5: 双语标签格式化
print("\n[Test 5] bilingual_label 函数")
test_labels = ["安倍", "統一教会", "bishop", "神棍"]
for entity in test_labels:
    label = bilingual_label(
        entity,
        translation=trans_result.get(entity),
        max_cjk_len=10,
        max_en_len=20
    )
    # 用 repr 显示换行符
    display = repr(label)
    print(f"  {entity:<12} → {display}")

# Test 6: 检查 Gemini API 可用性（如有 API key）
print("\n[Test 6] Gemini API 可用性检查")
if gemini_key:
    try:
        from google import genai
        print("  ✓ google-genai 库已安装")
        client = genai.Client(api_key=gemini_key)
        print("  ✓ Gemini 客户端可初始化")
        print("  💡 API 集成就绪。运行完整管线时将使用 Gemini 进行翻译")
    except ImportError:
        print("  ⚠ google-genai 库未安装 (pip install google-genai)")
    except Exception as e:
        print(f"  ⚠ Gemini 客户端初始化失败: {e}")
else:
    print("  ℹ 未设置 GEMINI_API_KEY，使用本地缓存和静态词典")

# Test 7: 可视化函数导入
print("\n[Test 7] 可视化依赖检查")
try:
    from unsupervised_classification.RQ1.viz_utils import (
        setup_matplotlib,
        setup_cjk_font,
        get_cjk_font_prop
    )
    print("  ✓ viz_utils 核心函数可导入")
    font = setup_cjk_font()
    if font:
        print(f"  ✓ CJK 字体已检测: {font}")
    else:
        print(f"  ⚠ CJK 字体未检测，将使用默认字体")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")

print("\n" + "=" * 60)
print("验证完成！")
print("=" * 60)
print("\n推荐工作流：")
print("  1. 运行 Layer1+2: python unsupervised_classification/RQ1/target_extraction_v3.py --stage layer12")
print("  2. 查看结果:     python unsupervised_classification/RQ1/target_extraction_v3.py --stage viz")
print("  3. (可选) 运行 LLM: python unsupervised_classification/RQ1/target_extraction_v3.py --stage llm")
print("  4. 更新图表:     python unsupervised_classification/RQ1/target_extraction_v3.py --stage viz")
