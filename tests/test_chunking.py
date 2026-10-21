import random

import pytest

from src.ingest.chunking import Chunk, split_text


def test_offsets_refer_to_original_text_with_outer_whitespace():
    text = "  中文段落。\n\n第二段。  \n"

    assert split_text(text, chunk_size=30, chunk_overlap=0) == [
        Chunk(text="中文段落。\n\n第二段。", start=2, end=13)
    ]


def test_short_paragraphs_are_not_cut_to_fill_a_window():
    text = "第一段落。\n\n第二段落。\n\n第三段落。"

    assert [chunk.text for chunk in split_text(text, 9, 0)] == [
        "第一段落。", "第二段落。", "第三段落。"
    ]


def test_markdown_heading_starts_a_new_chunk_and_stays_with_its_content():
    text = "前言。\n# 新章节\n内容。"

    assert [chunk.text for chunk in split_text(text, 30, 10)] == [
        "前言。", "# 新章节\n内容。"
    ]


def test_markdown_table_rows_stay_whole():
    text = "| 项目 | 值 |\n| --- | --- |\n| 收入 | 100 |\n| 利润 | 20 |"

    assert [chunk.text for chunk in split_text(text, 27, 0)] == [
        "| 项目 | 值 |\n| --- | --- |", "| 收入 | 100 |\n| 利润 | 20 |"
    ]


def test_fenced_code_does_not_treat_comments_or_pipes_as_structure():
    text = "介绍。\n```python\n# 保留注释\nvalue = a | b\n```\n正文。"

    assert split_text(text, 80, 0) == [Chunk(text, 0, len(text))]


def test_fenced_code_with_blank_lines_remains_a_whole_unit():
    text = "AAAAAA\n```\nx = 1\n\nx = 2\n```\nBBBBBB"

    assert [chunk.text for chunk in split_text(text, 23, 0)] == [
        "AAAAAA", "```\nx = 1\n\nx = 2\n```", "BBBBBB"
    ]


@pytest.mark.parametrize("opening, invalid_close", [
    ("````", "```"), ("~~~~", "~~~"), ("```", "~~~"), ("~~~", "```"),
])
def test_fence_closing_marker_must_match_character_and_minimum_length(opening, invalid_close):
    code = f"{opening}lang\n# first\n{invalid_close}\n# second\n{opening}"
    text = code + "\n# outside\n正文。"

    assert [chunk.text for chunk in split_text(text, 100, 0)] == [
        code, "# outside\n正文。"
    ]


def test_a_fence_with_trailing_info_does_not_close_a_code_block():
    code = "```\n# first\n```python\n# second\n```"

    assert [chunk.text for chunk in split_text(code + "\n# outside", 100, 0)] == [
        code, "# outside"
    ]


def test_unclosed_indented_fence_extends_to_eof_with_original_offsets():
    text = "  ~~~python\r\n# inside\r\n\r\n| a | b |\r\n尾部\r\n"
    expected = "~~~python\r\n# inside\r\n\r\n| a | b |\r\n尾部"

    assert split_text(text, 100, 0) == [Chunk(expected, 2, len(text) - 2)]


def test_oversized_fenced_code_falls_back_to_overlapping_windows():
    assert split_text("```\nabcdefghij\n```", 8, 2) == [
        Chunk("```\nabcd", 0, 8),
        Chunk("cdefghij", 6, 14),
        Chunk("ij\n```", 12, 18),
    ]


def test_whitespace_only_windows_are_not_embedded():
    text = "A         B"

    assert split_text(text, 3, 1) == [
        Chunk("A  ", 0, 3), Chunk("  B", 8, 11)
    ]


def test_long_paragraph_uses_overlapping_character_windows():
    assert split_text("甲乙丙丁戊己庚辛壬癸", 4, 1) == [
        Chunk("甲乙丙丁", 0, 4),
        Chunk("丁戊己庚", 3, 7),
        Chunk("庚辛壬癸", 6, 10),
    ]


def test_structure_overlap_reuses_a_whole_paragraph():
    assert split_text("AAAA\n\nBBBB\n\nCCCC", 10, 4) == [
        Chunk("AAAA\n\nBBBB", 0, 10),
        Chunk("BBBB\n\nCCCC", 6, 16),
    ]


def test_structure_overlap_does_not_cut_a_paragraph_to_meet_the_budget():
    assert [chunk.text for chunk in split_text("AAAAA\n\nBBBBB\n\nCCCCC", 11, 3)] == [
        "AAAAA", "BBBBB", "CCCCC"
    ]


def test_overlap_is_dropped_when_the_next_whole_block_would_not_fit():
    assert [chunk.text for chunk in split_text("AA\n\nBB\n\nCCCCCC", 8, 6)] == [
        "AA\n\nBB", "CCCCCC"
    ]


def test_repeated_text_keeps_distinct_original_offsets():
    assert split_text(" 重复\n\n重复\n\n重复 ", 2, 0) == [
        Chunk("重复", 1, 3), Chunk("重复", 5, 7), Chunk("重复", 9, 11)
    ]


def test_consecutive_nonblank_lines_remain_one_paragraph():
    assert [chunk.text for chunk in split_text("第一行。\n第二行。\n\n第三段。", 11, 0)] == [
        "第一行。\n第二行。", "第三段。"
    ]


@pytest.mark.parametrize("text", ["", " \t\n", "\r\n  \t"])
def test_empty_or_whitespace_input_has_no_chunks(text):
    assert split_text(text, 4, 1) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (-1, 0), (4, -1)])
def test_invalid_chunk_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("正文", size, overlap)


@pytest.mark.parametrize("size, overlap, expected", [
    (3, 20, [Chunk("abc", 0, 3), Chunk("bcd", 1, 4), Chunk("cde", 2, 5)]),
    (1, 20, [Chunk("a", 0, 1), Chunk("b", 1, 2), Chunk("c", 2, 3),
             Chunk("d", 3, 4), Chunk("e", 4, 5)]),
])
def test_excessive_overlap_is_clamped_and_always_makes_progress(size, overlap, expected):
    assert split_text("abcde", size, overlap) == expected


@pytest.mark.parametrize("size", [1, 3, 8, 16, 50])
@pytest.mark.parametrize("overlap", [0, 1, 4, 100])
def test_chunks_are_bounded_deterministic_and_cover_all_nonwhitespace(size, overlap):
    rng = random.Random(42)
    documents = [
        " \r\n# 财报\r\n中文段落。\r\n\r\n| 年份 | 收入 |\r\n| 2025 | 100 |\r\n\r\n重复\n\n重复 ",
        "| 很长的表格单元格 | " + "中文数值123" * 10 + " |\n下一段。",
        "a | b\n--- | ---\n100 | 200",
        "开始\n```python\n# 注释\n\nvalue = a | b\n```\n# 章节\n正文。",
        "~~~\n# 长代码块\n" + "中文数值123" * 10 + "\n~~~",
        "".join(rng.choice("中文abc \t\n#|") for _ in range(250)),
    ]
    for text in documents:
        chunks = split_text(text, size, overlap)
        assert chunks == split_text(text, size, overlap)
        covered = set()
        for chunk in chunks:
            assert 0 <= chunk.start < chunk.end <= len(text)
            assert chunk.text == text[chunk.start:chunk.end]
            assert len(chunk.text) <= size
            assert chunk.text.strip()
            covered.update(range(chunk.start, chunk.end))
        assert all(index in covered for index, char in enumerate(text) if not char.isspace())
        for previous, current in zip(chunks, chunks[1:]):
            assert current.start > previous.start
            assert current.end > previous.end
            assert previous.end - current.start <= min(overlap, size - 1)
