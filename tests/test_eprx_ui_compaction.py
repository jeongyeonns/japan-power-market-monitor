import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_national_summary_and_kpi_cards_are_not_rendered():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    overview = _function(tree, "render_national_overview")
    overview_source = ast.get_source_segment(source, overview)
    called_names = {
        node.func.id
        for node in ast.walk(overview)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "render_national_summary" not in called_names
    assert "전국 1차 조정력 시장 핵심지표" not in overview_source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "render_national_summary"
        for node in tree.body
    )


def test_national_data_warning_is_after_tables_and_before_basis_explanation():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    overview_source = ast.get_source_segment(
        source, _function(tree, "render_national_overview")
    )

    regional_position = overview_source.index("render_national_regional_table(")
    detail_position = overview_source.index('target.subheader("전국 시간대별 상세 데이터")')
    warning_position = overview_source.rindex("render_excess_award_warning(")
    explanation_position = overview_source.index('target.expander("데이터 기준 설명")')
    assert regional_position < detail_position < warning_position < explanation_position


def test_metric_label_hierarchy_uses_theme_text_and_small_gray_detail():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_source = ast.get_source_segment(
        source, _function(tree, "render_hierarchical_metric_table")
    )

    assert "metric-label-main" in helper_source
    assert "metric-label-sub" in helper_source
    assert "color: inherit" in helper_source
    assert "color: #8a8f98" in helper_source
    assert "font-size: 0.75rem" in helper_source
    assert "escape(" in helper_source
